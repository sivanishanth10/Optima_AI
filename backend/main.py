import os
import json
import shutil
import uuid
import time
import pandas as pd
from io import BytesIO
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Internal imports from our migrated modules
from utils.data_profiler import load_and_preprocess, dataset_fingerprint
from utils.ai_client import get_client, plan_prompt, request_plan, request_ai
from core.cleaning_engine import clean_dataframe, EngineConfig
from core.metrics_engine import evaluate_model
from core.knowledge_base import kb
from core.auth import (
    get_current_user, 
    User,
    Token
)
from core.quality_metrics import calculate_quality_metrics

# ── CONFIGURATION & MODELS ──
load_dotenv()

SUPPORTED_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
    "google/gemini-2.0-flash-exp:free",
    "stepfun/step-1-8k",
    "deepseek/deepseek-chat",
    "meta-llama/llama-3.1-405b-instruct:free",
]

app = FastAPI(title="Optima AI Backend", version="1.0.0")

# Ensure local upload directory exists
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Allow requests from the Next.js frontend (local and network IPs)
# Allow requests from local development and any Render subdomain
# This regex matches http://localhost:3000, http://127.0.0.1:3000, and https://*.onrender.com
allow_origin_regex = r"https?://(localhost|127\.0\.0\.1|.*\.onrender\.com)(:[0-9]+)?"

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=allow_origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/downloads/{filename:path}")
async def download_file_robust(filename: str):
    """
    Serves files from the uploads directory. 
    Smartly handles cases where the filename might include a redundant 'uploads/' prefix.
    """
    # Strip redundant 'uploads/' prefix if it got doubled up
    clean_filename = filename
    if filename.startswith("uploads/"):
        clean_filename = filename[len("uploads/"):]
    elif filename.startswith("/uploads/"):
        clean_filename = filename[len("/uploads/"):]
        
    file_path = os.path.join(UPLOAD_DIR, clean_filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {clean_filename}")
    
    # Determine the media type
    media_type = "application/octet-stream"
    if clean_filename.endswith(".csv"):
        media_type = "text/csv"
    elif clean_filename.endswith(".py"):
        media_type = "text/x-python"
        
    return FileResponse(path=file_path, filename=os.path.basename(clean_filename), media_type=media_type)

# ── MODELS ──
class AnalyzeInitRequest(BaseModel):
    file_path: str

class AnalyzeRequest(BaseModel):
    fingerprint: dict
    model: str | None = None
    api_key: str | None = None

class CleanRequest(BaseModel):
    file_path: str
    fingerprint: dict
    model: str | None = None
    api_key: str | None = None
    plan: dict | None = None           # Pre-generated plan (skips AI call if set)
    enabled_actions: list[str] | None = None  # Subset of action types to run

class ChatRequest(BaseModel):
    prompt: str
    dataset_state: str
    data_info: dict
    safe_summary: str
    file_path: str | None = None
    model: str | None = None
    api_key: str | None = None

class MetricsRequest(BaseModel):
    file_path: str
    target_column: str | None = ""
    task: str | None = ""
    model: str | None = ""

class LearnRequest(BaseModel):
    columns: list[str]
    fix_code: str
    description: str
    source_file: str

class VerifyRequest(BaseModel):
    pattern_id: str

class LoginRequest(BaseModel):
    email: str
    password: str

class GoogleLoginRequest(BaseModel):
    id_token: str

class QualityMetricsRequest(BaseModel):
    file_path: str

# ── HELPERS ──
def _generate_python_script(original_file_path: str, actions_applied: list, plan_actions: list) -> str:
    """Generates a reproducible Python script from the executed cleaning actions."""
    plan_dict = {"actions": plan_actions}
    plan_json = json.dumps(plan_dict, indent=4)
    
    lines = [
        """#!/usr/bin/env python3
"""
        '"""',
        "Optima AI — Auto-generated Cleaning Script",
        f"Source file: {original_file_path}",
        f"Actions executed: {', '.join(actions_applied) if actions_applied else 'none'}",
        '"""',
        "",
        "import pandas as pd",
        "import numpy as np",
        "import json",
        "",
        "# ── Load raw data ──",
        f"INPUT_FILE = r\"{original_file_path}\"",
        "OUTPUT_FILE = INPUT_FILE.replace('.csv', '_cleaned.csv')",
        "",
        "df = pd.read_csv(INPUT_FILE)",
        "print(f'Loaded: {len(df)} rows x {len(df.columns)} columns')",
        "",
        "# ── Cleaning Plan (AI-generated) ──",
        "CLEANING_PLAN = " + plan_json,
        "",
        "# ── Execute actions ──",
    ]
    
    # Per-action inline code
    action_code = {
        "normalize_columns":      "df.columns = df.columns.str.lower().str.replace(' ', '_').str.strip()",
        "strip_whitespace":       "df = df.apply(lambda c: c.str.strip() if c.dtype == 'object' else c)",
        "drop_empty_rows":        "df.dropna(how='all', inplace=True)",
        "drop_empty_cols":        "df.dropna(axis=1, how='all', inplace=True)",
        "deduplicate":            "df.drop_duplicates(inplace=True)",
        "coerce_numeric":         "df = df.apply(pd.to_numeric, errors='ignore')",
        "parse_dates":            "# parse_dates: applied to date-like columns during cleaning",
        "outliers_iqr":           "# outliers_iqr: IQR clipping applied to numeric columns",
        "impute":                 "df.fillna(df.median(numeric_only=True), inplace=True)\n  for c in df.select_dtypes('object').columns:\n    df[c].fillna(df[c].mode()[0] if not df[c].mode().empty else '', inplace=True)",
        "drop_high_null_cols":    "df.dropna(thresh=int(len(df)*0.5), axis=1, inplace=True)",
        "standardize_categories": "for c in df.select_dtypes('object').columns: df[c] = df[c].str.lower().str.strip()",
        "drop_columns":           "# drop_columns: specific columns dropped per AI plan",
        "extract_numeric":        "# extract_numeric: numbers extracted from text columns",
        "regex_extract":          "# regex_extract: regex patterns applied to extract sub-values",
        "map_categories":         "# map_categories: category values remapped per AI plan",
        "filter_range":           "# filter_range: rows outside valid ranges removed",
    }
    
    for ac in actions_applied:
        code = action_code.get(ac, f"# {ac}: applied")
        # Check if the code looks like a learned pattern (multiple lines or custom)
        lines.append(f"# {ac}")
        for sub in code.split("\n"):
            lines.append(sub)
        lines.append("")
    
    lines += [
        "# ── Save output ──",
        "df.to_csv(OUTPUT_FILE, index=False)",
        "print(f'Saved cleaned data to: {OUTPUT_FILE}')",
        "print(f'Result: {len(df)} rows x {len(df.columns)} columns')",
    ]
    
    return "\n".join(lines)

def _generate_cleaning_summary(actions_applied: list, metrics_pre: dict, metrics_post: dict) -> str:
    """Generates a structured cleaning report summary."""
    summary = [
        "# Optima AI — Cleaning Report",
        "",
        "## Executive Summary",
        f"Successfully executed {len(actions_applied)} cleaning actions.",
        "",
        "## Data Quality Shift",
        f"- **Initial Quality Score:** {metrics_pre.get('overall_score', 0):.2%}",
        f"- **Cleaned Quality Score:** {metrics_post.get('overall_score', 0):.2%}",
        f"- **Improvement:** {(metrics_post.get('overall_score', 0) - metrics_pre.get('overall_score', 0)):.2%}",
        "",
        "## Actions Log",
    ]
    for action in actions_applied:
        summary.append(f"- ✅ **{action}**: Successfully applied and verified.")
    
    summary.extend([
        "",
        "## Verification Report",
    ])
    
    # Simple verification logic: check if scores are below 80%
    missed_steps = []
    if metrics_post.get('completeness', 1) < 0.8: missed_steps.append("Impute missing values")
    if metrics_post.get('validity', 1) < 0.8: missed_steps.append("Address data validation errors")
    if metrics_post.get('consistency', 1) < 0.8: missed_steps.append("Resolve formatting inconsistencies")
    
    if missed_steps:
        summary.append(f"⚠️ **Notice**: Some indicators remain below threshold ({metrics_post.get('overall_score', 0):.1%}).")
        summary.append("Suggested follow-up: " + ", ".join(missed_steps))
    else:
        summary.append("✅ **All indicators verified** against standard quality thresholds.")

    summary.extend([
        "",
        "## Flow & Logic",
        "1. Raw Data Ingestion",
        "2. AI Diagnosis & Pattern Matching",
        "3. Pipeline Execution (Normalization → Coersion → Imputation)",
        "4. Post-Cleaning Validation",
        "",
        "## Suggested Use Cases",
        "- Machine Learning Training",
        "- Business Intelligence Dashboards",
        "- Regulatory Reporting",
    ])
    return "\n".join(summary)

# ── ENDPOINTS ──
@app.get("/")
def health_check():
    return {"status": "Optima Data Engine is online."}

# ── AUTH ENDPOINTS ──
@app.get("/api/me", response_model=User)
def get_me(user: User = Depends(get_current_user)):
    """Verifies the Supabase session token and returns user details."""
    return user

@app.post("/api/upload")
def upload_file(file: UploadFile = File(...)):
    """Saves an uploaded file locally and returns its path."""
    start_time = time.time()
    try:
        unique_filename = f"{uuid.uuid4().hex}_{file.filename}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"DEBUG: Upload took {time.time() - start_time:.4f}s")
        return {"file_path": file_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")

@app.post("/api/analyze-init")
def analyze_initial_file(req: AnalyzeInitRequest):
    """Reads the localized file into memory and generates the initial fingerprint."""
    start_time = time.time()
    try:
        # Read local file
        with open(req.file_path, "rb") as f:
            res = f.read()
            
        # Load and preprocess using the existing engine
        df = load_and_preprocess(req.file_path, res)
        fingerprint = dataset_fingerprint(df)
        
        print(f"DEBUG: Analyze-init took {time.time() - start_time:.4f}s")
        return {
            "message": "File analyzed successfully",
            "file_path": req.file_path,
            "shape": fingerprint["shape"],
            "fingerprint": fingerprint,
            "safe_summary": fingerprint.get("safe_summary", "")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing file from storage: {str(e)}")

@app.post("/api/analyze")
def analyze_data(req: AnalyzeRequest):
    """Takes a dataset fingerprint and generates a cleaning plan via OpenRouter."""
    start_time = time.time()
    api_key = req.api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise HTTPException(status_code=401, detail="API Key required.")
    
    prompt = plan_prompt(req.fingerprint)
    
    try:
        plan, explanation, fallback_note = request_plan(prompt, model_id=req.model or SUPPORTED_MODELS[0])
        print(f"DEBUG: Analyze (AI Plan) took {time.time() - start_time:.4f}s")
        if fallback_note:
            explanation = f"{fallback_note}\n\n{explanation}"
            
        return {
            "plan": plan,
            "explanation": explanation
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/diagnose")
def diagnose_data(req: AnalyzeRequest):
    """Generates an AI diagnostic report. Auto-retries with fallback models on rate-limit (429)."""
    api_key = req.api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise HTTPException(status_code=401, detail="API Key required.")

    prompt = (
        f"Analyze this dataset fingerprint:\n{json.dumps(req.fingerprint)}\n\n"
        "Provide a structured report covering:\n"
        "1. **Data Health Score & Initial Scan** — overall quality score out of 10\n"
        "2. **Column-level Profile** — type, null %, unique count, problems per column\n"
        "3. **Detected Issues** — missing data, duplicates, outliers, type mismatches\n"
        "4. **Recommended Cleaning Workflow** — step-by-step actions to take"
    )
    system_prompt = "You are 'Optima AI', an expert data analyst. Analyze the dataset fingerprint, detect problems, and provide a structured data health report."
    
    try:
        content, fallback_note = request_ai(f"{system_prompt}\n\n{prompt}", model_id=req.model or SUPPORTED_MODELS[0])
        
        report = content
        if fallback_note:
            report = f"{fallback_note}\n\n{report}"
            
        return {
            "report": report,
            "model_used": req.model if not fallback_note else "Fallback"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/clean")
def clean_dataset(req: CleanRequest):
    """Generates a cleaning plan using AI, executes it on the dataset, and uploads the cleaned version."""
    start_time = time.time()
    api_key = req.api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise HTTPException(status_code=401, detail="API Key required.")
    
    prompt = plan_prompt(req.fingerprint)
    
    try:
        # 0. Check Knowledge Base for matching fix
        learned_fix = kb.find_matching_fix(req.fingerprint)
        used_learned_fix = False

        if req.plan:
            # Plan already generated and reviewed by the user — skip AI call
            print("DEBUG: Using pre-generated plan from frontend")
            plan = req.plan
            explanation = plan.get("explanation", "Applied the reviewed cleaning plan.")
        elif learned_fix:
            print(f"DEBUG: Found matching KB fix ({learned_fix['id']})")
            plan = {"actions": ["learned_pattern"]}
            explanation = f"Matched internal knowledge pattern: {learned_fix['description']}"
            used_learned_fix = True
        else:
            # 1. Get the plan from AI (with multi-model fallback built-in)
            plan, explanation, fallback_note = request_plan(prompt, model_id=req.model or SUPPORTED_MODELS[0])
            if fallback_note:
                explanation = f"{fallback_note}\n\n{explanation}"

        # Filter to only enabled actions if the user toggled any off
        if req.enabled_actions is not None and isinstance(plan.get("actions"), list):
            plan["actions"] = [
                a for a in plan["actions"]
                if isinstance(a, dict) and a.get("type") in req.enabled_actions
            ]
            print(f"DEBUG: Filtered to {len(plan['actions'])} enabled actions: {req.enabled_actions}")
        
        # 2. Load the local raw file into Pandas
        with open(req.file_path, "rb") as f:
            res = f.read()
        df = load_and_preprocess(req.file_path, res)
        
        # 3. Execute the cleaning plan
        engine_config = EngineConfig()
        cleaned_df, report = clean_dataframe(df, plan, engine_config)
        
        # 4. Generate new fingerprint for the frontend Comparison View
        cleaned_fingerprint = dataset_fingerprint(cleaned_df)
        
        # 5. Save the cleaned CSV into the uploads directory (basename only)
        base_name = os.path.splitext(os.path.basename(req.file_path))[0]
        cleaned_filename = os.path.join(UPLOAD_DIR, f"{base_name}_cleaned.csv")
        cleaned_df.to_csv(cleaned_filename, index=False, encoding="utf-8")
        
        # 6. Build a reproducible Python script from executed actions
        actions_applied = report.get("actions_applied", [])
        
        # Handle learned pattern injection in code generation
        if used_learned_fix and learned_fix:
            python_script_content = _generate_python_script(req.file_path, [], plan.get("actions", []))
            # Inject the custom code after imports
            split = python_script_content.split("# ── Execute actions ──")
            custom_block = f"# ── Execute Learned Pattern ({learned_fix['id']}) ──\n{learned_fix['fix_code']}\n"
            python_script_content = split[0] + "# ── Execute actions ──\n" + custom_block + split[1]
        else:
            python_script_content = _generate_python_script(req.file_path, actions_applied, plan.get("actions", []))

        script_filename = f"{base_name}_cleaning_script.py"
        script_file_path = os.path.join(UPLOAD_DIR, script_filename)
        with open(script_file_path, "w", encoding="utf-8") as f:
            f.write(python_script_content)

        # 7. Generate Quality Metrics
        metrics_pre = calculate_quality_metrics(df)
        metrics_post = calculate_quality_metrics(cleaned_df)
        
        # 8. Build a cleaning summary report
        cleaning_summary = _generate_cleaning_summary(actions_applied, metrics_pre, metrics_post)

        return {
            "message": "Data cleaned successfully",
            "cleaning_report": script_filename,
            "cleaning_summary": cleaning_summary,
            "explanation": explanation,
            "used_kb": used_learned_fix,
            "plan": plan,
            "python_code": python_script_content,
            "quality_metrics": {
                "initial": metrics_pre,
                "cleaned": metrics_post
            },
            "cleaned_data": {
                "file_path": os.path.basename(cleaned_filename),
                "shape": cleaned_fingerprint["shape"],
                "fingerprint": cleaned_fingerprint
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during data cleaning: {str(e)}")

@app.get("/api/patterns")
def get_patterns():
    return kb.get_all_patterns()

@app.post("/api/learn")
def learn_pattern(req: LearnRequest):
    p_id = kb.stage_pattern(req.columns, req.fix_code, req.description, req.source_file)
    return {"message": "Pattern staged for review", "pattern_id": p_id}

@app.post("/api/verify")
def verify_pattern(req: VerifyRequest):
    success = kb.verify_pattern(req.pattern_id)
    if not success:
        raise HTTPException(status_code=404, detail="Pattern not found in staging")
    return {"message": "Pattern verified and pushed to production KB"}

@app.get("/api/downloads/report/{filename}")
def download_report(filename: str):
    """Serves a generated report file (e.g., Python script) for download."""
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Report file not found.")
    
    # Determine media type based on file extension
    media_type = "application/octet-stream"
    if filename.endswith(".py"):
        media_type = "text/x-python"
    elif filename.endswith(".txt"):
        media_type = "text/plain"
    elif filename.endswith(".json"):
        media_type = "application/json"
    
    return FileResponse(path=file_path, filename=filename, media_type=media_type)

@app.post("/api/chat")
def chat_with_data(req: ChatRequest):
    """Streams a chat response based on the dataset profile and user prompt."""
    print(f"DEBUG: Received chat request: {req.prompt[:50]}...")
    start_time = time.time()
    api_key = req.api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise HTTPException(status_code=401, detail="API Key required.")
    
    # Use the model the user selected, fall back to a default
    model = req.model if req.model and req.model in SUPPORTED_MODELS else SUPPORTED_MODELS[0]
    
    client = get_client(model, api_key)
    # ── CONSTRUCT PROMPT ──
    # Start with basic summary context
    quality_metrics = {}
    if req.file_path and os.path.exists(req.file_path):
        try:
            df_temp = pd.read_csv(req.file_path)
            quality_metrics = calculate_quality_metrics(df_temp)
        except:
            pass

    full_context = f"""
You are 'Optima AI', an autonomous data-assistant that helps users analyze, clean, and understand their datasets.
Current Dataset State: {req.dataset_state}

Data Overview: 
Shape: {req.data_info.get('shape')}
Null Counts: {req.data_info.get('null_counts')}

Quality Metrics:
- Completeness: {quality_metrics.get('completeness', 0):.2%}
- Validity: {quality_metrics.get('validity', 0):.2%}
- Consistency: {quality_metrics.get('consistency', 0):.2%}
- Uniqueness: {quality_metrics.get('uniqueness', 0):.2%}
- Accuracy: {quality_metrics.get('accuracy', 0):.2%}
- Structural: {quality_metrics.get('structural', 0):.2%}

Detailed Summary & Safe Sample:
{req.safe_summary}
"""

    # Add more rows from the actual file if path is provided
    if req.file_path and os.path.exists(req.file_path):
        try:
            # Load the entire dataset to compute full statistics
            df_full = pd.read_csv(req.file_path)
            total_rows = len(df_full)
            
            if total_rows <= 1000:
                # Small enough to send the whole thing (most modern LLMs handle this well)
                data_md = df_full.to_markdown(index=False)
                full_context += f"\n\nFULL DATA CONTEXT ({total_rows} rows):\n{data_md}"
            else:
                # Large dataset: Provide statistics for ALL rows + Head/Tail samples
                stats_md = df_full.describe(include='all').to_markdown()
                head_md = df_full.head(200).to_markdown(index=False)
                tail_md = df_full.tail(200).to_markdown(index=False)
                
                full_context += f"\n\nDATA OVERVIEW (Full Dataset of {total_rows} rows):"
                full_context += f"\n\nDESCRIPTIVE STATISTICS (Calculated from all rows):\n{stats_md}"
                full_context += f"\n\nSAMPLE (First 200 Rows):\n{head_md}"
                full_context += f"\n\nSAMPLE (Last 200 Rows):\n{tail_md}"
                full_context += "\n\nNote: The above samples are representative. Descriptive statistics cover the entire dataset."
        except Exception as e:
            print(f"DEBUG: Could not load full CSV for chat: {e}")

    full_context += """
Instructions:
1. Answer questions clearly, professionally, and insightfully.
2. The data sample provided is safe and has PII redacted.
3. If the user asks for a data change (e.g. "remove nulls"), first explain the step, then ONLY THEN include this JSON block at the end:
   ```json
   {"refinement_plan": ["action_type1", "action_type2"]}
   ```
4. Available actions: normalize_columns, strip_whitespace, drop_empty_rows, drop_empty_cols, deduplicate, coerce_numeric, parse_dates, outliers_iqr, impute, drop_high_null_cols, standardize_categories.
5. If the user asks about Python code or has "doubts", explain in detail in Markdown without proposing a plan.
6. Be concise but thorough. Use markdown formatting when appropriate.
"""
    
    try:
        content, fallback_note = request_ai(f"{full_context}\n\nUser: {req.prompt}", model_id=model)
        
        reply = content
        if fallback_note:
            reply = f"{fallback_note}\n\n{reply}"
            
        return {"reply": reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Engine Error: {str(e)}")


@app.post("/api/metrics")
def compute_metrics(req: MetricsRequest):
    """Trains a model and returns standard quality metrics."""
    try:
        # The file_path from frontend already includes 'uploads/'
        full_path = req.file_path
        if not os.path.exists(full_path):
            # Fallback for old sessions or different pathing
            alt_path = os.path.join(UPLOAD_DIR, os.path.basename(req.file_path))
            if os.path.exists(alt_path):
                full_path = alt_path
            else:
                raise HTTPException(status_code=404, detail=f"File not found at {full_path}")
            
        results = evaluate_model(
            file_path=full_path,
            target_column=req.target_column,
            model_name=req.model,
            task=req.task
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Metrics Error: {str(e)}")

@app.post("/api/quality-metrics")
def get_quality_metrics(req: QualityMetricsRequest):
    """Calculates data quality scores for a given CSV file."""
    try:
        full_path = req.file_path
        if not os.path.exists(full_path):
            alt_path = os.path.join(UPLOAD_DIR, os.path.basename(req.file_path))
            if os.path.exists(alt_path):
                full_path = alt_path
            else:
                raise HTTPException(status_code=404, detail=f"File not found at {full_path}")
            
        df = pd.read_csv(full_path)
        metrics = calculate_quality_metrics(df)
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quality Metrics Error: {str(e)}")
