import json
import re
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def map_model_id(model_id: str) -> str:
    mapping = {
        "llama-3.3-70b-versatile": "meta-llama/llama-3.3-70b-instruct",
        "llama-3.1-8b-instant": "meta-llama/llama-3.1-8b-instruct",
        "mixtral-8x7b-32768": "mistralai/mixtral-8x7b-instruct",
        "llama3-70b-8192": "meta-llama/llama-3-70b-instruct",
        "llama3-8b-8192": "meta-llama/llama-3-8b-instruct",
        "gemma2-9b-it": "google/gemma-2-9b-it",
    }
    return mapping.get(model_id, model_id)

def get_client(model_id: str, api_key: str = None) -> OpenAI:
    """
    Returns an OpenAI-compatible client pointing to OpenRouter.
    """
    base_url = "https://openrouter.ai/api/v1"
    key = api_key or os.getenv("OPENROUTER_API_KEY")
    
    return OpenAI(
        base_url=base_url,
        api_key=(key or "").strip()
    )

def plan_prompt(fingerprint: dict, max_steps: int = 15) -> str:
    """
    Constructs the specialized system instruction prompt for "DataClean-Agent",
    modeled after Julius AI's methodology.
    """
    return f"""
You are "DataClean-Agent", an autonomous data-analysis and cleaning agent modeled after Julius AI's real-world data-cleaning workflow. Your task is to analyze, detect, clean, and describe dataset issues exactly as Julius AI does.

========================
AGENT RULES & CONSTRAINTS
========================
- Always analyze the data before cleaning.
- Always produce a reversible cleaning plan.
- Never remove data without justification.
- Assume user wants "Julius-like full cleaning" unless specified.
- Be transparent: explain each transformation.
- Use multi-step reasoning.
- Never hallucinate data; operate strictly on provided tables.

========================
CORE BEHAVIOR MODEL
========================
Follow Julius AI's documented cleaning workflow:
1. Perform an automatic initial scan:
   - Detect inconsistent formatting, missing entries, duplicate rows.
   - Identify dirty columns and potential schema problems.

2. Generate a complete column-level profile:
   - Infer column types (numeric, categorical, datetime, boolean, text).
   - Identify pattern consistency, null counts, unique counts.

3. Detect problems:
   - Missing data patterns, type mismatches.
   - Category inconsistencies (e.g. "US", "U.S.A.", "United States").
   - Duplicates, outliers, formatting inconsistencies.

========================
CLEANING WORKFLOW
========================
Follow Julius AI's multi-step cleaning logic:
1. Fix formatting (standardize case, whitespace).
2. Drop unneeded/leftover index columns using "drop_columns".
3. Extract precise numbers from strings (e.g. "8GB" -> 8, "1.2kg" -> 1.2) using "extract_numeric".
4. Extract advanced features (e.g. Parse Memory into SSD/HDD, extract CPU/GPU brands) using "regex_extract" with "pattern" and a "new_columns" list.
5. Standardize categories & OS names using "map_categories" with a "mapping" dict {{"TargetName": ["variant1", "variant2"]}}.
6. Handle missing values (impute median/mode or drop responsibly).
7. Remove exact and near duplicates.
8. Remove impossible outliers strictly using "filter_range" (e.g., negative prices, impossible weights) or "outliers_iqr".
9. Type correction exactly.

========================
OUTPUT & EXPLANATION REQUIREMENTS
========================
Your outputs must include MULTIPLE PARTS:

A. A structured JSON cleaning plan (in a markdown ```json block):
   - You MUST provide a valid JSON object with an "actions" key.
   - Each action in "actions" MUST have a "type" string from the allowed list: 
     "normalize_columns", "strip_whitespace", "drop_empty_rows", "drop_empty_cols", "deduplicate", "coerce_numeric", "parse_dates", "standardize_categories", "outliers_iqr", "drop_high_null_cols", "impute", "drop_columns", "extract_numeric", "regex_extract", "map_categories", "filter_range".
   - Each action should include a "params" object if applicable (e.g. "columns", "mapping", "pattern").
   - DO NOT omit the JSON block. It is CRITICAL for the pipeline.

B. A natural-language explanation (Julius-style):
   - Describe what was wrong in a professional, concise manner.
   - Explain cleaning choices clearly and simply.
   - Summarize the improvements made.

C. Expected result summary text.

Dataset Fingerprint:
{json.dumps(fingerprint)}
"""

def _robust_parse_json(raw: str) -> dict:
    """
    Attempt to parse a JSON string produced by an LLM, applying progressive
    auto-fix strategies.
    """
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    cleaned = re.sub(r",\s*([}\]])", r"\1", raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    cleaned2 = re.sub(r"(?<![\\])'", '"', cleaned)
    cleaned2 = cleaned2.replace("True", "true").replace("False", "false").replace("None", "null")
    try:
        return json.loads(cleaned2)
    except json.JSONDecodeError:
        pass

    cleaned3 = re.sub(r'(?<=")([^"\\]*(?:\\.[^"\\]*)*)"', 
                      lambda m: m.group(0).replace('\n', ' ').replace('\r', ' '), 
                      cleaned2)
    try:
        return json.loads(cleaned3)
    except json.JSONDecodeError as e:
        raise ValueError(f"Could not repair malformed JSON from AI: {e}") from e

def request_ai(prompt: str, model_id: str = "llama-3.3-70b-versatile") -> tuple[str, str | None]:
    """
    Generic AI request using OpenRouter.
    Returns: (content_str, fallback_note)
    """
    model_id = map_model_id(model_id)
    fallback_chain = [
        model_id,
        "meta-llama/llama-3.3-70b-instruct",
        "google/gemini-2.0-flash-exp:free",
        "meta-llama/llama-3.1-8b-instruct",
        "deepseek/deepseek-chat",     
        "mistralai/mixtral-8x7b-instruct",  
    ]
    
    unique_models = []
    for m in fallback_chain:
        if m and m not in unique_models:
            unique_models.append(m)

    last_error = "Unknown error"
    used_fallback = False

    for current_model in unique_models:
        try:
            if current_model != model_id:
                used_fallback = True
                print(f"DEBUG: Falling back to model: {current_model}")

            client = get_client(current_model)
            resp = client.chat.completions.create(
                model=current_model,
                messages=[{"role": "user", "content": prompt}],
                timeout=30, # Prevent hanging
            )
            content = resp.choices[0].message.content.strip()
            
            fallback_note = None
            if used_fallback:
                fallback_note = f"Note: Selected model ({model_id}) failed. Using fallback: {current_model}."

            return content, fallback_note

        except Exception as e:
            err_str = str(e)
            last_error = err_str
            print(f"DEBUG [request_ai] {current_model} failed: {err_str[:200]}")
            continue

    raise RuntimeError(f"All AI models failed. Last error: {last_error}")

def request_plan(prompt: str, model_id: str = "llama-3.3-70b-versatile") -> tuple[dict, str, str | None]:
    """
    Specialized for JSON plans. Returns: (plan_dict, explanation_str, fallback_note)
    """
    # Use the generic request_ai for the actual LLM call
    full_content, fallback_note = request_ai(prompt, model_id)

    # Parsing logic remains the same
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", full_content, re.DOTALL)
    if json_match:
        plan_str = json_match.group(1).strip()
    elif full_content.startswith("{") and full_content.endswith("}"):
        plan_str = full_content
    else:
        # If no JSON block, we might need to retry with a different model if possible,
        # but for now we'll just fail cleanly.
        raise ValueError("No valid JSON cleaning plan found in AI response.")

    plan = _robust_parse_json(plan_str)

    if isinstance(plan, dict) and "actions" in plan and isinstance(plan["actions"], list):
        plan["actions"] = [
            a for a in plan["actions"]
            if isinstance(a, dict) and isinstance(a.get("type"), str) and a["type"].strip()
        ]

    explanation = re.sub(r"```(?:json)?\s*\{.*?\}\s*```", "", full_content, flags=re.DOTALL).strip()
    if not explanation:
        explanation = "Cleaned the dataset based on the generated plan."

    return plan, explanation, fallback_note
