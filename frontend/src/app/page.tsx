"use client";

/**
 * page.tsx — The Root Application Entry Point
 *
 * PURPOSE: This file is the "brain" of the app. It:
 *   1. Holds ALL shared state (datasets, messages, UI toggles)
 *   2. Contains ALL API call functions
 *   3. Composes (assembles) all the components into one layout
 *
 * CONCEPTS USED:
 * - "Lifting state up": State like `messages`, `activeDataset`, and `isSidebarOpen`
 *   is defined HERE (the parent) and passed DOWN to child components as props.
 *   This makes the parent the single "source of truth" — multiple children
 *   can read from and react to the same state.
 *
 * - Props as callbacks: Functions like `handleFileUpload` are defined here and
 *   passed to child components. When a child calls `onFileSelect(...)`, it
 *   is actually calling `handleFileUpload` from this parent. This lets children
 *   trigger actions without knowing the implementation details.
 *
 * - Conditional rendering: We show LandingScreen OR WorkflowLayout depending
 *   on whether `activeDataset` is set. This is the core navigation pattern
 *   for this single-page application.
 *
 * HOW API CALLS WORK:
 * - `fetch()` is the browser's built-in HTTP client. We use it to call our
 *   FastAPI backend at `http://localhost:8000`.
 * - `await` pauses execution until the Promise resolves (the server responds).
 * - We wrap calls in `try/catch` to handle network errors gracefully.
 */

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import Sidebar, { SessionRecord } from "@/components/Sidebar";
import Header from "@/components/Header";
import LandingScreen from "@/components/LandingScreen";
import ChatPanel from "@/components/ChatPanel";
import DataPanel from "@/components/DataPanel";
import WorkflowLayout from "@/components/WorkflowLayout";
import KnowledgeBasePane from "@/components/KnowledgeBasePane";
import { useRole } from "@/lib/hooks";
import * as api from "@/lib/api";
import { DatasetFingerprint, TabId } from "@/lib/schemas";
import { supabase } from "@/lib/supabase";
import { AuthChangeEvent, Session } from "@supabase/supabase-js";

// ── Type definitions ──
type Message = { role: "user" | "assistant"; content: string };
type Dataset = {
  fileName: string;
  filePath: string;
  shape: [number, number];
  fingerprint: any;
};

// ── localStorage helpers ──
// These keep session history in the browser across page refreshes.
const STORAGE_KEY = "optima_sessions";

function loadSessions(): SessionRecord[] {
  // typeof window check prevents SSR crash (Next.js renders on server too)
  if (typeof window === "undefined") return [];
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "[]");
  } catch {
    return [];
  }
}

function saveSessions(sessions: SessionRecord[]): void {
  if (typeof window === "undefined") return;
  // Keep only the last 20 sessions to avoid filling up localStorage
  localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions.slice(0, 20)));
}

export default function Home() {
  const router = useRouter();
  // ── Sidebar State ──
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);

  // ── Session History State ──
  // Loaded from localStorage on mount, saved whenever a new dataset is uploaded.
  const [sessions, setSessions] = useState<SessionRecord[]>([]);
  const [activeSessionId, setActiveSessionIdState] = useState<string | null>(null);

  // Wrapper to also persist to sessionStorage
  const setActiveSessionId = (id: string | null) => {
    setActiveSessionIdState(id);
    if (id) sessionStorage.setItem("optima_active_session_id", id);
    else sessionStorage.removeItem("optima_active_session_id");
  };

  const [isInitializing, setIsInitializing] = useState(true);

  // Load sessions from localStorage on first render (client-side only)
  useEffect(() => {
    const loaded = loadSessions();
    setSessions(loaded);
    
    // Check if we should restore the last active session
    const savedActiveId = sessionStorage.getItem("optima_active_session_id");
    if (savedActiveId) {
      const session = loaded.find(s => s.id === savedActiveId);
      if (session) {
        // Trigger restoration in the next frame to avoid React strict mode issues
        setTimeout(() => handleLoadSession(session), 0);
      }
    }
    setIsInitializing(false);
  }, []);

  // ── Upload & Dataset State ──
  const [uploading, setUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<{ msg: string; isError: boolean } | null>(null);
  const [activeDataset, setActiveDataset] = useState<Dataset | null>(null);

  // ── Chat State ──
  const [messages, setMessages] = useState<Message[]>([]);
  const [chatInput, setChatInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);

  const { copy } = useRole();

  // ── Workflow State ──
  const [activeTab, setActiveTab] = useState<TabId>("diagnostics");
  const [activeLeftView, setActiveLeftView] = useState<"chat" | "kb">("chat");
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [diagnosticReport, setDiagnosticReport] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [cleanedDataset, setCleanedDataset] = useState<any | null>(null);
  const [isCleaning, setIsCleaning] = useState(false);
  const [selectedModel, setSelectedModel] = useState("llama-3.3-70b-versatile");
  const [theme, setTheme] = useState<"dark" | "light">("dark");
  // Stage 1 state: plan generated but not yet executed
  const [generatedPlan, setGeneratedPlan] = useState<{ plan: any; explanation: string } | null>(null);
  const [isPlanLoading, setIsPlanLoading] = useState(false);

  // ── Quality Metrics State ──
  const [qualityMetricsPre, setQualityMetricsPre] = useState<any | null>(null);
  const [qualityMetricsPost, setQualityMetricsPost] = useState<any | null>(null);
  const [cleaningSummary, setCleaningSummary] = useState<string | null>(null);

  // ── Auth & User State ──
  const [user, setUser] = useState<any>(null);

  // ── Auth Guard ──
  useEffect(() => {
    // Initial check
    const checkUser = async () => {
      try {
        const { data: { session }, error } = await supabase.auth.getSession();
        
        // Handle explicit auth errors (like invalid refresh tokens)
        if (error) {
          console.warn("Auth session error, signing out:", error.message);
          await supabase.auth.signOut();
          router.push("/login");
          return;
        }

        if (!session) {
          router.push("/login");
        } else {
          localStorage.setItem("optima_token", session.access_token);
          setUser(session.user);
        }
      } catch (err) {
        console.error("Auth initialization failed:", err);
        await supabase.auth.signOut();
        router.push("/login");
      }
    };
    checkUser();

    // Listen for changes
    const { data: { subscription } } = supabase.auth.onAuthStateChange((_event: AuthChangeEvent, session: Session | null) => {
      if (!session) {
        localStorage.removeItem("optima_token");
        setUser(null);
        router.push("/login");
      } else {
        localStorage.setItem("optima_token", session.access_token);
        setUser(session.user);
      }
    });

    return () => subscription.unsubscribe();
  }, [router]);

  const handleLogout = async () => {
    await supabase.auth.signOut();
    localStorage.removeItem("optima_token");
    sessionStorage.removeItem("optima_active_session_id");
    router.push("/login");
  };

  // ── Close sidebar on mobile when dataset loads ──
  useEffect(() => {
    if (activeDataset && window.innerWidth < 768) {
      setIsSidebarOpen(false);
    }
  }, [activeDataset]);

  // ── Auto-save Session State ──
  // Watches for changes and continuously updates the active session in localStorage
  useEffect(() => {
    if (!activeSessionId || !activeDataset) return;
    
    setSessions((prev) => {
      const idx = prev.findIndex(s => s.id === activeSessionId);
      if (idx === -1) return prev;
      
      const updatedSessions = [...prev];
      updatedSessions[idx] = {
        ...updatedSessions[idx],
        filePath: activeDataset.filePath,
        messages,
        diagnosticReport,
        cleanedDataset,
        qualityMetricsPre,
        qualityMetricsPost,
        cleaningSummary,
      };
      
      saveSessions(updatedSessions);
      return updatedSessions;
    });
  }, [activeSessionId, activeDataset, messages, diagnosticReport, cleanedDataset]);

  // ────────────────────────────────────────────
  // SESSION MANAGEMENT
  // ────────────────────────────────────────────

  // Save a new session to history after a successful upload
  const saveSession = (dataset: Dataset, sessionId: string) => {
    const newSession: SessionRecord = {
      id: sessionId,
      fileName: dataset.fileName,
      filePath: dataset.filePath,
      shape: dataset.shape,
      fingerprint: dataset.fingerprint,
      createdAt: new Date().toISOString(),
    };
    // Prepend new session (most recent first), remove duplicates by filename
    const updated = [
      newSession,
      ...sessions.filter((s) => s.fileName !== dataset.fileName),
    ];
    setSessions(updated);
    saveSessions(updated);
  };

  // "New Session" button — resets all state and goes back to landing screen
  const handleNewSession = () => {
    setActiveDataset(null);
    setCleanedDataset(null);
    setGeneratedPlan(null);
    setDiagnosticReport(null);
    setMessages([]);
    setActiveTab("diagnostics");
    setUploadStatus(null);
    setChatInput("");
    setActiveSessionId(null);
  };

  // Load a past session from the history list in the sidebar
  const handleLoadSession = (session: SessionRecord) => {
    // Reconstruct the dataset object from the stored session
    const dataset: Dataset = {
      fileName: session.fileName,
      filePath: session.filePath || "",
      shape: session.shape,
      fingerprint: session.fingerprint,
    };
    setActiveDataset(dataset);
    setActiveSessionId(session.id);
    
    // Restore chat and panels from the saved session
    if (session.messages && session.messages.length > 0) {
      setMessages(session.messages);
    } else {
      setMessages([
        {
          role: "assistant",
          content: `📂 Loaded session: **${session.fileName}** (${session.shape[0].toLocaleString()} rows × ${session.shape[1]} cols). Note: the cleaned file may need to be re-processed if the backend was restarted.`,
        },
      ]);
    }
    
    setCleanedDataset(session.cleanedDataset || null);
    setDiagnosticReport(session.diagnosticReport || null);
    setQualityMetricsPre((session as any).qualityMetricsPre || null);
    setQualityMetricsPost((session as any).qualityMetricsPost || null);
    setCleaningSummary((session as any).cleaningSummary || null);
    
    // Auto-switch to the most relevant tab
    if (session.cleanedDataset) {
      setActiveTab("dashboard");
    } else if (session.diagnosticReport) {
      setActiveTab("diagnostics");
    } else {
      setActiveTab("diagnostics");
    }
    
    if (window.innerWidth < 768) setIsSidebarOpen(false);
  };

  // Remove one session from history
  const handleDeleteSession = (id: string) => {
    const updated = sessions.filter((s) => s.id !== id);
    setSessions(updated);
    saveSessions(updated);
    // If the deleted session was active, go back to landing
    if (id === activeSessionId) handleNewSession();
  };

  // ────────────────────────────────────────────
  // FILE UPLOAD HANDLER
  // Called when: user picks a file from LandingScreen or ChatPanel
  //
  // Steps:
  //  1. POST the file to /api/upload → get back `file_path` on disk
  //  2. POST file_path to /api/analyze-init → get shape + fingerprint
  //  3. Build the `activeDataset` object and set it in state
  //  4. If the user had typed a prompt before uploading, auto-send it
  // ────────────────────────────────────────────
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploading(true);
    setUploadStatus(null);
    const initialPrompt = chatInput.trim();
    if (initialPrompt) setChatInput("");

    try {
      console.log("DEBUG: Starting file upload for", file.name);
      // Step 1: Upload file (Must be first to get the path)
      const uploadData = await api.uploadFile(file);
      console.log("DEBUG: Upload successful, path:", uploadData.file_path);

      // Step 2: Analyze & Metrics in Parallel
      // We start both but await analyzeInit first as it's required for the workspace UI
      const analyzePromise = api.analyzeInit(uploadData.file_path);
      const metricsPromise = api.getQualityMetrics(uploadData.file_path);

      const analyzeData = await analyzePromise;
      
      // Step 3: Store dataset in state (Workspace shows up now)
      const newDataset: Dataset = {
        fileName: file.name,
        filePath: uploadData.file_path,
        shape: analyzeData.fingerprint.shape,
        fingerprint: analyzeData.fingerprint,
      };
      
      // Fallback for crypto.randomUUID() in non-secure contexts or older browsers
      const newSessionId = (typeof crypto !== "undefined" && crypto.randomUUID) 
        ? crypto.randomUUID() 
        : Date.now().toString(36) + Math.random().toString(36).substring(2, 9);
      setActiveDataset(newDataset);
      setActiveSessionId(newSessionId);
      saveSession(newDataset, newSessionId);

      setUploadStatus({ msg: "Dataset loaded. Finalizing analysis...", isError: false });
      setActiveTab("diagnostics");
      setUploading(false); // UI Unlocks here!

      // Step 4: Await and set quality metrics in the background
      try {
        const qm = await metricsPromise;
        setQualityMetricsPre(qm);
        setUploadStatus({ msg: "Analysis complete.", isError: false });
      } catch (qmErr) {
        console.error("Failed to fetch quality metrics", qmErr);
      }

      // Step 4: Initial messages
      const baseMessages: Message[] = [
        {
          role: "assistant",
          content: `Dataset **${file.name}** loaded! It has **${newDataset.shape[0].toLocaleString()} rows** and **${newDataset.shape[1]} columns**. What would you like to analyze or clean?`,
        },
      ];

      if (initialPrompt) {
        baseMessages.push({ role: "user", content: initialPrompt });
        setMessages(baseMessages);
        setIsTyping(true);
        try {
          const chatData = await api.chat(
            initialPrompt,
            `Loaded file: ${newDataset.fileName}`,
            newDataset.fingerprint as any,
            analyzeData.safe_summary,
            selectedModel
          );
          setMessages((prev) => [
            ...prev,
            { role: "assistant", content: chatData.reply },
          ]);
        } catch (chatErr: any) {
          setMessages((prev) => [
            ...prev,
            { role: "assistant", content: `❌ Chat error: ${chatErr.message}` },
          ]);
        } finally {
          setIsTyping(false);
        }
      } else {
        setMessages(baseMessages);
      }
    } catch (err: any) {
      console.error("DEBUG: Upload failed error:", err);
      const errorMessage = err.message || "Upload failed (Network Error or CORS)";
      setUploadStatus({ msg: errorMessage, isError: true });
      setUploading(false);
    }
  };

  // ────────────────────────────────────────────
  // SEND MESSAGE HANDLER
  // Called when: user submits the chat input form
  //
  // Process:
  //   1. Add user message to the messages array immediately (optimistic update)
  //   2. Show the typing indicator
  //   3. POST to /api/chat and await the AI reply
  //   4. Append the AI reply to messages, hide typing indicator
  // ────────────────────────────────────────────
  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!chatInput.trim() || isTyping) return;

    const userMsg = chatInput;
    setChatInput("");
    setMessages((prev) => [...prev, { role: "user", content: userMsg }]);
    setIsTyping(true);

    try {
      const chatData = await api.chat(
        userMsg,
        activeDataset ? `Loaded file: ${activeDataset.fileName}` : "mock",
        (activeDataset?.fingerprint ?? {}) as any,
        JSON.stringify(activeDataset?.fingerprint?.safe_sample ?? {}),
        selectedModel,
        undefined, // apiKey
        activeDataset?.filePath
      );
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: chatData.reply },
      ]);
    } catch (err: any) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: `❌ Network error: ${err.message}` },
      ]);
    } finally {
      setIsTyping(false);
    }
  };

  // ────────────────────────────────────────────
  // RUN DIAGNOSTICS
  // POSTs the dataset fingerprint to /api/diagnose and gets back a text report
  // ────────────────────────────────────────────
  const runDiagnostics = async () => {
    if (!activeDataset) return;
    setIsAnalyzing(true);
    try {
      const data = await api.diagnose(
        activeDataset.fingerprint as any,
        selectedModel
      );
      setDiagnosticReport(data.report);
    } catch (err: any) {
      setDiagnosticReport(`❌ Error: ${err.message}`);
    } finally {
      setIsAnalyzing(false);
    }
  };

  // ────────────────────────────────────────────
  // STAGE 1: GENERATE PLAN (no execution)
  // POSTs to /api/analyze to get the AI cleaning plan, then switches to Plan tab.
  // The user can then review and toggle steps before deploying.
  // ────────────────────────────────────────────
  const generatePlan = async () => {
    if (!activeDataset) return;
    setIsPlanLoading(true);
    setGeneratedPlan(null); // clear any previous plan
    try {
      const data = await api.analyzePlan(
        activeDataset.fingerprint as any,
        selectedModel
      );
      setGeneratedPlan({ plan: data.plan, explanation: data.explanation });
      setActiveTab("plan"); // auto-switch to Plan tab
    } catch (err: any) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: `❌ **Plan generation failed:** ${err.message}` },
      ]);
    } finally {
      setIsPlanLoading(false);
    }
  };

  // ────────────────────────────────────────────
  // STAGE 2: RUN REFINERY (execute the reviewed plan)
  // Takes the enabledActions list (from CleaningPlanPane toggles),
  // passes the pre-generated plan + filtered actions to /api/clean.
  // ────────────────────────────────────────────
  const runRefinery = async (enabledActions?: string[]) => {
    if (!activeDataset) return;
    setIsCleaning(true);
    try {
      const planToUse = generatedPlan?.plan ?? undefined;
      const data = await api.cleanDataset(
        activeDataset.filePath,
        activeDataset.fingerprint as any,
        selectedModel,
        undefined,       // apiKey (uses backend .env)
        planToUse,       // pre-reviewed plan — skips AI call
        enabledActions   // only execute these action types
      );

      const finalCleanedDataset = {
        ...data.cleaned_data,
        plan: data.plan,
        python_code: data.python_code,
        explanation: data.explanation,
        used_kb: data.used_kb,
      };
      setCleanedDataset(finalCleanedDataset);
      setGeneratedPlan(null); // plan has been executed — clear it
      setCleaningSummary(data.cleaning_summary || null);
      if (data.quality_metrics) {
        setQualityMetricsPre(data.quality_metrics.initial);
        setQualityMetricsPost(data.quality_metrics.cleaned);
      }
      setActiveTab("dashboard");
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: `✨ Pipeline deployed successfully!\n\n${data.explanation ?? ""}` },
      ]);
    } catch (err: any) {
      const errMsg = err?.message ?? "Unknown error";
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: `❌ **Refinement failed:** ${errMsg}` },
      ]);
    } finally {
      setTimeout(() => setIsCleaning(false), 500);
    }
  };

  // ────────────────────────────────────────────
  // RENDER
  // This is the JSX tree — React converts this to real DOM elements.
  // ────────────────────────────────────────────
  return (
    /*
     * Root container: full viewport height, dark background, dark text.
     * `overflow-hidden` prevents any global scrollbar — each panel scrolls independently.
     * `flex` makes children lay out as a row (sidebar + main).
     */
    <div className="flex h-screen bg-[#080e1a] text-slate-200 font-sans overflow-hidden">

      {/* ── Sidebar ── */}
      <Sidebar
        isOpen={isSidebarOpen}
        onToggle={() => setIsSidebarOpen((v) => !v)}
        sessions={sessions}
        activeSessionId={activeSessionId}
        onNewSession={handleNewSession}
        onLoadSession={handleLoadSession}
        onDeleteSession={handleDeleteSession}
        activeLeftView={activeLeftView}
        onViewChange={setActiveLeftView}
      />

      {/* ── Mobile overlay backdrop (darkens content behind open sidebar) ── */}
      {isSidebarOpen && (
        <div
          className="fixed inset-0 bg-black/60 z-40 md:hidden"
          onClick={() => setIsSidebarOpen(false)}
        />
      )}

      {/* ── Main Content Area ── */}
      {/*
       * `flex-1`: Takes all remaining width after the sidebar.
       * `flex flex-col`: Stack header + page content vertically.
       * `min-w-0`: Prevents flex children from overflowing (important with flex-1).
       */}
      <main className="flex-1 flex flex-col overflow-hidden min-w-0">

        {/* ── Header ── */}
        <Header 
          fileName={activeDataset?.fileName} 
          onMobileMenuToggle={() => setIsSidebarOpen((v) => !v)}
          isSidebarOpen={isSidebarOpen}
          theme={theme}
          onThemeToggle={() => setTheme(theme === "dark" ? "light" : "dark")}
          onLogout={handleLogout}
          user={user}
        />

        {/* ── Page Body: Landing OR Workspace ── */}
        {activeDataset ? (
          /*
           * WORKSPACE VIEW — shown when a dataset is active.
           * WorkflowLayout provides the two-column split.
           * We pass ChatPanel to `leftPanel` and DataPanel to `rightPanel`.
           */
           <WorkflowLayout
             left={
               activeLeftView === "chat" ? (
                 <ChatPanel
                   messages={messages}
                   chatInput={chatInput}
                   onChatInputChange={setChatInput}
                   onSendMessage={handleSendMessage}
                   isTyping={isTyping}
                   cleanedDataset={cleanedDataset}
                   onFileSelect={handleFileUpload}
                   selectedModel={selectedModel}
                   onModelChange={setSelectedModel}
                   chatPlaceholder={copy.chatPlaceholder}
                   onRefine={runRefinery}
                 />
               ) : (
                 <KnowledgeBasePane />
               )
             }
            right={
              <DataPanel
                activeTab={activeTab}
                onTabChange={setActiveTab}
                activeDataset={activeDataset}
                cleanedDataset={cleanedDataset}
                generatedPlan={generatedPlan}
                diagnosticReport={diagnosticReport}
                isAnalyzing={isAnalyzing}
                isPlanLoading={isPlanLoading}
                isCleaning={isCleaning}
                onRunDiagnostics={runDiagnostics}
                onGeneratePlan={generatePlan}
                onRunRefinery={runRefinery}
                showAdvanced={copy.showAdvancedOptions}
                usedKb={cleanedDataset?.used_kb}
                qualityMetricsPre={qualityMetricsPre}
                qualityMetricsPost={qualityMetricsPost}
                cleaningSummary={cleaningSummary}
                selectedModel={selectedModel}
                onModelChange={setSelectedModel}
                onClose={() => {
                  setActiveDataset(null);
                  setCleanedDataset(null);
                  setGeneratedPlan(null);
                  setDiagnosticReport(null);
                  setMessages([]);
                  setActiveTab("diagnostics");
                }}
              />
            }
            onUploadTrigger={() => document.getElementById("file-upload-input")?.click()}
            onClean={runRefinery}
            onToggleDiff={() => setActiveTab("dashboard")}
          />
        ) : (
          /*
           * LANDING VIEW — shown when no dataset is loaded yet.
           */
          <LandingScreen
            chatInput={chatInput}
            onChatInputChange={setChatInput}
            onFileSelect={handleFileUpload}
            onSubmit={handleSendMessage}
            uploading={uploading}
            uploadStatus={uploadStatus}
            selectedModel={selectedModel}
            onModelChange={setSelectedModel}
          />
        )}
      </main>
    </div>
  );
}
