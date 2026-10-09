import { useCallback, useEffect, useRef, useState } from "react";
import FormattedAnswer from "./formatAnswer";

const EXAMPLES = [
  "What are the symptoms of leukemia?",
  "How is breast cancer treated?",
  "What causes lung cancer?",
  "What is the capital of France?",
];

function metricClass(value, invert = false) {
  if (value == null) return "";
  const v = invert ? 1 - value : value;
  if (v >= 0.75) return "good";
  if (v >= 0.5) return "warn";
  return "bad";
}

function formatTime(iso) {
  try {
    return new Date(iso).toLocaleString(undefined, {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return iso;
  }
}

function pct(value) {
  if (value == null) return "—";
  return `${Math.round(value * 100)}%`;
}

function metricsMethodLabel(method) {
  if (method === "ragas") return "Scored by RAGAS";
  if (method === "refusal") return "Safe refusal — no unsupported claims";
  if (method === "model_refusal") return "Model declined to answer from the sources — no unsupported claims";
  if (method === "skipped") return "Metrics not computed";
  return "Estimated from word overlap — RAGAS unavailable for this answer";
}

function MetricCard({ label, value, invert, hint }) {
  return (
    <div className={`metric-card ${metricClass(value, invert)}`}>
      <span className="metric-label">{label}</span>
      <span className="metric-value">{pct(value)}</span>
      {hint && <span className="metric-hint">{hint}</span>}
    </div>
  );
}

function BotMessage({ display }) {
  return (
    <div className="message bot">
      <div className="avatar bot-avatar" aria-hidden>
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M12 2v4M8 6h8M6 10h12v8a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2v-8z" />
          <path d="M9 14h6M9 17h4" />
        </svg>
      </div>
      <div className="message-content">
        <div className={`bubble bot-bubble ${display.refused ? "refused" : ""}`}>
          {display.refused && (
            <div className="refusal-badge">
              <span className="refusal-dot" />
              Out of knowledge base — safe refusal
            </div>
          )}
          <FormattedAnswer text={display.answer} refused={display.refused} />
        </div>

        {display.metrics && (
          <>
            <div className="metrics-grid">
              <MetricCard label="Faithfulness" value={display.metrics.faithfulness} hint="Grounded in sources" />
              <MetricCard label="Relevance" value={display.metrics.relevance} hint="Answers your question" />
              <MetricCard label="Hallucination" value={display.metrics.hallucination} invert hint="Lower is better" />
            </div>
            <p className="metrics-method">{metricsMethodLabel(display.metrics.method)}</p>
          </>
        )}

        {display.sources?.length > 0 && (
          <details className="sources">
            <summary>
              <span className="sources-icon">📚</span>
              {display.sources.length} NIH sources retrieved
            </summary>
            <div className="sources-list">
              {display.sources.map((s, i) => (
                <article className="source-card" key={`${s.url}-${i}`}>
                  <div className="source-head">
                    <span className="source-topic">{s.focus}</span>
                    <span className="source-score">
                      {s.rerank_score != null
                        ? `rerank ${s.rerank_score.toFixed(2)}`
                        : s.similarity != null
                          ? `similarity ${s.similarity.toFixed(2)}`
                          : `distance ${s.distance}`}
                    </span>
                  </div>
                  {s.doc_type && (
                    <p className="source-meta">
                      {s.doc_type} · page {s.page} of {s.total_pages}
                    </p>
                  )}
                  <a href={s.url} target="_blank" rel="noreferrer" className="source-link">
                    {s.url.replace(/^https?:\/\//, "")}
                  </a>
                  <p className="source-excerpt">{s.chunk}</p>
                </article>
              ))}
            </div>
          </details>
        )}
      </div>
    </div>
  );
}

export default function App() {
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [current, setCurrent] = useState(null);
  const [history, setHistory] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const chatEndRef = useRef(null);

  const loadHistory = useCallback(async () => {
    try {
      const res = await fetch("/api/history?limit=50");
      const data = await res.json();
      setHistory(data.items || []);
    } catch {
      /* backend may be starting */
    }
  }, []);

  useEffect(() => {
    loadHistory();
  }, [loadHistory]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [current, loading]);

  async function askQuestion(q) {
    const text = (q || question).trim();
    if (!text || loading) return;
    setLoading(true);
    setError("");
    setSelectedId(null);
    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: text, save_history: true, compute_metrics: true }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Request failed");
      setCurrent(data);
      setQuestion("");
      await loadHistory();
    } catch (e) {
      setError(e.message || "Could not reach the API. Start the backend first.");
    } finally {
      setLoading(false);
    }
  }

  function selectHistory(item) {
    setSelectedId(item.id);
    setCurrent(item);
    setError("");
  }

  function startNewChat() {
    setSelectedId(null);
    setCurrent(null);
    setError("");
    setQuestion("");
  }

  async function clearHistory() {
    if (!history.length) return;
    await fetch("/api/history", { method: "DELETE" });
    setHistory([]);
    setCurrent(null);
    setSelectedId(null);
  }

  const display = current;
  const isEmpty = !display && !loading;

  return (
    <div className="app-layout">
      <aside className="sidebar">
        <div className="sidebar-inner">
          <div className="brand">
            <div className="brand-icon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2">
                <path d="M12 21s-6-4.35-6-10a6 6 0 0 1 12 0c0 5.65-6 10-6 10z" />
                <path d="M12 11v-4M10 9h4" />
              </svg>
            </div>
            <div>
              <h1>Health FAQ</h1>
              <p>NIH Cancer.gov · RAGAS</p>
            </div>
          </div>

          <button type="button" className="btn btn-new" onClick={startNewChat}>
            + New question
          </button>

          <div className="history-section">
            <div className="history-header">
              <span>Recent chats</span>
              {history.length > 0 && (
                <button type="button" className="link-btn" onClick={clearHistory}>
                  Clear
                </button>
              )}
            </div>
            <div className="history-list">
              {history.length === 0 ? (
                <p className="history-empty">Your questions will appear here.</p>
              ) : (
                history.map((item) => (
                  <button
                    type="button"
                    key={item.id}
                    className={`history-item ${selectedId === item.id ? "active" : ""}`}
                    onClick={() => selectHistory(item)}
                  >
                    <span className="history-q">{item.question}</span>
                    <span className="history-meta">
                      {formatTime(item.created_at)}
                      {item.refused && " · Refused"}
                    </span>
                  </button>
                ))
              )}
            </div>
          </div>
        </div>

        <div className="sidebar-footer">
          <span>MedQuAD · Group 63</span>
        </div>
      </aside>

      <main className="main">
        <header className="topbar">
          <div>
            <h2>Health Information Assistant</h2>
            <p>Trusted answers from NIH sources with quality metrics on every response</p>
          </div>
          <div className="status-pill">
            <span className="status-dot" />
            Live
          </div>
        </header>

        <div className="chat-scroll">
          {error && <div className="error-banner">{error}</div>}

          {isEmpty && (
            <div className="welcome">
              <div className="welcome-icon">
                <svg viewBox="0 0 48 48" fill="none">
                  <circle cx="24" cy="24" r="22" stroke="currentColor" strokeWidth="1.5" opacity="0.2" />
                  <path d="M24 14v20M14 24h20" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" />
                </svg>
              </div>
              <h3>How can I help you today?</h3>
              <p>
                Ask about cancer symptoms, treatments, diagnosis, or risk factors.
                Questions outside our health knowledge base are safely refused.
              </p>
              <div className="examples">
                <span className="examples-label">Try asking</span>
                <div className="examples-grid">
                  {EXAMPLES.map((ex) => (
                    <button type="button" key={ex} className="example-card" onClick={() => askQuestion(ex)}>
                      {ex}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          )}

          {display && (
            <div className="thread">
              <div className="message user">
                <div className="message-content user-content">
                  <div className="bubble user-bubble">{display.question}</div>
                </div>
                <div className="avatar user-avatar" aria-hidden>You</div>
              </div>
              <BotMessage display={display} />
            </div>
          )}

          {loading && (
            <div className="message bot">
              <div className="avatar bot-avatar" aria-hidden>
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M12 2v4M8 6h8M6 10h12v8a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2v-8z" />
                </svg>
              </div>
              <div className="message-content">
                <div className="bubble bot-bubble loading-bubble">
                  <span className="typing">
                    <span /><span /><span />
                  </span>
                  Retrieving sources and generating answer…
                </div>
              </div>
            </div>
          )}

          <div ref={chatEndRef} />
        </div>

        <form
          className="composer"
          onSubmit={(e) => {
            e.preventDefault();
            askQuestion();
          }}
        >
          <div className="composer-box">
            <input
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Ask a health question…"
              disabled={loading}
              aria-label="Your question"
            />
            <button type="submit" className="btn btn-send" disabled={loading || !question.trim()}>
              {loading ? "…" : "Send"}
            </button>
          </div>
          <p className="composer-disclaimer">
            For educational purposes only. Not a substitute for professional medical advice.
          </p>
        </form>
      </main>
    </div>
  );
}
