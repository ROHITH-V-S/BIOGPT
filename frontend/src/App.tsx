import type React from "react"
import { useState, useEffect, useRef } from "react"
import {
  Search, Loader2, FileText, Sparkles, Activity,
  ExternalLink, ChevronDown, Zap, AlertTriangle, X,
} from "lucide-react"
import "./index.css"

// ─── Configuration ───────────────────────────────────────────
const API_BASE_URL = import.meta.env.VITE_API_URL || ""

// ─── Types ───────────────────────────────────────────────────
interface Source { title: string; summary: string; link: string }
interface HealthData { status: string; embedding_backend: string; llm_models: string[]; index_loaded: boolean }
interface Entities { genes?: string[]; diseases?: string[]; drugs?: string[] }

// ─── App ─────────────────────────────────────────────────────
export default function App() {
  const [query, setQuery] = useState("")
  const [answer, setAnswer] = useState("")
  const [chunks, setChunks] = useState<string[]>([])
  const [sources, setSources] = useState<Source[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [hasSearched, setHasSearched] = useState(false)
  const [entityAware, setEntityAware] = useState(false)
  const [expandedChunks, setExpandedChunks] = useState<Set<number>>(new Set())
  const [health, setHealth] = useState<HealthData | null>(null)
  const [modelUsed, setModelUsed] = useState<string | null>(null)
  const [entities, setEntities] = useState<Entities | null>(null)

  const answerRef = useRef<HTMLDivElement>(null)

  // Health check
  useEffect(() => {
    fetch(`${API_BASE_URL}/health`)
      .then(r => r.json())
      .then(setHealth)
      .catch(() => {})
  }, [])

  // Auto-scroll answer into view while streaming
  useEffect(() => {
    if (loading && answerRef.current) {
      answerRef.current.scrollIntoView({ behavior: "smooth", block: "nearest" })
    }
  }, [answer])

  const toggleChunk = (i: number) => {
    const next = new Set(expandedChunks)
    next.has(i) ? next.delete(i) : next.add(i)
    setExpandedChunks(next)
  }

  // ─── Submit handler (SSE) ────────────────────────────────
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!query.trim()) return

    setLoading(true)
    setError(null)
    setHasSearched(true)
    setAnswer("")
    setChunks([])
    setSources([])
    setModelUsed(null)
    setEntities(null)
    setExpandedChunks(new Set())

    try {
      const res = await fetch(`${API_BASE_URL}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: query.trim(),
          max_results: 5,
          stream: true,
          entity_aware: entityAware,
        }),
      })
      if (!res.ok) throw new Error(`Server error (${res.status})`)

      const reader = res.body?.getReader()
      const decoder = new TextDecoder()
      if (!reader) throw new Error("Stream unavailable")

      let fullAnswer = ""
      let currentEvent = ""

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        for (const line of decoder.decode(value, { stream: true }).split("\n")) {
          if (line.startsWith("event: ")) {
            currentEvent = line.slice(7).trim()
          } else if (line.startsWith("data: ")) {
            try {
              const data = JSON.parse(line.slice(6))
              if (currentEvent === "chunk") {
                if (data.chunks) setChunks(data.chunks)
                if (data.sources) setSources(data.sources)
                if (data.entities) setEntities(data.entities)
              } else if (currentEvent === "token") {
                fullAnswer += data.token
                setAnswer(fullAnswer)
              } else if (currentEvent === "done") {
                if (data.answer) setAnswer(data.answer)
                if (data.model) setModelUsed(data.model)
              } else if (currentEvent === "error") {
                setError(data.error)
              }
            } catch { /* skip malformed lines */ }
          }
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed")
    } finally {
      setLoading(false)
    }
  }

  // ─── Render helpers ──────────────────────────────────────
  const hasEntities = entities && (
    (entities.genes?.length ?? 0) +
    (entities.diseases?.length ?? 0) +
    (entities.drugs?.length ?? 0) > 0
  )

  return (
    <div className="app">
      {/* Ambient background effects */}
      <div className="ambient-bg" />
      <div className="noise-overlay" />

      <div className="main-content">
        {/* ─── Header ─────────────────────────────────── */}
        <header className="header">
          <div className="logo-container">
            <span className="logo-dna">🧬</span>
            <span className="logo-text">BioGPT Explorer</span>
          </div>

          <p className="tagline">
            Advanced biomedical RAG engine — ask complex scientific questions,
            explore literature, and synthesize research insights in real-time.
          </p>

          <div className={`health-badge ${health ? '' : 'health-offline'}`}>
            <div className="health-dot" />
            {health ? (
              <>
                <span>System Online</span>
                <div className="health-divider" />
                <span>{health.index_loaded ? "Index Ready" : "No Index"}</span>
                <div className="health-divider" />
                <span style={{ fontFamily: "'JetBrains Mono', monospace" }}>
                  {health.llm_models?.[0]?.split("/").pop() ?? "—"}
                </span>
              </>
            ) : (
              <>
                <Activity style={{ width: 12, height: 12, color: "var(--accent-amber)" }} className="spin" />
                <span>Connecting…</span>
              </>
            )}
          </div>
        </header>

        {/* ─── Search ─────────────────────────────────── */}
        <section className="search-section">
          <form onSubmit={handleSubmit} className="search-form">

            {/* Toggle */}
            <div className="search-options">
              <div className="toggle-group" onClick={() => setEntityAware(!entityAware)}>
                <div className={`toggle-track ${entityAware ? "active" : ""}`}>
                  <div className="toggle-thumb" />
                </div>
                <span className={`toggle-label ${entityAware ? "active" : ""}`}>
                  Entity-Aware Retrieval
                </span>
              </div>
            </div>

            {/* Input + Button */}
            <div className="search-input-group">
              <div className="search-input-wrapper">
                <textarea
                  className="search-input"
                  value={query}
                  onChange={e => setQuery(e.target.value)}
                  onKeyDown={e => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault()
                      handleSubmit(e)
                    }
                  }}
                  placeholder="What is the mechanism of action of mRNA vaccines?  How does BRCA1 mutation affect cancer risk?  Describe p53's role in apoptosis..."
                  disabled={loading}
                />
                <div className="search-hint">
                  <kbd>↵</kbd> to search
                </div>
              </div>

              <button type="submit" className="submit-btn" disabled={loading || !query.trim()}>
                {loading ? (
                  <Loader2 className="icon spin" />
                ) : (
                  <Search className="icon" />
                )}
                <span>{loading ? "Analyzing…" : "Explore"}</span>
              </button>
            </div>
          </form>
        </section>

        {/* ─── Results ────────────────────────────────── */}
        {hasSearched && (
          <div className="results-area">
            {/* Error */}
            {error && (
              <div className="error-banner">
                <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                  <div className="error-icon">
                    <AlertTriangle style={{ width: 18, height: 18, color: "var(--accent-red)" }} />
                  </div>
                  <div className="error-content">
                    <h3>Analysis Error</h3>
                    <p>{error}</p>
                  </div>
                </div>
                <button className="error-dismiss" onClick={() => setError(null)}>
                  <X style={{ width: 16, height: 16 }} />
                </button>
              </div>
            )}

            {/* Answer Card */}
            {(answer || loading) && !error && (
              <div className="answer-card" ref={answerRef}>
                {/* Entities */}
                {hasEntities && (
                  <div className="entities-container">
                    {entities!.genes?.map(g => (
                      <span key={`g-${g}`} className="entity-tag entity-gene">
                        <span className="dot" />{g}
                      </span>
                    ))}
                    {entities!.diseases?.map(d => (
                      <span key={`d-${d}`} className="entity-tag entity-disease">
                        <span className="dot" />{d}
                      </span>
                    ))}
                    {entities!.drugs?.map(d => (
                      <span key={`dr-${d}`} className="entity-tag entity-drug">
                        <span className="dot" />{d}
                      </span>
                    ))}
                  </div>
                )}

                <div className="answer-header">
                  <div className="icon-wrapper">
                    <Sparkles style={{ width: 16, height: 16 }} />
                  </div>
                  <h2>Synthesized Answer</h2>
                </div>

                {answer ? (
                  <div className={`answer-text ${loading ? "streaming" : ""}`}>
                    {answer}
                  </div>
                ) : (
                  <div className="skeleton-group">
                    <div className="skeleton-line" style={{ width: "78%" }} />
                    <div className="skeleton-line" style={{ width: "100%" }} />
                    <div className="skeleton-line" style={{ width: "65%" }} />
                    <div className="skeleton-line" style={{ width: "90%" }} />
                  </div>
                )}

                {modelUsed && !loading && (
                  <div className="model-badge">
                    <Zap className="icon" />
                    {modelUsed}
                  </div>
                )}
              </div>
            )}

            {/* Sources + Chunks */}
            {!loading && !error && (sources.length > 0 || chunks.length > 0) && (
              <div className="results-grid">
                {/* Sources */}
                {sources.length > 0 && (
                  <div className="result-panel">
                    <div className="panel-header">
                      <div className="icon-wrapper" style={{ background: "rgba(34,211,238,0.1)", color: "var(--accent-cyan)" }}>
                        <ExternalLink style={{ width: 14, height: 14 }} />
                      </div>
                      <h3>Literature Sources</h3>
                      <span className="count">{sources.length}</span>
                    </div>
                    <div className="source-list">
                      {sources.map((s, i) => (
                        <a key={i} href={s.link} target="_blank" rel="noopener noreferrer" className="source-card">
                          <h4>{s.title}</h4>
                          <div className="source-meta">
                            <ExternalLink style={{ width: 10, height: 10 }} />
                            View on PubMed
                          </div>
                        </a>
                      ))}
                    </div>
                  </div>
                )}

                {/* Chunks */}
                {chunks.length > 0 && (
                  <div className="result-panel">
                    <div className="panel-header">
                      <div className="icon-wrapper" style={{ background: "rgba(168,85,247,0.1)", color: "var(--accent-purple)" }}>
                        <FileText style={{ width: 14, height: 14 }} />
                      </div>
                      <h3>Retrieved Context</h3>
                      <span className="count">{chunks.length}</span>
                    </div>
                    <div className="chunk-list">
                      {chunks.map((chunk, i) => (
                        <div key={i} className="chunk-item">
                          <button className="chunk-trigger" onClick={() => toggleChunk(i)}>
                            <span style={{ display: "flex", alignItems: "center" }}>
                              <span className="chunk-num">{i + 1}</span>
                              Passage #{i + 1}
                            </span>
                            <ChevronDown className={`chevron ${expandedChunks.has(i) ? "open" : ""}`} />
                          </button>
                          {expandedChunks.has(i) && (
                            <div className="chunk-body">
                              <p>{chunk}</p>
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Empty state */}
            {!loading && !error && !answer && chunks.length === 0 && (
              <div className="empty-state">
                <FileText className="icon" />
                <h3>No Results Found</h3>
                <p>Try rephrasing your query or ingest documents first to build the knowledge base.</p>
              </div>
            )}
          </div>
        )}
      </div>

      {/* ─── Footer ─────────────────────────────────── */}
      <footer className="footer">
        <p>
          Built by <a href="#">Rohith</a> · BioGPT Explorer · Biomedical RAG Pipeline
        </p>
      </footer>
    </div>
  )
}
