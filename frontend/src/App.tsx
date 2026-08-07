"use client"

import type React from "react"
import { useState, useEffect, useRef } from "react"
import { Search, Loader2, FileText, Sparkles, Activity, ExternalLink, ChevronDown, ChevronUp } from "lucide-react"

// Configuration
const API_BASE_URL = import.meta.env.VITE_API_URL || ""

// Types
interface Source {
  title: string
  summary: string
  link: string
}

interface HealthData {
  status: string
  embedding_backend: string
  llm_models: string[]
  index_loaded: boolean
}

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
  const [entities, setEntities] = useState<{genes?: string[], diseases?: string[], drugs?: string[]} | null>(null)

  // Fetch health status
  useEffect(() => {
    fetch(`${API_BASE_URL}/health`)
      .then(res => res.json())
      .then(data => setHealth(data))
      .catch(err => console.error("Health check failed:", err))
  }, [])

  const toggleChunk = (index: number) => {
    const newExpanded = new Set(expandedChunks)
    if (newExpanded.has(index)) {
      newExpanded.delete(index)
    } else {
      newExpanded.add(index)
    }
    setExpandedChunks(newExpanded)
  }

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

    try {
      const response = await fetch(`${API_BASE_URL}/query`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query: query.trim(), max_results: 5, stream: true, entity_aware: entityAware }),
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      
      const reader = response.body?.getReader()
      const decoder = new TextDecoder("utf-8")
      if (!reader) throw new Error("No reader available")

      let currentAnswer = ""
      
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        
        const chunk = decoder.decode(value, { stream: true })
        const lines = chunk.split("\n")
        
        let currentEvent = ""
        for (const line of lines) {
          if (line.startsWith("event: ")) {
            currentEvent = line.substring(7).trim()
          } else if (line.startsWith("data: ")) {
            const dataStr = line.substring(6).trim()
            if (!dataStr) continue
            
            try {
              const data = JSON.parse(dataStr)
              if (currentEvent === "chunk") {
                if (data.chunks) setChunks(data.chunks)
                if (data.sources) setSources(data.sources)
                if (data.entities) setEntities(data.entities)
              } else if (currentEvent === "token") {
                currentAnswer += data.token
                setAnswer(currentAnswer)
              } else if (currentEvent === "done") {
                if (data.answer) {
                  setAnswer(data.answer)
                }
                if (data.model) {
                  setModelUsed(data.model)
                }
              } else if (currentEvent === "error") {
                setError(data.error)
              }
            } catch (e) {
              console.error("Error parsing JSON data line:", dataStr, e)
            }
          }
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to fetch results")
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-background text-foreground relative overflow-hidden font-sans">
      {/* Background Effects */}
      <div className="absolute inset-0 bg-background z-0"></div>
      <div className="absolute top-[-10%] left-[-10%] w-[50%] h-[50%] bg-bioblue/10 rounded-full blur-[120px] pointer-events-none"></div>
      <div className="absolute bottom-[-10%] right-[-10%] w-[50%] h-[50%] bg-biopurple/10 rounded-full blur-[120px] pointer-events-none"></div>

      {/* Main Content */}
      <div className="relative z-10 flex flex-col min-h-screen max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        
        {/* Header */}
        <header className="py-10 text-center animate-fade-in flex flex-col items-center">
          <div className="inline-flex items-center justify-center gap-3 mb-4 p-4 glass rounded-2xl border-white/10 shadow-[0_0_40px_rgba(6,182,212,0.15)]">
            <Sparkles className="w-8 h-8 text-biocyan animate-pulse-glow" />
            <h1 className="text-4xl md:text-5xl font-bold bg-gradient-to-r from-biocyan via-bioblue to-biopurple bg-clip-text text-transparent tracking-tight">
              BioGPT Explorer 🧬
            </h1>
          </div>
          <p className="text-gray-400 text-lg max-w-2xl mx-auto mt-2 font-light">
            An advanced biomedical retrieval-augmented generation engine. Ask complex questions, explore literature, and synthesize scientific insights.
          </p>
          
          {/* Health indicator */}
          <div className="mt-6 flex items-center gap-2 text-xs font-medium px-4 py-2 rounded-full bg-gray-900/50 border border-gray-800 shadow-lg">
            {health ? (
              <>
                <span className="relative flex h-2.5 w-2.5">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-green-500"></span>
                </span>
                <span className="text-gray-300">System Online</span>
                <span className="text-gray-600 mx-1">•</span>
                <span className="text-gray-400">{health.index_loaded ? "Index Loaded" : "No Index"}</span>
                <span className="text-gray-600 mx-1">•</span>
                <span className="text-gray-400">{health.llm_models.length > 0 ? health.llm_models[0] : "No LLM"}</span>
              </>
            ) : (
              <>
                <Activity className="w-3.5 h-3.5 text-yellow-500 animate-pulse" />
                <span className="text-gray-400">Connecting to Backend...</span>
              </>
            )}
          </div>
        </header>

        {/* Search Section */}
        <div className="flex-1 pb-16 animate-slide-up" style={{ animationDelay: '0.1s' }}>
          <div className="w-full mx-auto max-w-4xl">
            <form onSubmit={handleSubmit} className="mb-10 group">
              <div className="flex flex-col mb-3">
                <div className="flex items-center gap-3 px-2 mb-2 self-start cursor-pointer group/toggle" onClick={() => setEntityAware(!entityAware)}>
                  <div className={`w-11 h-6 rounded-full relative transition-colors duration-300 shadow-inner flex items-center ${entityAware ? 'bg-biocyan' : 'bg-gray-800'}`}>
                    <div className={`absolute bg-white w-4 h-4 rounded-full transition-transform duration-300 shadow-sm ${entityAware ? 'translate-x-6' : 'translate-x-1'}`}></div>
                  </div>
                  <span className={`text-sm font-medium transition-colors ${entityAware ? 'text-biocyan' : 'text-gray-400 group-hover/toggle:text-gray-300'}`}>
                    Entity-Aware Retrieval
                  </span>
                </div>
              </div>
              <div className="relative flex flex-col sm:flex-row gap-4">
                <div className="flex-1 relative">
                  <textarea
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        handleSubmit(e);
                      }
                    }}
                    placeholder="Enter your biomedical query here (e.g. 'What is the role of TP53 in cancer?')"
                    className="w-full h-32 px-6 py-5 glass-card rounded-2xl text-foreground placeholder-gray-500 resize-none focus:outline-none focus:ring-2 focus:ring-bioblue/50 focus:border-bioblue/30 transition-all duration-300 shadow-[0_8px_30px_rgb(0,0,0,0.12)] text-lg"
                    disabled={loading}
                  />
                  <div className="absolute bottom-4 right-4 flex items-center gap-2 text-gray-500 text-sm pointer-events-none">
                    <kbd className="px-2 py-1 bg-gray-800 rounded-md border border-gray-700 text-xs">Enter</kbd> to search
                  </div>
                </div>
                <button
                  type="submit"
                  disabled={loading || !query.trim()}
                  className="sm:w-40 px-6 py-4 bg-gradient-to-br from-bioblue to-biopurple hover:from-blue-500 hover:to-purple-500 disabled:from-gray-700 disabled:to-gray-800 disabled:text-gray-500 text-white font-semibold rounded-2xl transition-all duration-300 disabled:cursor-not-allowed flex flex-col items-center justify-center gap-2 shadow-lg hover:shadow-bioblue/20 hover:-translate-y-0.5 active:translate-y-0"
                >
                  {loading ? (
                    <>
                      <Loader2 className="w-6 h-6 animate-spin" />
                      <span className="text-sm">Synthesizing...</span>
                    </>
                  ) : (
                    <>
                      <Search className="w-6 h-6 group-hover:scale-110 transition-transform duration-300" />
                      <span className="text-sm">Explore</span>
                    </>
                  )}
                </button>
              </div>
            </form>

            {/* Results Section */}
            {hasSearched && (
              <div className="space-y-8 animate-slide-up" style={{ animationDelay: '0.2s' }}>
                
                {/* Error Banner (Toast-like) */}
                {error && (
                  <div className="bg-red-950/40 border border-red-500/30 rounded-2xl p-4 flex items-center justify-between gap-4 shadow-lg backdrop-blur-md">
                    <div className="flex items-center gap-3">
                      <div className="p-2 bg-red-500/20 rounded-lg">
                        <Activity className="w-5 h-5 text-red-400" />
                      </div>
                      <div>
                        <h3 className="text-red-400 font-medium">Analysis Error</h3>
                        <p className="text-sm text-red-300/80 mt-0.5">{error}</p>
                      </div>
                    </div>
                    <button onClick={() => setError(null)} className="text-red-400 hover:text-red-300 px-2 py-1">
                      Dismiss
                    </button>
                  </div>
                )}

                {/* Answer Card */}
                {(answer || loading) && !error && (
                  <div className="glass-card rounded-2xl p-6 sm:p-8 relative overflow-hidden">
                    <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-biocyan via-bioblue to-biopurple opacity-70"></div>
                    
                    {/* Entities Display */}
                    {entities && (entities.genes?.length || entities.diseases?.length || entities.drugs?.length) ? (
                      <div className="flex flex-wrap gap-2.5 mb-8 p-4 bg-gray-950/40 rounded-xl border border-gray-800/60 shadow-inner">
                        {entities.genes?.map(g => (
                          <span key={`gene-${g}`} className="px-3 py-1.5 bg-blue-900/30 text-blue-300 text-xs font-semibold rounded-full border border-blue-700/50 flex items-center gap-1.5 shadow-sm">
                            <span className="w-1.5 h-1.5 rounded-full bg-blue-400"></span> {g}
                          </span>
                        ))}
                        {entities.diseases?.map(d => (
                          <span key={`disease-${d}`} className="px-3 py-1.5 bg-red-900/30 text-red-300 text-xs font-semibold rounded-full border border-red-700/50 flex items-center gap-1.5 shadow-sm">
                            <span className="w-1.5 h-1.5 rounded-full bg-red-400"></span> {d}
                          </span>
                        ))}
                        {entities.drugs?.map(d => (
                          <span key={`drug-${d}`} className="px-3 py-1.5 bg-green-900/30 text-green-300 text-xs font-semibold rounded-full border border-green-700/50 flex items-center gap-1.5 shadow-sm">
                            <span className="w-1.5 h-1.5 rounded-full bg-green-400"></span> {d}
                          </span>
                        ))}
                      </div>
                    ) : null}

                    <h2 className="text-xl font-semibold mb-5 flex items-center gap-2 text-gray-100">
                      <Sparkles className="w-5 h-5 text-bioblue" />
                      Synthesized Answer
                    </h2>
                    <div className="prose prose-invert max-w-none">
                      {answer ? (
                        <div className={`text-gray-200 text-[1.05rem] leading-relaxed ${loading ? 'typing-cursor' : ''} whitespace-pre-wrap`}>
                          {answer}
                        </div>
                      ) : (
                        <div className="flex flex-col gap-4 py-2">
                          <div className="h-3 bg-gray-800/80 rounded animate-pulse w-3/4"></div>
                          <div className="h-3 bg-gray-800/80 rounded animate-pulse w-full"></div>
                          <div className="h-3 bg-gray-800/80 rounded animate-pulse w-5/6"></div>
                        </div>
                      )}
                    </div>
                    {modelUsed && !loading && (
                      <div className="mt-8 pt-4 border-t border-gray-800/50 flex justify-end">
                        <span className="text-xs text-gray-400 bg-gray-900/80 px-3 py-1.5 rounded-full border border-gray-700/50 flex items-center gap-1.5 shadow-sm">
                          <Activity className="w-3 h-3 text-biocyan" />
                          Generated by: {modelUsed}
                        </span>
                      </div>
                    )}
                  </div>
                )}

                {/* Sources & Chunks Columns */}
                {!loading && !error && (sources.length > 0 || chunks.length > 0) && (
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-start">
                    
                    {/* Sources */}
                    {sources.length > 0 && (
                      <div className="glass-card rounded-2xl p-6 shadow-xl">
                        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2 border-b border-gray-800 pb-3 text-gray-200">
                          <ExternalLink className="w-5 h-5 text-biocyan" />
                          Literature Sources ({sources.length})
                        </h3>
                        <div className="space-y-4">
                          {sources.map((source, idx) => (
                            <a 
                              key={idx} 
                              href={source.link} 
                              target="_blank" 
                              rel="noopener noreferrer"
                              className="block p-4 bg-gray-900/50 border border-gray-800 hover:border-biocyan/40 rounded-xl transition-all duration-200 group hover:bg-gray-800/50"
                            >
                              <h4 className="font-medium text-bioblue group-hover:text-biocyan transition-colors line-clamp-2 leading-snug">
                                {source.title}
                              </h4>
                              <p className="text-sm text-gray-400 mt-2 line-clamp-3 leading-relaxed">
                                {source.summary}
                              </p>
                              <div className="mt-3 flex items-center text-xs text-biocyan font-medium opacity-0 group-hover:opacity-100 transition-opacity transform translate-y-1 group-hover:translate-y-0 duration-200">
                                View Full Paper <ExternalLink className="w-3 h-3 ml-1.5" />
                              </div>
                            </a>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Retrieved Context Chunks */}
                    {chunks.length > 0 && (
                      <div className="glass-card rounded-2xl p-6 shadow-xl">
                        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2 border-b border-gray-800 pb-3 text-gray-200">
                          <FileText className="w-5 h-5 text-biopurple" />
                          Retrieved Context ({chunks.length})
                        </h3>
                        <div className="space-y-3">
                          {chunks.map((chunk, idx) => {
                            const isExpanded = expandedChunks.has(idx);
                            return (
                              <div key={idx} className="bg-gray-900/50 border border-gray-800 rounded-xl overflow-hidden transition-colors hover:border-gray-700">
                                <button 
                                  onClick={() => toggleChunk(idx)}
                                  className="w-full px-4 py-3.5 flex items-center justify-between text-left hover:bg-gray-800/50 transition-colors"
                                >
                                  <span className="text-sm font-medium text-gray-300 flex items-center gap-2">
                                    <span className="w-5 h-5 rounded-full bg-gray-800 flex items-center justify-center text-xs text-gray-400">{idx + 1}</span>
                                    Research Chunk
                                  </span>
                                  {isExpanded ? <ChevronUp className="w-4 h-4 text-gray-500" /> : <ChevronDown className="w-4 h-4 text-gray-500" />}
                                </button>
                                {isExpanded && (
                                  <div className="p-4 border-t border-gray-800 bg-gray-950/30">
                                    <p className="text-sm text-gray-400 leading-relaxed whitespace-pre-wrap">{chunk}</p>
                                  </div>
                                )}
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    )}

                  </div>
                )}
                
                {/* No Results Empty State */}
                {!loading && !error && answer === "" && chunks.length === 0 && hasSearched && (
                  <div className="glass-card rounded-2xl p-12 text-center shadow-lg border-dashed border-gray-700">
                    <FileText className="w-12 h-12 text-gray-600 mx-auto mb-4 opacity-50" />
                    <h3 className="text-xl font-medium text-gray-300">No Information Found</h3>
                    <p className="text-gray-500 mt-2 max-w-sm mx-auto">Try adjusting your query or expanding your search terms to match our database.</p>
                  </div>
                )}

              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <footer className="mt-auto py-6 border-t border-white/5 relative z-10">
          <div className="text-center">
            <p className="text-sm text-gray-500/70 font-medium">
              Made by Rohith · BioGPT Explorer · Biomedical RAG
            </p>
          </div>
        </footer>
      </div>
    </div>
  )
}
