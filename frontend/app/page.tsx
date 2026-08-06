// frontend/app/page.tsx
"use client";

import { useState } from "react";

interface PaperSummary {
  title: string;
  summary: string;
  link: string;
}

export default function Home() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<PaperSummary[]>([]);
  const [loading, setLoading] = useState(false);

  const handleSearch = async () => {
    if (!query.trim()) return;
    setLoading(true);
    setResults([]);

    try {
      const res = await fetch("http://localhost:8000/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, max_results: 5 }),
      });

      const data = await res.json();
      setResults(data.results || []);
    } catch (error) {
      console.error("Fetch error:", error);
      setResults([
        { title: "Error", summary: "Could not fetch results.", link: "#" },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-gray-900 p-8 text-gray-100 flex flex-col items-center">
      <h1 className="text-3xl font-bold mb-6 text-cyan-400">🧬 BioGPT Explorer</h1>

      <div className="w-full max-w-2xl flex gap-2 mb-6">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search PubMed papers..."
          className="flex-1 px-4 py-2 rounded-xl border border-gray-700 bg-gray-800 text-gray-100 focus:outline-none focus:ring-2 focus:ring-cyan-400"
        />
        <button
          onClick={handleSearch}
          disabled={loading}
          className="px-4 py-2 bg-cyan-500 text-white rounded-xl hover:bg-cyan-400 disabled:opacity-50"
        >
          {loading ? "Searching..." : "Search"}
        </button>
      </div>

      <div className="w-full max-w-2xl flex flex-col gap-4">
        {results.length > 0 ? (
          results.map((paper, idx) => (
            <div key={idx} className="p-4 bg-gray-800 rounded-xl shadow">
              <h2 className="text-lg font-bold text-cyan-300">{paper.title}</h2>
              <p className="mt-2 text-gray-200">{paper.summary}</p>
              {paper.link !== "#" && (
                <a
                  href={paper.link}
                  target="_blank"
                  className="mt-2 inline-block text-cyan-400 underline"
                >
                  Read on PubMed
                </a>
              )}
            </div>
          ))
        ) : (
          <p className="text-gray-500 mt-4">No results yet — ask something!</p>
        )}
      </div>

      <footer className="mt-12 w-full text-right text-sm text-gray-500">
        Built by Rohith •{" "}
        <a href="https://github.com/23BCB0017" target="_blank" className="underline">
          GitHub
        </a>
      </footer>
    </main>
  );
}
