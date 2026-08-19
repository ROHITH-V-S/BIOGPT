"""
Build the evaluation corpus from live PubMed abstracts.

Experiment design
-----------------
Every question's abstracts go into ONE shared index rather than a per-question
one. If each question were evaluated against only its own topic's abstracts,
retrieval would be trivial — everything in the index is already relevant, and
precision would be ~1.0 for any method. Pooling all topics forces each query
to discriminate against seven other topics' worth of biomedical text.

That mixture is also deliberately adversarial for pure vector search: TP53,
BRCA1/BRCA2, and ACE2 abstracts all share heavily overlapping vocabulary
(gene, mutation, expression, tumour, receptor), so embeddings alone tend to
blur them together. That is exactly the confusion entity-aware re-ranking is
supposed to resolve.

Usage::

    python scripts/build_eval_corpus.py [--per-topic 8] [--force]
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.embeddings import EmbeddingService  # noqa: E402
from app.pubmed import search_abstracts  # noqa: E402
from app.rag.embedder import embed_and_store  # noqa: E402
from app.rag.loader import split_text  # noqa: E402

BACKEND = Path(__file__).resolve().parent.parent

#: Two suites. "general" spreads eight unrelated topics, so vector search can
#: separate them on topic alone. "hard" pools confusable siblings (BRCA1 vs
#: BRCA2, ACE vs ACE2, Cas9 vs Cas12a) into one corpus, where topical
#: similarity is high and only the exact symbol disambiguates.
SUITES = {
    "general": ("eval_questions.json", "eval"),
    "hard": ("eval_questions_hard.json", "eval_hard"),
}


def suite_paths(suite: str):
    questions, folder = SUITES[suite]
    base = BACKEND / "data" / folder
    return (
        BACKEND / "data" / questions,
        base / "vector.index",
        base / "chunk_data.pkl",
        base / "corpus_manifest.json",
    )


async def build(suite: str, per_topic: int, force: bool) -> int:
    EVAL_FILE, EVAL_INDEX, EVAL_CHUNKS, MANIFEST = suite_paths(suite)

    if EVAL_INDEX.exists() and not force:
        print(f"Eval corpus already present at {EVAL_INDEX} (use --force to rebuild).")
        return 0

    for path in (EVAL_INDEX, EVAL_CHUNKS):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            path.unlink()

    dataset = json.loads(EVAL_FILE.read_text(encoding="utf-8"))
    all_chunks: list[str] = []
    manifest: list[dict] = []

    for i, item in enumerate(dataset, 1):
        topic = item.get("pubmed_topic") or item["question"]
        print(f"[{i}/{len(dataset)}] PubMed: {topic}")

        articles = await search_abstracts(topic, per_topic)
        if not articles:
            print("    !! no abstracts returned")
            continue

        topic_chunks: list[str] = []
        for art in articles:
            header = f"[PubMed {art.pmid}] {art.title}"
            topic_chunks.extend(f"{header}\n{p}" for p in split_text(art.abstract))

        all_chunks.extend(topic_chunks)
        manifest.append({
            "question": item["question"],
            "topic": topic,
            "pmids": [a.pmid for a in articles],
            "chunks": len(topic_chunks),
        })
        print(f"    {len(articles)} abstracts -> {len(topic_chunks)} chunks")

        # Stay well inside NCBI's ~3 req/s courtesy limit.
        await asyncio.sleep(0.5)

    if not all_chunks:
        print("ERROR: no chunks collected; corpus not built.")
        return 1

    print(f"\nEmbedding {len(all_chunks)} chunks …")
    await embed_and_store(
        chunks=all_chunks,
        index_path=str(EVAL_INDEX),
        chunk_path=str(EVAL_CHUNKS),
        embedder=EmbeddingService(),
    )

    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"suite: {suite}")
    print(f"\nCorpus ready: {len(all_chunks)} chunks from {len(manifest)} topics")
    print(f"  index    -> {EVAL_INDEX}")
    print(f"  manifest -> {MANIFEST}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=sorted(SUITES), default="general")
    ap.add_argument("--per-topic", type=int, default=8, help="abstracts per topic")
    ap.add_argument("--force", action="store_true", help="rebuild if it exists")
    args = ap.parse_args()
    raise SystemExit(asyncio.run(build(args.suite, args.per_topic, args.force)))
