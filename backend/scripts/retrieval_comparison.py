"""
Embedding-model comparison — does a domain-specific encoder retrieve better?

The question
-----------
``EMBEDDING_BACKEND`` offers a general-purpose API model, a general-purpose
local model, and a biomedical one. The intuition is that a PubMed-pretrained
encoder should win on PubMed text. This script measures whether it does.

Only the *encoder* varies. The chunk set, the queries, the gold answers, k, and
the relevance threshold are held fixed, and entity-aware re-ranking is off in
every arm — so any difference is attributable to the embedding space.

Metrics are imported from ``run_eval`` rather than redefined. Two harnesses
with two subtly different notions of "precision" produce numbers that cannot
be compared, and the whole point of both is comparison.

Why there is no distance column
-------------------------------
The previous version of this script reported mean L2 distance per model. That
number is meaningless across encoders: distances live in different spaces with
different dimensionalities and norms, so a smaller mean says nothing about
retrieval quality. Rank-based metrics are the only fair comparison.

Cost note
---------
The OpenRouter arm reuses the FAISS index ``build_eval_corpus.py`` already
wrote (it was built with exactly this model), so only the queries are embedded
through the API — 8 requests per suite rather than ~40. The local arms cost
nothing but CPU. First run downloads PubMedBERT (~440 MB) into the
HuggingFace cache.

Usage::

    python scripts/build_eval_corpus.py --suite general   # once
    python scripts/retrieval_comparison.py --suite general
"""

import argparse
import asyncio
import json
import logging
import os
import pickle
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List

import faiss
import numpy as np
from tabulate import tabulate

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import settings  # noqa: E402
from app.embeddings import EmbeddingService  # noqa: E402

# Shared metric definitions — see module docstring.
from run_eval import (  # noqa: E402
    RELEVANCE_THRESHOLD,
    SUITES,
    TOP_K,
    context_precision,
    context_recall,
    hit_and_rr,
    suite_paths,
)

logger = logging.getLogger(__name__)

# See the same note in run_eval.py: a redirected stdout on Windows defaults to
# cp1252 and cannot encode the "·" this script prints.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BACKEND_DIR = Path(__file__).resolve().parent.parent
REPORT_FILE = BACKEND_DIR.parent / "docs" / "retrieval_comparison.md"

#: (display label, EMBEDDING_BACKEND value, config attribute holding the model)
BACKENDS = [
    ("OpenRouter (general, API)", "openrouter", "EMBEDDING_MODEL"),
    ("MiniLM (general, local)", "local", "LOCAL_EMBEDDING_MODEL"),
    ("PubMedBERT (biomedical, local)", "biomedical", "BIOMEDICAL_EMBEDDING_MODEL"),
]

#: The arm every other arm's top-1 agreement is measured against.
REFERENCE = "OpenRouter (general, API)"


def _embedder(backend: str) -> EmbeddingService:
    """EmbeddingService reads the backend flag at construction time."""
    settings.EMBEDDING_BACKEND = backend
    return EmbeddingService()


async def build_index(
    backend: str, chunks: List[str], prebuilt: Path
) -> tuple[faiss.Index, EmbeddingService]:
    """
    Return a FAISS index over ``chunks`` for ``backend``.

    Reuses ``prebuilt`` for the OpenRouter arm — that index was written by
    build_eval_corpus.py with this exact model, so re-embedding would burn
    ~40 free-tier requests to reproduce vectors already on disk.
    """
    embedder = _embedder(backend)

    if backend == "openrouter" and prebuilt.exists():
        index = faiss.read_index(str(prebuilt))
        if index.ntotal == len(chunks):
            print(f"  reusing prebuilt index ({index.ntotal} vectors, dim {index.d})")
            return index, embedder
        print(f"  prebuilt index is stale ({index.ntotal} != {len(chunks)}) — rebuilding")

    print(f"  embedding {len(chunks)} chunks …")
    started = time.perf_counter()
    vectors = await embedder.embed_texts(chunks)
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    print(f"  built in {time.perf_counter() - started:.1f}s (dim {index.d})")
    return index, embedder


async def evaluate(
    label: str, backend: str, model: str, chunks: List[str], dataset: List[dict],
    prebuilt: Path,
) -> dict:
    print(f"\n--- {label}: {model}")
    index, embedder = await build_index(backend, chunks, prebuilt)

    acc = {"precision": 0.0, "recall": 0.0, "hit": 0.0, "mrr": 0.0}
    top1: List[str] = []
    latencies: List[float] = []

    for i, item in enumerate(dataset, 1):
        question = item["question"]

        started = time.perf_counter()
        query_vec = await embedder.embed_query(question)
        latencies.append(time.perf_counter() - started)

        _, indices = index.search(query_vec, min(TOP_K, index.ntotal))
        retrieved = [chunks[j] for j in indices[0] if 0 <= j < len(chunks)]
        top1.append(retrieved[0] if retrieved else "")

        p = context_precision(retrieved, item["ground_truth"])
        r = context_recall(retrieved, item["ground_truth_context"])
        hit, rr = hit_and_rr(retrieved, item["ground_truth_context"])
        acc["precision"] += p
        acc["recall"] += r
        acc["hit"] += hit
        acc["mrr"] += rr

        print(f"  [{i}/{len(dataset)}] P={p:.3f} R={r:.3f} hit={hit:.0f} rr={rr:.2f}"
              f"  | {question[:48]}")

    n = len(dataset)
    return {
        "label": label,
        "model": model,
        "dim": index.d,
        "latency": sum(latencies) / n,
        "top1": top1,
        **{k: v / n for k, v in acc.items()},
    }


def build_table(results: List[dict]) -> str:
    reference = next((r for r in results if r["label"] == REFERENCE), None)

    rows = []
    for r in results:
        if reference is None or r is reference:
            agreement = "—"
        else:
            same = sum(1 for a, b in zip(r["top1"], reference["top1"]) if a == b)
            agreement = f"{same}/{len(r['top1'])}"
        rows.append([
            r["label"], r["dim"],
            f"{r['precision']:.3f}", f"{r['recall']:.3f}",
            f"{r['hit']:.3f}", f"{r['mrr']:.3f}",
            agreement, f"{r['latency'] * 1000:.0f} ms",
        ])

    return tabulate(
        rows,
        headers=["Embedding model", "Dim", "Precision@3", "Recall",
                 "Hit@3", "MRR", "Same top-1 as ref.", "Query latency"],
        tablefmt="pipe",
    )


def write_report(suite: str, results: List[dict], corpus: dict) -> None:
    if not REPORT_FILE.exists():
        logger.warning("Report file missing: %s", REPORT_FILE)
        return

    meta = (
        f"- **Suite**: `{suite}`\n"
        f"- **Corpus**: {corpus['chunks']} chunks from {corpus['abstracts']} PubMed "
        f"abstracts across {corpus['topics']} topics\n"
        f"- **Questions**: {corpus['questions']}  ·  **k**: {TOP_K}  ·  "
        f"**Relevance threshold**: {RELEVANCE_THRESHOLD} content-word coverage\n"
        f"- **Re-ranking**: disabled in every arm (encoder is the only variable)\n"
        f"- **Reference arm for top-1 agreement**: {REFERENCE}\n"
    )
    block = (f"*Last run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
             f"{meta}\n{build_table(results)}\n")

    start, end = f"<!-- RESULTS:{suite} -->", f"<!-- /RESULTS:{suite} -->"
    content = REPORT_FILE.read_text(encoding="utf-8")

    if start in content and end in content:
        content = re.sub(re.escape(start) + r".*?" + re.escape(end),
                         f"{start}\n{block}{end}", content, flags=re.S)
    else:
        content += f"\n\n### Suite: `{suite}`\n\n{start}\n{block}{end}\n"

    REPORT_FILE.write_text(content, encoding="utf-8")
    print(f"\nReport written -> {REPORT_FILE} (suite: {suite})")


async def main(suite: str) -> int:
    questions_file, index_path, chunk_path = suite_paths(suite)

    if not chunk_path.exists():
        print(f"No '{suite}' corpus found. Run:  "
              f"python scripts/build_eval_corpus.py --suite {suite}")
        return 1

    dataset = json.loads(questions_file.read_text(encoding="utf-8"))
    with open(chunk_path, "rb") as fh:
        chunks = pickle.load(fh)

    manifest_path = index_path.parent / "corpus_manifest.json"
    manifest = (json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest_path.exists() else [])
    corpus = {
        "chunks": len(chunks),
        "abstracts": sum(len(m["pmids"]) for m in manifest),
        "topics": len(manifest),
        "questions": len(dataset),
    }

    print(f"Suite: {suite} — {len(chunks)} chunks, {len(dataset)} questions, k={TOP_K}")

    original_backend = settings.EMBEDDING_BACKEND
    results = []
    try:
        for label, backend, model_attr in BACKENDS:
            model = getattr(settings, model_attr)
            try:
                results.append(
                    await evaluate(label, backend, model, chunks, dataset, index_path)
                )
            except Exception as exc:
                # One unavailable encoder (no quota, failed model download)
                # should not discard the arms that did run.
                print(f"  SKIPPED — {type(exc).__name__}: {exc}")
                logger.warning("Backend %s failed: %s", backend, exc)
    finally:
        settings.EMBEDDING_BACKEND = original_backend

    if not results:
        print("\nEvery backend failed — nothing to report.")
        return 1

    print("\n" + build_table(results))
    write_report(suite, results, corpus)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=sorted(SUITES), default="general")
    args = ap.parse_args()
    raise SystemExit(asyncio.run(main(args.suite)))
