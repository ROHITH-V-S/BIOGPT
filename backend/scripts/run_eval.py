"""
RAG evaluation harness — baseline vector search vs entity-aware re-ranking.

What it measures
----------------
The headline experiment is a *retrieval* comparison, so the primary metrics
are deterministic and need no LLM:

* **Context Precision@k** — mean lexical overlap between each retrieved chunk
  and the gold answer.
* **Context Recall** — how much of the gold context's vocabulary the retrieved
  set covers.
* **Hit@k** — fraction of questions where at least one retrieved chunk clears
  ``RELEVANCE_THRESHOLD``.
* **MRR** — mean reciprocal rank of the first such chunk. This is the metric
  re-ranking should move: re-ranking cannot retrieve anything new, it can only
  reorder what vector search already surfaced.

Overlap is computed on content words (stopwords stripped) — otherwise "the",
"of" and "in" dominate every score and the metric mostly measures chunk length.

Generation quality (faithfulness / answer relevancy) is judged by an LLM and
is therefore opt-in via ``--judge``: it costs 2 extra model calls per question
per mode, and the binary 0/1 verdicts are noisy enough that small differences
should not be over-read.

``--llm ollama`` routes those calls to a locally served model. That is not
merely a cost saving: a full judge pass is ~48 calls, which does not fit in
OpenRouter's 50-request daily free tier alongside anything else, so the pass
could not be completed at all against the hosted backend. ``--judge-model``
further allows the judge to be a *different* model than the one under test —
a model grading its own output is a known source of self-preference bias, and
the single-chain default cannot express that separation.

Usage::

    python scripts/build_eval_corpus.py         # once, builds the corpus
    python scripts/run_eval.py                  # retrieval metrics only
    python scripts/run_eval.py --backend local  # same test, weaker encoder
    python scripts/run_eval.py --judge --llm ollama \
        --judge-model gemma3n:e4b               # generation metrics, local

``--backend`` exists to test this harness's own conclusion. The headline result
is that re-ranking does not help because dense retrieval has already saturated;
if that explanation is right, a weaker encoder should create headroom and the
re-ranker should start to pay. Swapping the encoder while holding the chunks,
questions and re-ranker fixed is how that prediction gets checked instead of
assumed. Local backends cost no API quota.
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import settings  # noqa: E402
from app.embeddings import EmbeddingService  # noqa: E402
from app.llm import generate_answer  # noqa: E402
from app.rag.engine import RAGEngine  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# The summary table prints "Δ" and "·". On Windows a redirected stdout defaults
# to cp1252, which cannot encode either, and the run dies *after* computing
# every metric. Reports are written as UTF-8 regardless; this only fixes console
# output.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BACKEND = Path(__file__).resolve().parent.parent
REPORT_FILE = BACKEND.parent / "docs" / "eval_report.md"

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
    )

TOP_K = 3

#: A retrieved chunk counts as relevant when it covers at least this fraction
#: of the gold context's content words. Chosen so that a chunk discussing the
#: same mechanism scores a hit while a merely same-field chunk does not.
RELEVANCE_THRESHOLD = 0.20

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "in", "to", "is", "are", "was", "were",
    "be", "been", "being", "for", "on", "with", "as", "by", "that", "this",
    "these", "those", "it", "its", "from", "at", "which", "can", "has", "have",
    "had", "not", "but", "also", "we", "our", "their", "they", "them", "such",
    "may", "more", "most", "than", "into", "when", "how", "what", "does", "do",
}


def content_words(text: str) -> set[str]:
    """Lowercase alphanumeric tokens, stopwords and very short tokens removed."""
    tokens = re.findall(r"[a-z0-9][a-z0-9\-]*", text.lower())
    return {t for t in tokens if len(t) > 2 and t not in STOPWORDS}


def coverage(chunk: str, reference: str) -> float:
    """Fraction of the reference's content words present in the chunk."""
    ref = content_words(reference)
    if not ref:
        return 0.0
    return len(ref & content_words(chunk)) / len(ref)


# ---------------------------------------------------------------------------
# Retrieval metrics (deterministic)
# ---------------------------------------------------------------------------
def context_precision(chunks: List[str], ground_truth: str) -> float:
    if not chunks:
        return 0.0
    return sum(coverage(c, ground_truth) for c in chunks) / len(chunks)


def context_recall(chunks: List[str], gt_contexts: List[str]) -> float:
    if not chunks or not gt_contexts:
        return 0.0
    joined = " ".join(chunks)
    return sum(coverage(joined, gt) for gt in gt_contexts) / len(gt_contexts)


def hit_and_rr(chunks: List[str], gt_contexts: List[str]) -> tuple[float, float]:
    """Return (hit@k, reciprocal rank) for the first relevant chunk."""
    reference = " ".join(gt_contexts)
    for rank, chunk in enumerate(chunks, start=1):
        if coverage(chunk, reference) >= RELEVANCE_THRESHOLD:
            return 1.0, 1.0 / rank
    return 0.0, 0.0


# ---------------------------------------------------------------------------
# Generation metrics (LLM-as-judge, opt-in)
# ---------------------------------------------------------------------------
#: Model used for judging, when it should differ from the answering model.
#: Set by ``--judge-model``.
JUDGE_MODEL: str | None = None


async def _judge(prompt: str) -> float | None:
    """
    Return the judge's 0/1 verdict, or ``None`` if the call failed.

    ``None`` rather than ``0.0``: a quota-exhausted or errored call is *not*
    evidence of an unfaithful answer, and scoring it 0 silently drags the mean
    down in a way that is indistinguishable from a real negative verdict.
    Failed calls are excluded from the denominator and reported separately.
    """
    try:
        verdict = await generate_answer(
            "Return only 0 or 1.", [prompt],
            models=[JUDGE_MODEL] if JUDGE_MODEL else None,
        )
        return 1.0 if "1" in verdict.strip()[:5] else 0.0
    except Exception as exc:
        logger.error("Judge call failed: %s", exc)
        return None


async def judge_faithfulness(question: str, answer: str, context: List[str]) -> float | None:
    return await _judge(
        "Evaluate whether the ANSWER is entirely supported by the CONTEXT "
        "(no outside knowledge, no invented detail). Reply ONLY '1' if fully "
        "faithful, or '0' otherwise.\n\n"
        f"CONTEXT:\n{chr(10).join(context)}\n\nQUESTION: {question}\n\n"
        f"ANSWER: {answer}\n\nScore (0 or 1):"
    )


async def judge_relevancy(question: str, answer: str) -> float | None:
    return await _judge(
        "Evaluate whether the ANSWER directly addresses the QUESTION. Reply "
        "ONLY '1' if highly relevant, or '0' if evasive or off-topic.\n\n"
        f"QUESTION: {question}\n\nANSWER: {answer}\n\nScore (0 or 1):"
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
#: Judgments keyed on (question, retrieved chunks). The two modes retrieve the
#: *same* context for every query the re-ranker leaves unchanged, so judging
#: both would spend three identical model calls for an identical verdict. On a
#: 50-request/day free tier that difference decides whether the pass completes.
_JUDGE_CACHE: Dict[tuple, tuple[float | None, float | None]] = {}


async def _judged_scores(
    question: str, chunks: List[str]
) -> tuple[float | None, float | None]:
    key = (question, tuple(chunks))
    if key not in _JUDGE_CACHE:
        try:
            answer = await generate_answer(question, chunks) if chunks else "No context."
        except Exception as exc:
            # Generating the answer under test is as quota-bound as judging it.
            # Without this guard an exhausted quota aborts the whole run and the
            # retrieval metrics — which cost nothing and were already computed —
            # are lost with it.
            logger.error("Answer generation failed: %s", exc)
            _JUDGE_CACHE[key] = (None, None)
            return _JUDGE_CACHE[key]

        _JUDGE_CACHE[key] = (
            await judge_faithfulness(question, answer, chunks),
            await judge_relevancy(question, answer),
        )
    return _JUDGE_CACHE[key]


async def prepare_judgments(engine: RAGEngine, dataset: List[dict]) -> None:
    """
    Pre-compute every judgment in two model-batched phases.

    Generating an answer and judging it can use different models. Doing that
    per-question interleaves them, and two models rarely fit in VRAM at once —
    a 2 GB and a 7.5 GB model on a 6 GB card means Ollama evicts and reloads on
    every single call, at 30-200s per load. Answering everything first and
    judging everything second costs two model loads instead of two per question.

    Results land in ``_JUDGE_CACHE``, so ``run_mode`` then makes no LLM calls.
    """
    # Distinct (question, context) pairs across both retrieval modes. The two
    # modes agree on most queries, and an identical context yields an identical
    # verdict, so deduplicating here is what keeps the pass affordable.
    keys: list[tuple[str, tuple[str, ...]]] = []
    for item in dataset:
        question = item["question"]
        for chunks in (
            await engine.retrieve(question, k=TOP_K),
            await engine.retrieve_entity_aware(question, k=TOP_K),
        ):
            key = (question, tuple(chunks))
            if key not in keys:
                keys.append(key)

    print(f"\n=== Judge prepass: {len(keys)} distinct (question, context) pairs ===")

    answers: dict[tuple[str, tuple[str, ...]], str | None] = {}
    for i, key in enumerate(keys, 1):
        question, chunks = key
        try:
            answers[key] = await generate_answer(question, list(chunks))
            status = "ok"
        except Exception as exc:
            logger.error("Answer generation failed: %s", exc)
            answers[key] = None
            status = f"FAILED ({type(exc).__name__})"
        print(f"  answer [{i}/{len(keys)}] {status}  | {question[:46]}")

    for i, (key, answer) in enumerate(answers.items(), 1):
        question, chunks = key
        if answer is None:
            _JUDGE_CACHE[key] = (None, None)
            continue
        faith = await judge_faithfulness(question, answer, list(chunks))
        rel = await judge_relevancy(question, answer)
        _JUDGE_CACHE[key] = (faith, rel)
        fmt = lambda s: "--" if s is None else f"{s:.0f}"  # noqa: E731
        print(f"  judge  [{i}/{len(keys)}] faith={fmt(faith)} rel={fmt(rel)}"
              f"  | {question[:46]}")


async def run_mode(
    engine: RAGEngine, dataset: List[dict], entity_aware: bool, judge: bool
) -> Dict[str, float]:
    label = "entity-aware" if entity_aware else "baseline"
    print(f"\n=== Mode: {label} (k={TOP_K}) ===")

    acc = {"precision": 0.0, "recall": 0.0, "hit": 0.0, "mrr": 0.0, "reordered": 0.0}
    # Judge scores are averaged over *successful* calls only, so these carry
    # their own denominators.
    judged: Dict[str, list[float]] = {"faithfulness": [], "relevancy": []}

    for i, item in enumerate(dataset, 1):
        q, gt, gt_ctx = item["question"], item["ground_truth"], item["ground_truth_context"]

        baseline_chunks = await engine.retrieve(q, k=TOP_K)
        if entity_aware:
            chunks = await engine.retrieve_entity_aware(q, k=TOP_K)
            # A re-ranker that never changes the order cannot change the
            # score; tracking this separates "no effect" from "bad effect".
            acc["reordered"] += 1.0 if chunks != baseline_chunks else 0.0
        else:
            chunks = baseline_chunks

        p = context_precision(chunks, gt)
        r = context_recall(chunks, gt_ctx)
        hit, rr = hit_and_rr(chunks, gt_ctx)

        acc["precision"] += p
        acc["recall"] += r
        acc["hit"] += hit
        acc["mrr"] += rr

        line = f"[{i}/{len(dataset)}] P={p:.3f} R={r:.3f} hit={hit:.0f} rr={rr:.2f}"

        if judge:
            f, a = await _judged_scores(q, chunks)
            for name, score in (("faithfulness", f), ("relevancy", a)):
                if score is not None:
                    judged[name].append(score)
            fmt = lambda s: "--" if s is None else f"{s:.0f}"  # noqa: E731
            line += f" faith={fmt(f)} rel={fmt(a)}"

        print(f"{line}  | {q[:52]}")

    n = len(dataset)
    result = {k: v / n for k, v in acc.items()}
    for name, scores in judged.items():
        result[name] = sum(scores) / len(scores) if scores else 0.0
        result[f"{name}_n"] = float(len(scores))
    return result


def _delta(base: float, new: float) -> str:
    diff = new - base
    if abs(diff) < 1e-9:
        return "±0.000"
    return f"{'+' if diff > 0 else ''}{diff:.3f}"


def write_report(base: Dict[str, float], ent: Dict[str, float], judge: bool, corpus: dict) -> None:
    rows = [
        ("Context Precision@3", "precision"),
        ("Context Recall", "recall"),
        ("Hit@3", "hit"),
        ("MRR", "mrr"),
    ]
    if judge:
        rows += [("Faithfulness (LLM judge)", "faithfulness"),
                 ("Answer Relevancy (LLM judge)", "relevancy")]
        from app.llm import active_backend, active_models
        answering = active_models()[0]
        judging = JUDGE_MODEL or answering
        judge_note = (
            f"enabled — {int(min(base['faithfulness_n'], ent['faithfulness_n']))}"
            f"/{corpus['questions']} questions scored per mode · "
            f"answers by `{answering}`, judged by `{judging}` "
            f"(backend: {active_backend()})"
        )
    else:
        judge_note = "not run (retrieval-only pass)"

    table = "| Metric | Baseline (vector only) | Entity-aware re-rank | Δ |\n|---|---|---|---|\n"
    for label, key in rows:
        table += f"| {label} | {base[key]:.3f} | {ent[key]:.3f} | {_delta(base[key], ent[key])} |\n"
    # Not a scored metric, but the methodology section promises it: without this
    # a no-op re-rank and a neutral re-rank are indistinguishable in the report.
    n = corpus["questions"]
    table += (f"| Queries reordered | — | {ent['reordered'] * n:.0f}/{n} | |\n")

    meta = (
        f"- **Suite**: `{corpus.get('suite', 'general')}`\n"
        f"- **Corpus**: {corpus['chunks']} chunks from {corpus['abstracts']} PubMed "
        f"abstracts across {corpus['topics']} topics (built by "
        "`scripts/build_eval_corpus.py`)\n"
        f"- **Questions**: {corpus['questions']}\n"
        f"- **k**: {TOP_K}  ·  **Relevance threshold**: {RELEVANCE_THRESHOLD} "
        "content-word coverage\n"
        f"- **Embeddings**: `{corpus['model']}` "
        f"(backend: {corpus['backend']})\n"
        f"- **LLM judge**: {judge_note}\n"
    )

    if not REPORT_FILE.exists():
        logger.warning("Report file missing: %s", REPORT_FILE)
        return

    suite = corpus.get("suite", "general")
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    block = f"*Last run: {ts}*\n\n{meta}\n{table}"

    # Each suite+encoder owns a marked region, so re-running one configuration
    # never clobbers another's results.
    region = suite if corpus["backend"] == "openrouter" else f"{suite}-{corpus['backend']}"
    start, end = f"<!-- RESULTS:{region} -->", f"<!-- /RESULTS:{region} -->"
    content = REPORT_FILE.read_text(encoding="utf-8")

    if start in content and end in content:
        content = re.sub(
            re.escape(start) + r".*?" + re.escape(end),
            f"{start}\n{block}{end}",
            content,
            flags=re.S,
        )
    else:
        content += f"\n\n### Suite: `{region}`\n\n{start}\n{block}{end}\n"

    REPORT_FILE.write_text(content, encoding="utf-8")
    print(f"\nReport written -> {REPORT_FILE} (region: {region})")


#: Config attribute holding each backend's model name.
BACKEND_MODEL_ATTR = {
    "openrouter": "EMBEDDING_MODEL",
    "local": "LOCAL_EMBEDDING_MODEL",
    "biomedical": "BIOMEDICAL_EMBEDDING_MODEL",
}


async def index_for_backend(backend: str, index_path: Path, chunk_path: Path) -> Path:
    """
    Return a FAISS index path for ``backend``, building it if necessary.

    The default corpus index was written with the OpenRouter encoder, so any
    other backend needs its own — the dimensionalities differ (2048 / 768 / 384)
    and a mismatched index fails at search time rather than silently.

    Alternate indexes are cached next to the corpus as ``vector.<backend>.index``
    so repeat runs are free. The chunk list is shared and never rebuilt: holding
    the chunk text fixed is what makes the encoders comparable.
    """
    if backend == "openrouter":
        return index_path

    import pickle

    import faiss

    alt_path = index_path.with_suffix(f".{backend}.index")
    with open(chunk_path, "rb") as fh:
        chunks = pickle.load(fh)

    if alt_path.exists():
        index = faiss.read_index(str(alt_path))
        if index.ntotal == len(chunks):
            print(f"Using cached {backend} index: {alt_path.name}")
            return alt_path
        print(f"Cached {backend} index is stale ({index.ntotal} != {len(chunks)}) — rebuilding")

    settings.EMBEDDING_BACKEND = backend
    print(f"Building {backend} index over {len(chunks)} chunks …")
    vectors = await EmbeddingService().embed_texts(chunks)
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(alt_path))
    print(f"Wrote {alt_path.name} (dim {index.d})")
    return alt_path


def memoize_query_embeddings(engine: RAGEngine) -> None:
    """
    Cache query embeddings for the duration of the run.

    Each question is embedded up to three times per suite: once for the
    baseline mode, and twice more in entity-aware mode (which retrieves the
    baseline order as its reordering control). Embeddings are deterministic, so
    recomputing them changes no result — it just triples the API cost of the
    experiment against a 50-request/day free tier.

    Applied to the harness only. The live engine must not cache: its corpus and
    index change under it at ingest time.
    """
    embed_query = engine.embedder.embed_query
    cache: Dict[str, object] = {}

    async def cached(query: str):
        if query not in cache:
            cache[query] = await embed_query(query)
        return cache[query]

    engine.embedder.embed_query = cached


async def main(suite: str, judge: bool, backend: str) -> int:
    EVAL_FILE, EVAL_INDEX, EVAL_CHUNKS = suite_paths(suite)

    if not EVAL_INDEX.exists():
        print(f"No '{suite}' corpus found. Run:  "
              f"python scripts/build_eval_corpus.py --suite {suite}")
        return 1
    print(f"Suite: {suite}  ·  encoder: {backend}")

    dataset = json.loads(EVAL_FILE.read_text(encoding="utf-8"))
    index_path = await index_for_backend(backend, EVAL_INDEX, EVAL_CHUNKS)

    # Set after index construction so the engine's own EmbeddingService picks up
    # the same encoder that built the index it is about to search.
    settings.EMBEDDING_BACKEND = backend
    engine = RAGEngine(index_path=str(index_path), chunk_path=str(EVAL_CHUNKS))
    engine.initialize()
    memoize_query_embeddings(engine)
    print(f"Eval corpus: {engine.index.ntotal} vectors, {len(engine.chunks)} chunks")

    manifest_path = EVAL_INDEX.parent / "corpus_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else []
    corpus = {
        "chunks": len(engine.chunks),
        "abstracts": sum(len(m["pmids"]) for m in manifest),
        "topics": len(manifest),
        "questions": len(dataset),
        "backend": backend,
        "model": getattr(settings, BACKEND_MODEL_ATTR[backend]),
    }

    if judge:
        await prepare_judgments(engine, dataset)

    base = await run_mode(engine, dataset, entity_aware=False, judge=judge)
    ent = await run_mode(engine, dataset, entity_aware=True, judge=judge)

    print("\n" + "=" * 66)
    print(f"{'Metric':<28}{'Baseline':>12}{'Entity-aware':>15}{'Δ':>11}")
    print("-" * 66)
    for label, key in [("Context Precision@3", "precision"), ("Context Recall", "recall"),
                       ("Hit@3", "hit"), ("MRR", "mrr")] + (
                      [("Faithfulness", "faithfulness"), ("Answer Relevancy", "relevancy")] if judge else []):
        print(f"{label:<28}{base[key]:>12.3f}{ent[key]:>15.3f}{_delta(base[key], ent[key]):>11}")
    print("-" * 66)
    reordered = f"{ent['reordered'] * len(dataset):.0f}/{len(dataset)}"
    print(f"{'Queries reordered':<28}{'—':>12}{reordered:>15}")
    print("=" * 66)

    corpus["suite"] = suite
    write_report(base, ent, judge, corpus)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=sorted(SUITES), default="general")
    ap.add_argument("--judge", action="store_true", help="also run LLM-as-judge metrics")
    ap.add_argument(
        "--backend", choices=sorted(BACKEND_MODEL_ATTR), default="openrouter",
        help="embedding backend to retrieve with; non-default builds and caches "
             "its own index over the same chunks (local backends cost no quota)",
    )
    ap.add_argument(
        "--llm", choices=("openrouter", "ollama"), default=None,
        help="LLM backend for --judge (default: whatever LLM_BACKEND is set to)",
    )
    ap.add_argument(
        "--judge-model", default=None,
        help="judge with this model instead of the answering model, to avoid "
             "a model grading its own output",
    )
    args = ap.parse_args()

    if args.llm:
        settings.LLM_BACKEND = args.llm
    # Module-level scope, so this rebinds the global the judge helpers read.
    JUDGE_MODEL = args.judge_model

    raise SystemExit(asyncio.run(main(args.suite, args.judge, args.backend)))
