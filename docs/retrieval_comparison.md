# Embedding Model Comparison

Does a domain-specific encoder retrieve biomedical text better than a
general-purpose one? `EMBEDDING_BACKEND` offers three options, so this is a
question the project can answer rather than assume.

Reproduce with:

```bash
cd backend
make corpus                                        # build corpora from live PubMed
python scripts/retrieval_comparison.py --suite general
python scripts/retrieval_comparison.py --suite hard
```

## Methodology

Only the **encoder** varies. The chunk set, queries, gold answers, `k`, and the
relevance threshold are identical across arms, and entity-aware re-ranking is
disabled everywhere — so any difference is attributable to the embedding space.

| Arm | Model | Type |
|---|---|---|
| OpenRouter | `nvidia/nemotron-3-embed-1b:free` | general-purpose, hosted API |
| MiniLM | `sentence-transformers/all-MiniLM-L6-v2` | general-purpose, local |
| PubMedBERT | `pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb` | biomedical, local |

Metrics are imported from `scripts/run_eval.py` rather than redefined, so the
numbers here are directly comparable with
[the re-ranking evaluation](eval_report.md): mean content-word overlap for
Precision@3 and Recall, and a 0.20-coverage threshold for Hit@3 / MRR.

Two deliberate choices:

- **No mean-distance column.** An earlier version of this script reported mean
  L2 distance per model. That comparison is invalid — distances live in
  different spaces with different dimensionality and norms, so a smaller mean
  says nothing about ranking quality. Rank-based metrics are the only fair
  basis.
- **Corpora are topic-matched to the questions.** The first version of this
  experiment scored ten oncology and pharmacology queries against a paper about
  bee gut microbiota. Every arm retrieved irrelevant chunks, and the resulting
  table compared noise with noise.

**Query latency** is wall-clock time to embed one query, averaged over the
suite. It is indicative only — the API arm includes network round-trip and the
local arms are CPU-bound on one machine — but the gap is the practical
trade-off, so it is reported.

## Results

### Suite: `general`

Eight unrelated biomedical topics.

<!-- RESULTS:general -->
*Last run: 2026-08-17 20:43:49*

- **Suite**: `general`
- **Corpus**: 303 chunks from 64 PubMed abstracts across 8 topics
- **Questions**: 8  ·  **k**: 3  ·  **Relevance threshold**: 0.2 content-word coverage
- **Re-ranking**: disabled in every arm (encoder is the only variable)
- **Reference arm for top-1 agreement**: OpenRouter (general, API)

| Embedding model                |   Dim |   Precision@3 |   Recall |   Hit@3 |   MRR | Same top-1 as ref.   | Query latency   |
|:-------------------------------|------:|--------------:|---------:|--------:|------:|:---------------------|:----------------|
| OpenRouter (general, API)      |  2048 |         0.321 |    0.493 |   0.875 | 0.812 | —                    | 707 ms          |
| MiniLM (general, local)        |   384 |         0.264 |    0.411 |   0.5   | 0.5   | 0/8                  | 87 ms           |
| PubMedBERT (biomedical, local) |   768 |         0.31  |    0.477 |   0.875 | 0.875 | 3/8                  | 76 ms           |
<!-- /RESULTS:general -->

### Suite: `hard`

Confusable siblings pooled in one corpus (BRCA1/BRCA2, ACE/ACE2, Cas9/Cas12a,
HER2/EGFR) so that near-miss chunks compete with correct ones.

<!-- RESULTS:hard -->
*Last run: 2026-08-17 20:45:40*

- **Suite**: `hard`
- **Corpus**: 244 chunks from 58 PubMed abstracts across 8 topics
- **Questions**: 8  ·  **k**: 3  ·  **Relevance threshold**: 0.2 content-word coverage
- **Re-ranking**: disabled in every arm (encoder is the only variable)
- **Reference arm for top-1 agreement**: OpenRouter (general, API)

| Embedding model                |   Dim |   Precision@3 |   Recall |   Hit@3 |   MRR | Same top-1 as ref.   | Query latency   |
|:-------------------------------|------:|--------------:|---------:|--------:|------:|:---------------------|:----------------|
| OpenRouter (general, API)      |  2048 |         0.407 |    0.477 |   1     | 1     | —                    | 513 ms          |
| MiniLM (general, local)        |   384 |         0.331 |    0.457 |   0.875 | 0.75  | 3/8                  | 22 ms           |
| PubMedBERT (biomedical, local) |   768 |         0.422 |    0.547 |   1     | 0.938 | 1/8                  | 82 ms           |
<!-- /RESULTS:hard -->

## Analysis

**A 768-dimension biomedical encoder running locally matches a 2048-dimension
hosted general-purpose model, and a general-purpose local model of the same
convenience does not.**

| | General suite | Hard suite |
|---|---|---|
| PubMedBERT vs OpenRouter — Precision@3 | −0.011 | **+0.015** |
| PubMedBERT vs OpenRouter — Recall | −0.016 | **+0.070** |
| PubMedBERT vs OpenRouter — MRR | **+0.063** | −0.062 |
| MiniLM vs OpenRouter — MRR | −0.312 | −0.250 |

PubMedBERT trades small amounts in both directions against OpenRouter and comes
out even; MiniLM loses decisively on both suites. MiniLM is the control that
makes this interpretable: it is local, small and fast in exactly the way
PubMedBERT is, so the gap between them cannot be explained by hosting,
dimensionality or inference budget. What separates them is domain pretraining.

### Where the domain model actually helps

The general suite's CRISPR question is the clearest case. It is OpenRouter's
only outright failure on that suite (Precision 0.100, no relevant chunk in the
top 3); PubMedBERT scores 0.317 and puts a relevant chunk at rank 1. On the
hard suite, the biggest single gain is the gefitinib/erlotinib question
(Precision 0.364 → 0.606, Recall 0.556 → 0.833), where distinguishing EGFR from
HER2 requires knowing that both are receptor tyrosine kinases.

The failure direction is just as informative. On the hard suite's Cas12a
question, PubMedBERT drops the correct chunk to rank 2 while OpenRouter holds
rank 1 — which is what costs it the MRR column. The domain model is not
uniformly better; it is better at fine-grained biomedical distinctions and
occasionally worse at coarse ones.

### The two encoders retrieve different evidence

Despite near-identical aggregate scores, PubMedBERT's top-1 chunk matches
OpenRouter's on only **3/8** questions (general) and **1/8** (hard). Two models
scoring the same are not doing the same thing — they are surfacing largely
disjoint evidence of comparable quality.

That has a concrete implication: the aggregate table alone would justify
treating the encoders as interchangeable, and the agreement column shows that
conclusion would be wrong. It is also the strongest argument in this repo for
hybrid or ensemble retrieval — two encoders with 1/8 overlap and equal accuracy
have complementary recall that neither exploits alone.

### Practical consequence

Answer generation is the only part of this system that genuinely needs a hosted
model. Retrieval does not: on this evidence the local biomedical encoder is
quality-neutral, **6–9× faster per query** (76–82 ms vs 513–707 ms), and
consumes none of the 50-request/day free-tier budget that is otherwise the
system's binding constraint.

The default remains `EMBEDDING_BACKEND=openrouter` regardless, for two reasons
worth stating rather than hiding: the biomedical backend needs a ~440 MB model
download before first use, and switching changes the vector dimensionality from
2048 to 768, which invalidates every index already on disk. Eight questions per
suite is not enough evidence to impose that migration cost by default. It is
enough to document the option and the measurement behind it.

### Relationship to the re-ranking result

[`eval_report.md`](eval_report.md) found that entity-aware re-ranking buys
nothing on these corpora, because dense retrieval already saturates Hit@3 and
MRR. The two experiments together make the sharpest available comparison:

- Changing the **encoder** moves Hit@3 by up to **0.375** (MiniLM vs the other
  two on the `general` suite).
- Changing the **ranking** on top of a fixed encoder moves Hit@3 by exactly
  **0.000** — in all six configurations of that experiment.

All three encoders in this comparison were fed back into the re-ranking harness
to produce those six arms, which is what makes the contrast a measurement rather
than a juxtaposition. Where retrieval quality is concerned, the embedding space
is doing the work.

## Limitations

- **Eight questions per suite.** Directional, not statistically significant.
  A single question flipping changes Hit@3 by 0.125.
- **Lexical-overlap proxies, not human relevance judgments.** A chunk that
  paraphrases the gold answer without sharing vocabulary is scored as a miss.
- **The OpenRouter arm reuses the corpus index** rather than re-embedding, to
  stay inside the free-tier request budget. The vectors are from the same model
  and the same chunk text, so this is a cost optimisation, not a methodological
  shortcut.
- **Latency is measured on one machine, unbatched**, and the API arm's figure
  depends on network conditions at run time.
