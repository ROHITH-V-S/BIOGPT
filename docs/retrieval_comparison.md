# Retrieval Quality Comparison

This document compares the retrieval quality of the general-purpose OpenRouter embedding model vs. a domain-specific PubMedBERT model on a sample biomedical paper.

## Methodology

- **Corpus**: `data/sample_paper.pdf`
- **Queries**: 10 varied biomedical questions (e.g., cancer genetics, drug interactions, protein structure).
- **Models**:
  1. OpenRouter (General): `nvidia/nemotron-3-embed-1b:free`
  2. PubMedBERT (Biomedical): `pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb`
- **Metrics**: 
  - Precision@3 and Precision@5: Fraction of retrieved chunks containing domain keywords.
  - Average Distance: L2 distance in FAISS.
  - Qualitative comparison of chunks retrieved by both models.

## Results

<!-- RESULTS_TABLE -->

## Analysis & Conclusions

<!-- ANALYSIS -->
