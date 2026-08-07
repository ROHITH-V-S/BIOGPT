# BioGPT Explorer - Evaluation Report

This document contains the evaluation results of the BioGPT Explorer RAG pipeline.

## Methodology

We evaluate the RAG pipeline using a held-out dataset of biomedical Q&A pairs. We use custom implementations of the RAGAS metrics:

- **Faithfulness**: Does the answer only state things supported by the retrieved context? (Evaluated via LLM-as-judge)
- **Answer Relevancy**: Is the answer relevant to the user's query? (Evaluated via LLM-as-judge)
- **Context Precision**: Are the retrieved chunks relevant to the ground truth answer? (Keyword overlap heuristic)
- **Context Recall**: Do the retrieved chunks cover the ground truth context? (Overlap measure)

## Results

*Last evaluated:* <!-- TIMESTAMP -->

<!-- EVAL_RESULTS -->
