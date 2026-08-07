"""
Domain-specific retrieval quality comparison script.
Compares OpenRouter embeddings against a biomedical-specific PubMedBERT model.
"""

import asyncio
import os
import sys
import faiss
import numpy as np
from tabulate import tabulate

# Add backend to path so we can import from app
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from app.config import settings
from app.embeddings import EmbeddingService
from app.rag.loader import load_pdf_chunks

QUERIES = [
    "What are the genetic markers for breast cancer?",
    "How does aspirin interact with blood thinners?",
    "Describe the secondary structure of hemoglobin.",
    "What is the mechanism of action for mRNA vaccines?",
    "How do ACE inhibitors affect blood pressure?",
    "What are the side effects of chemotherapy in pediatric patients?",
    "Explain the role of p53 in tumor suppression.",
    "What is the impact of vitamin D on bone density?",
    "How does CRISPR-Cas9 edit genes?",
    "What is the correlation between cholesterol levels and heart disease?",
]

KEYWORDS = [
    "gene", "cancer", "tumor", "protein", "vaccine", "inhibitor", "blood", 
    "chemotherapy", "vitamin", "crispr", "cholesterol", "heart", "disease",
    "structure", "mechanism", "pressure", "pediatric", "density", "edit"
]

def compute_precision(retrieved_chunks, k):
    """Compute Precision@K based on heuristic keyword matching."""
    chunks_at_k = retrieved_chunks[:k]
    if not chunks_at_k:
        return 0.0
    relevant = 0
    for chunk in chunks_at_k:
        chunk_lower = chunk.lower()
        if any(kw in chunk_lower for kw in KEYWORDS):
            relevant += 1
    return relevant / k

async def build_index(chunks, embedder):
    embeddings = await embedder.embed_texts(chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

async def main():
    # Setup paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    pdf_path = os.path.join(base_dir, "backend", "data", "sample_paper.pdf")
    docs_path = os.path.join(base_dir, "docs", "retrieval_comparison.md")

    if not os.path.exists(pdf_path):
        print(f"Error: Could not find {pdf_path}")
        return

    # Load chunks
    chunks = load_pdf_chunks(pdf_path)
    if not chunks:
        print("No chunks loaded.")
        return

    print(f"Loaded {len(chunks)} chunks from {pdf_path}")

    # Setup EmbeddingServices
    settings.EMBEDDING_BACKEND = "openrouter"
    openrouter_embedder = EmbeddingService()
    
    settings.EMBEDDING_BACKEND = "biomedical"
    biomedical_embedder = EmbeddingService()

    print("Building OpenRouter index...")
    openrouter_index = await build_index(chunks, openrouter_embedder)
    
    print("Building Biomedical index...")
    biomedical_index = await build_index(chunks, biomedical_embedder)

    results = []
    
    print("\nStarting comparison...")
    for query in QUERIES:
        print(f"\nQuery: {query}")
        # OpenRouter
        or_q_emb = await openrouter_embedder.embed_query(query)
        or_distances, or_indices = openrouter_index.search(or_q_emb, 5)
        or_chunks = [chunks[i] for i in or_indices[0] if i < len(chunks)]
        or_p3 = compute_precision(or_chunks, 3)
        or_p5 = compute_precision(or_chunks, 5)
        or_avg_dist = float(np.mean(or_distances[0]))
        
        # Biomedical
        bio_q_emb = await biomedical_embedder.embed_query(query)
        bio_distances, bio_indices = biomedical_index.search(bio_q_emb, 5)
        bio_chunks = [chunks[i] for i in bio_indices[0] if i < len(chunks)]
        bio_p3 = compute_precision(bio_chunks, 3)
        bio_p5 = compute_precision(bio_chunks, 5)
        bio_avg_dist = float(np.mean(bio_distances[0]))

        results.append({
            "Query": query,
            "OR P@3": or_p3,
            "OR P@5": or_p5,
            "OR AvgDist": or_avg_dist,
            "Bio P@3": bio_p3,
            "Bio P@5": bio_p5,
            "Bio AvgDist": bio_avg_dist,
        })
        
        # Qualitative
        print("OpenRouter Top Chunk:")
        print(f"  {or_chunks[0][:100]}..." if or_chunks else "  None")
        print("Biomedical Top Chunk:")
        print(f"  {bio_chunks[0][:100]}..." if bio_chunks else "  None")

    # Generate Markdown Table
    headers = ["Query", "OpenRouter P@3", "OpenRouter P@5", "OpenRouter AvgDist", "Biomedical P@3", "Biomedical P@5", "Biomedical AvgDist"]
    table_data = [
        [
            r["Query"],
            f"{r['OR P@3']:.2f}",
            f"{r['OR P@5']:.2f}",
            f"{r['OR AvgDist']:.2f}",
            f"{r['Bio P@3']:.2f}",
            f"{r['Bio P@5']:.2f}",
            f"{r['Bio AvgDist']:.2f}",
        ]
        for r in results
    ]
    md_table = tabulate(table_data, headers=headers, tablefmt="pipe")
    
    print("\n" + md_table)
    
    # Update docs/retrieval_comparison.md
    if os.path.exists(docs_path):
        with open(docs_path, "r", encoding="utf-8") as f:
            content = f.read()
        if "<!-- RESULTS_TABLE -->" in content:
            new_content = content.replace("<!-- RESULTS_TABLE -->", md_table)
            with open(docs_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            print(f"\nUpdated {docs_path}")
        else:
            print("\nCould not find <!-- RESULTS_TABLE --> placeholder in markdown file.")
    else:
        print(f"\nCould not find {docs_path} to update.")

if __name__ == "__main__":
    asyncio.run(main())
