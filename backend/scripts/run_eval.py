import json
import asyncio
import sys
import os
import logging
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime

# Add backend dir to sys.path to import app modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.rag.engine import RAGEngine
from app.llm import generate_answer
from app.config import settings

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

EVAL_FILE = Path(__file__).parent.parent / "data" / "eval_questions.json"
REPORT_FILE = Path(__file__).parent.parent.parent / "docs" / "eval_report.md"

async def eval_faithfulness(question: str, answer: str, context: List[str]) -> float:
    """Uses LLM-as-judge to check if the answer is derived strictly from the context."""
    context_str = "\n".join(context)
    prompt = (
        "Given the following context and an answer to a question, evaluate if the answer is "
        "entirely faithful to the context (i.e., it does not hallucinate or use outside knowledge). "
        "Answer ONLY '1' if it is completely faithful, or '0' if it includes information not found in the context.\n\n"
        f"Context:\n{context_str}\n\n"
        f"Question: {question}\n\n"
        f"Answer: {answer}\n\n"
        "Score (0 or 1):"
    )
    # Using existing LLM to judge
    try:
        res = await generate_answer("Evaluate faithfulness", [prompt])
        score_str = res.strip()
        if "1" in score_str: return 1.0
        return 0.0
    except Exception as e:
        logger.error(f"Faithfulness eval failed: {e}")
        return 0.0

async def eval_answer_relevancy(question: str, answer: str) -> float:
    """Uses LLM-as-judge to check if the answer directly addresses the question."""
    prompt = (
        "Given the following question and an answer, evaluate if the answer directly and appropriately "
        "answers the question. Answer ONLY '1' if it is highly relevant and addresses the core question, "
        "or '0' if it is evasive, off-topic, or irrelevant.\n\n"
        f"Question: {question}\n\n"
        f"Answer: {answer}\n\n"
        "Score (0 or 1):"
    )
    try:
        res = await generate_answer("Evaluate relevancy", [prompt])
        score_str = res.strip()
        if "1" in score_str: return 1.0
        return 0.0
    except Exception as e:
        logger.error(f"Answer relevancy eval failed: {e}")
        return 0.0

def eval_context_precision(retrieved_contexts: List[str], ground_truth: str) -> float:
    """Keyword overlap heuristic to check if retrieved chunks are relevant to the ground truth."""
    if not retrieved_contexts:
        return 0.0
    
    gt_words = set(ground_truth.lower().split())
    if not gt_words: return 0.0
    
    total_score = 0
    for chunk in retrieved_contexts:
        chunk_words = set(chunk.lower().split())
        overlap = len(gt_words.intersection(chunk_words))
        score = overlap / len(gt_words)
        total_score += min(score, 1.0)
    
    return min(total_score / len(retrieved_contexts), 1.0)

def eval_context_recall(retrieved_contexts: List[str], ground_truth_contexts: List[str]) -> float:
    """Overlap measure to check if retrieved chunks cover the ground truth context."""
    if not ground_truth_contexts or not retrieved_contexts:
        return 0.0
        
    retrieved_text = " ".join(retrieved_contexts).lower()
    
    total_score = 0
    for gt_ctx in ground_truth_contexts:
        gt_words = set(gt_ctx.lower().split())
        if not gt_words: continue
        
        retrieved_words = set(retrieved_text.split())
        overlap = len(gt_words.intersection(retrieved_words))
        score = overlap / len(gt_words)
        total_score += score
        
    return total_score / len(ground_truth_contexts)

async def main():
    logger.info("Starting RAG Evaluation Harness...")
    
    if not os.path.exists(settings.FAISS_INDEX_PATH):
        logger.warning(f"FAISS index not found at {settings.FAISS_INDEX_PATH}. Skipping evaluation.")
        print("\n[WARNING] No FAISS index exists. Run ingestion first to evaluate retrieval.\n")
        return
        
    if not EVAL_FILE.exists():
        logger.error(f"Eval questions not found at {EVAL_FILE}")
        return
        
    with open(EVAL_FILE, "r") as f:
        dataset = json.load(f)
        
    engine = RAGEngine()
    engine.initialize()
    
    results = []
    
    print("\nRunning Evaluation...\n")
    
    for i, item in enumerate(dataset):
        q = item["question"]
        gt_answer = item["ground_truth"]
        gt_context = item["ground_truth_context"]
        
        print(f"[{i+1}/{len(dataset)}] Q: {q}")
        
        # 1. RAG pipeline
        response = await engine.query(q, k=3)
        retrieved_chunks = response.chunks
        answer = response.answer
        
        # 2. Metrics
        faithfulness = await eval_faithfulness(q, answer, retrieved_chunks)
        relevancy = await eval_answer_relevancy(q, answer)
        precision = eval_context_precision(retrieved_chunks, gt_answer)
        recall = eval_context_recall(retrieved_chunks, gt_context)
        
        res_dict = {
            "question": q,
            "faithfulness": faithfulness,
            "answer_relevancy": relevancy,
            "context_precision": precision,
            "context_recall": recall
        }
        results.append(res_dict)
        print(f"  -> Faithfulness: {faithfulness}, Relevancy: {relevancy}, Precision: {precision:.2f}, Recall: {recall:.2f}")

    # Averages
    avg_faithfulness = sum(r["faithfulness"] for r in results) / len(results)
    avg_relevancy = sum(r["answer_relevancy"] for r in results) / len(results)
    avg_precision = sum(r["context_precision"] for r in results) / len(results)
    avg_recall = sum(r["context_recall"] for r in results) / len(results)
    
    print("\n=== EVALUATION RESULTS ===")
    print(f"Faithfulness:     {avg_faithfulness:.2f}")
    print(f"Answer Relevancy: {avg_relevancy:.2f}")
    print(f"Context Precision:{avg_precision:.2f}")
    print(f"Context Recall:   {avg_recall:.2f}")
    print("==========================\n")
    
    # Write to report
    if REPORT_FILE.exists():
        with open(REPORT_FILE, "r") as f:
            report_content = f.read()
            
        # Create table
        table = "| Metric | Score |\n|---|---|\n"
        table += f"| Faithfulness | {avg_faithfulness:.2f} |\n"
        table += f"| Answer Relevancy | {avg_relevancy:.2f} |\n"
        table += f"| Context Precision | {avg_precision:.2f} |\n"
        table += f"| Context Recall | {avg_recall:.2f} |\n"
        
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Simple replace for placeholders
        if "<!-- EVAL_RESULTS -->" in report_content:
            report_content = report_content.replace("<!-- EVAL_RESULTS -->", table)
        if "<!-- TIMESTAMP -->" in report_content:
            report_content = report_content.replace("<!-- TIMESTAMP -->", ts)
            
        with open(REPORT_FILE, "w") as f:
            f.write(report_content)
        logger.info(f"Report updated at {REPORT_FILE}")

if __name__ == "__main__":
    asyncio.run(main())
