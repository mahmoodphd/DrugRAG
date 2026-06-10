#!/usr/bin/env python3
"""
Cold re-run of DrugRAG + RAGAS faithfulness for both DrugRAG and Naive RAG.

Steps:
  1. Clear LightRAG LLM response cache (backup first)
  2. Re-run DrugRAG on 20 questions (cold, real latency), capturing KG contexts
  3. Re-run naive RAG retrieval to capture contexts
  4. Compute RAGAS faithfulness for both
  5. Update eval_grounded_qa_summary.json with cold latency + faithfulness
"""

import asyncio
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dashboard"))

import config
import rag_chat
from tools.run_grounded_eval import EVAL_QUESTIONS, NaiveRAG, extract_pmids, verify_pmids

WORKING_DIR = ROOT / "data" / "phase3" / "lightrag_working_v2"
CACHE_FILE = WORKING_DIR / "kv_store_llm_response_cache.json"
OUT_DIR = ROOT / "data" / "phase4"


def clear_cache():
    """Backup and clear the LightRAG LLM response cache."""
    backup = CACHE_FILE.with_suffix(".json.bak")
    if CACHE_FILE.exists():
        print(f"  Backing up cache -> {backup.name}")
        shutil.copy2(CACHE_FILE, backup)
        CACHE_FILE.write_text("{}")
        print(f"  Cache cleared (was {CACHE_FILE.stat().st_size / 1e6:.0f} MB)")
    else:
        print("  No cache file found")


async def run_drugrag_cold():
    """Re-run all 20 Qs through DrugRAG with cleared cache, capturing contexts."""
    from lightrag import QueryParam
    rag = await rag_chat.get_rag()

    results = []
    total = len(EVAL_QUESTIONS)
    for i, q in enumerate(EVAL_QUESTIONS, 1):
        print(f"  [{i}/{total}] {q[:60]}...", end=" ", flush=True)
        t_start = time.perf_counter()
        try:
            result = await rag.aquery_llm(q, param=QueryParam(mode="hybrid"))
            ans = result.get("llm_response", {}).get("content", "") or ""
            lat = time.perf_counter() - t_start

            # Extract context chunks from raw_data
            raw = result.get("data", {})
            context_chunks = []
            if "chunks" in raw:
                for chunk in raw["chunks"]:
                    if isinstance(chunk, dict) and "content" in chunk:
                        context_chunks.append(chunk["content"][:800])
            if not context_chunks and "entities" in raw:
                for ent in raw["entities"][:5]:
                    if isinstance(ent, dict):
                        desc = ent.get("description", ent.get("content", ""))
                        if desc:
                            context_chunks.append(desc[:500])
            if not context_chunks and ans:
                context_chunks = [ans[:1500]]

        except Exception as e:
            print(f"ERROR: {e}")
            ans, lat = "", time.perf_counter() - t_start
            context_chunks = []

        pmids = extract_pmids(ans or "")
        verdict = verify_pmids(pmids)
        n_verified = sum(1 for v in verdict.values() if v)

        print(f"lat={lat:.1f}s, cites={len(pmids)} ({n_verified} verif.), ctx={len(context_chunks)}")
        results.append({
            "q_id": i,
            "question": q,
            "answer": ans or "",
            "latency_s": round(lat, 2),
            "pmids": pmids,
            "n_verified": n_verified,
            "contexts": context_chunks,
        })
        if i < total:
            await asyncio.sleep(1.5)

    return results


def capture_naive_contexts(naive_rag):
    """Get the retrieved passage texts for each question (for RAGAS contexts)."""
    contexts = []
    for q in EVAL_QUESTIONS:
        hits = naive_rag.retrieve(q, top_k=5)
        ctx_parts = []
        for doc_text, doc_pmid in hits:
            prefix = f"[PMID: {doc_pmid}] " if doc_pmid else ""
            ctx_parts.append(f"{prefix}{doc_text[:800]}")
        contexts.append(ctx_parts)
    return contexts


async def compute_ragas_faithfulness(questions, answers, contexts_list):
    """Compute RAGAS faithfulness for a set of Q&A with contexts."""
    from datasets import Dataset
    from ragas import evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    valid = [(q, a, c) for q, a, c in zip(questions, answers, contexts_list)
             if a and len(a.strip()) > 10 and c]
    if not valid:
        return None, []

    valid_q, valid_a, valid_c = zip(*valid)
    data = {
        "question": list(valid_q),
        "answer": list(valid_a),
        "contexts": list(valid_c),
    }
    dataset = Dataset.from_dict(data)

    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    llm = ChatOpenAI(
        model=rag_chat.LLM_MODEL,
        api_key=config.OPENROUTER_API_KEY,
        base_url=config.OPENROUTER_BASE_URL,
        temperature=0,
    )
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=config.OPENROUTER_API_KEY,
        base_url=config.OPENROUTER_BASE_URL,
    )

    from ragas.metrics import Faithfulness
    ragas_llm = LangchainLLMWrapper(llm)
    ragas_emb = LangchainEmbeddingsWrapper(embeddings)

    try:
        result = evaluate(
            dataset,
            metrics=[Faithfulness()],
            llm=ragas_llm,
            embeddings=ragas_emb,
        )
        scores = result.to_pandas()["faithfulness"].tolist()
        mean_score = sum(scores) / len(scores) if scores else 0.0
        return round(mean_score, 4), scores
    except Exception as e:
        print(f"  RAGAS error: {e}")
        import traceback
        traceback.print_exc()
        return None, []


async def main_async():
    print("=" * 60)
    print("  Cold Re-run + RAGAS Faithfulness")
    print("=" * 60)

    # Step 1: Clear cache
    print("\n[1/4] Clearing LightRAG LLM response cache...")
    clear_cache()

    # Step 2: Cold DrugRAG
    print("\n[2/4] Running DrugRAG cold (20 questions)...")
    drugrag_results = await run_drugrag_cold()

    cold_path = OUT_DIR / "eval_drugrag_cold.json"
    with open(cold_path, "w") as f:
        json.dump(drugrag_results, f, indent=2)
    print(f"  Saved cold DrugRAG results: {cold_path}")

    lats = [r["latency_s"] for r in drugrag_results if r["latency_s"] > 0]
    drugrag_median_lat = round(sorted(lats)[len(lats) // 2], 1) if lats else 0
    drugrag_mean_lat = round(sum(lats) / len(lats), 1) if lats else 0
    print(f"  DrugRAG cold latency: median={drugrag_median_lat}s, mean={drugrag_mean_lat}s")

    # Step 3: Naive RAG contexts
    print("\n[3/4] Capturing naive RAG contexts...")
    naive_rag = NaiveRAG()
    naive_contexts = capture_naive_contexts(naive_rag)
    print(f"  Captured contexts for {len(naive_contexts)} questions")

    # DrugRAG contexts from aquery_llm raw_data (captured during cold run)
    drugrag_contexts = [r["contexts"] for r in drugrag_results]

    # Load existing naive answers from CSV
    import pandas as pd
    existing_csv = OUT_DIR / "eval_grounded_qa.csv"
    if existing_csv.exists():
        df = pd.read_csv(existing_csv)
        naive_answers = df["naive_answer"].fillna("").tolist()[:20]
    else:
        print("  ERROR: eval_grounded_qa.csv not found")
        naive_answers = [""] * 20

    drugrag_answers = [r["answer"] for r in drugrag_results]
    questions = EVAL_QUESTIONS

    # Step 4: RAGAS faithfulness
    print("\n[4/4] Computing RAGAS faithfulness...")

    print("  Naive RAG faithfulness...")
    naive_faith, naive_scores = await compute_ragas_faithfulness(
        questions, naive_answers, naive_contexts
    )
    if naive_faith is not None:
        print(f"  -> Naive RAG faithfulness: {naive_faith:.4f} ({len(naive_scores)} scored)")
    else:
        print("  -> Naive RAG faithfulness: FAILED")

    print("  DrugRAG faithfulness...")
    drugrag_faith, drugrag_scores = await compute_ragas_faithfulness(
        questions, drugrag_answers, drugrag_contexts
    )
    if drugrag_faith is not None:
        print(f"  -> DrugRAG faithfulness: {drugrag_faith:.4f} ({len(drugrag_scores)} scored)")
    else:
        print("  -> DrugRAG faithfulness: FAILED")

    # Update summary JSON
    summary_path = OUT_DIR / "eval_grounded_qa_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
    else:
        summary = {}

    if "DrugRAG (LightRAG hybrid)" in summary:
        summary["DrugRAG (LightRAG hybrid)"]["latency_mean_s"] = drugrag_mean_lat
        summary["DrugRAG (LightRAG hybrid)"]["latency_median_s"] = drugrag_median_lat

    if naive_faith is not None and "Naive RAG" in summary:
        summary["Naive RAG"]["faithfulness_this_run"] = naive_faith
    if drugrag_faith is not None and "DrugRAG (LightRAG hybrid)" in summary:
        summary["DrugRAG (LightRAG hybrid)"]["faithfulness_this_run"] = drugrag_faith

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nUpdated summary: {summary_path}")

    print("\n" + "=" * 60)
    print("  UPDATED SUMMARY")
    print("=" * 60)
    for name, stats in summary.items():
        faith = stats.get("faithfulness_this_run", "N/A")
        lat = stats.get("latency_median_s", "N/A")
        print(f"  {name:30s}  verif={stats.get('verifiable_citation_rate_pct','?'):>8s}  "
              f"faith={faith}  lat={lat}s")

    return summary


if __name__ == "__main__":
    rag_chat._run(main_async())
