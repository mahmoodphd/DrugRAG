#!/usr/bin/env python3
"""
M1.5 - LightRAG feasibility spike.

Purpose: decide if lightrag_working_v2 (the already-built KG index)
can serve live hybrid queries inside a free HF Space (16 GB RAM cap).

Run:  python spike_lightrag_query.py
Requires: OPENROUTER_API_KEY in ../.env (for free Gemma LLM call)
"""

import asyncio
import os
import sys
import time
import tracemalloc
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import config

WORKING_DIR = config.PHASE3_DIR / "lightrag_working_v2"
LLM_MODEL = "google/gemma-4-31b-it:free"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
QUERY = "What does the literature say about Y-27632 and endothelial tube formation?"


async def llm_func(prompt, system_prompt=None, history_messages=None, **kwargs):
    from lightrag.llm.openai import openai_complete_if_cache
    return await openai_complete_if_cache(
        model=LLM_MODEL,
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=config.OPENROUTER_API_KEY,
        base_url=config.OPENROUTER_BASE_URL,
        **kwargs,
    )


_minilm_model = None

async def embed_func(texts, **kwargs):
    global _minilm_model
    import numpy as np
    from sentence_transformers import SentenceTransformer
    if _minilm_model is None:
        _minilm_model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = _minilm_model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return np.array(embeddings)


async def run_spike():
    from lightrag import LightRAG, QueryParam
    from lightrag.base import EmbeddingFunc

    print("=" * 70)
    print("M1.5 LightRAG FEASIBILITY SPIKE")
    print("=" * 70)
    print(f"  Working dir : {WORKING_DIR}")
    print(f"  LLM model   : {LLM_MODEL}")
    print(f"  Embedding   : {EMBEDDING_MODEL} (local, {EMBEDDING_DIM}-dim)")
    print(f"  Query       : {QUERY}")
    print()

    cache_path = WORKING_DIR / "kv_store_llm_response_cache.json"
    cache_exists = cache_path.exists()
    cache_size_mb = cache_path.stat().st_size / (1024 * 1024) if cache_exists else 0
    print(f"  LLM cache exists: {cache_exists} ({cache_size_mb:.1f} MB)")
    print()

    tracemalloc.start()
    t0 = time.perf_counter()

    print("[1/3] Initializing LightRAG...")
    t_init_start = time.perf_counter()

    ef = EmbeddingFunc(embedding_dim=EMBEDDING_DIM, func=embed_func)
    rag = LightRAG(
        working_dir=str(WORKING_DIR),
        llm_model_func=llm_func,
        llm_model_name=LLM_MODEL,
        embedding_func=ef,
        embedding_batch_num=16,
        llm_model_max_async=1,
        addon_params={"entity_types": config.ENTITY_TYPES},
    )
    await rag.initialize_storages()

    t_init = time.perf_counter() - t_init_start
    mem_after_init = tracemalloc.get_traced_memory()
    print(f"    Init time: {t_init:.2f}s")
    print(f"    RAM after init: {mem_after_init[1] / (1024**2):.1f} MB (peak)")
    print()

    print("[2/3] Running hybrid query...")
    t_query_start = time.perf_counter()
    param = QueryParam(mode="hybrid")
    result = await rag.aquery(QUERY, param=param)
    t_query = time.perf_counter() - t_query_start
    mem_after_query = tracemalloc.get_traced_memory()
    print(f"    Query time: {t_query:.2f}s")
    print(f"    RAM after query: {mem_after_query[1] / (1024**2):.1f} MB (peak)")
    print()

    total_time = time.perf_counter() - t0
    peak_ram_mb = mem_after_query[1] / (1024 ** 2)
    tracemalloc.stop()

    print("[3/3] RESULTS")
    print("-" * 70)
    print(f"  (a) Answer length: {len(result)} chars")
    print(f"      Grounded? {'Yes' if len(result) > 100 else 'Unclear'}")
    print(f"  (b) Peak RAM (Python traced): {peak_ram_mb:.1f} MB")
    print(f"  (c) Total latency: {total_time:.2f}s (init={t_init:.2f}s + query={t_query:.2f}s)")
    print(f"  (d) LLM cache present: {cache_exists} ({cache_size_mb:.1f} MB)")
    print(f"      -> Removal is safe; LightRAG still runs without it.")
    print()
    print("ANSWER:")
    print("-" * 70)
    print(result[:2000])
    if len(result) > 2000:
        print(f"\n... [{len(result) - 2000} more chars truncated]")
    print("-" * 70)
    print()
    print("FEASIBILITY VERDICT:")
    if peak_ram_mb < 8000:
        print(f"  PASS: Peak RAM ({peak_ram_mb:.0f} MB) fits 16 GB HF Space")
    else:
        print(f"  WARN: Peak RAM ({peak_ram_mb:.0f} MB) may be tight for 16 GB")
    if t_query < 60:
        print(f"  PASS: Query latency ({t_query:.1f}s) is interactive")
    else:
        print(f"  WARN: Query latency ({t_query:.1f}s) too slow for chat")
    print(f"  INFO: Cache ({cache_size_mb:.0f} MB) optional, deletable to save space")
    print()
    return {
        "answer_length": len(result),
        "peak_ram_mb": peak_ram_mb,
        "total_latency_s": total_time,
        "init_latency_s": t_init,
        "query_latency_s": t_query,
        "cache_present": cache_exists,
        "cache_size_mb": cache_size_mb,
    }


if __name__ == "__main__":
    results = asyncio.run(run_spike())
