#!/usr/bin/env python3
"""
Full re-index of unique papers with gpt-4o-mini + MiniLM.

Resumable from kv_store_doc_status (LightRAG built-in dedup):
  - On restart, re-calling ainsert() with the same list will skip
    already-processed docs (status=processed in doc_status)
  - PENDING/FAILED/PROCESSING docs are retried

4× parallel via LightRAG's max_parallel_insert=4 (4 chunks in flight at once).

Run under nohup:
  nohup .venv/bin/python full_reindex.py > reindex.log 2>&1 &

Kill/resume:
  - SIGTERM is safe; the doc_status is updated per paper
  - To resume, just re-run the same command
  - Already-processed papers are skipped in microseconds (content-hash dedup)
"""

import asyncio
import json
import os
import signal
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

import config
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
UNIQUE_PAPERS_PATH = config.PHASE3_DIR / "unique_papers.json"
WORKING_DIR = config.PHASE3_DIR / "lightrag_working_v2"  # Fresh dir
COST_REPORT_PATH = config.PHASE3_DIR / "full_reindex_cost.json"
PROGRESS_PATH = config.PHASE3_DIR / "full_reindex_progress.json"

GPT4O_MINI = "openai/gpt-4o-mini"
MINILM_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# LightRAG parallelism: 4 chunks in parallel
LLM_MODEL_MAX_ASYNC = 8
MAX_PARALLEL_INSERT = 4

# Cost tracking
INPUT_PRICE_PER_M = 0.15
OUTPUT_PRICE_PER_M = 0.60

_cost_tracker = {
    "total_calls": 0,
    "total_input_tokens": 0,
    "total_output_tokens": 0,
    "total_input_cost": 0.0,
    "total_output_cost": 0.0,
    "errors": [],
}

_shutdown_requested = False


def _signal_handler(sig, frame):
    global _shutdown_requested
    print(f"\n[!] Signal {sig} received — finishing current operation and exiting gracefully")
    _shutdown_requested = True


# ---------------------------------------------------------------------------
# Wrappers
# ---------------------------------------------------------------------------
async def llm_wrapper(prompt, system_prompt=None, history_messages=None, **kwargs):
    """LLM wrapper using gpt-4o-mini with cost tracking."""
    from lightrag.llm.openai import openai_complete_if_cache

    result = await openai_complete_if_cache(
        model=GPT4O_MINI,
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=config.OPENROUTER_API_KEY,
        base_url=config.OPENROUTER_BASE_URL,
        **kwargs,
    )

    # Estimate tokens (4 chars ≈ 1 token for English)
    input_chars = len(prompt) + (len(system_prompt or ""))
    output_chars = len(result) if result else 0
    est_input_tokens = input_chars // 4
    est_output_tokens = output_chars // 4

    input_cost = est_input_tokens * INPUT_PRICE_PER_M / 1_000_000
    output_cost = est_output_tokens * OUTPUT_PRICE_PER_M / 1_000_000

    _cost_tracker["total_calls"] += 1
    _cost_tracker["total_input_tokens"] += est_input_tokens
    _cost_tracker["total_output_tokens"] += est_output_tokens
    _cost_tracker["total_input_cost"] += input_cost
    _cost_tracker["total_output_cost"] += output_cost

    return result


_minilm_model = None


async def embed_minilm_wrapper(texts, **kwargs):
    """Embedding wrapper using local MiniLM (no API calls, $0). Cached singleton."""
    global _minilm_model
    import numpy as np
    from sentence_transformers import SentenceTransformer

    if _minilm_model is None:
        _minilm_model = SentenceTransformer(MINILM_MODEL)
    embeddings = _minilm_model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return np.array(embeddings)


# ---------------------------------------------------------------------------
# Progress save/load
# ---------------------------------------------------------------------------
def load_progress():
    if PROGRESS_PATH.exists():
        try:
            with open(PROGRESS_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "started_at": None,
        "last_update": None,
        "papers_total": 0,
        "papers_processed": 0,
        "errors": 0,
        "cost_usd": 0.0,
    }


def save_progress(extra=None):
    """Save progress atomically."""
    progress = load_progress()
    progress.update({
        "last_update": time.strftime("%Y-%m-%d %H:%M:%S"),
        "papers_total": _cost_tracker.get("papers_total", 0),
        "papers_processed": _cost_tracker.get("papers_processed", 0),
        "errors": len(_cost_tracker["errors"]),
        "cost_usd": round(
            _cost_tracker["total_input_cost"] + _cost_tracker["total_output_cost"], 4
        ),
    })
    if extra:
        progress.update(extra)
    tmp = PROGRESS_PATH.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(progress, f, indent=2)
    tmp.replace(PROGRESS_PATH)


def save_cost_report():
    total_cost = _cost_tracker["total_input_cost"] + _cost_tracker["total_output_cost"]
    papers_done = _cost_tracker.get("papers_processed", 0)
    report = {
        "label": "full_reindex",
        "config": {
            "llm_model": GPT4O_MINI,
            "embedding_model": MINILM_MODEL,
            "max_parallel_insert": MAX_PARALLEL_INSERT,
            "llm_model_max_async": LLM_MODEL_MAX_ASYNC,
            "working_dir": str(WORKING_DIR),
        },
        "progress": {
            "papers_total": _cost_tracker.get("papers_total", 0),
            "papers_processed": papers_done,
            "errors": len(_cost_tracker["errors"]),
        },
        "cost": {
            "total_usd": round(total_cost, 4),
            "cost_per_paper": round(total_cost / max(papers_done, 1), 5),
            "input_tokens": _cost_tracker["total_input_tokens"],
            "output_tokens": _cost_tracker["total_output_tokens"],
            "calls": _cost_tracker["total_calls"],
        },
    }
    with open(COST_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)


# ---------------------------------------------------------------------------
# Initialize LightRAG
# ---------------------------------------------------------------------------
async def ainit_lightrag():
    """Async-initialize LightRAG. Must be called from async context."""
    from lightrag import LightRAG
    from lightrag.base import EmbeddingFunc

    WORKING_DIR.mkdir(parents=True, exist_ok=True)

    embed_func = EmbeddingFunc(
        embedding_dim=EMBEDDING_DIM,
        func=embed_minilm_wrapper,
    )

    rag = LightRAG(
        working_dir=str(WORKING_DIR),
        llm_model_func=llm_wrapper,
        llm_model_name=GPT4O_MINI,
        embedding_func=embed_func,
        embedding_batch_num=16,
        llm_model_max_async=LLM_MODEL_MAX_ASYNC,
        max_parallel_insert=MAX_PARALLEL_INSERT,
        addon_params={
            "entity_types": config.ENTITY_TYPES,
        },
        auto_manage_storages_states=True,
    )

    await rag.initialize_storages()

    print(f"LightRAG initialized at {WORKING_DIR}")
    return rag


def count_processed(rag):
    """Count already-processed papers in doc_status."""
    doc_status_path = WORKING_DIR / "kv_store_doc_status.json"
    if doc_status_path.exists():
        try:
            with open(doc_status_path) as f:
                status = json.load(f)
            return sum(1 for v in status.values() if v.get("status") == "processed")
        except Exception:
            pass
    return 0


# ---------------------------------------------------------------------------
# Periodic progress poller (runs in background while ainsert is blocking)
# ---------------------------------------------------------------------------
async def progress_poller(rag, t_start, total_papers, last_save):
    """Background task to print progress every 30s while ingest is running."""
    while not _shutdown_requested:
        await asyncio.sleep(30)
        try:
            processed = count_processed(rag)
            _cost_tracker["papers_processed"] = processed
            elapsed = time.time() - t_start
            rate = processed / max(elapsed, 1)  # papers/sec
            remaining = total_papers - processed
            eta_sec = remaining / max(rate, 0.001)
            total_cost = _cost_tracker["total_input_cost"] + _cost_tracker["total_output_cost"]
            cost_per_paper = total_cost / max(processed, 1)
            print(
                f"  [progress] {processed}/{total_papers} "
                f"({100*processed/total_papers:.1f}%) | "
                f"{rate*60:.1f} papers/min | "
                f"ETA {eta_sec/3600:.1f}h | "
                f"cost: ${total_cost:.3f} (${cost_per_paper:.5f}/paper) | "
                f"calls: {_cost_tracker['total_calls']}"
            )
            save_progress()
            save_cost_report()
            last_save[0] = time.time()
        except Exception as e:
            print(f"  [poller] {e}")
    # Final save
    save_progress()
    save_cost_report()


# ---------------------------------------------------------------------------
# Main ingest
# ---------------------------------------------------------------------------
async def run_ingest(papers):
    """Run the full ingest. Resumable on re-run."""
    progress = load_progress()
    if progress.get("started_at") is None:
        progress["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    _cost_tracker["papers_total"] = len(papers)

    print(f"\n{'='*70}")
    print(f"FULL RE-INDEX: gpt-4o-mini + MiniLM")
    print(f"{'='*70}")
    print(f"  Total unique papers: {len(papers)}")
    print(f"  Max parallel insert: {MAX_PARALLEL_INSERT} chunks")
    print(f"  LLM model max async: {LLM_MODEL_MAX_ASYNC}")
    print(f"  Working dir: {WORKING_DIR}")

    rag = await ainit_lightrag()
    already = count_processed(rag)
    print(f"  Already processed (resume): {already}")
    print(f"  Remaining: {len(papers) - already}")
    print(f"\nStarting ingest at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    # Signal handlers
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    # Prepare (text, file_path) pairs, skipping empty
    pairs = []
    skipped_empty = 0
    for p in papers:
        p_path = Path(p["file_path"])
        if not p_path.exists():
            skipped_empty += 1
            continue
        text = p_path.read_text(encoding="utf-8", errors="ignore").strip()
        if len(text) < 50:
            skipped_empty += 1
            continue
        pairs.append((text, p["file_path"]))

    print(f"  Valid papers to insert: {len(pairs)} (skipped {skipped_empty} empty/missing)")

    if not pairs:
        print("  Nothing to do!")
        return

    texts = [t for t, _ in pairs]
    file_paths = [fp for _, fp in pairs]

    t_start = time.time()
    last_save = [t_start]

    # Start background progress poller
    poller = asyncio.create_task(
        progress_poller(rag, t_start, len(papers), last_save)
    )

    try:
        # Call ainsert with all papers at once — LightRAG handles 4× parallel
        # internally. Resumability: already-processed papers are skipped at
        # enqueue time (filter_keys checks doc_status).
        await rag.ainsert(input=texts, file_paths=file_paths)
    except Exception as e:
        _cost_tracker["errors"].append(str(e)[:300])
        print(f"\n[!] ainsert error: {e}")
    finally:
        _shutdown_requested = True
        try:
            await asyncio.wait_for(poller, timeout=5)
        except asyncio.TimeoutError:
            poller.cancel()

    # Final stats
    processed = count_processed(rag)
    _cost_tracker["papers_processed"] = processed
    elapsed = time.time() - t_start
    total_cost = _cost_tracker["total_input_cost"] + _cost_tracker["total_output_cost"]

    print(f"\n{'='*70}")
    print(f"INGEST {'COMPLETE' if not _shutdown_requested else 'INTERRUPTED'}")
    print(f"{'='*70}")
    print(f"  Papers processed (total in doc_status): {processed}/{len(papers)}")
    print(f"  Elapsed: {elapsed/60:.1f} min ({elapsed/3600:.2f}h)")
    print(f"  Total cost: ${total_cost:.3f}")
    print(f"  Cost per paper: ${total_cost/max(processed,1):.5f}")
    print(f"  LLM calls: {_cost_tracker['total_calls']}")
    print(f"  Input tokens: ~{_cost_tracker['total_input_tokens']:,}")
    print(f"  Output tokens: ~{_cost_tracker['total_output_tokens']:,}")
    print(f"  Errors: {len(_cost_tracker['errors'])}")
    if _cost_tracker["errors"]:
        for err in _cost_tracker["errors"][:5]:
            print(f"    - {err}")

    save_progress()
    save_cost_report()

    # Verify target entities
    print(f"\n{'='*70}")
    print(f"ENTITY VERIFICATION")
    print(f"{'='*70}")
    verify_entities_in_kg()

    if processed < len(papers):
        print(f"\n[!] {len(papers) - processed} papers remaining. Re-run to resume.")
    else:
        print(f"\n[OK] All papers processed. Ready for v4 evaluation.")


def verify_entities_in_kg():
    """Check target entities in vdb_entities.json."""
    vdb_path = WORKING_DIR / "vdb_entities.json"
    if not vdb_path.exists():
        print("  No vdb_entities.json found")
        return
    with open(vdb_path) as f:
        vdb = json.load(f)
    entities = vdb.get("data", [])
    entity_names = [e.get("entity_name", "") for e in entities]
    entity_names_lower = [n.lower() for n in entity_names]
    print(f"  Total entities in KG: {len(entities)}")
    targets = ["SU5416", "Y-27632", "ROCK", "VEGF", "Sunitinib", "Thalidomide"]
    for target in targets:
        t_lower = target.lower()
        found = t_lower in entity_names_lower
        partial = [n for n in entity_names if t_lower in n.lower() or n.lower() in t_lower]
        status = "FOUND" if found else ("PARTIAL" if partial else "MISSING")
        partial_str = f" (partial: {partial[:3]})" if partial and not found else ""
        print(f"  {target:15s}: {status}{partial_str}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    with open(UNIQUE_PAPERS_PATH) as f:
        unique_data = json.load(f)
    papers = unique_data["papers"]
    print(f"Loaded {len(papers)} unique papers from {UNIQUE_PAPERS_PATH}")

    # Verify sunitinib/thalidomide are present
    for drug_search in ["sunitinib", "thalidomide"]:
        matching = [
            p for p in papers
            if any(drug_search.lower() in d.lower() for d in p.get("drugs", []))
        ]
        print(f"  {drug_search}: {len(matching)} papers in unique set")

    asyncio.run(run_ingest(papers))


if __name__ == "__main__":
    main()
