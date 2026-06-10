#!/usr/bin/env python3
"""diag_loop.py — decisive test for the event-loop-binding bug.

Simulates two consecutive Gradio "Run" presses: each run_both/lightrag_answer
goes through its own asyncio.run() (a fresh event loop), while _rag is a cached
singleton. If query #1 works and query #2 comes back empty with
'bound to a different event loop', that's the production bug.
"""
import asyncio
import sys
import time
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "dashboard"))
import rag_chat  # noqa: E402

Q = "What evidence connects glutamate signaling to endothelial tube formation?"


def one(n):
    t0 = time.perf_counter()
    try:
        ans, dt = rag_chat._run(rag_chat.lightrag_answer(Q, mode="hybrid"))
        ok = bool(ans and ans.strip())
        print(f"QUERY {n}: aquery={dt:.2f}s wall={time.perf_counter()-t0:.2f}s "
              f"len={len(ans or '')} ok={ok}\n   head={(ans or '')[:90]!r}")
    except Exception as e:  # noqa: BLE001
        print(f"QUERY {n}: RAISED {type(e).__name__}: {e}")


print("=== simulating two consecutive Run presses (fresh asyncio.run each) ===")
one(1)
print("-" * 60)
one(2)
