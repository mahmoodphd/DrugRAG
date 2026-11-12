"""LightRAG chat helpers — query, plain-LLM baseline, citation verification."""

import asyncio
import os
import re
import sys
import threading
import time
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config

WORKING_DIR = ROOT / "data" / "phase3" / "lightrag_working_v2"
LLM_MODEL = "google/gemma-4-31b-it:free"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Lazy singletons
_rag = None
_minilm = None
_corpus_pmids = None  # set of PMIDs in our 6,500-abstract corpus

# ─── One persistent event loop for ALL RAG work ─────────────────────────────
# LightRAG's async storage locks + embedding-worker queues bind to the event
# loop that first initializes them. Using asyncio.run() per query creates a new
# loop each time, so the cached _rag's primitives raise "bound to a different
# event loop" on the 2nd query onward (1st works, rest return empty). Keeping a
# single long-lived loop in a background thread fixes this — _rag binds once.
_loop = None
_loop_lock = threading.Lock()


def _get_loop():
    global _loop
    with _loop_lock:
        if _loop is None:
            _loop = asyncio.new_event_loop()
            threading.Thread(target=_loop.run_forever, daemon=True).start()
    return _loop


def _run(coro):
    """Run a coroutine to completion on the persistent loop, from any thread."""
    return asyncio.run_coroutine_threadsafe(coro, _get_loop()).result()


def _load_corpus_pmids():
    """Load the set of PMIDs that ARE in our local corpus (verified)."""
    global _corpus_pmids
    if _corpus_pmids is None:
        import pandas as pd
        df = pd.read_csv(ROOT / "data" / "phase2" / "corpus_index.csv", usecols=["pmid"])
        _corpus_pmids = set(str(p) for p in df["pmid"].dropna().astype(str).tolist())
    return _corpus_pmids


async def _embed_func(texts, **kwargs):
    global _minilm
    import numpy as np
    from sentence_transformers import SentenceTransformer
    if _minilm is None:
        _minilm = SentenceTransformer(EMBEDDING_MODEL)
    arr = _minilm.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return np.array(arr)


async def _llm_func(prompt, system_prompt=None, history_messages=None, **kwargs):
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


async def get_rag():
    global _rag
    if _rag is None:
        from lightrag import LightRAG
        from lightrag.base import EmbeddingFunc
        ef = EmbeddingFunc(embedding_dim=EMBEDDING_DIM, func=_embed_func)
        _rag = LightRAG(
            working_dir=str(WORKING_DIR),
            llm_model_func=_llm_func,
            llm_model_name=LLM_MODEL,
            embedding_func=ef,
            embedding_batch_num=16,
            llm_model_max_async=1,
            addon_params={"entity_types": config.ENTITY_TYPES},
        )
        await _rag.initialize_storages()
    return _rag


async def lightrag_answer(question: str, mode: str = "hybrid") -> tuple[str, float]:
    """Run a LightRAG query with given mode. Returns (answer, latency_seconds)."""
    from lightrag import QueryParam
    rag = await get_rag()
    t0 = time.perf_counter()
    answer = await rag.aquery(question, param=QueryParam(mode=mode))
    return answer, time.perf_counter() - t0


async def plain_llm_answer(question: str) -> tuple[str, float]:
    """Same Gemma model, no retrieval. Asked to cite if it has supporting sources."""
    from lightrag.llm.openai import openai_complete_if_cache
    sys_prompt = (
        "You are a biomedical literature assistant. Answer the question. "
        "If you cite specific papers, give a PMID or DOI inline. "
        "If you don't have direct evidence, say so explicitly — do not invent citations."
    )
    t0 = time.perf_counter()
    answer = await openai_complete_if_cache(
        model=LLM_MODEL,
        prompt=question,
        system_prompt=sys_prompt,
        api_key=config.OPENROUTER_API_KEY,
        base_url=config.OPENROUTER_BASE_URL,
    )
    return answer, time.perf_counter() - t0


# ─── Citation extraction + verification ─────────────────────────────────────
PMID_PAT = re.compile(r"pmid[_\s:]?(\d{6,9})", re.IGNORECASE)
PMID_PURE = re.compile(r"\b(\d{7,9})\b")  # pure 7-9 digit numbers


def extract_pmids(text: str) -> list[str]:
    """Pull PMIDs from text. Looks at file_paths (pmid_NNNNN_*) and standalone numbers."""
    found = set()
    for m in PMID_PAT.finditer(text):
        found.add(m.group(1))
    # Also catch standalone PMIDs near "PMID" keyword
    for m in re.finditer(r"PMID[:\s]*(\d{6,9})", text, re.IGNORECASE):
        found.add(m.group(1))
    return sorted(found)


def verify_pmids(pmids: list[str]) -> dict[str, bool]:
    """Mark each PMID as True (in our corpus) or False (cannot verify)."""
    corpus = _load_corpus_pmids()
    return {p: (p in corpus) for p in pmids}


def colorize_citations(text: str, verdict: dict[str, bool]) -> str:
    """Wrap each PMID in green/red markdown spans based on verdict.
    Returns markdown text safe for gr.Markdown(render_inline=True)."""
    if not verdict:
        return text

    # Replace each PMID occurrence with colored badge
    def repl(m):
        # Match could be from PMID_PAT (group1) or the number itself
        full = m.group(0)
        num = m.group(1) if m.lastindex else full
        if num in verdict:
            color = "#5aae61" if verdict[num] else "#d6604d"
            # "in corpus", not "verified": corpus membership is one of the four
            # levels the paper reports (Table III), and on its own it says nothing
            # about whether the paper exists in PubMed or is relevant to the answer.
            label = "✓ in corpus" if verdict[num] else "✗ not in corpus"
            return f'<span style="background:{color};color:white;padding:2px 6px;border-radius:3px;font-weight:600;">PMID {num} ({label})</span>'
        return full

    text = PMID_PAT.sub(repl, text)
    text = re.sub(r"PMID[:\s]*(\d{6,9})", repl, text, flags=re.IGNORECASE)
    return text


def run_both(question: str, mode: str = "hybrid") -> dict:
    """Synchronous helper — run RAG + plain LLM, return both answers + verifications."""
    async def go():
        rag_task = asyncio.create_task(lightrag_answer(question, mode=mode))
        plain_task = asyncio.create_task(plain_llm_answer(question))
        (rag_ans, rag_t), (plain_ans, plain_t) = await asyncio.gather(rag_task, plain_task)
        return rag_ans, rag_t, plain_ans, plain_t

    rag_ans, rag_t, plain_ans, plain_t = _run(go())

    # Guard against None responses (transient API failures / rate limits)
    rag_ans = rag_ans or "(No response — the model may be rate-limited. Try again in a moment.)"
    plain_ans = plain_ans or "(No response — the model may be rate-limited. Try again in a moment.)"

    rag_pmids = extract_pmids(rag_ans)
    plain_pmids = extract_pmids(plain_ans)

    rag_verdict = verify_pmids(rag_pmids)
    plain_verdict = verify_pmids(plain_pmids)

    return {
        "rag_answer": rag_ans,
        "rag_latency": rag_t,
        "rag_pmids": rag_pmids,
        "rag_verdict": rag_verdict,
        "plain_answer": plain_ans,
        "plain_latency": plain_t,
        "plain_pmids": plain_pmids,
        "plain_verdict": plain_verdict,
    }


# ─── Streaming path (Feature 3) ─────────────────────────────────────────────────────

async def _stream_plain_llm(question: str):
    """Stream plain Gemma answer token-by-token via OpenRouter."""
    from openai import AsyncOpenAI
    client = AsyncOpenAI(
        base_url=config.OPENROUTER_BASE_URL,
        api_key=config.OPENROUTER_API_KEY,
    )
    sys_prompt = (
        "You are a biomedical literature assistant. Answer the question. "
        "If you cite specific papers, give a PMID or DOI inline. "
        "If you don't have direct evidence, say so explicitly — do not invent citations."
    )
    stream = await client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": question},
        ],
        stream=True,
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta and delta.content:
            yield delta.content


async def _stream_rag(question: str, mode: str = "hybrid"):
    """Stream LightRAG answer. Returns async iterator of text chunks."""
    from lightrag import QueryParam
    rag = await get_rag()
    result = await rag.aquery(question, param=QueryParam(mode=mode, stream=True))
    # result is an AsyncIterator[str] when stream=True
    async for chunk in result:
        yield chunk


def run_both_stream(question: str, mode: str = "hybrid"):
    """Generator that yields (rag_md, plain_md, status) tuples for Gradio streaming.

    Streams both answers live, then does a final yield with colorized citations.
    The existing run_both() is preserved as a non-streaming fallback.
    """
    if not question or not question.strip():
        yield ("*Type a question above and press Run.*",
               "*Type a question above and press Run.*", "")
        return

    t0 = time.perf_counter()

    # Collect chunks from both streams
    rag_chunks = []
    plain_chunks = []
    rag_done = False
    plain_done = False
    rag_error = None
    plain_error = None

    async def collect():
        nonlocal rag_done, plain_done, rag_error, plain_error

        async def collect_rag():
            nonlocal rag_done, rag_error
            try:
                async for chunk in _stream_rag(question, mode):
                    rag_chunks.append(chunk)
            except Exception as e:
                rag_error = e
            finally:
                rag_done = True

        async def collect_plain():
            nonlocal plain_done, plain_error
            try:
                async for chunk in _stream_plain_llm(question):
                    plain_chunks.append(chunk)
            except Exception as e:
                plain_error = e
            finally:
                plain_done = True

        await asyncio.gather(collect_rag(), collect_plain())

    # Run both streams and yield intermediate state periodically
    import threading
    import queue

    done_event = threading.Event()

    def run_async():
        try:
            asyncio.run_coroutine_threadsafe(collect(), _get_loop()).result()
        finally:
            done_event.set()

    thread = threading.Thread(target=run_async, daemon=True)
    thread.start()

    # Yield progress while streams are running
    last_rag_len = 0
    last_plain_len = 0
    yield ("*🔄 Retrieving from knowledge graph…*",
           "*🔄 Generating plain answer…*",
           "Streaming…")

    while not done_event.is_set():
        done_event.wait(timeout=0.4)
        rag_text = "".join(rag_chunks)
        plain_text = "".join(plain_chunks)
        cur_rag_len = len(rag_chunks)
        cur_plain_len = len(plain_chunks)
        if cur_rag_len != last_rag_len or cur_plain_len != last_plain_len:
            rag_display = rag_text if rag_text else "*🔄 Retrieving from knowledge graph…*"
            plain_display = plain_text if plain_text else "*🔄 Generating plain answer…*"
            elapsed = time.perf_counter() - t0
            yield (rag_display, plain_display, f"Streaming… ({elapsed:.0f} s)")
            last_rag_len = cur_rag_len
            last_plain_len = cur_plain_len

    # Final: colorize citations
    elapsed = time.perf_counter() - t0

    rag_ans = "".join(rag_chunks) if rag_chunks else None
    plain_ans = "".join(plain_chunks) if plain_chunks else None

    # None guard
    if rag_error or not rag_ans:
        rag_ans = "(No response — the model may be rate-limited. Try again in a moment.)"
    if plain_error or not plain_ans:
        plain_ans = "(No response — the model may be rate-limited. Try again in a moment.)"

    rag_pmids = extract_pmids(rag_ans)
    plain_pmids = extract_pmids(plain_ans)
    rag_verdict = verify_pmids(rag_pmids)
    plain_verdict = verify_pmids(plain_pmids)

    rag_md = colorize_citations(rag_ans, rag_verdict)
    plain_md = colorize_citations(plain_ans, plain_verdict)

    rag_v = sum(1 for v in rag_verdict.values() if v)
    rag_n = len(rag_verdict)
    plain_v = sum(1 for v in plain_verdict.values() if v)
    plain_n = len(plain_verdict)

    rag_header = (
        f"### DrugRAG (LightRAG + Gemma) — {elapsed:.1f} s\n"
        f"**Citations in corpus:** {rag_v}/{rag_n} "
        f"<span style='color:#5aae61'>green = in our corpus</span> \u00b7 "
        f"<span style='color:#d6604d'>red = not in our corpus</span>\n\n"
    )
    plain_header = (
        f"### Plain Gemma (no retrieval) — {elapsed:.1f} s\n"
        f"**Citations in corpus:** {plain_v}/{plain_n} "
        f"<span style='color:#5aae61'>green = in our corpus</span> \u00b7 "
        f"<span style='color:#d6604d'>red = not in our corpus</span>\n\n"
    )
    status = f"Total wall time: **{elapsed:.1f} s** (streamed in parallel)."
    yield (rag_header + rag_md, plain_header + plain_md, status)
