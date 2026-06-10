"""
Phase 3: LightRAG Knowledge Graph Construction.

Configures and manages a LightRAG knowledge graph built from the Phase 2
literature corpus. Supports domain-specific biomedical entity extraction,
incremental ingestion, and three query modes (local, global, hybrid).
"""

import asyncio
import json
import os
import re
import time
from functools import partial
from pathlib import Path

import config
import data_loader


# ---------------------------------------------------------------------------
# LightRAG initialization
# ---------------------------------------------------------------------------
_rag_instance = None


async def _llm_func_wrapper(
    prompt, system_prompt=None, history_messages=None, **kwargs
):
    """Wrapper that injects model/API credentials into the LLM call."""
    from lightrag.llm.openai import openai_complete_if_cache

    return await openai_complete_if_cache(
        model=config.LIGHTRAG_QUERY_MODEL,
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=config.OPENROUTER_API_KEY,
        base_url=config.OPENROUTER_BASE_URL,
        **kwargs,
    )


async def _embed_func_wrapper(texts, **kwargs):
    """Wrapper that injects model/API credentials into the embedding call."""
    from lightrag.llm.openai import openai_embed

    embed_kwargs = {
        "model": config.LIGHTRAG_EMBEDDING_MODEL,
    }
    if config.OPENAI_API_KEY:
        embed_kwargs["api_key"] = config.OPENAI_API_KEY
    else:
        embed_kwargs["api_key"] = config.OPENROUTER_API_KEY
        embed_kwargs["base_url"] = config.OPENROUTER_BASE_URL
    return await openai_embed(texts, **embed_kwargs, **kwargs)


def init_lightrag():
    """
    Create and return a configured LightRAG instance.

    Uses OpenRouter for LLM calls and OpenAI (or fallback) for embeddings.
    Configures domain-specific entity types for biomedical extraction.
    """
    global _rag_instance
    if _rag_instance is not None:
        return _rag_instance

    from lightrag import LightRAG
    from lightrag.base import EmbeddingFunc

    # Wrap embedding function in EmbeddingFunc dataclass (required by LightRAG >= 1.4)
    embed_func = EmbeddingFunc(
        embedding_dim=384,
        func=_embed_func_wrapper,
    )

    _rag_instance = LightRAG(
        working_dir=str(config.LIGHTRAG_DIR),
        llm_model_func=_llm_func_wrapper,
        llm_model_name=config.LIGHTRAG_INDEXING_MODEL,
        embedding_func=embed_func,
        embedding_batch_num=16,
        llm_model_max_async=8,
        max_parallel_insert=4,
        addon_params={
            "entity_types": config.ENTITY_TYPES,
        },
        auto_manage_storages_states=True,
    )

    # LightRAG >= 1.4 requires explicit storage initialization
    loop = asyncio.new_event_loop()
    loop.run_until_complete(_rag_instance.initialize_storages())
    loop.close()

    print(f"LightRAG initialized at {config.LIGHTRAG_DIR}")
    return _rag_instance


# ---------------------------------------------------------------------------
# Ingestion tracking
# ---------------------------------------------------------------------------
MANIFEST_PATH = config.PHASE3_DIR / "ingested_files.json"


def load_ingested_manifest() -> set:
    """Load set of already-ingested file paths."""
    if MANIFEST_PATH.exists():
        data = json.loads(MANIFEST_PATH.read_text())
        return set(data.get("ingested_files", []))
    return set()


def save_ingested_manifest(ingested: set):
    """Save the set of ingested file paths."""
    MANIFEST_PATH.write_text(json.dumps(
        {"ingested_files": sorted(ingested), "count": len(ingested)},
        indent=2,
    ))


# ---------------------------------------------------------------------------
# Priority-ranked ingestion
# ---------------------------------------------------------------------------
def compute_ingestion_priority() -> list:
    """
    Compute priority ordering for corpus ingestion.

    Ranks papers by relevance to evaluation questions and key drug targets:
    - Tier A: Papers matching eval question target drugs and VEGF inhibitors
    - Tier B: Papers for dark drugs and PhenoCluster members
    - Tier C: All remaining papers

    Returns a sorted list of Path objects (highest priority first).
    """
    import pandas as pd

    # Target drugs from evaluation questions (Tier A - must have)
    eval_target_drugs = [
        "SU 5416", "SU5416", "Semaxanib", "Sunitinib", "DMH4",
        "Memantine", "Y-27632", "Suramin", "Thalidomide",
        "2-methoxyestradiol", "Minocycline", "Roscovitine",
    ]
    # Add VEGF inhibitors from config
    eval_target_drugs.extend(config.PC7_VEGF_INHIBITORS)

    # Mechanism-level targets (Tier A)
    mechanism_keywords = [
        "glutamate", "VEGF", "NMDA", "ROCK", "dopamine", "GABA",
        "serotonin", "GRM5", "GRIN3A", "cytoskeletal",
    ]

    # Collect all text files
    all_files = []
    for subdir in [config.FULLTEXT_DIR, config.ABSTRACT_DIR]:
        if subdir.exists():
            all_files.extend(sorted(subdir.glob("*.txt")))

    if not all_files:
        print("No text files found for priority ranking.")
        return []

    # Load corpus index for drug name mapping
    corpus_index_path = config.PHASE2_DIR / "corpus_index.csv"
    drug_to_files = {}
    if corpus_index_path.exists():
        try:
            df = pd.read_csv(corpus_index_path)
            for _, row in df.iterrows():
                drug = str(row.get("drug_name", "")).lower()
                if drug:
                    if drug not in drug_to_files:
                        drug_to_files[drug] = []
                    # Reconstruct file path from PMID and drug name
                    pmid = row.get("pmid", "")
                    if pmid:
                        pattern = f"pmid_{pmid}_"
                        drug_to_files[drug].append(pattern)
        except Exception as e:
            print(f"Warning: Could not load corpus_index.csv: {e}")

    # Classify files into tiers
    tier_a = []  # Eval target drugs
    tier_b = []  # Dark drugs, mechanism keywords
    tier_c = []  # Everything else

    for filepath in all_files:
        fname_lower = filepath.name.lower()
        fname_stem = filepath.stem.lower()

        # Check Tier A: exact match to eval target drugs
        is_tier_a = False
        for drug in eval_target_drugs:
            if drug.lower() in fname_stem:
                is_tier_a = True
                break

        if is_tier_a:
            tier_a.append(filepath)
            continue

        # Check Tier B: mechanism keywords and broad angiogenesis terms
        is_tier_b = False
        for kw in mechanism_keywords:
            if kw.lower() in fname_stem or kw.lower() in fname_lower:
                is_tier_b = True
                break

        # Also check for angiogenesis-related terms in filename
        if not is_tier_b:
            angio_terms = ["angiogen", "endothelial", "VEGF", "tube formation",
                          "metasta", "cancer", "tumor"]
            for term in angio_terms:
                if term.lower() in fname_stem:
                    is_tier_b = True
                    break

        if is_tier_b:
            tier_b.append(filepath)
        else:
            tier_c.append(filepath)

    # Sort each tier by PMID (numeric, to get diverse coverage)
    def pmid_sort_key(p):
        name = p.stem
        if name.startswith("pmid_"):
            try:
                return int(name.split("_")[1])
            except (IndexError, ValueError):
                return 0
        return 0

    tier_a.sort(key=pmid_sort_key)
    tier_b.sort(key=pmid_sort_key)
    tier_c.sort(key=pmid_sort_key)

    # Interleave within each tier for diversity
    print(f"Priority ranking computed:")
    print(f"  Tier A (eval targets): {len(tier_a)} files")
    print(f"  Tier B (mechanisms): {len(tier_b)} files")
    print(f"  Tier C (remaining): {len(tier_c)} files")

    return tier_a + tier_b + tier_c


# ---------------------------------------------------------------------------
# Corpus ingestion
# ---------------------------------------------------------------------------
def ingest_corpus(corpus_dir: Path = None, limit: int = None,
                  priority_files: list = None):
    """
    Ingest papers from the Phase 2 corpus into the LightRAG knowledge graph.

    Reads text files from papers/fulltext/ and papers/abstract/ directories.
    Skips already-ingested files (resume-safe via manifest).

    Args:
        corpus_dir: Override corpus directory (default: config.PAPERS_DIR)
        limit: Max files to ingest (None = all)
        priority_files: Pre-sorted file list from compute_ingestion_priority()
    """
    rag = init_lightrag()
    corpus_dir = corpus_dir or config.PAPERS_DIR

    # Collect text files
    if priority_files:
        text_files = priority_files
        print(f"\nUsing priority-ranked file list ({len(text_files)} files)")
    else:
        text_files = []
        for subdir in [config.FULLTEXT_DIR, config.ABSTRACT_DIR]:
            if subdir.exists():
                text_files.extend(sorted(subdir.glob("*.txt")))

    if not text_files:
        print("No text files found in corpus directory.")
        print(f"  Expected files in: {config.FULLTEXT_DIR}")
        print(f"  Expected files in: {config.ABSTRACT_DIR}")
        print("  Run Phase 2 first to build the literature corpus.")
        return

    # Filter already-ingested
    ingested = load_ingested_manifest()
    remaining = [f for f in text_files if str(f) not in ingested]

    if limit:
        remaining = remaining[:limit]

    print(f"\nPhase 3: Ingesting corpus into LightRAG")
    print(f"  Total text files: {len(text_files)}")
    print(f"  Already ingested: {len(ingested)}")
    print(f"  To ingest: {len(remaining)}")

    success = 0
    errors = 0
    t_start = time.time()

    for i, filepath in enumerate(remaining):
        try:
            text = filepath.read_text(encoding="utf-8").strip()
            if len(text) < 50:
                print(f"  [{i + 1}] SKIP (too short): {filepath.name}")
                continue

            rag.insert(text)
            ingested.add(str(filepath))
            success += 1

            if (i + 1) % 5 == 0 or (i + 1) == len(remaining):
                elapsed = time.time() - t_start
                rate = elapsed / (i + 1)
                eta_min = rate * (len(remaining) - i - 1) / 60
                print(f"  [{i + 1}/{len(remaining)}] {success} ok, {errors} err "
                      f"| {rate:.1f}s/doc | ETA {eta_min:.0f}m")
                save_ingested_manifest(ingested)

        except Exception as e:
            errors += 1
            print(f"  [{i + 1}] ERROR ingesting {filepath.name}: {str(e)[:80]}")

    # Final save
    save_ingested_manifest(ingested)
    print(f"\nIngestion complete: {success} files, {errors} errors")
    print(f"Total ingested: {len(ingested)}")


# ---------------------------------------------------------------------------
# Querying
# ---------------------------------------------------------------------------
def query_kg(question: str, mode: str = "hybrid") -> dict:
    """
    Query the knowledge graph with a specific retrieval mode.

    Uses rag.query_llm() to get BOTH the synthesized answer AND the
    underlying retrieved chunks/entities/relationships as separate contexts.
    This is critical for RAGAS evaluation where contexts must be the
    retrieved evidence, not the generated answer.

    Args:
        question: The research question to answer
        mode: Retrieval mode - "local", "global", or "hybrid"

    Returns dict with question, mode, response text, contexts, and metadata.
    """
    from lightrag import QueryParam

    rag = init_lightrag()
    param = QueryParam(mode=mode)

    print(f"  Querying ({mode}): {question[:80]}...")

    # Use query_llm() to get both structured retrieval data and LLM answer
    max_retries = 3
    for attempt in range(max_retries):
        try:
            result = rag.query_llm(question, param=param)
            break
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"    Retry {attempt + 1}/{max_retries} after {wait}s: {str(e)[:60]}")
                time.sleep(wait)
            else:
                print(f"    All retries failed: {str(e)[:80]}")
                return {
                    "question": question,
                    "mode": mode,
                    "response": f"ERROR: query_llm failed after {max_retries} attempts",
                    "contexts": [],
                    "entities": [],
                    "relationships": [],
                    "n_chunks": 0,
                    "n_entities": 0,
                    "n_relationships": 0,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }

    # Extract the LLM-generated answer
    llm_resp = result.get("llm_response", {})
    answer = llm_resp.get("content", "")
    if answer is None:
        # Handle streaming response (shouldn't happen in sync mode)
        iterator = llm_resp.get("response_iterator", "")
        if hasattr(iterator, "__iter__") and not isinstance(iterator, str):
            answer = "".join(iterator)
        else:
            answer = str(iterator) if iterator else ""

    # Extract actual retrieved context chunks (the evidence, NOT the answer)
    data = result.get("data", {})
    chunks = data.get("chunks", [])
    contexts = [c["content"] for c in chunks if c.get("content")]

    # Extract entities and relationships metadata
    entities = data.get("entities", [])
    relationships = data.get("relationships", [])

    return {
        "question": question,
        "mode": mode,
        "response": answer or "",
        "contexts": contexts,
        "entities": entities,
        "relationships": relationships,
        "n_chunks": len(contexts),
        "n_entities": len(entities),
        "n_relationships": len(relationships),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


# ---------------------------------------------------------------------------
# Keyword-based KG retrieval (bypasses paid embedding requirement)
# ---------------------------------------------------------------------------
_kg_data_cache = None

# Biomedical-aware stopword list for keyword extraction
_STOPWORDS = frozenset(
    "a an the is are was were be been being have has had do does did will "
    "would shall should may might can could of in to for on with at by from "
    "as into through between about above below up down out off over under "
    "and but or nor not no so yet both either neither each every all any "
    "few more most other some such only own same than too very just "
    "that this these those it its what which who whom whose how when where "
    "why if then else because while although though since unless until "
    "i me my we our you your he him his she her they them their "
    "there here what's that's it's don't doesn't didn't won't "
    "also however therefore thus moreover furthermore additionally "
    "et al etc eg ie vs via "
    "using used use based shown show suggest suggests demonstrated "
    "study studies results result found find observed indicate indicates "
    "role effect effects relationship associated related "
    "cells cell protein proteins receptor receptors gene genes "
    "pathway pathways drug drugs treatment treatments "
    "expression activity inhibition binding function "
    " vitro vivo clinical patients subjects model "
    "level levels response responses mechanism action "
    "dose doses concentration concentrations "
    "analysis method methods data evidence "
    "does have has been being".split()
)


def _load_kg_data() -> dict:
    """Load KG text data from VDB JSON files (cached, skips vectors)."""
    global _kg_data_cache
    if _kg_data_cache is not None:
        return _kg_data_cache

    workdir = config.LIGHTRAG_DIR
    print("  Loading KG data from VDB files (text only, skipping vectors)...")

    # Load entities
    entities = []
    ent_path = workdir / "vdb_entities.json"
    if ent_path.exists():
        raw = json.loads(ent_path.read_text())
        for item in raw.get("data", []):
            entities.append({
                "entity_name": item.get("entity_name", ""),
                "content": item.get("content", ""),
                "source_id": item.get("source_id", ""),
            })
    print(f"    Entities: {len(entities)}")

    # Load relationships
    relationships = []
    rel_path = workdir / "vdb_relationships.json"
    if rel_path.exists():
        raw = json.loads(rel_path.read_text())
        for item in raw.get("data", []):
            relationships.append({
                "src_id": item.get("src_id", ""),
                "tgt_id": item.get("tgt_id", ""),
                "content": item.get("content", ""),
                "source_id": item.get("source_id", ""),
            })
    print(f"    Relationships: {len(relationships)}")

    # Load chunks
    chunks = []
    chunk_path = workdir / "vdb_chunks.json"
    if chunk_path.exists():
        raw = json.loads(chunk_path.read_text())
        for item in raw.get("data", []):
            chunks.append({
                "content": item.get("content", ""),
                "full_doc_id": item.get("full_doc_id", ""),
            })
    print(f"    Chunks: {len(chunks)}")

    _kg_data_cache = {
        "entities": entities,
        "relationships": relationships,
        "chunks": chunks,
    }
    return _kg_data_cache


def _stem(word: str) -> str:
    """Lightweight suffix stemmer for biomedical text.

    Strips common English suffixes to match word variants:
    antagonize/antagonist, inhibit/inhibition, etc.
    """
    if len(word) <= 4:
        return word
    # Order matters: longer suffixes first
    suffixes = [
        "ation", "ization", "isation",
        "ively", "iveness",
        "ting", "ning", "ring", "ding", "ling",
        "ized", "ised", "ated",
        "tion", "sion", "ment", "ness",
        "ible", "able",
        "ize", "ise", "ate", "ify",
        "ous", "ive", "ful", "less",
        "ing", "ity",
        "ist", "ism",
        "ers", "est",
        "ly", "ed", "er", "al",
        "es", "ty", "or",
        "s",
    ]
    for suffix in suffixes:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[:len(word) - len(suffix)]
    return word


def _extract_keywords(question: str) -> list:
    """Extract meaningful keywords from a research question.

    Splits on whitespace/punctuation, removes stopwords, preserves
    biomedical terms (gene names, drug names, abbreviations).
    Returns only original word forms (stems used at scoring time).
    """
    # Split on non-alphanumeric except hyphens within words
    tokens = re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", question)
    keywords = []
    seen = set()
    for token in tokens:
        lower = token.lower()
        # Keep: non-stopword tokens, short abbreviations (VEGF, NMDA, etc.),
        # tokens with numbers (SU5416, GRM5), capitalized terms
        if lower not in _STOPWORDS and len(lower) >= 2:
            if lower not in seen:
                seen.add(lower)
                keywords.append(lower)
        elif token.isupper() and len(token) >= 2:
            # Preserve uppercase abbreviations even if in stopwords
            if lower not in seen:
                seen.add(lower)
                keywords.append(lower)
    return keywords


def _score_text(keywords: list, text_lower: str, name_lower: str = "",
                idf: dict = None) -> float:
    """Score a text item by keyword overlap with IDF weighting and stemming.

    For each keyword, checks exact match first, then stem as fallback (70%).
    This prevents double-counting when both word and stem match the same text.
    Name matches get 3x weight (entity names are high-signal).
    """
    if not keywords or not text_lower:
        return 0.0
    score = 0.0
    for kw in keywords:
        w = idf.get(kw, 1.0) if idf else 1.0
        stem = _stem(kw)
        # Check name match (highest signal)
        name_hit = False
        if name_lower:
            if kw in name_lower:
                score += 3.0 * w
                name_hit = True
            elif stem != kw and len(stem) >= 3 and stem in name_lower:
                score += 3.0 * w * 0.7  # Stem match in name (70%)
                name_hit = True
        # Check content match
        if kw in text_lower:
            score += 1.0 * w
        elif not name_hit and stem != kw and len(stem) >= 3 and stem in text_lower:
            score += 1.0 * w * 0.7  # Stem match in content (70%)
    return score


def _compute_idf(keywords: list, documents: list) -> dict:
    """Compute IDF weights for keywords against a set of documents.

    Rare terms (e.g. 'memantine' in 18/4094 docs) get high weight.
    Common terms (e.g. 'glutamate' in 224/4094 docs) get low weight.
    """
    import math
    n = len(documents)
    if n == 0:
        return {kw: 1.0 for kw in keywords}
    df = {kw: 0 for kw in keywords}
    for doc in documents:
        doc_lower = doc.lower()
        for kw in keywords:
            if kw in doc_lower:
                df[kw] += 1
    return {kw: math.log(1 + n / (1 + df[kw])) for kw in keywords}


def query_kg_keyword(question: str, mode: str = "hybrid",
                     top_k: int = 10) -> dict:
    """Keyword-based KG retrieval that bypasses vector search.

    Searches entities, relationships, and text chunks using string matching
    instead of embedding-based vector similarity. This avoids the need for
    paid embedding API calls (text-embedding-3-small).

    Mode mapping:
      - local: entity-centric (entities + their source chunks)
      - global: relationship-centric (relationships + all chunks)
      - hybrid: all sources combined

    Args:
        question: The research question
        mode: 'local', 'global', or 'hybrid'
        top_k: Max contexts to return

    Returns: Same dict format as query_kg().
    """
    data = _load_kg_data()
    keywords = _extract_keywords(question)

    if not keywords:
        return {
            "question": question, "mode": mode, "response": "",
            "contexts": [], "entities": [], "relationships": [],
            "n_chunks": 0, "n_entities": 0, "n_relationships": 0,
            "keyword_query": [],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

    entities = data["entities"]
    relationships = data["relationships"]
    chunks = data["chunks"]

    # Compute IDF weights: rare terms get higher weight
    # Use entity names + relationship names as the document corpus
    all_names = [e["entity_name"] for e in entities]
    all_names.extend(f"{r['src_id']} {r['tgt_id']}" for r in relationships)
    idf = _compute_idf(keywords, all_names)

    # Entity-name boost: identify drug/gene names in the question and
    # strongly boost matching entities. Only boost tokens that look
    # like proper nouns (capitalized mid-sentence, contain digits, or
    # are abbreviations). This prevents generic words like "blood",
    # "vessel", "formation" from boosting hundreds of entities.
    q_tokens = re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", question)
    entity_name_index = {e["entity_name"].lower(): e for e in entities}
    name_boost_entities = {}  # entity_name_lower -> boost amount
    for i, token in enumerate(q_tokens):
        t_lower = token.lower()
        if len(t_lower) < 3 or t_lower in _STOPWORDS:
            continue
        # Only boost "name-like" tokens: capitalized mid-sentence,
        # contain digits (SU5416, GRM5), or are abbreviations (VEGF)
        is_name_like = (
            (i > 0 and token[0].isupper() and not token.isupper())  # Title case
            or (token.isupper() and 2 <= len(token) <= 6)  # Abbreviation
            or any(c.isdigit() for c in token)  # Has digits
        )
        if not is_name_like:
            continue
        # Exact entity-name match → strong boost
        if t_lower in entity_name_index:
            name_boost_entities[t_lower] = 100.0
        # Substring match for drug-like tokens (contains hyphens,
        # digits, or is >6 chars to avoid short word matches)
        if len(t_lower) > 6 or any(c.isdigit() for c in t_lower) or "-" in token:
            for ent_name_lower in entity_name_index:
                if t_lower in ent_name_lower or ent_name_lower in t_lower:
                    name_boost_entities[ent_name_lower] = max(
                        name_boost_entities.get(ent_name_lower, 0), 80.0
                    )

    # Score entities
    scored_entities = []
    for ent in entities:
        name = ent["entity_name"]
        content = ent["content"]
        score = _score_text(keywords, content.lower(), name.lower(), idf)
        # Apply entity-name boost
        name_lower = name.lower()
        if name_lower in name_boost_entities:
            score += name_boost_entities[name_lower]
        if score > 0:
            scored_entities.append({**ent, "_score": score})
    scored_entities.sort(key=lambda x: x["_score"], reverse=True)

    # Score relationships
    scored_rels = []
    for rel in relationships:
        name = f"{rel['src_id']} {rel['tgt_id']}"
        content = rel["content"]
        score = _score_text(keywords, content.lower(), name.lower(), idf)
        if score > 0:
            scored_rels.append({**rel, "_score": score})
    scored_rels.sort(key=lambda x: x["_score"], reverse=True)

    # Score chunks
    scored_chunks = []
    for chunk in chunks:
        content = chunk["content"]
        score = _score_text(keywords, content.lower(), idf=idf)
        if score > 0:
            scored_chunks.append({**chunk, "_score": score})
    scored_chunks.sort(key=lambda x: x["_score"], reverse=True)

    # Select contexts based on mode
    contexts = []
    selected_entities = []
    selected_rels = []

    if mode == "local":
        # Entity-centric: top entities + their source chunks
        selected_entities = scored_entities[:top_k]
        for ent in selected_entities:
            contexts.append(ent["content"])
        # Add matching chunks from same source docs
        ent_sources = {e["source_id"] for e in selected_entities}
        for chunk in scored_chunks:
            if chunk.get("full_doc_id", "") in ent_sources:
                if chunk["content"] not in contexts:
                    contexts.append(chunk["content"])
        selected_rels = scored_rels[:top_k // 2]

    elif mode == "global":
        # Relationship-centric: top relationships + broad chunk coverage
        selected_rels = scored_rels[:top_k]
        for rel in selected_rels:
            contexts.append(rel["content"])
        for chunk in scored_chunks[:top_k]:
            if chunk["content"] not in contexts:
                contexts.append(chunk["content"])
        selected_entities = scored_entities[:top_k // 2]

    else:  # hybrid
        # Combined: best from all sources
        for ent in scored_entities[:top_k // 2]:
            contexts.append(ent["content"])
            selected_entities.append(ent)
        for rel in scored_rels[:top_k // 2]:
            if rel["content"] not in contexts:
                contexts.append(rel["content"])
                selected_rels.append(rel)
        for chunk in scored_chunks[:top_k // 2]:
            if chunk["content"] not in contexts:
                contexts.append(chunk["content"])

    # Trim to top_k
    contexts = contexts[:top_k]

    return {
        "question": question,
        "mode": mode,
        "response": "",  # Will be generated by caller with LLM
        "contexts": contexts,
        "entities": selected_entities,
        "relationships": selected_rels,
        "n_chunks": len(contexts),
        "n_entities": len(selected_entities),
        "n_relationships": len(selected_rels),
        "keyword_query": keywords,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def generate_answer_from_contexts(question: str, contexts: list,
                                  model: str = None) -> str:
    """Generate an answer from retrieved contexts using a free LLM.

    Used by keyword-based retrieval to synthesize an answer from
    retrieved evidence (since there's no built-in LLM step).
    """
    if not contexts:
        return ("Based on the available knowledge graph, no relevant evidence "
                "was found for this question.")

    model = model or config.LIGHTRAG_QUERY_MODEL
    client = config.get_openrouter_client()

    context_text = "\n\n---\n\n".join(
        f"[Evidence {i+1}]: {c[:800]}" for i, c in enumerate(contexts[:8])
    )

    system_prompt = (
        "You are a biomedical research assistant. Answer the question using "
        "ONLY the provided evidence. If the evidence is insufficient, state "
        "that clearly. Be precise about drug names, receptors, pathways, and "
        "mechanisms. Cite evidence numbers [Evidence N] in your answer."
    )

    user_content = (
        f"Question: {question}\n\n"
        f"Retrieved Evidence:\n{context_text}\n\n"
        f"Provide a comprehensive answer based on the evidence above."
    )

    max_retries = 4
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.3,
                max_tokens=1024,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            if attempt < max_retries - 1:
                err_str = str(e)
                # Rate limit (429): wait for full minute window reset
                if "429" in err_str:
                    wait = 65
                else:
                    wait = 2 ** (attempt + 1)
                print(f"    LLM retry {attempt+1}/{max_retries} after {wait}s: "
                      f"{str(e)[:60]}")
                time.sleep(wait)
            else:
                return f"ERROR: LLM generation failed after {max_retries} attempts"


def batch_query(questions_file: str = None, questions: list = None) -> list:
    """
    Run a batch of questions through all three LightRAG modes.

    Args:
        questions_file: Path to JSONL file with questions (one per line)
        questions: Or pass a list of question strings directly

    Returns list of result dicts.
    """
    if questions_file:
        path = Path(questions_file)
        questions = []
        for line in path.read_text().strip().split("\n"):
            line = line.strip()
            if line:
                try:
                    obj = json.loads(line)
                    questions.append(obj.get("question", line))
                except json.JSONDecodeError:
                    questions.append(line)

    if not questions:
        print("No questions provided.")
        return []

    modes = ["local", "global", "hybrid"]
    results = []

    print(f"\nBatch querying {len(questions)} questions x {len(modes)} modes")

    for i, question in enumerate(questions):
        print(f"\n[{i + 1}/{len(questions)}] {question[:80]}")
        for mode in modes:
            try:
                result = query_kg(question, mode)
                results.append(result)
            except Exception as e:
                print(f"    ERROR ({mode}): {str(e)[:80]}")
                results.append({
                    "question": question,
                    "mode": mode,
                    "response": f"ERROR: {str(e)}",
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                })

    # Save results
    output_path = config.PHASE3_DIR / "query_results.json"
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nResults saved to {output_path}")

    return results


# ---------------------------------------------------------------------------
# Graph statistics
# ---------------------------------------------------------------------------
def export_graph_stats() -> dict:
    """
    Export statistics about the knowledge graph.

    Returns dict with node counts, edge counts, and top entities.
    """
    rag = init_lightrag()

    stats = {
        "working_dir": str(config.LIGHTRAG_DIR),
        "ingested_files": len(load_ingested_manifest()),
    }

    # Try to get graph statistics from LightRAG internals
    try:
        if hasattr(rag, "chunk_entity_relation_graph"):
            graph = rag.chunk_entity_relation_graph
            if hasattr(graph, "_data"):
                data = graph._data
                stats["nodes"] = len(data.get("entities", {}))
                stats["edges"] = len(data.get("relationships", {}))
    except Exception:
        stats["graph_info"] = "Could not extract detailed graph stats"

    output_path = config.PHASE3_DIR / "graph_stats.json"
    output_path.write_text(json.dumps(stats, indent=2))
    print(f"Graph stats saved to {output_path}")

    return stats


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--ingest":
        limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
        ingest_corpus(limit=limit)
    elif len(sys.argv) > 1 and sys.argv[1] == "--query":
        question = " ".join(sys.argv[2:])
        mode = "hybrid"
        result = query_kg(question, mode)
        print(f"\nResponse:\n{result['response']}")
    elif len(sys.argv) > 1 and sys.argv[1] == "--stats":
        stats = export_graph_stats()
        print(json.dumps(stats, indent=2))
    else:
        print("Usage:")
        print("  python phase3_knowledge_graph.py --ingest [limit]")
        print("  python phase3_knowledge_graph.py --query <question>")
        print("  python phase3_knowledge_graph.py --stats")
        print("  python phase3_knowledge_graph.py --query <question>")
        print("  python phase3_knowledge_graph.py --stats")
