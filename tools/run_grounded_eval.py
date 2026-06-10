#!/usr/bin/env python3
"""
T-EVAL: Grounded-QA head-to-head evaluation for BHI 2026 Table II.

Runs the same 20 angiogenesis questions through three systems:
  1. Plain Gemma (no retrieval) — via dashboard/rag_chat.py plain_llm_answer()
  2. Naive RAG (TF-IDF + Gemma)  — local TF-IDF retriever + Gemma generation
  3. DrugRAG (LightRAG hybrid)   — via dashboard/rag_chat.py lightrag_answer()

Scores:
  - Verifiable-citation rate (PMID in corpus)
  - Latency (seconds)
  - RAGAS faithfulness + answer relevancy (from v4 results for Gemma 4)

Output:
  - data/phase4/eval_grounded_qa.csv      (per-question detail)
  - data/phase4/eval_grounded_qa_summary.json  (aggregated for Table II)

Usage:
  source .venv/bin/activate
  python tools/run_grounded_eval.py
"""

import asyncio
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dashboard"))

import config
import rag_chat

# ─── Evaluation questions (20: mix of factual + mechanistic + discovery) ───
EVAL_QUESTIONS = [
    # Factual (should find specific drug info in KG)
    "What receptor does SU5416 inhibit and what is its role in angiogenesis?",
    "Which VEGF receptor subtypes does Sunitinib target?",
    "What is the mechanism of DMH4 as a VEGF inhibitor?",
    "What type of glutamate receptor does Memantine antagonize?",
    "What kinase does Y-27632 inhibit and what is its effect on cell morphology?",
    "What is the mechanism of action of Suramin as an anti-angiogenic agent?",
    "How does Thalidomide affect blood vessel formation?",
    # Mechanistic (multi-hop)
    "How does glutamate receptor inhibition affect endothelial tube formation?",
    "What is the relationship between VEGF signaling and endothelial cell network formation?",
    "How do NMDA receptor antagonists influence angiogenesis in vitro?",
    "What is the role of ROCK signaling in endothelial tube morphogenesis?",
    "How do dopamine receptor antagonists affect endothelial cell migration and branching?",
    "What is the connection between GABA signaling and vascular biology?",
    "How do serotonin receptor antagonists influence angiogenic processes?",
    # Discovery / synthesis
    "Which LOPAC1280 drug classes modulate angiogenesis through neurotransmitter pathways?",
    "Which drugs in the LOPAC library have phenotypic effects but lack angiogenesis literature?",
    "What is the overall evidence that glutamate signaling plays a role in vascular biology?",
    "What are the most promising drug repurposing candidates for anti-angiogenic therapy based on LOPAC1280 screening?",
    "What common mechanisms do PhenoCluster 7 drugs share that cause complete disruption of tube formation?",
    "How does the Matrigel tube formation assay capture different aspects of endothelial cell behavior?",
]


# ─── Naive RAG (TF-IDF + Gemma) ─────────────────────────────────────────────
class NaiveRAG:
    """TF-IDF retriever + Gemma generation. Each passage tagged with its PMID for fair citation."""

    _PMID_RE = re.compile(r"^pmid_(\d+)_")

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        self.vectorizer = TfidfVectorizer(
            max_features=10000, stop_words="english", ngram_range=(1, 2)
        )
        self.cosine_similarity = cosine_similarity
        self.documents = []   # list of text strings
        self.doc_pmids = []   # parallel list of PMID strings (or "")
        self.tfidf_matrix = None
        self._load_corpus()

    def _load_corpus(self):
        for subdir in [config.FULLTEXT_DIR, config.ABSTRACT_DIR]:
            if subdir.exists():
                for f in sorted(subdir.glob("*.txt")):
                    text = f.read_text(encoding="utf-8").strip()
                    if len(text) > 50:
                        m = self._PMID_RE.match(f.name)
                        pmid = m.group(1) if m else ""
                        self.documents.append(text[:3000])  # cap per doc
                        self.doc_pmids.append(pmid)
        if self.documents:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)
            n_tagged = sum(1 for p in self.doc_pmids if p)
            print(f"  Naive RAG: loaded {len(self.documents)} documents ({n_tagged} with PMID tags)")

    def retrieve(self, query, top_k=5):
        if not self.documents or self.tfidf_matrix is None:
            return []
        query_vec = self.vectorizer.transform([query])
        sims = self.cosine_similarity(query_vec, self.tfidf_matrix)[0]
        top_idx = sims.argsort()[-top_k:][::-1]
        return [(self.documents[i], self.doc_pmids[i])
                for i in top_idx if sims[i] > 0]

    async def answer(self, question):
        from lightrag.llm.openai import openai_complete_if_cache

        hits = self.retrieve(question, top_k=5)
        if not hits:
            context_block = "(No relevant passages found in corpus.)"
        else:
            parts = []
            for doc_text, doc_pmid in hits:
                prefix = f"[PMID: {doc_pmid}] " if doc_pmid else ""
                parts.append(f"{prefix}{doc_text[:800]}")
            context_block = "\n\n---\n\n".join(parts)

        sys_prompt = (
            "You are a biomedical literature assistant. "
            "Based on the retrieved passages below, answer the question. "
            "Cite specific papers using their PMID tags (e.g. [PMID: 13829]) where possible. "
            "If the passages don't contain relevant information, say so."
        )
        user_msg = f"RETRIEVED PASSAGES:\n{context_block}\n\nQUESTION: {question}"

        t0 = time.perf_counter()
        ans = await openai_complete_if_cache(
            model=rag_chat.LLM_MODEL,
            prompt=user_msg,
            system_prompt=sys_prompt,
            api_key=config.OPENROUTER_API_KEY,
            base_url=config.OPENROUTER_BASE_URL,
        )
        latency = time.perf_counter() - t0
        return ans or "", latency


# ─── Citation extraction (reuses rag_chat patterns) ─────────────────────────
def extract_pmids(text):
    """Same PMID extraction as dashboard uses."""
    return rag_chat.extract_pmids(text)


def verify_pmids(pmids):
    """Check each PMID against corpus."""
    return rag_chat.verify_pmids(pmids)


# ─── Main eval loop ─────────────────────────────────────────────────────────
async def run_eval():
    naive_rag = NaiveRAG()
    results = []

    total = len(EVAL_QUESTIONS)
    for i, q in enumerate(EVAL_QUESTIONS, 1):
        print(f"\n[{i}/{total}] {q[:70]}...")

        # 1. Plain Gemma (no retrieval)
        print("  Plain Gemma...", end=" ", flush=True)
        try:
            plain_ans, plain_lat = await rag_chat.plain_llm_answer(q)
        except Exception as e:
            print(f"ERROR: {e}")
            plain_ans, plain_lat = "", 0.0

        # 2. Naive RAG
        print("Naive RAG...", end=" ", flush=True)
        try:
            naive_ans, naive_lat = await naive_rag.answer(q)
        except Exception as e:
            print(f"ERROR: {e}")
            naive_ans, naive_lat = "", 0.0

        # 3. DrugRAG (LightRAG hybrid)
        print("DrugRAG...", end=" ", flush=True)
        try:
            drugrag_ans, drugrag_lat = await rag_chat.lightrag_answer(q, mode="hybrid")
        except Exception as e:
            print(f"ERROR: {e}")
            drugrag_ans, drugrag_lat = "", 0.0

        # Extract and verify PMIDs
        plain_pmids = extract_pmids(plain_ans or "")
        naive_pmids = extract_pmids(naive_ans or "")
        drugrag_pmids = extract_pmids(drugrag_ans or "")

        plain_v = verify_pmids(plain_pmids)
        naive_v = verify_pmids(naive_pmids)
        drugrag_v = verify_pmids(drugrag_pmids)

        def _stats(pmids, verdict):
            n = len(pmids)
            nv = sum(1 for v in verdict.values() if v)
            return n, nv, (nv / n if n > 0 else 0.0)

        pn, pnv, prate = _stats(plain_pmids, plain_v)
        nn, nnv, nrate = _stats(naive_pmids, naive_v)
        dn, dnv, drate = _stats(drugrag_pmids, drugrag_v)

        print(f"done. PMIDs: plain={pnv}/{pn}, naive={nnv}/{nn}, drugrag={dnv}/{dn}")

        results.append({
            "q_id": i,
            "question": q,
            "plain_answer": plain_ans,
            "plain_latency_s": round(plain_lat, 2),
            "plain_pmids_total": pn,
            "plain_pmids_verified": pnv,
            "naive_answer": naive_ans,
            "naive_latency_s": round(naive_lat, 2),
            "naive_pmids_total": nn,
            "naive_pmids_verified": nnv,
            "drugrag_answer": drugrag_ans,
            "drugrag_latency_s": round(drugrag_lat, 2),
            "drugrag_pmids_total": dn,
            "drugrag_pmids_verified": dnv,
        })

        # Rate-limit politeness between questions
        if i < total:
            await asyncio.sleep(1.5)

    return results


def compute_summary(results):
    """Aggregate per-system stats for Table II."""
    systems = {}
    for sys_name, prefix in [("Plain Gemma", "plain"),
                              ("Naive RAG", "naive"),
                              ("DrugRAG (LightRAG hybrid)", "drugrag")]:
        total_pmids = sum(r[f"{prefix}_pmids_total"] for r in results)
        total_verified = sum(r[f"{prefix}_pmids_verified"] for r in results)
        latency_vals = [r[f"{prefix}_latency_s"] for r in results if r[f"{prefix}_latency_s"] > 0]
        verif_rate = total_verified / total_pmids if total_pmids > 0 else 0.0

        systems[sys_name] = {
            "n_questions": len(results),
            "total_citations": total_pmids,
            "verified_citations": total_verified,
            "verifiable_citation_rate": round(verif_rate, 4),
            "verifiable_citation_rate_pct": f"{verif_rate * 100:.1f}%",
            "latency_mean_s": round(sum(latency_vals) / len(latency_vals), 1) if latency_vals else 0,
            "latency_median_s": round(sorted(latency_vals)[len(latency_vals) // 2], 1) if latency_vals else 0,
        }

    # NOTE: RAGAS faithfulness / answer-relevancy are NOT computed here.
    # They come from the separate phase-4 evaluation (30-Q set, all models).
    # Table II reports only what this 20-Q run directly measures.

    return systems


def main():
    print("=" * 60)
    print("  T-EVAL: Grounded-QA Head-to-Head Evaluation")
    print(f"  {len(EVAL_QUESTIONS)} questions × 3 systems")
    print("=" * 60)

    results = rag_chat._run(run_eval())

    # Save per-question detail
    out_csv = ROOT / "data" / "phase4" / "eval_grounded_qa.csv"
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)
    print(f"\nSaved per-question detail: {out_csv}")

    # Save summary
    summary = compute_summary(results)
    out_json = ROOT / "data" / "phase4" / "eval_grounded_qa_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary: {out_json}")

    # Print Table II shell (20-Q measured metrics only)
    print("\n" + "=" * 60)
    print("  TABLE II — Grounded QA, head-to-head (20-Q run)")
    print("=" * 60)
    print(f"{'System':30s} {'Citations':>10s} {'Verif.':>7s} {'Verif. rate':>12s} {'Latency':>8s}")
    print("-" * 70)
    for name, stats in summary.items():
        print(f"{name:30s} {stats['total_citations']:>10d} {stats['verified_citations']:>7d} "
              f"{stats['verifiable_citation_rate_pct']:>12s} "
              f"{stats['latency_median_s']:>7.1f}s")
    print("\n  RAGAS faithfulness/answer-relevancy reported separately")
    print("  (phase-4 30-Q evaluation, see ragas_summary_v4.csv)")


if __name__ == "__main__":
    main()
