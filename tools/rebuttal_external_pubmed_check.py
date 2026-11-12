#!/usr/bin/env python
"""Check every cited PMID from the 20-question grounded-QA run against real
PubMed records (the same external reference source for all three systems).

Corpus membership alone would judge the ungrounded model on a different
reference than the citation audit, which checks against PubMed. This script
puts all three systems on the same external reference, and reports corpus membership as a
separate, additional metric.

Output: data/phase4/rebuttal_external_citation_check.json
"""
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dashboard"))

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"


def extract_pmids(text: str) -> list[str]:
    """Same extraction the dashboard/eval uses (dashboard/rag_chat.py)."""
    import rag_chat
    return rag_chat.extract_pmids(text or "")


def pubmed_exists(pmids: list[str]) -> dict[str, dict]:
    """Batch esummary. A PMID 'exists' if PubMed returns a record with a title."""
    out = {}
    for i in range(0, len(pmids), 50):
        batch = pmids[i:i + 50]
        q = urllib.parse.urlencode(
            {"db": "pubmed", "id": ",".join(batch), "retmode": "json"})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(f"{EUTILS}?{q}", timeout=60) as r:
                    data = json.loads(r.read().decode())
                break
            except Exception as e:                       # transient NCBI throttling
                if attempt == 3:
                    raise
                print(f"  retry {attempt+1} after {e}", flush=True)
                time.sleep(3 * (attempt + 1))
        res = data.get("result", {})
        for p in batch:
            rec = res.get(p)
            ok = bool(rec) and not rec.get("error") and bool(rec.get("title"))
            out[p] = {
                "exists": ok,
                "title": (rec or {}).get("title", ""),
                "journal": (rec or {}).get("source", ""),
                "year": (rec or {}).get("pubdate", ""),
            }
        time.sleep(0.5)                                   # be polite, no API key
    return out


def main():
    df = pd.read_csv(ROOT / "data" / "phase4" / "eval_grounded_qa.csv")
    corpus = set(
        pd.read_csv(ROOT / "data" / "phase2" / "corpus_index.csv", usecols=["pmid"])
        ["pmid"].dropna().astype(str)
    )
    print(f"corpus PMIDs: {len(corpus)}")

    systems = {"Plain Gemma": "plain", "Naive RAG": "naive", "DrugRAG": "drugrag"}
    per_system_pmids = {}
    for name, pre in systems.items():
        pm = []
        for txt in df[f"{pre}_answer"].fillna(""):
            pm += extract_pmids(txt)
        per_system_pmids[name] = pm                       # keep duplicates = citation instances
        print(f"{name}: {len(pm)} citation instances, {len(set(pm))} unique PMIDs")

    every = sorted({p for v in per_system_pmids.values() for p in v})
    print(f"\nchecking {len(every)} unique PMIDs against real PubMed...")
    verdict = pubmed_exists(every)

    summary = {}
    for name, pm in per_system_pmids.items():
        n = len(pm)
        n_pubmed = sum(1 for p in pm if verdict[p]["exists"])
        n_corpus = sum(1 for p in pm if p in corpus)
        summary[name] = {
            "citations_generated": n,
            "exists_in_real_pubmed": n_pubmed,
            "pubmed_existence_rate": round(n_pubmed / n, 4) if n else None,
            "in_drugrag_corpus": n_corpus,
            "corpus_membership_rate": round(n_corpus / n, 4) if n else None,
            "fabricated_not_in_pubmed": n - n_pubmed,
            "real_but_outside_corpus": n_pubmed - n_corpus,
        }
        print(f"\n{name}: generated={n} | in PubMed={n_pubmed} "
              f"({100*n_pubmed/n if n else 0:.1f}%) | in corpus={n_corpus} "
              f"| fabricated={n - n_pubmed} | real-but-outside-corpus={n_pubmed - n_corpus}")

    out = {
        "note": ("All three systems checked against the same external reference "
                 "(NCBI PubMed esummary). Corpus membership reported separately. "
                 "Denominator = every PMID emitted in the answer text; nothing is "
                 "filtered before counting."),
        "n_questions": int(len(df)),
        "summary": summary,
        "per_pmid": {p: {**verdict[p],
                         "in_corpus": p in corpus,
                         "cited_by": [n for n, v in per_system_pmids.items() if p in v]}
                     for p in every},
    }
    dest = ROOT / "data" / "phase4" / "rebuttal_external_citation_check.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
