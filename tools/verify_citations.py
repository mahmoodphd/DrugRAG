#!/usr/bin/env python3
"""
verify_citations.py — verify PMIDs/DOIs in any text against the local corpus.

The core hallucination-audit mechanic, used by both the review pipeline and
the chat badge. Extracts every PMID/DOI from input text and marks each as
IN-CORPUS (verifiable) or NOT-IN-CORPUS. With --pubmed it also checks real
PubMed, separating a fabricated PMID from one that's simply not in our corpus.

Usage:
    python tools/verify_citations.py --file answer.txt
    echo "...PMID 28472639..." | python tools/verify_citations.py
    python tools/verify_citations.py --text "PMID 12441250, doi 10.1016/j.x" --pubmed
"""
import argparse
import pathlib
import re
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "phase2" / "corpus_index.csv"

PMID_RE = re.compile(r"pmid[\s:_]*?(\d{6,9})", re.IGNORECASE)
DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", re.IGNORECASE)


def load_corpus():
    df = pd.read_csv(CORPUS, usecols=["pmid", "doi"])
    pmids = {
        str(int(float(p)))
        for p in df["pmid"].dropna()
        if str(p).strip().replace(".0", "").isdigit()
    }
    dois = {str(d).strip().lower().rstrip(".") for d in df["doi"].dropna()}
    return pmids, dois


def extract(text):
    pmids = list(dict.fromkeys(m.group(1) for m in PMID_RE.finditer(text)))
    dois = list(dict.fromkeys(d.lower().rstrip(".") for d in DOI_RE.findall(text)))
    return pmids, dois


def pubmed_exists(pmid, email="drugrag@example.com"):
    try:
        from Bio import Entrez
        Entrez.email = email
        h = Entrez.esummary(db="pubmed", id=pmid)
        rec = Entrez.read(h)
        h.close()
        return bool(rec)
    except Exception:
        return None  # could not check (offline / no Bio)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--file")
    g.add_argument("--text")
    ap.add_argument("--pubmed", action="store_true",
                    help="also check real PubMed for not-in-corpus PMIDs")
    a = ap.parse_args()

    if a.file:
        text = pathlib.Path(a.file).read_text(errors="ignore")
    elif a.text:
        text = a.text
    else:
        text = sys.stdin.read()

    pmids, dois = extract(text)
    cp, cd = load_corpus()
    n_ok = n_bad = 0
    print(f"Citations found: {len(pmids)} PMID(s), {len(dois)} DOI(s)\n")

    for p in pmids:
        in_corpus = p in cp
        tag = "IN-CORPUS ✓" if in_corpus else "NOT-IN-CORPUS ✗"
        extra = ""
        if not in_corpus and a.pubmed:
            ex = pubmed_exists(p)
            extra = ("  [real on PubMed, just not in our corpus]" if ex
                     else "  [NOT on PubMed → likely fabricated]" if ex is False
                     else "  [PubMed check unavailable]")
        print(f"  PMID {p:<10} {tag}{extra}")
        n_ok += in_corpus
        n_bad += not in_corpus

    for d in dois:
        in_corpus = d in cd
        tag = "IN-CORPUS ✓" if in_corpus else "NOT-IN-CORPUS ✗"
        print(f"  DOI  {d[:48]:<48} {tag}")
        n_ok += in_corpus
        n_bad += not in_corpus

    total = len(pmids) + len(dois)
    print(f"\nVerified in corpus: {n_ok}/{total}"
          + (f"  |  not in corpus: {n_bad}" if total else ""))
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
