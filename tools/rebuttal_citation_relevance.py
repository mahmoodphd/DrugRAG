#!/usr/bin/env python
"""Four-level citation relevance audit for the 20-question grounded-QA run
(Table III of the paper).

A citation can exist and still be irrelevant, so existence alone is not enough.
This script grades every emitted citation on four escalating
levels, using the same external reference source for all three systems:

  L1 existence      PMID resolves to a real PubMed record.
  L2 topicality     the resolved record is about vascular/angiogenesis biology
                    (curated lexicon, data/phase4/angio_lexicon_v4.json).
  L3 question-match the record mentions the specific entity the question asks
                    about (drug / receptor / pathway token from the question).
  L4 traceability   the PMID is in the corpus the system actually retrieved
                    from, so a reader can open the exact evidence used.

L1-L3 are judged on the real PubMed title+abstract, independent of our corpus,
so no system is advantaged by the choice of reference source.

Output: data/phase4/rebuttal_citation_relevance.json
"""
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dashboard"))

EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# Entity tokens each question is about. Alternatives within a question are
# separated so a hit on any one counts (e.g. Y-27632 papers may say "ROCK").
QUESTION_ENTITIES = [
    ["su5416", "semaxanib", "vegfr"],
    ["sunitinib", "vegfr"],
    ["dmh4", "vegf"],
    ["memantine", "nmda"],
    ["y-27632", "y27632", "rock", "rho-kinase", "rho kinase"],
    ["suramin"],
    ["thalidomide"],
    ["glutamate", "grm", "grin", "gria", "nmda", "ampa"],
    ["vegf"],
    ["nmda", "glutamate"],
    ["rock", "rho-kinase", "rho kinase", "rhoa"],
    ["dopamine", "dopaminergic", "d2 receptor"],
    ["gaba", "gabaergic"],
    ["serotonin", "5-ht", "5ht", "serotonergic"],
    ["neurotransmitter", "glutamate", "dopamine", "serotonin", "gaba",
     "adrenergic", "cholinergic"],
    ["lopac", "phenotypic screen", "drug screen", "compound library"],
    ["glutamate", "grm", "grin", "gria", "nmda"],
    ["repurpos", "anti-angiogenic", "antiangiogenic", "screen"],
    ["tube formation", "tube disruption", "endothelial network", "phenocluster"],
    ["matrigel", "tube formation", "endothelial"],
]


def load_lexicon() -> list[str]:
    d = json.loads((ROOT / "data" / "phase4" / "angio_lexicon_v4.json").read_text())
    terms = set()
    for cat in d["categories"].values():
        for t in cat:
            terms.add(t.lower())
    # the query anchors the corpus itself was built on
    terms |= {"angiogenesis", "angiogenic", "endothelial", "vascular",
              "blood vessel", "neovascularization", "vasculogenesis",
              "tube formation", "microvessel", "capillary"}
    return sorted(terms)


def fetch_records(pmids: list[str]) -> dict[str, dict]:
    """efetch title+abstract for each PMID (XML, batched)."""
    out = {}
    for i in range(0, len(pmids), 40):
        batch = pmids[i:i + 40]
        q = urllib.parse.urlencode(
            {"db": "pubmed", "id": ",".join(batch), "retmode": "xml"})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(f"{EFETCH}?{q}", timeout=90) as r:
                    xml = r.read().decode("utf-8", "replace")
                break
            except Exception as e:
                if attempt == 3:
                    raise
                print(f"  retry {attempt+1} after {e}", flush=True)
                time.sleep(3 * (attempt + 1))
        for art in re.findall(r"<PubmedArticle>.*?</PubmedArticle>", xml, re.S):
            pm = re.search(r"<PMID[^>]*>(\d+)</PMID>", art)
            if not pm:
                continue
            title = " ".join(re.findall(r"<ArticleTitle[^>]*>(.*?)</ArticleTitle>", art, re.S))
            abst = " ".join(re.findall(r"<AbstractText[^>]*>(.*?)</AbstractText>", art, re.S))
            clean = lambda s: re.sub(r"<[^>]+>", " ", s).strip()
            out[pm.group(1)] = {"title": clean(title), "abstract": clean(abst)}
        time.sleep(0.5)
    return out


def main():
    df = pd.read_csv(ROOT / "data" / "phase4" / "eval_grounded_qa.csv")
    assert len(df) == len(QUESTION_ENTITIES), "question/entity list out of sync"
    corpus = set(
        pd.read_csv(ROOT / "data" / "phase2" / "corpus_index.csv", usecols=["pmid"])
        ["pmid"].dropna().astype(str)
    )
    lex = load_lexicon()
    import rag_chat

    # (system, q_index, pmid) for every citation instance
    cites = []
    for pre, name in [("plain", "Plain Gemma"), ("naive", "Naive RAG"),
                      ("drugrag", "DrugRAG")]:
        for qi, txt in enumerate(df[f"{pre}_answer"].fillna("")):
            for pm in rag_chat.extract_pmids(txt):
                cites.append({"system": name, "q": qi, "pmid": pm})

    uniq = sorted({c["pmid"] for c in cites})
    print(f"{len(cites)} citation instances, {len(uniq)} unique PMIDs; fetching...")
    recs = fetch_records(uniq)
    print(f"fetched {len(recs)} records")

    for c in cites:
        r = recs.get(c["pmid"])
        text = f"{r['title']} {r['abstract']}".lower() if r else ""
        c["L1_exists"] = bool(r)
        c["L2_topical"] = bool(text) and any(t in text for t in lex)
        c["L3_question_match"] = bool(text) and any(
            e in text for e in QUESTION_ENTITIES[c["q"]])
        c["L4_traceable"] = c["pmid"] in corpus
        c["title"] = (r or {}).get("title", "")

    summary = {}
    for name in ["Plain Gemma", "Naive RAG", "DrugRAG"]:
        s = [c for c in cites if c["system"] == name]
        n = len(s)
        row = {"citations_generated": n}
        for lvl, key in [("L1_exists_in_pubmed", "L1_exists"),
                         ("L2_vascular_topical", "L2_topical"),
                         ("L3_matches_question_entity", "L3_question_match"),
                         ("L4_traceable_to_retrieved_corpus", "L4_traceable")]:
            k = sum(1 for c in s if c[key])
            row[lvl] = {"n": k, "rate": round(k / n, 4) if n else None}
        summary[name] = row
        print(f"\n{name} (n={n})")
        for lvl in ["L1_exists_in_pubmed", "L2_vascular_topical",
                    "L3_matches_question_entity", "L4_traceable_to_retrieved_corpus"]:
            print(f"   {lvl:38s} {row[lvl]['n']:3d}/{n}  {100*row[lvl]['rate']:5.1f}%")

    out = {
        "note": ("Four-level citation audit. L1-L3 graded on the real PubMed "
                 "record (title+abstract) for every system, so the reference "
                 "source is identical across systems. L4 is corpus membership, "
                 "reported separately rather than conflated with existence. "
                 "Denominator is every PMID emitted; nothing filtered."),
        "levels": {
            "L1": "PMID resolves to a real PubMed record",
            "L2": "resolved record is vascular/angiogenesis-topical (curated lexicon)",
            "L3": "resolved record mentions the entity the question asks about",
            "L4": "PMID is in the retrieved corpus, so the evidence is openable",
        },
        "n_questions": int(len(df)),
        "summary": summary,
        "citations": cites,
    }
    dest = ROOT / "data" / "phase4" / "rebuttal_citation_relevance.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
