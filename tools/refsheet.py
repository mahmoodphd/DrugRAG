#!/usr/bin/env python3
"""
refsheet.py — single source of truth for DrugRAG labels & reference facts.

Shared by the review and build pipelines: regenerate the canonical
reference sheet straight from the locked data so labels never get retyped or
drift between the dashboard, the figures, and the paper.

Usage:
    python tools/refsheet.py            # print human-readable summary
    python tools/refsheet.py --json     # also write data/phase5/reference_labels.json

Outputs: cluster meanings + direction, the confirmed TRUE-DARK drugs (with the
verified active-dark centerpieces flagged), KG coverage + best case-study drugs
for the KG explorer, and PubMed false-match outliers to exclude from charts.
"""
import argparse
import json
import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
V5 = ROOT / "data" / "phase5" / "drug_repurposing_candidates_v5_final.csv"
PV = ROOT / "data" / "phase5" / "tables" / "pubmed_verification_v5.csv"
OUT = ROOT / "data" / "phase5" / "reference_labels.json"

# Cluster meanings: PC2/PC4/PC7 are paper-grounded; PC1/3/5/6 are direction-based
# from our own net_direction analysis (do NOT invent phenotype names for those).
CLUSTER_LABELS = {
    1: "Near-neutral (topology remodel, little net change)",
    2: "Control-like / minimal effect",
    3: "Pro-angiogenic",
    4: "Pro-angiogenic (incl. glutamate-antagonist group)",
    5: "Strongly pro-angiogenic",
    6: "Pro-angiogenic",
    7: "Tube disruption / anti-angiogenic (most active)",
}
PAPER_GROUNDED = {2, 4, 7}  # the rest are direction-derived labels


def build():
    if not V5.exists() or not PV.exists():
        sys.exit(f"Missing data: {V5} or {PV}")
    df = pd.read_csv(V5)
    pv = pd.read_csv(PV)

    # 1) Clusters: size + data-driven direction
    clusters = {}
    for c, g in df.groupby("cluster"):
        mean_dir = float(g["net_direction"].mean())
        clusters[int(c)] = {
            "label": CLUSTER_LABELS.get(int(c), f"PC{int(c)}"),
            "paper_grounded": int(c) in PAPER_GROUNDED,
            "n": int(len(g)),
            "mean_direction": round(mean_dir, 3),
            "direction": "anti" if mean_dir < 0 else "pro",
            "mean_activity": round(float(g["activity_magnitude"].mean()), 3),
        }

    # 2) Confirmed TRUE-DARK drugs (flag the verified active-dark centerpieces)
    td = pv[pv["verification"] == "TRUE-DARK"].sort_values(
        "activity_magnitude", ascending=False
    )
    true_dark = [
        {
            "drug_name": r["drug_name"],
            "cluster": int(r["cluster"]),
            "direction": "anti" if r["net_direction"] < 0 else "pro",
            "activity_magnitude": round(float(r["activity_magnitude"]), 3),
            "real_angio_pubmed_hits": int(r["real_angio_pubmed_hits"]),
            "active_dark_centerpiece": bool(r["is_active_dark"]),
        }
        for _, r in td.iterrows()
    ]
    centerpieces = [d["drug_name"] for d in true_dark if d["active_dark_centerpiece"]]

    # 3) KG coverage + best case-study drugs for the KG explorer
    has_kg = df[df["angio_edges"] > 0]
    kg_rich = [
        {"drug_name": r["drug_name"], "cluster": int(r["cluster"]),
         "angio_edges": int(r["angio_edges"])}
        for _, r in has_kg.sort_values("angio_edges", ascending=False).head(12).iterrows()
    ]

    # 4) PubMed false-match outliers: angio hits > total hits is impossible
    bad = df[df["real_angio_pubmed_hits"] > df["real_total_pubmed_hits"]]
    outliers = [
        {"drug_name": r["drug_name"],
         "real_angio_pubmed_hits": int(r["real_angio_pubmed_hits"]),
         "real_total_pubmed_hits": int(r["real_total_pubmed_hits"])}
        for _, r in bad.sort_values("real_angio_pubmed_hits", ascending=False).iterrows()
    ]

    return {
        "n_drugs": int(len(df)),
        "clusters": clusters,
        "true_dark": true_dark,
        "active_dark_centerpieces": centerpieces,
        "kg_coverage": {
            "drugs_with_angio_edges": int((df["angio_edges"] > 0).sum()),
            "pct": round(float((df["angio_edges"] > 0).mean()) * 100, 1),
            "case_study_drugs": kg_rich,
        },
        "pubmed_false_match_outliers": outliers,
    }


def summary(ref):
    print(f"DrugRAG reference sheet — {ref['n_drugs']} drugs\n")
    print("CLUSTERS (PC2/PC4/PC7 paper-grounded; others direction-based):")
    for c, v in ref["clusters"].items():
        tag = "*" if v["paper_grounded"] else " "
        print(f"  {tag}PC{c}: n={v['n']:>3}  dir={v['direction']:<4} "
              f"({v['mean_direction']:+.2f})  — {v['label']}")
    print(f"\nTRUE-DARK drugs: {len(ref['true_dark'])}  "
          f"(centerpieces: {', '.join(ref['active_dark_centerpieces'])})")
    for d in ref["true_dark"]:
        star = "*" if d["active_dark_centerpiece"] else " "
        print(f"  {star}{d['drug_name']:<42} PC{d['cluster']} {d['direction']:<4} "
              f"act={d['activity_magnitude']:.2f} angio={d['real_angio_pubmed_hits']}")
    kg = ref["kg_coverage"]
    print(f"\nKG coverage: {kg['drugs_with_angio_edges']}/{ref['n_drugs']} "
          f"({kg['pct']}%) have angio edges. Case-study drugs:")
    for d in kg["case_study_drugs"]:
        print(f"   {d['drug_name']:<42} PC{d['cluster']} edges={d['angio_edges']}")
    print(f"\nPubMed false-match outliers (exclude from charts): "
          f"{len(ref['pubmed_false_match_outliers'])}")
    for o in ref["pubmed_false_match_outliers"][:5]:
        print(f"   {o['drug_name']:<42} angio={o['real_angio_pubmed_hits']} "
              f"> total={o['real_total_pubmed_hits']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="write reference_labels.json")
    args = ap.parse_args()
    ref = build()
    summary(ref)
    if args.json:
        OUT.write_text(json.dumps(ref, indent=2))
        print(f"\n[written] {OUT}")
