#!/usr/bin/env python
"""Score ablation, weight sensitivity, and interval estimates.

Computes, from locked artifacts: Wilson 95% intervals for the citation audit
(Table II), the Fisher test on question match (Table III), the ablation of the
knowledge-graph term, and the sweep over the hand-set score weights.

Output: data/phase5/tables/rebuttal_ablation_sensitivity.json
"""
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
W = (0.40, 0.35, 0.25)          # a, nu, k  (D14, set a priori)


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0.0, c - h), 4), round(min(1.0, c + h), 4))


def main():
    out = {}

    # ── 1. KG ablation over the full 1,278-drug ranking ─────────────────────
    df = pd.read_csv(ROOT / "data" / "phase5" / "drug_repurposing_candidates_v5_final.csv")
    a, nu, k = df["activity_magnitude"], df["novelty_real"], df["kg_plausibility"]

    full = W[0] * a + W[1] * nu + W[2] * k
    # drop the KG term and renormalise the remaining weights so scores stay on
    # a comparable scale (0.40/0.35 -> 0.533/0.467)
    s = W[0] + W[1]
    noKG = (W[0] / s) * a + (W[1] / s) * nu

    r_full = full.rank(ascending=False, method="min")
    r_noKG = noKG.rank(ascending=False, method="min")
    delta = (r_noKG - r_full).abs()

    rho, p_rho = stats.spearmanr(full, noKG)
    top50_full = set(df.loc[r_full <= 50, "drug_name"])
    top50_noKG = set(df.loc[r_noKG <= 50, "drug_name"])

    movers = (df.assign(rank_full=r_full, rank_noKG=r_noKG, d=delta,
                        kg=k)
                .sort_values("d", ascending=False)
                .head(10)[["drug_name", "kg", "rank_full", "rank_noKG", "d"]])

    out["kg_ablation"] = {
        "n_drugs": int(len(df)),
        "spearman_rho_full_vs_noKG": round(float(rho), 4),
        "spearman_p": float(p_rho),
        "median_abs_rank_change": float(delta.median()),
        "mean_abs_rank_change": round(float(delta.mean()), 2),
        "n_drugs_moved_gt_50_ranks": int((delta > 50).sum()),
        "top50_overlap": len(top50_full & top50_noKG),
        "n_drugs_with_kg_gt_0": int((k > 0).sum()),
        "n_drugs_with_kg_eq_0": int((k == 0).sum()),
        "kg_mean_nonzero": round(float(k[k > 0].mean()), 4) if (k > 0).any() else None,
        "true_dark_kg_all_zero": bool((df.loc[df["novelty_status"] == "TRUE-DARK",
                                              "kg_plausibility"] == 0).all())
        if "novelty_status" in df.columns else None,
        "largest_movers": movers.to_dict("records"),
    }

    # did the 13 TRUE-DARK set change under ablation?
    dark = pd.read_csv(ROOT / "data" / "phase5" / "tables" / "pubmed_verification_v5.csv")
    dcol = [c for c in dark.columns if "verification" in c.lower()]
    if dcol:
        names = set(dark.loc[dark[dcol[0]] == "TRUE-DARK", dark.columns[0]])
        sub = df[df["drug_name"].isin(names)]
        out["kg_ablation"]["true_dark_n_matched"] = int(len(sub))
        out["kg_ablation"]["true_dark_rank_full"] = \
            sorted(int(x) for x in r_full[sub.index])
        out["kg_ablation"]["true_dark_rank_noKG"] = \
            sorted(int(x) for x in r_noKG[sub.index])

    # ── 2. Weight sensitivity: grid over plausible weightings ───────────────
    grid, rows = np.arange(0.20, 0.61, 0.05), []
    for wa in grid:
        for wn in grid:
            wk = 1 - wa - wn
            if wk < 0.05 or wk > 0.45:
                continue
            sc = wa * a + wn * nu + wk * k
            r = sc.rank(ascending=False, method="min")
            rows.append({
                "w_a": round(float(wa), 2), "w_nu": round(float(wn), 2),
                "w_k": round(float(wk), 2),
                "spearman_vs_default": round(float(stats.spearmanr(sc, full)[0]), 4),
                "top50_overlap": len(set(df.loc[r <= 50, "drug_name"]) & top50_full),
            })
    sens = pd.DataFrame(rows)
    out["weight_sensitivity"] = {
        "n_weightings": int(len(sens)),
        "spearman_vs_default_min": round(float(sens["spearman_vs_default"].min()), 4),
        "spearman_vs_default_median": round(float(sens["spearman_vs_default"].median()), 4),
        "top50_overlap_min": int(sens["top50_overlap"].min()),
        "top50_overlap_median": float(sens["top50_overlap"].median()),
        "grid": sens.to_dict("records"),
    }

    # ── 3. Interval estimates on every rate the paper reports ───────────────
    audit = [("Parametric (closed-book)", 12, 33),
             ("Cite-sources prompt", 68, 346),
             ("Cite-sources prompt (intensive)", 121, 506)]
    out["phase1_audit_ci"] = {
        name: {"unverifiable": kk, "citations": nn,
               "rate": round(kk / nn, 4), "wilson95": wilson(kk, nn)}
        for name, kk, nn in audit
    }

    rel = json.loads((ROOT / "data" / "phase4" /
                      "rebuttal_citation_relevance.json").read_text())
    out["citation_audit_ci"] = {}
    for sysname, row in rel["summary"].items():
        n = row["citations_generated"]
        out["citation_audit_ci"][sysname] = {
            lvl: {"n": row[lvl]["n"], "of": n, "rate": row[lvl]["rate"],
                  "wilson95": wilson(row[lvl]["n"], n)}
            for lvl in row if lvl != "citations_generated"
        }

    # DrugRAG vs naive RAG on question-entity match: Fisher exact
    d_ok = rel["summary"]["DrugRAG"]["L3_matches_question_entity"]["n"]
    d_n = rel["summary"]["DrugRAG"]["citations_generated"]
    n_ok = rel["summary"]["Naive RAG"]["L3_matches_question_entity"]["n"]
    n_n = rel["summary"]["Naive RAG"]["citations_generated"]
    odds, pval = stats.fisher_exact([[d_ok, d_n - d_ok], [n_ok, n_n - n_ok]])
    out["drugrag_vs_naive_relevance"] = {
        "drugrag": f"{d_ok}/{d_n}", "naive": f"{n_ok}/{n_n}",
        "fisher_odds_ratio": round(float(odds), 3), "fisher_p": round(float(pval), 4),
        "relevant_citations_per_answer": {
            "DrugRAG": round(d_ok / rel["n_questions"], 2),
            "Naive RAG": round(n_ok / rel["n_questions"], 2)},
        "interpretation": ("absolute count of question-relevant citations is the "
                           "comparison that matters; the rate difference alone is "
                           "not significant at this sample size"),
    }

    dest = ROOT / "data" / "phase5" / "tables" / "rebuttal_ablation_sensitivity.json"
    dest.write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps({kk: v for kk, v in out.items() if kk != "weight_sensitivity"},
                     indent=2, default=str)[:4000])
    print(f"\nweight sensitivity: {out['weight_sensitivity']['n_weightings']} weightings, "
          f"min rho={out['weight_sensitivity']['spearman_vs_default_min']}, "
          f"min top-50 overlap={out['weight_sensitivity']['top50_overlap_min']}")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
