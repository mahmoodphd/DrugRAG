"""Full v4 RAGAS report — local, no API calls.

Pulls authoritative numbers from data/phase4/ragas_scores_v4.csv and writes
publication-ready tables into data/phase4/tables/.

Outputs:
  - v4_ragas_per_config_metric.csv/.md   (main matrix: 5 configs × 4 metrics + CI + n_valid)
  - v4_ragas_per_tier_config.csv/.md     (3 tiers × 5 configs × 4 metrics)
  - v4_relevancy_disentangled.csv/.md    (no-info rate vs relevancy, recomputed excluding no-info)
  - v4_v3_side_by_side.csv/.md           (side-by-side, labeled NOT a controlled comparison)
  - v4_question_faithfulness_gap.csv/.md (30 questions ranked by LightRAG-hybrid - baseline gap)
  - v4_full_ragas_report.json            (machine-readable summary of all of the above)

Reproduces the no-info regex and bootstrap CI from rerun_v4_entity_boost.py exactly.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
TABLES_DIR = ROOT / "data" / "phase4" / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

V4_CSV = ROOT / "data" / "phase4" / "ragas_scores_v4.csv"
V4_RESULTS = ROOT / "data" / "phase4" / "eval_results_v4.json"
V3_CSV = ROOT / "data" / "phase4" / "ragas_scores_v3.csv"
V3_SUMMARY = ROOT / "data" / "phase4" / "analysis_summary_v3.csv"

CONFIG_ORDER = ["baseline", "naive_rag", "lightrag_local", "lightrag_global", "lightrag_hybrid"]
CONFIG_LABEL = {
    "baseline":        "A: Baseline (no retrieval)",
    "naive_rag":       "B: Naive RAG (TF-IDF)",
    "lightrag_local":  "C: LightRAG local",
    "lightrag_global": "D: LightRAG global",
    "lightrag_hybrid": "E: LightRAG hybrid",
}
METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
TIER_LABEL = {1: "Tier 1 (factual)", 2: "Tier 2 (mechanistic)", 3: "Tier 3 (discovery)"}
# tier strings in the CSV → int
TIER_MAP = {
    "tier1_factual": 1, "tier2_mechanistic": 2, "tier3_discovery": 3,
    1: 1, 2: 2, 3: 3,
}

# Official no-info regex from rerun_v4_entity_boost.py:372
NO_INFO_REGEX = (
    "no relevant evidence|insufficient|does not contain|"
    "not possible to determine|no specific|not available|"
    "not contain any information"
)


def bootstrap_ci(values, n_boot=1000, ci=0.95, seed=42):
    """Match rerun_v4_entity_boost.py:388 exactly."""
    arr = np.array(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan"), 0
    rng = np.random.default_rng(seed)
    means = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(means, [(1 - ci) / 2 * 100, (1 + (ci - 1) / 2 + ci) * 100 - 100])
    # cleaner: percentile(2.5, 97.5)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(arr.mean()), float(lo), float(hi), int(len(arr))


def df_to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    rows = df.astype(object).where(pd.notnull(df), "").values.tolist()
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    align = ["---:" if pd.api.types.is_numeric_dtype(df[c]) else "---" for c in cols]
    sep = "| " + " | ".join(align) + " |"
    body = "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in rows)
    return "\n".join([header, sep, body])


def write_table(name: str, df: pd.DataFrame, columns_md=None) -> None:
    df.to_csv(TABLES_DIR / f"{name}.csv", index=False)
    md_df = df if columns_md is None else df[columns_md]
    (TABLES_DIR / f"{name}.md").write_text(df_to_markdown(md_df) + "\n")
    print(f"  wrote {TABLES_DIR / name}.csv / .md")


def fmt_ci(mean, lo, hi):
    if pd.isna(mean):
        return "n/a"
    return f"{mean:.3f} [{lo:.3f}-{hi:.3f}]"


# ================================================================
# 1. Per-(config × metric) matrix with 95% CIs and n_valid
# ================================================================
def task_per_config_metric(df: pd.DataFrame) -> dict:
    print("\n[1] per-(config × metric) matrix")
    rows = []
    for cfg in CONFIG_ORDER:
        sub = df[df.config == cfg]
        row = {"config": cfg, "label": CONFIG_LABEL[cfg]}
        for m in METRICS:
            mean, lo, hi, n = bootstrap_ci(sub[m].values)
            row[f"{m}_mean"] = round(mean, 4)
            row[f"{m}_ci_lo"] = round(lo, 4)
            row[f"{m}_ci_hi"] = round(hi, 4)
            row[f"{m}_n"] = n
        rows.append(row)
    out = pd.DataFrame(rows)

    pretty_rows = []
    for _, r in out.iterrows():
        prow = {"config": r["label"]}
        for m in METRICS:
            prow[m] = fmt_ci(r[f"{m}_mean"], r[f"{m}_ci_lo"], r[f"{m}_ci_hi"])
        prow["n_valid"] = int(r["faithfulness_n"])
        pretty_rows.append(prow)
    pretty = pd.DataFrame(pretty_rows)
    write_table("v4_ragas_per_config_metric", pretty)
    out.to_csv(TABLES_DIR / "v4_ragas_per_config_metric_raw.csv", index=False)

    # winners
    winners = {}
    for m in METRICS:
        means = {cfg: float(out[out.config == cfg][f"{m}_mean"].iloc[0]) for cfg in CONFIG_ORDER}
        winner = max(means, key=means.get)
        winners[m] = {"winner": winner, "mean": means[winner], "all": means}
    return {"matrix": out.to_dict(orient="records"), "winners": winners}


# ================================================================
# 2. Within-v4 headline ranking + plain narrative
# ================================================================
def task_headline_ranking(matrix_results: dict) -> dict:
    print("\n[2] within-v4 headline ranking")
    winners = matrix_results["winners"]

    print(f"  faithfulness winner   : {winners['faithfulness']['winner']}  ({winners['faithfulness']['mean']:.3f})")
    print(f"  relevancy winner      : {winners['answer_relevancy']['winner']}  ({winners['answer_relevancy']['mean']:.3f})")
    print(f"  ctx_precision winner  : {winners['context_precision']['winner']}  ({winners['context_precision']['mean']:.3f})")
    print(f"  ctx_recall winner     : {winners['context_recall']['winner']}  ({winners['context_recall']['mean']:.3f})")

    # Composite: equally weighted (just an indicator)
    f = winners["faithfulness"]["all"]
    r = winners["answer_relevancy"]["all"]
    cp = winners["context_precision"]["all"]
    cr = winners["context_recall"]["all"]
    composite = {cfg: (f[cfg] + r[cfg] + cp[cfg] + cr[cfg]) / 4 for cfg in CONFIG_ORDER}
    overall_winner = max(composite, key=composite.get)

    # check monotone baseline < naive < LightRAG on faithfulness
    fb = f["baseline"]
    fn = f["naive_rag"]
    fl = max(f["lightrag_local"], f["lightrag_global"], f["lightrag_hybrid"])
    monotone = (fb < fn < fl)

    headline = {
        "faithfulness_ranking": sorted(f.items(), key=lambda kv: -kv[1]),
        "composite_equal_weight": sorted(composite.items(), key=lambda kv: -kv[1]),
        "overall_winner": overall_winner,
        "monotone_baseline_lt_naive_lt_lightrag_on_faithfulness": monotone,
        "narrative": "monotone improvement: baseline < naive < LightRAG (clean)" if monotone
                     else "NOT monotone — some inversion between configs (messier than expected)",
    }
    print(f"  composite winner      : {overall_winner}")
    print(f"  baseline<naive<LR (faith)? {monotone}")
    print(f"  narrative: {headline['narrative']}")
    return headline


# ================================================================
# 3. Per-tier × config breakdown
# ================================================================
def task_per_tier_config(df: pd.DataFrame) -> dict:
    print("\n[3] per-tier × config breakdown")
    rows = []
    for tier in [1, 2, 3]:
        for cfg in CONFIG_ORDER:
            sub = df[(df.tier == tier) & (df.config == cfg)]
            row = {"tier": tier, "tier_label": TIER_LABEL[tier], "config": cfg, "label": CONFIG_LABEL[cfg]}
            for m in METRICS:
                mean, lo, hi, n = bootstrap_ci(sub[m].values)
                row[f"{m}_mean"] = round(mean, 4)
                row[f"{m}_ci_lo"] = round(lo, 4)
                row[f"{m}_ci_hi"] = round(hi, 4)
                row[f"{m}_n"] = n
            rows.append(row)
    out = pd.DataFrame(rows)

    pretty_rows = []
    for _, r in out.iterrows():
        prow = {"tier": r["tier_label"], "config": r["label"]}
        for m in METRICS:
            prow[m] = fmt_ci(r[f"{m}_mean"], r[f"{m}_ci_lo"], r[f"{m}_ci_hi"])
        prow["n_valid"] = int(r["faithfulness_n"])
        pretty_rows.append(prow)
    pretty = pd.DataFrame(pretty_rows)
    write_table("v4_ragas_per_tier_config", pretty)
    out.to_csv(TABLES_DIR / "v4_ragas_per_tier_config_raw.csv", index=False)

    # where does the KG help most? compute LightRAG-hybrid - baseline faithfulness gain by tier
    gains = {}
    for tier in [1, 2, 3]:
        baseline = float(out[(out.tier == tier) & (out.config == "baseline")]["faithfulness_mean"].iloc[0])
        lr_hyb = float(out[(out.tier == tier) & (out.config == "lightrag_hybrid")]["faithfulness_mean"].iloc[0])
        gains[TIER_LABEL[tier]] = {
            "baseline_faith": round(baseline, 3),
            "lightrag_hybrid_faith": round(lr_hyb, 3),
            "gap": round(lr_hyb - baseline, 3),
        }
    print("  faithfulness gain by tier:")
    for k, v in gains.items():
        print(f"    {k:30s} baseline={v['baseline_faith']:.3f}  LR-hyb={v['lightrag_hybrid_faith']:.3f}  Δ={v['gap']:+.3f}")
    return {"matrix": out.to_dict(orient="records"), "gains": gains}


# ================================================================
# 4. Disentangle answer_relevancy via no-info rate
# ================================================================
def task_relevancy_disentangle(df: pd.DataFrame, results_records: list) -> dict:
    print("\n[4] disentangle answer_relevancy via no-info filter")
    # join eval_results (which has the answer text) with ragas scores by (question_id, config, model)
    er = pd.DataFrame(results_records)[["question_id", "config", "model", "answer"]]
    er["is_no_info"] = er["answer"].str.contains(NO_INFO_REGEX, case=False, na=False)
    merged = df.merge(er, on=["question_id", "config", "model"], how="left")

    rows = []
    for cfg in CONFIG_ORDER:
        sub = merged[merged.config == cfg]
        n_total = len(sub)
        n_no_info = int(sub["is_no_info"].sum())
        no_info_rate = n_no_info / n_total if n_total else float("nan")

        # relevancy as-reported (all answers)
        rel_all_mean, rel_all_lo, rel_all_hi, rel_n = bootstrap_ci(sub["answer_relevancy"].values)
        # relevancy excluding no-info
        rel_excl = sub[~sub["is_no_info"]]["answer_relevancy"].values
        rel_excl_mean, rel_excl_lo, rel_excl_hi, rel_excl_n = bootstrap_ci(rel_excl)
        # relevancy among no-info answers
        rel_ni = sub[sub["is_no_info"]]["answer_relevancy"].values
        rel_ni_mean, rel_ni_lo, rel_ni_hi, rel_ni_n = bootstrap_ci(rel_ni) if len(rel_ni) else (float("nan"),)*3 + (0,)

        # faithfulness excluding no-info (since faithfulness over abstention is defined by RAGAS as well)
        faith_excl = sub[~sub["is_no_info"]]["faithfulness"].values
        f_excl_mean, f_excl_lo, f_excl_hi, f_excl_n = bootstrap_ci(faith_excl)

        rows.append({
            "config": CONFIG_LABEL[cfg],
            "n_total": n_total,
            "n_no_info": n_no_info,
            "no_info_rate": f"{no_info_rate:.3f}",
            "relevancy_all": fmt_ci(rel_all_mean, rel_all_lo, rel_all_hi),
            "relevancy_excl_no_info": fmt_ci(rel_excl_mean, rel_excl_lo, rel_excl_hi) + f"  (n={rel_excl_n})",
            "relevancy_among_no_info": fmt_ci(rel_ni_mean, rel_ni_lo, rel_ni_hi) + f"  (n={rel_ni_n})",
            "faithfulness_excl_no_info": fmt_ci(f_excl_mean, f_excl_lo, f_excl_hi) + f"  (n={f_excl_n})",
        })
    pretty = pd.DataFrame(rows)
    write_table("v4_relevancy_disentangled", pretty)

    # narrative
    print("\n  per-config no-info-rate × relevancy-recomputed:")
    for r in rows:
        print(f"    {r['config']:35s}  no-info={r['no_info_rate']}  rel-all={r['relevancy_all']:30s}  rel-excl={r['relevancy_excl_no_info']}")

    return {"rows": rows}


# ================================================================
# 5. v3 ↔ v4 side-by-side (NOT a controlled comparison)
# ================================================================
def task_v3_v4_side_by_side(v4_df: pd.DataFrame) -> dict:
    print("\n[5] v3↔v4 side-by-side (NOT a controlled comparison)")
    if not V3_CSV.exists():
        print(f"  ! {V3_CSV} not found; skipping")
        return {"skipped": True}

    v3 = pd.read_csv(V3_CSV)
    rows = []
    for cfg in CONFIG_ORDER:
        v3sub = v3[v3.config == cfg]
        v4sub = v4_df[v4_df.config == cfg]
        row = {"config": CONFIG_LABEL[cfg]}
        for m in METRICS:
            v3_mean, v3_lo, v3_hi, v3_n = bootstrap_ci(v3sub[m].values) if m in v3.columns else (float("nan"),)*3+(0,)
            v4_mean, v4_lo, v4_hi, v4_n = bootstrap_ci(v4sub[m].values)
            row[f"v3_{m}"] = fmt_ci(v3_mean, v3_lo, v3_hi)
            row[f"v4_{m}"] = fmt_ci(v4_mean, v4_lo, v4_hi)
            row[f"Δ_{m}"] = (
                f"{v4_mean - v3_mean:+.3f}" if not (pd.isna(v3_mean) or pd.isna(v4_mean)) else "n/a"
            )
        rows.append(row)
    pretty = pd.DataFrame(rows)
    write_table("v4_v3_side_by_side", pretty)
    print("  ⚠ This is NOT a controlled comparison — KG was rebuilt, judge changed (free→paid gpt-4o-mini),")
    print("    top_k changed (10→3), and the question set/ground truth was regenerated. v3↔v4 deltas are")
    print("    confounded by all four simultaneous changes. See DECISIONS.md D10.")
    return {"rows": rows}


# ================================================================
# 6. Question-level LightRAG-hybrid − baseline faithfulness gap
# ================================================================
def task_question_faithfulness_gap(df: pd.DataFrame) -> dict:
    print("\n[6] question-level faithfulness gap (lightrag_hybrid - baseline)")
    # Average over models within (question_id, config); then compute gap
    pivot = df.groupby(["question_id", "config"])["faithfulness"].mean().unstack("config")
    pivot = pivot.reindex(columns=CONFIG_ORDER)

    # tier + question text
    qmeta = df.groupby("question_id").agg({"tier": "first", "user_input": "first"}).reset_index()

    out = pivot.reset_index()
    out = out.merge(qmeta, on="question_id", how="left")
    out["faith_gap_LR_hyb_minus_baseline"] = (out["lightrag_hybrid"] - out["baseline"]).round(4)
    out = out.sort_values("faith_gap_LR_hyb_minus_baseline", ascending=False)

    pretty_cols = ["question_id", "tier", "user_input",
                   "baseline", "naive_rag", "lightrag_hybrid",
                   "faith_gap_LR_hyb_minus_baseline"]
    pretty = out[pretty_cols].copy()
    # truncate question text for md readability
    pretty["user_input"] = pretty["user_input"].str.slice(0, 80) + pretty["user_input"].apply(
        lambda s: "…" if len(s) > 80 else "")
    for c in ["baseline", "naive_rag", "lightrag_hybrid", "faith_gap_LR_hyb_minus_baseline"]:
        pretty[c] = pretty[c].round(3)
    write_table("v4_question_faithfulness_gap", pretty)
    out.to_csv(TABLES_DIR / "v4_question_faithfulness_gap_raw.csv", index=False)

    print(f"  top 5 questions by gap (LR_hyb − baseline):")
    for _, r in pretty.head(5).iterrows():
        print(f"    Q{int(r['question_id']):2d}  tier={int(r['tier']) if pd.notna(r['tier']) else '?'}  Δ={r['faith_gap_LR_hyb_minus_baseline']:+.3f}  {r['user_input']}")
    print(f"  bottom 5:")
    for _, r in pretty.tail(5).iterrows():
        print(f"    Q{int(r['question_id']):2d}  tier={int(r['tier']) if pd.notna(r['tier']) else '?'}  Δ={r['faith_gap_LR_hyb_minus_baseline']:+.3f}  {r['user_input']}")

    return {
        "n_questions": len(pretty),
        "top10_question_ids": pretty.head(10)["question_id"].astype(int).tolist(),
        "median_gap": float(pretty["faith_gap_LR_hyb_minus_baseline"].median()),
        "mean_gap": float(pretty["faith_gap_LR_hyb_minus_baseline"].mean()),
    }


def main():
    print("=" * 70)
    print("V4 RAGAS REPORT  (local, no API)")
    print("=" * 70)
    df = pd.read_csv(V4_CSV)
    df["tier_str"] = df["tier"]
    df["tier"] = df["tier"].map(TIER_MAP)
    print(f'loaded {V4_CSV.name}: {len(df)} rows  configs={df.config.nunique()}  '
          f'models={df.model.nunique()}  questions={df.question_id.nunique()}  tiers={sorted(df.tier.unique())}')

    with open(V4_RESULTS) as f:
        results_records = json.load(f)
    print(f"loaded {V4_RESULTS.name}: {len(results_records)} records")

    summary = {}
    summary["per_config_metric"] = task_per_config_metric(df)
    summary["headline"] = task_headline_ranking(summary["per_config_metric"])
    summary["per_tier_config"] = task_per_tier_config(df)
    summary["relevancy_disentangled"] = task_relevancy_disentangle(df, results_records)
    summary["v3_v4_side_by_side"] = task_v3_v4_side_by_side(df)
    summary["question_faith_gap"] = task_question_faithfulness_gap(df)

    out = ROOT / "data" / "phase4" / "v4_full_ragas_report.json"
    out.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n[summary] {out}")
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
