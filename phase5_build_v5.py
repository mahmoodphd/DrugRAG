"""Phase 5 step 4: build v5 ranking + dual-list deliverables.

INPUTS (read-only):
    data/external/41598_2020_70440_MOESM9_ESM.xlsx  -> 10 network-quantity features
                                                       (falls back to Supplementary*/)
    data/phase5/drug_activity.csv                   -> activity_magnitude + cluster
    data/phase4/drug_repurposing_candidates_v4.csv  -> n_papers, edges_all, angio_edges,
                                                       novelty, kg_plausibility (carried)
    data/phase1/drug_master_list.csv                -> backstop for drugs v4 dropped
                                                       under (n_papers==0 AND edges_all==0)

OUTPUTS:
    data/phase5/drug_activity.csv                   -> rewritten with net_direction column
    data/phase5/drug_repurposing_candidates_v5.csv  -> the v5 ranking (no exclusions)
    data/phase5/tables/top20_anti_angiogenic_v5.{csv,md}
    data/phase5/tables/top20_pro_angiogenic_v5.{csv,md}
    data/phase5/tables/active_dark_anti_angiogenic_v5.{csv,md}
    data/phase5/tables/active_dark_pro_angiogenic_v5.{csv,md}
    data/phase5/tables/v5_validation_gate.{json,md}

POLICY: v3 + locked v4 are READ-ONLY. No API calls. Phase 4 untouched.

Score formula:
    repurposing_score_v5 = 0.40 * activity_magnitude
                         + 0.35 * novelty
                         + 0.25 * kg_plausibility
where novelty + kg_plausibility are carried EXACTLY from the locked v4 method
(topical lexicon, alias-union edges); activity_magnitude is the Phase-5 z-score-
to-DMSO Euclidean magnitude over 101 features (computed in
phase5_activity_magnitude.py).
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from drug_normalize import normalize_drug_name_v2  # type: ignore

PHASE4 = ROOT / "data" / "phase4"
PHASE5 = ROOT / "data" / "phase5"
TABLES = PHASE5 / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

V4_CSV = PHASE4 / "drug_repurposing_candidates_v4.csv"
PHENO_MAP = PHASE5 / "drug_phenocluster_map.csv"
ACTIVITY_CSV = PHASE5 / "drug_activity.csv"          # rewritten with net_direction
PIPELINE_CSV = ROOT / "data" / "phase1" / "drug_master_list.csv"

V5_CSV = PHASE5 / "drug_repurposing_candidates_v5.csv"
GATE_JSON = TABLES / "v5_validation_gate.json"
GATE_MD = TABLES / "v5_validation_gate.md"


def _screen_xlsx(name: str) -> str:
    """Screen supplementary table: data/external/ first, then the legacy
    'Supplementary*' folder."""
    p = ROOT / "data" / "external" / name
    if p.is_file():
        return str(p)
    legacy = sorted(glob.glob(str(ROOT / "Supplementary*" / name)))
    if legacy:
        return legacy[0]
    raise FileNotFoundError(f"{name} not found in data/external/ or Supplementary*/")


MOESM9 = _screen_xlsx("41598_2020_70440_MOESM9_ESM.xlsx")

# ---- Network-quantity features (10) for net_direction --------------------
NETWORK_FEATURES = [
    "Analysedarea", "Nbnodes", "NbJunctions",
    "Nbmasterjunction", "Nbmastersegments", "Totmastersegmentslength",
    "Totlength", "Nbmeshes", "Totmeshesarea", "Nbbranches",
]

# ---- v5 weights (TUNABLE; not hand-tuned to force any drug's rank) ----------
W_ACTIVITY = 0.40
W_NOVELTY = 0.35
W_KG = 0.25

# ---- "active dark" thresholds ---------------------------------------------
# is_active_dark = (n_papers == 0) AND (activity_magnitude >= 75th percentile)
ACTIVE_DARK_PCT = 0.75


def strip_quotes(s) -> str:
    if s is None:
        return ""
    return str(s).strip().strip("'\"")


def norm_key(name: str) -> str:
    return normalize_drug_name_v2(name).lower()


# ============================================================================
# Step 1: net_direction per drug from the 10 network features
# ============================================================================
def compute_net_direction() -> pd.DataFrame:
    print("=" * 72)
    print("STEP 1: net_direction = mean signed z-score over 10 network features")
    print("=" * 72)
    wb = openpyxl.load_workbook(MOESM9, read_only=True, data_only=True)
    ws = wb["Supplementary Table 6"]
    it = ws.iter_rows(values_only=True)
    header = [strip_quotes(c) for c in next(it)]
    body = [[strip_quotes(r[0])] + list(r[1:]) for r in it if r and r[0] is not None]
    wb.close()
    df = pd.DataFrame(body, columns=header)
    feat_cols = list(df.columns[1:])
    df[feat_cols] = df[feat_cols].astype(float)

    missing = [c for c in NETWORK_FEATURES if c not in df.columns]
    if missing:
        raise RuntimeError(f"network features not found in Table 6 columns: {missing}")
    print(f"[load] Table 6: {len(df)} drugs x {len(feat_cols)} features  "
          f"(network subset = {len(NETWORK_FEATURES)})")

    drug_col = df.columns[0]
    dmso_mask = df[drug_col].astype(str).str.fullmatch("DMSO", case=False)
    dmso_row = df.loc[dmso_mask, NETWORK_FEATURES].to_numpy(dtype=float)[0]    # (10,)

    feats = df[NETWORK_FEATURES].to_numpy(dtype=float)                          # (1280, 10)
    std = feats.std(axis=0, ddof=1)                                             # (10,)
    if (std <= 1e-12).any():
        raise RuntimeError("zero-variance network feature in MOESM9; cannot z-score")

    z = (feats - dmso_row) / std                                                # signed z
    net_direction = z.mean(axis=1)                                              # (1280,)
    print(f"        DMSO net_direction = {net_direction[int(np.where(dmso_mask)[0][0])]:.4f}  "
          f"(should be 0)")
    print(f"        net_direction range [{net_direction.min():.3f}, {net_direction.max():.3f}]")
    n_neg = int((net_direction < 0).sum())
    n_pos = int((net_direction > 0).sum())
    print(f"        anti-angiogenic (neg): {n_neg}   pro-angiogenic (pos): {n_pos}")

    nd = pd.DataFrame({drug_col: df[drug_col].values, "net_direction": net_direction})
    nd["norm"] = nd[drug_col].map(norm_key)
    # average across salt variants on the norm key
    nd_g = nd.groupby("norm", as_index=False)["net_direction"].mean()
    return nd_g


def merge_net_direction_into_activity(net_df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n[merge] {ACTIVITY_CSV.relative_to(ROOT)}  +  net_direction")
    act = pd.read_csv(ACTIVITY_CSV)
    act["norm"] = act["drug_name"].map(norm_key)
    merged = act.merge(net_df, on="norm", how="left")
    n_hits = int(merged["net_direction"].notna().sum())
    print(f"        {n_hits}/{len(merged)} pipeline drugs got a net_direction")
    # rewrite drug_activity.csv with the new column
    out_cols = ["drug_name", "canonical_name", "cluster",
                "activity_magnitude", "net_direction", "matched_flag"]
    merged = merged.reindex(columns=out_cols + ["norm"]).drop_duplicates(subset="drug_name")
    merged[out_cols].to_csv(ACTIVITY_CSV, index=False)
    print(f"        rewrote {ACTIVITY_CSV.relative_to(ROOT)}  ({len(merged)} rows)")
    return merged


# ============================================================================
# Step 2: build v5 ranking, no exclusions
# ============================================================================
def load_v4_carry() -> pd.DataFrame:
    print("\n" + "=" * 72)
    print("STEP 2: build v5 ranking (no exclusions)")
    print("=" * 72)
    print(f"[load] {V4_CSV.relative_to(ROOT)} (carry novelty / kg_plausibility / counts)")
    # v4 first line is a leading comment -- skip it
    v4 = pd.read_csv(V4_CSV, comment="#")
    keep = ["drug_name", "n_papers", "edges_all", "angio_edges",
            "novelty", "kg_plausibility"]
    v4 = v4[keep].copy()
    print(f"        v4 ranked rows: {len(v4)}")
    return v4


def build_v5(activity_with_dir: pd.DataFrame, v4: pd.DataFrame) -> pd.DataFrame:
    df = activity_with_dir.merge(v4, on="drug_name", how="left")
    n_dropped_recovered = int(df["n_papers"].isna().sum())
    print(f"        v4-dropped drugs recovered : {n_dropped_recovered}  "
          f"(n_papers==0 AND edges_all==0 in v4)")

    # For drugs the v4 build dropped: they had n_papers==0 AND edges_all==0,
    # which means n_papers=0, edges_all=0, angio_edges=0,
    # novelty = max(0.1, 1 - log10(1)/3) = 1.0,
    # kg_plausibility = min(1, log10(1)/log10(11)) = 0.0.
    fill_map = {"n_papers": 0, "edges_all": 0, "angio_edges": 0,
                "novelty": 1.0, "kg_plausibility": 0.0}
    for col, val in fill_map.items():
        df[col] = df[col].fillna(val)

    df[["n_papers", "edges_all", "angio_edges"]] = df[["n_papers", "edges_all", "angio_edges"]].astype(int)

    # v5 score
    df["repurposing_score_v5"] = (
        W_ACTIVITY * df["activity_magnitude"].astype(float)
        + W_NOVELTY * df["novelty"].astype(float)
        + W_KG * df["kg_plausibility"].astype(float)
    ).round(4)

    # rank
    df = df.sort_values(
        by=["repurposing_score_v5", "activity_magnitude", "novelty", "drug_name"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    df["rank"] = df.index + 1

    # is_active_dark: dark literature + top-quartile activity
    act_threshold = float(df["activity_magnitude"].quantile(ACTIVE_DARK_PCT))
    df["is_active_dark"] = (df["n_papers"] == 0) & (df["activity_magnitude"] >= act_threshold)
    n_ad = int(df["is_active_dark"].sum())
    print(f"        active-dark threshold (75th-pct activity): {act_threshold:.4f}")
    print(f"        is_active_dark drugs: {n_ad}")

    out_cols = [
        "drug_name", "canonical_name", "cluster", "net_direction",
        "activity_magnitude", "n_papers", "angio_edges",
        "novelty", "kg_plausibility", "repurposing_score_v5", "rank",
        "is_active_dark",
    ]
    df_out = df[out_cols].copy()
    for c in ["net_direction", "activity_magnitude", "novelty",
              "kg_plausibility", "repurposing_score_v5"]:
        df_out[c] = df_out[c].round(4)
    df_out.to_csv(V5_CSV, index=False)
    print(f"\n[write] {V5_CSV.relative_to(ROOT)}  ({len(df_out)} rows, no exclusions)")
    return df_out


# ============================================================================
# Step 3: headline lists
# ============================================================================
def df_to_md(df: pd.DataFrame, fp: Path, title: str, note: str = "") -> None:
    lines = [f"# {title}", ""]
    if note:
        lines += [note, ""]
    cols = list(df.columns)
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append(f"{v:.4f}")
            elif pd.isna(v):
                cells.append("")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    fp.write_text("\n".join(lines) + "\n")


def emit_headline_lists(df: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("STEP 3: headline lists")
    print("=" * 72)

    show = ["rank", "drug_name", "cluster", "net_direction", "activity_magnitude",
            "n_papers", "angio_edges", "novelty", "kg_plausibility",
            "repurposing_score_v5", "is_active_dark"]

    anti = df[df["net_direction"] < 0].sort_values(
        "repurposing_score_v5", ascending=False).head(20).copy()
    pro = df[df["net_direction"] > 0].sort_values(
        "repurposing_score_v5", ascending=False).head(20).copy()
    # within-list rank for readability
    anti.insert(0, "list_rank", range(1, len(anti) + 1))
    pro.insert(0, "list_rank", range(1, len(pro) + 1))

    anti[show + ["list_rank"]].rename(columns={"list_rank": "list_rank"})

    # write tables
    cols_anti = ["list_rank"] + show
    anti[cols_anti].to_csv(TABLES / "top20_anti_angiogenic_v5.csv", index=False)
    pro[cols_anti].to_csv(TABLES / "top20_pro_angiogenic_v5.csv", index=False)
    df_to_md(
        anti[cols_anti], TABLES / "top20_anti_angiogenic_v5.md",
        "Top 20 anti-angiogenic candidates (v5)",
        "Filter: net_direction < 0. Sort: repurposing_score_v5 desc. "
        "**No exclusions.** Score = 0.40 \u00b7 activity + 0.35 \u00b7 novelty + 0.25 \u00b7 kg_plausibility.",
    )
    df_to_md(
        pro[cols_anti], TABLES / "top20_pro_angiogenic_v5.md",
        "Top 20 pro-angiogenic candidates (v5)",
        "Filter: net_direction > 0. Sort: repurposing_score_v5 desc. "
        "**No exclusions.** Score = 0.40 \u00b7 activity + 0.35 \u00b7 novelty + 0.25 \u00b7 kg_plausibility.",
    )

    # active-dark centerpiece, split by direction
    dark = df[df["n_papers"] == 0].copy()
    dark_anti = dark[dark["net_direction"] < 0].sort_values(
        "activity_magnitude", ascending=False).copy()
    dark_pro = dark[dark["net_direction"] > 0].sort_values(
        "activity_magnitude", ascending=False).copy()

    dark_cols = ["drug_name", "cluster", "net_direction", "activity_magnitude",
                 "angio_edges", "novelty", "kg_plausibility",
                 "repurposing_score_v5", "rank", "is_active_dark"]
    dark_anti.insert(0, "list_rank", range(1, len(dark_anti) + 1))
    dark_pro.insert(0, "list_rank", range(1, len(dark_pro) + 1))
    write_anti = dark_anti[["list_rank"] + dark_cols].head(40)   # cap reporting at 40
    write_pro = dark_pro[["list_rank"] + dark_cols].head(40)

    write_anti.to_csv(TABLES / "active_dark_anti_angiogenic_v5.csv", index=False)
    write_pro.to_csv(TABLES / "active_dark_pro_angiogenic_v5.csv", index=False)
    df_to_md(
        write_anti, TABLES / "active_dark_anti_angiogenic_v5.md",
        "Active-dark anti-angiogenic candidates (v5 centerpiece)",
        f"Filter: n_papers == 0 AND net_direction < 0. Sort: activity_magnitude desc. "
        f"Showing top {len(write_anti)} of {len(dark_anti)} total.",
    )
    df_to_md(
        write_pro, TABLES / "active_dark_pro_angiogenic_v5.md",
        "Active-dark pro-angiogenic candidates (v5 centerpiece)",
        f"Filter: n_papers == 0 AND net_direction > 0. Sort: activity_magnitude desc. "
        f"Showing top {len(write_pro)} of {len(dark_pro)} total.",
    )

    print(f"  top-20 anti  : {len(anti)}  -> tables/top20_anti_angiogenic_v5.{{csv,md}}")
    print(f"  top-20 pro   : {len(pro)}  -> tables/top20_pro_angiogenic_v5.{{csv,md}}")
    print(f"  active-dark anti : {len(dark_anti)} total  (writing top {len(write_anti)})")
    print(f"  active-dark pro  : {len(dark_pro)} total  (writing top {len(write_pro)})")
    return {
        "top20_anti": anti[cols_anti].to_dict("records"),
        "top20_pro":  pro[cols_anti].to_dict("records"),
        "active_dark_anti_total": int(len(dark_anti)),
        "active_dark_pro_total":  int(len(dark_pro)),
        "active_dark_anti_top10": dark_anti[["list_rank"] + dark_cols].head(10).to_dict("records"),
        "active_dark_pro_top10":  dark_pro[["list_rank"] + dark_cols].head(10).to_dict("records"),
    }


# ============================================================================
# Step 4: validation gate
# ============================================================================
PC7_CONTROLS = ["Sunitinib", "SU5416", "Suramin", "DMH4"]


def _search_key(s):
    return norm_key(s).replace(" ", "")


def validation_gate(df: pd.DataFrame, headlines: dict) -> dict:
    print("\n" + "=" * 72)
    print("STEP 4: VALIDATION GATE")
    print("=" * 72)

    canon_compact = df["canonical_name"].fillna("").str.replace(" ", "", regex=False)
    rows = []
    print("\n[gate-A] PC7 controls (well-studied) should be ANTI + high-activity"
          " but NOT #1 overall:")
    pc7_results = []
    for stem in PC7_CONTROLS:
        sk = _search_key(stem)
        hit = df[canon_compact.str.contains(sk, na=False, regex=False)]
        for _, r in hit.iterrows():
            anti_rank = (df[df["net_direction"] < 0]
                         .sort_values("repurposing_score_v5", ascending=False)
                         .reset_index(drop=True))
            anti_rank["pos"] = anti_rank.index + 1
            pos = int(anti_rank.loc[anti_rank["drug_name"] == r["drug_name"], "pos"].iloc[0]) \
                if (anti_rank["drug_name"] == r["drug_name"]).any() else None
            pc7_results.append({
                "stem": stem, "drug_name": r["drug_name"],
                "cluster": int(r["cluster"]) if pd.notna(r["cluster"]) else None,
                "net_direction": round(float(r["net_direction"]), 4),
                "activity_magnitude": round(float(r["activity_magnitude"]), 4),
                "n_papers": int(r["n_papers"]),
                "angio_edges": int(r["angio_edges"]),
                "novelty": round(float(r["novelty"]), 4),
                "kg_plausibility": round(float(r["kg_plausibility"]), 4),
                "repurposing_score_v5": round(float(r["repurposing_score_v5"]), 4),
                "rank_overall": int(r["rank"]),
                "rank_anti_list": pos,
            })
            print(f"  {stem:10s}  rank_anti={pos}  "
                  f"score={r['repurposing_score_v5']:.4f}  "
                  f"net_dir={r['net_direction']:+.3f}  "
                  f"activity={r['activity_magnitude']:.3f}  "
                  f"n_papers={int(r['n_papers'])}  novelty={r['novelty']:.3f}  "
                  f"({r['drug_name']})")
    rows.append(pc7_results)

    # Heuristic verdict: should be anti (net_direction<0) + activity in top-half
    # AND should NOT be top-3 in the anti list (because their novelty is low).
    anti_top3 = [r["drug_name"] for r in headlines["top20_anti"][:3]]
    pc7_in_top3 = [r for r in pc7_results if r["drug_name"] in anti_top3]
    pc7_anti_high_act = all(r["net_direction"] < 0 and r["activity_magnitude"] > 0.10
                            for r in pc7_results)
    print(f"\n  PC7 all anti+active     : {pc7_anti_high_act}")
    print(f"  PC7 in anti top-3       : "
          f"{[r['drug_name'] for r in pc7_in_top3] if pc7_in_top3 else 'NONE  (correct -- low novelty pushes them down)'}")
    gate_a_pass = pc7_anti_high_act and not pc7_in_top3

    # Gate B: top of each list should be high-activity + understudied
    print("\n[gate-B] top of each list -- high activity + low n_papers expected:")
    top_anti = headlines["top20_anti"][:5]
    top_pro = headlines["top20_pro"][:5]
    for r in top_anti:
        print(f"  ANTI #{r['list_rank']}  {r['drug_name']:35s}  "
              f"activity={r['activity_magnitude']:.3f}  "
              f"n_papers={int(r['n_papers'])}  novelty={r['novelty']:.3f}")
    for r in top_pro:
        print(f"  PRO  #{r['list_rank']}  {r['drug_name']:35s}  "
              f"activity={r['activity_magnitude']:.3f}  "
              f"n_papers={int(r['n_papers'])}  novelty={r['novelty']:.3f}")
    # Pass if top-5 of each list have median n_papers <= 2 (i.e., understudied)
    med_anti_pap = float(pd.Series([r["n_papers"] for r in top_anti]).median())
    med_pro_pap = float(pd.Series([r["n_papers"] for r in top_pro]).median())
    gate_b_pass = (med_anti_pap <= 2) and (med_pro_pap <= 2)
    print(f"  median n_papers (anti top-5): {med_anti_pap}")
    print(f"  median n_papers (pro  top-5): {med_pro_pap}")

    # Gate C: spot-check 4 top active-dark drugs (cluster + percentile)
    print("\n[gate-C] active-dark spot check (top 4 by activity, both directions):")
    act_pct_lookup = df["activity_magnitude"].rank(pct=True)
    df_p = df.assign(pct=act_pct_lookup.values)
    pool = (df_p[df_p["n_papers"] == 0]
            .sort_values("activity_magnitude", ascending=False)
            .head(4))
    spot_rows = []
    for _, r in pool.iterrows():
        spot_rows.append({
            "drug_name": r["drug_name"],
            "cluster": int(r["cluster"]) if pd.notna(r["cluster"]) else None,
            "net_direction": round(float(r["net_direction"]), 4),
            "activity_magnitude": round(float(r["activity_magnitude"]), 4),
            "activity_pct_rank": round(float(r["pct"]), 4),
            "rank_overall": int(r["rank"]),
            "is_active_dark": bool(r["is_active_dark"]),
        })
        print(f"  {r['drug_name']:40s}  PC{int(r['cluster'])}  "
              f"net_dir={r['net_direction']:+.3f}  "
              f"act={r['activity_magnitude']:.3f}  pct={r['pct']:.3f}  "
              f"overall_rank={int(r['rank'])}")

    verdict = ("PASS" if (gate_a_pass and gate_b_pass) else "REVIEW REQUIRED")
    print(f"\n[gate] verdict: {verdict}   (A:{gate_a_pass}  B:{gate_b_pass})")

    return {
        "gate_A_pc7_controls": pc7_results,
        "gate_A_pass": bool(gate_a_pass),
        "gate_A_pc7_in_anti_top3": [r["drug_name"] for r in pc7_in_top3],
        "gate_B_top_anti_top5": top_anti,
        "gate_B_top_pro_top5": top_pro,
        "gate_B_median_n_papers_anti_top5": med_anti_pap,
        "gate_B_median_n_papers_pro_top5": med_pro_pap,
        "gate_B_pass": bool(gate_b_pass),
        "gate_C_active_dark_spotcheck": spot_rows,
        "verdict": verdict,
    }


# ============================================================================
def main():
    nd = compute_net_direction()
    activity_with_dir = merge_net_direction_into_activity(nd)
    v4 = load_v4_carry()
    v5 = build_v5(activity_with_dir, v4)
    headlines = emit_headline_lists(v5)
    gate = validation_gate(v5, headlines)

    summary = {
        "weights": {"activity": W_ACTIVITY, "novelty": W_NOVELTY,
                    "kg_plausibility": W_KG, "tunable": True},
        "active_dark_threshold_pct": ACTIVE_DARK_PCT,
        "n_drugs_ranked": int(len(v5)),
        "n_active_dark": int(v5["is_active_dark"].sum()),
        "n_anti": int((v5["net_direction"] < 0).sum()),
        "n_pro": int((v5["net_direction"] > 0).sum()),
        "outputs": {
            "v5_csv": str(V5_CSV.relative_to(ROOT)),
            "tables_dir": str(TABLES.relative_to(ROOT)),
        },
        "validation_gate": gate,
    }
    GATE_JSON.write_text(json.dumps(summary, indent=2, default=str))
    # Markdown summary of the gate
    GATE_MD.write_text(
        "# v5 validation gate\n\n"
        f"- weights: activity={W_ACTIVITY}, novelty={W_NOVELTY}, kg={W_KG} (tunable)\n"
        f"- drugs ranked: {summary['n_drugs_ranked']}  "
        f"(anti={summary['n_anti']}, pro={summary['n_pro']})\n"
        f"- active-dark drugs: {summary['n_active_dark']} "
        f"(top-quartile activity AND n_papers==0)\n\n"
        f"## verdict: **{gate['verdict']}**\n\n"
        f"- gate A (PC7 controls anti+active, NOT in anti top-3): "
        f"**{'PASS' if gate['gate_A_pass'] else 'FAIL'}**\n"
        f"  PC7 in anti top-3: {gate['gate_A_pc7_in_anti_top3'] or 'NONE'}\n"
        f"- gate B (top-5 of each list median n_papers <= 2): "
        f"**{'PASS' if gate['gate_B_pass'] else 'FAIL'}**\n"
        f"  median n_papers anti top-5 = {gate['gate_B_median_n_papers_anti_top5']}, "
        f"pro top-5 = {gate['gate_B_median_n_papers_pro_top5']}\n"
        f"- gate C (active-dark spot check): see JSON\n"
    )
    print(f"\n[write] {GATE_JSON.relative_to(ROOT)}")
    print(f"[write] {GATE_MD.relative_to(ROOT)}")
    print("\nDONE")


if __name__ == "__main__":
    main()
