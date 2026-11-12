"""Phase 5 step 2 + 3: continuous activity magnitude + per-cluster validation.

Computes a single "did it do something" signal per drug from the 101 morphology
features in MOESM9 / Supplementary Table 6 (Sailem & Al Haj Zen 2020, Scientific
Reports), then joins it onto the PhenoCluster map produced by
phase5_phenocluster_join.py and validates against expected biology.

Method (per the user's spec):
    1. For each of the 101 features, compute the standard deviation across all
       1,280 perturbations (this is the population scale).
    2. Z-score each cell against the DMSO control row:  z_ij = (x_ij - dmso_j) / std_j
    3. Per-drug Euclidean distance from DMSO:  d_i = sqrt(sum_j z_ij^2)
    4. Min-max normalize d to [0,1] -> activity_magnitude.

Outputs:
    data/phase5/drug_activity.csv          (drug, canonical_name, cluster,
                                             activity_magnitude, matched_flag)
    data/phase5/activity_validation.json   (per-cluster means + spot checks)

Read-only on v3 / locked v4 / supplementary spreadsheets. No API. No score build.
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

PHASE5 = ROOT / "data" / "phase5"
PHASE5.mkdir(parents=True, exist_ok=True)


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
PHENO_MAP = PHASE5 / "drug_phenocluster_map.csv"
OUT_CSV = PHASE5 / "drug_activity.csv"
OUT_JSON = PHASE5 / "activity_validation.json"

# Spot-checks: expected to be HIGH (real perturbation)
EXPECTED_HIGH = ["Sunitinib", "SU5416", "Suramin", "DMH4"]
# Expected to be LOW (clinical drug, control-cluster PC2)
EXPECTED_LOW = ["Thalidomide"]

# Cluster biology (paper) — used to flag mean-magnitude inconsistencies.
# "high" => strong perturbation phenotype; "low" => control-like; "mid" => intermediate.
CLUSTER_DESCRIPTION = {
    1: ("high", "strong-phenotype"),
    2: ("low",  "control-like"),
    3: ("mid",  "intermediate"),
    4: ("high", "strong-phenotype (glutamate/NMDA)"),
    5: ("mid",  "intermediate (glutamate-related)"),
    6: ("mid",  "intermediate"),
    7: ("high", "strong-phenotype (anti-angiogenic)"),
}


def strip_quotes(s) -> str:
    if s is None:
        return ""
    return str(s).strip().strip("'\"")


def norm_key(name: str) -> str:
    return normalize_drug_name_v2(name).lower()


def _search_key(s: str) -> str:
    return norm_key(s).replace(" ", "")


def load_table6() -> pd.DataFrame:
    print(f"[load] {Path(MOESM9).name} / 'Supplementary Table 6'")
    wb = openpyxl.load_workbook(MOESM9, read_only=True, data_only=True)
    ws = wb["Supplementary Table 6"]
    it = ws.iter_rows(values_only=True)
    header = [strip_quotes(c) for c in next(it)]
    body = []
    for r in it:
        if r and r[0] is not None:
            row = [strip_quotes(r[0])] + list(r[1:])
            body.append(row)
    wb.close()
    df = pd.DataFrame(body, columns=header)
    feat_cols = list(df.columns[1:])
    df[feat_cols] = df[feat_cols].astype(float)
    print(f"        rows={len(df)}  features={len(feat_cols)}")
    return df


def compute_activity_magnitude(t6: pd.DataFrame) -> pd.DataFrame:
    print("\n[compute] z-scoring 101 features against DMSO")
    drug_col = t6.columns[0]
    feat_cols = list(t6.columns[1:])
    feats = t6[feat_cols].to_numpy(dtype=float)              # (1280, 101)

    # locate DMSO control
    dmso_mask = t6[drug_col].astype(str).str.fullmatch("DMSO", case=False)
    if dmso_mask.sum() != 1:
        raise RuntimeError(f"expected exactly 1 DMSO row, got {dmso_mask.sum()}")
    dmso_idx = int(np.where(dmso_mask.values)[0][0])
    dmso_row = feats[dmso_idx]                                # (101,)
    print(f"        DMSO row index = {dmso_idx}")

    # population std across ALL 1280 perturbations (paper convention)
    std = feats.std(axis=0, ddof=1)                           # (101,)
    # safeguard: any feature with zero variance is informationless -> drop it
    keep = std > 1e-12
    n_dropped = int((~keep).sum())
    if n_dropped:
        print(f"        ! dropped {n_dropped} zero-variance feature(s)")
    feats_kept = feats[:, keep]
    dmso_kept = dmso_row[keep]
    std_kept = std[keep]

    # z-score against DMSO
    z = (feats_kept - dmso_kept) / std_kept                   # (1280, k)
    # per-drug Euclidean distance from DMSO
    d = np.sqrt((z ** 2).sum(axis=1))                          # (1280,)

    # min-max normalize to [0,1]
    d_min, d_max = float(d.min()), float(d.max())
    activity = (d - d_min) / (d_max - d_min)
    print(f"        raw distance:  min={d_min:.3f}  max={d_max:.3f}  "
          f"DMSO={d[dmso_idx]:.3f}")
    print(f"        normalized:    DMSO_activity={activity[dmso_idx]:.3f}  "
          f"(should be 0)")

    out = pd.DataFrame({
        drug_col: t6[drug_col].values,
        "raw_distance": d,
        "activity_magnitude": activity,
    })
    out["norm"] = out[drug_col].map(norm_key)
    return out, n_dropped, d_min, d_max, dmso_idx


def join_on_phenocluster(activity_df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n[join] {PHENO_MAP.relative_to(ROOT)}")
    pheno = pd.read_csv(PHENO_MAP)
    pheno["norm"] = pheno["drug_name"].map(norm_key)

    # collapse activity duplicates on norm key (salt variants of same compound):
    # average the magnitudes.
    activity_grouped = (activity_df.groupby("norm", as_index=False)
                        .agg(activity_magnitude=("activity_magnitude", "mean"),
                             raw_distance=("raw_distance", "mean")))

    merged = pheno.merge(activity_grouped, on="norm", how="left")
    has_act = merged["activity_magnitude"].notna()
    n_hits = int(has_act.sum())
    print(f"        pipeline drugs with activity: {n_hits}/{len(merged)}")

    out = merged[["drug_name", "canonical_name", "cluster", "activity_magnitude",
                  "matched_flag"]].copy()
    out.to_csv(OUT_CSV, index=False)
    print(f"        wrote {OUT_CSV.relative_to(ROOT)}  ({len(out)} rows)")
    return out, n_hits


def validate_per_cluster(merged: pd.DataFrame) -> dict:
    print("\n[validate] mean activity_magnitude per PhenoCluster")
    print("           (PC2 should be lowest; PC1/PC4/PC7 should be highest)")
    grp = merged.dropna(subset=["cluster", "activity_magnitude"]).copy()
    grp["cluster"] = grp["cluster"].astype(int)
    summary = (grp.groupby("cluster")["activity_magnitude"]
               .agg(["mean", "median", "std", "count"])
               .round(4))
    summary["expected"] = summary.index.map(lambda c: CLUSTER_DESCRIPTION[c][0])
    summary["description"] = summary.index.map(lambda c: CLUSTER_DESCRIPTION[c][1])
    summary = summary.reset_index()

    # ranking sanity: PC2 should rank near the bottom; PC7 should rank in top-3.
    rank_by_mean = summary.sort_values("mean", ascending=False).reset_index(drop=True)
    rank_by_mean["rank"] = rank_by_mean.index + 1

    print()
    print("rank cluster  mean    median  std     n     expected  description")
    print("-" * 72)
    for _, r in rank_by_mean.iterrows():
        print(f" {int(r['rank']):2d}   PC{int(r['cluster'])}    "
              f"{r['mean']:.4f}  {r['median']:.4f}  {r['std']:.4f}  "
              f"{int(r['count']):3d}   {r['expected']:5s}    {r['description']}")

    # consistency flags
    flags = []
    rank_lookup = dict(zip(rank_by_mean["cluster"], rank_by_mean["rank"]))
    for c in range(1, 8):
        expected = CLUSTER_DESCRIPTION[c][0]
        rank = rank_lookup[c]
        # rough thresholds: high -> rank 1-3, low -> rank 5-7, mid -> middle
        if expected == "high" and rank > 3:
            flags.append(f"PC{c} expected HIGH but ranks #{rank}/7")
        elif expected == "low" and rank < 5:
            flags.append(f"PC{c} expected LOW but ranks #{rank}/7 (i.e. unexpectedly active)")
    if flags:
        print("\n  ! consistency flags:")
        for f in flags:
            print(f"    - {f}")
    else:
        print("\n  all clusters consistent with paper biology")
    return {
        "per_cluster": summary.to_dict(orient="records"),
        "ranked_by_mean": rank_by_mean[["rank", "cluster", "mean", "expected", "description"]]
            .to_dict(orient="records"),
        "flags": flags,
    }


def spot_check(merged: pd.DataFrame) -> dict:
    print("\n[spot-check] expected HIGH (PC7 strong-phenotype controls):")
    canon_compact = merged["canonical_name"].fillna("").str.replace(" ", "", regex=False)
    res = {"high": {}, "low": {}}
    for stem in EXPECTED_HIGH:
        sk = _search_key(stem)
        hits = merged[canon_compact.str.contains(sk, na=False, regex=False)]
        for _, r in hits.iterrows():
            am = r["activity_magnitude"]
            am_str = f"{am:.3f}" if pd.notna(am) else "NA"
            print(f"  {stem:10s}  PC{int(r['cluster']) if pd.notna(r['cluster']) else '?':<3}"
                  f"  activity={am_str}  ({r['drug_name']})")
        res["high"][stem] = hits[["drug_name", "cluster", "activity_magnitude"]].to_dict("records")

    print("\n[spot-check] expected LOW (PC2 control-like):")
    for stem in EXPECTED_LOW:
        sk = _search_key(stem)
        hits = merged[canon_compact.str.contains(sk, na=False, regex=False)]
        for _, r in hits.iterrows():
            am = r["activity_magnitude"]
            am_str = f"{am:.3f}" if pd.notna(am) else "NA"
            print(f"  {stem:10s}  PC{int(r['cluster']) if pd.notna(r['cluster']) else '?':<3}"
                  f"  activity={am_str}  ({r['drug_name']})")
        res["low"][stem] = hits[["drug_name", "cluster", "activity_magnitude"]].to_dict("records")

    # quick numeric verdict using percentile rank of each spot-checked drug
    grp = merged.dropna(subset=["activity_magnitude"]).copy()
    grp["pct_rank"] = grp["activity_magnitude"].rank(pct=True)
    print("\n[spot-check] percentile ranks within 1278-drug pipeline distribution:")
    for stem in EXPECTED_HIGH + EXPECTED_LOW:
        sk = _search_key(stem)
        m = grp["canonical_name"].fillna("").str.replace(" ", "", regex=False).str.contains(sk, na=False, regex=False)
        for _, r in grp[m].iterrows():
            print(f"  {stem:10s}  pct={r['pct_rank']:.3f}  activity={r['activity_magnitude']:.3f}"
                  f"   ({r['drug_name']})")
    return res


def main():
    print("=" * 72)
    print("Phase 5 step 2+3: ACTIVITY MAGNITUDE + per-cluster VALIDATION")
    print("=" * 72)
    t6 = load_table6()
    activity, n_dropped, d_min, d_max, dmso_idx = compute_activity_magnitude(t6)
    merged, n_with_activity = join_on_phenocluster(activity)
    cluster_validation = validate_per_cluster(merged)
    spot = spot_check(merged)

    OUT_JSON.write_text(json.dumps({
        "method": {
            "z_score_baseline": "DMSO",
            "z_score_scale": "population SD across 1280 drugs (ddof=1)",
            "distance": "Euclidean over z-scored features",
            "normalization": "min-max to [0,1]",
            "n_features_used": 101 - n_dropped,
            "n_features_dropped_zero_var": n_dropped,
            "raw_distance_range": [round(d_min, 4), round(d_max, 4)],
            "dmso_row_index": dmso_idx,
        },
        "join": {
            "n_pipeline_drugs": int(len(merged)),
            "n_with_activity": int(n_with_activity),
            "match_pct": round(n_with_activity / len(merged) * 100, 2),
        },
        "cluster_validation": cluster_validation,
        "spot_check": spot,
        "outputs": {
            "drug_activity_csv": str(OUT_CSV.relative_to(ROOT)),
            "validation_json": str(OUT_JSON.relative_to(ROOT)),
        },
        "policy": {
            "v3_locked": True,
            "v4_locked": True,
            "v5_score_built": False,
            "phase4_read_only": True,
        },
    }, indent=2, default=str))
    print(f"\n[report] {OUT_JSON.relative_to(ROOT)}")
    print("=" * 72)
    print("DONE")
    print("=" * 72)


if __name__ == "__main__":
    main()
