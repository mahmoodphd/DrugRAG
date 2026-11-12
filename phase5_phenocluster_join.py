"""Phase 5 step 1: PhenoCluster join + validation.

Joins LOPAC1280 PhenoCluster labels (MOESM5 / Suppl Table 2) onto the pipeline
drug master list, using the same alias-normalization (step1.normalize_drug_name)
as the v4 candidate build.

Also reports the structure of MOESM9 / Suppl Table 6 (101 morphology features
per drug) without computing on it -- just so we know whether continuous activity
magnitudes are available for the score-integration step.

Outputs:
  data/phase5/drug_phenocluster_map.csv     (drug, canonical_name, cluster, matched_flag)
  data/phase5/phenocluster_join_report.json (full report incl. unmatched names)

Read-only on v3, locked v4 ranking, and the supplementary spreadsheets.
No score integration. No API calls.
"""
from __future__ import annotations

import glob
import json
import re
import sys
from collections import Counter
from pathlib import Path

import openpyxl
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
# v2 normalizer: standalone, side-effect-free; adds ribose-position prime
# stripping (3', 5', 2', etc.) so nucleoside / nucleotide analogs join.
from drug_normalize import normalize_drug_name_v2 as normalize_drug_name  # type: ignore

PHASE5 = ROOT / "data" / "phase5"
PHASE5.mkdir(parents=True, exist_ok=True)
OUT_CSV = PHASE5 / "drug_phenocluster_map.csv"
REPORT_JSON = PHASE5 / "phenocluster_join_report.json"


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


MOESM5 = _screen_xlsx("41598_2020_70440_MOESM5_ESM.xlsx")
MOESM9 = _screen_xlsx("41598_2020_70440_MOESM9_ESM.xlsx")
PIPELINE_CSV = ROOT / "data" / "phase1" / "drug_master_list.csv"

# Sanity-check expectations from the paper / lab guidance
EXPECTED_PC7 = ["Sunitinib", "SU5416", "Suramin", "DMH4"]
EXPECTED_PC4_OR_5 = ["Memantine"]


def strip_quotes(s: str) -> str:
    """Table 6 wraps names in single quotes ('DMSO'); strip them."""
    if s is None:
        return ""
    return str(s).strip().strip("'\"")


def norm_key(name: str) -> str:
    """Canonical key for the join: strip stereo prefixes, salts, hydrates,
    ribose-position primes, collapse whitespace, lower-case. v2 normalizer.
    """
    return normalize_drug_name(name).lower()


def _search_key(s: str) -> str:
    """Aggressive form for sanity-check substring matching: drops whitespace
    so 'SU5416' matches 'SU 5416' and vice versa."""
    return norm_key(s).replace(" ", "")


def load_phenocluster_table() -> pd.DataFrame:
    print(f"[load] {Path(MOESM5).name} / 'Supplementary Table 2'")
    wb = openpyxl.load_workbook(MOESM5, read_only=True, data_only=True)
    ws = wb["Supplementary Table 2"]
    rows = list(ws.iter_rows(values_only=True))
    wb.close()
    header = rows[0]
    body = [r for r in rows[1:] if r and r[0] is not None]
    df = pd.DataFrame(body, columns=list(header))
    df["Drug"] = df["Drug"].astype(str).str.strip()
    df["Cluster"] = df["Cluster"].astype(int)
    print(f"        rows={len(df)}  columns={list(df.columns)}  "
          f"clusters={sorted(df['Cluster'].unique())}")
    return df


def load_pipeline_drugs() -> pd.DataFrame:
    print(f"[load] {PIPELINE_CSV.relative_to(ROOT)}")
    df = pd.read_csv(PIPELINE_CSV)
    df = df[["compound_clean"]].rename(columns={"compound_clean": "drug_name"})
    df = df.drop_duplicates(subset="drug_name").reset_index(drop=True)
    print(f"        unique pipeline drugs: {len(df)}")
    return df


def join_and_report(pheno: pd.DataFrame, pipeline: pd.DataFrame) -> dict:
    print("\n[join] normalizing both sides with step1.normalize_drug_name")
    pheno["norm"] = pheno["Drug"].map(norm_key)
    pipeline["norm"] = pipeline["drug_name"].map(norm_key)

    # build dict norm -> (drug_in_pheno, cluster). Collisions: keep all, count.
    pheno_dups = pheno["norm"].duplicated(keep=False)
    if pheno_dups.any():
        print(f"  ! {pheno_dups.sum()} pheno-side normalized-name duplicates "
              f"(salts of same compound landing on same key) -- merging by majority cluster")
    pheno_grouped = (pheno.groupby("norm")
                     .agg({"Drug": "first",
                           "Cluster": lambda c: Counter(c).most_common(1)[0][0]})
                     .reset_index()
                     .rename(columns={"Drug": "pheno_drug_name"}))

    merged = pipeline.merge(pheno_grouped, on="norm", how="left")
    merged["matched_flag"] = merged["Cluster"].notna()
    merged["cluster"] = merged["Cluster"].astype("Int64")
    merged["canonical_name"] = merged["norm"]

    n_total = len(merged)
    n_matched = int(merged["matched_flag"].sum())
    pct = n_matched / n_total * 100
    print(f"\n  match rate: {n_matched}/{n_total} = {pct:.1f}%")
    print(f"  unmatched : {n_total - n_matched}")

    unmatched = merged[~merged["matched_flag"]]["drug_name"].tolist()
    if unmatched:
        print(f"\n  unmatched names (first 30):")
        for name in unmatched[:30]:
            print(f"    - {name}")
        if len(unmatched) > 30:
            print(f"    ... and {len(unmatched) - 30} more")

    # per-cluster counts after join
    print("\n  per-cluster counts (matched pipeline drugs):")
    counts = merged.dropna(subset=["cluster"])["cluster"].value_counts().sort_index()
    for c, n in counts.items():
        print(f"    cluster {int(c)} : {n} drugs")

    # also: pheno-side per-cluster totals (denominators)
    pheno_totals = pheno_grouped["Cluster"].value_counts().sort_index()
    print("\n  pheno-side per-cluster totals (after norm-collapse):")
    for c, n in pheno_totals.items():
        print(f"    cluster {int(c)} : {n} drugs")

    # save the public artifact
    out = merged[["drug_name", "canonical_name", "cluster", "matched_flag"]].copy()
    out.to_csv(OUT_CSV, index=False)
    print(f"\n  wrote {OUT_CSV.relative_to(ROOT)}  ({len(out)} rows)")

    return {
        "n_total_pipeline": n_total,
        "n_matched": n_matched,
        "match_pct": round(pct, 2),
        "n_unmatched": n_total - n_matched,
        "unmatched_names": unmatched,
        "per_cluster_counts_matched": {int(c): int(n) for c, n in counts.items()},
        "pheno_side_totals": {int(c): int(n) for c, n in pheno_totals.items()},
        "merged_df": merged,
    }


def sanity_check(merged: pd.DataFrame) -> dict:
    print("\n[sanity] expected PC7 anti-angiogenics:")
    pc7_results = {}
    canon_compact = merged["canonical_name"].fillna("").str.replace(" ", "", regex=False)
    for stem in EXPECTED_PC7:
        stem_norm = _search_key(stem)
        hits = merged[canon_compact.str.contains(re.escape(stem_norm), na=False, regex=True)]
        if len(hits) == 0:
            print(f"  ! {stem:12s}  NOT IN PIPELINE LIST")
            pc7_results[stem] = {"found": False}
            continue
        clusters = hits.dropna(subset=["cluster"])["cluster"].astype(int).tolist()
        rows = hits[["drug_name", "cluster"]].to_dict("records")
        ok = any(c == 7 for c in clusters)
        flag = "✓" if ok else "✗"
        print(f"  {flag} {stem:12s}  {len(hits)} pipeline rows  clusters={clusters}")
        for r in rows:
            print(f"     - {r['drug_name']!r:40s}  cluster={r['cluster']}")
        pc7_results[stem] = {"found": True, "in_pc7": ok, "rows": rows}

    print("\n[sanity] expected PC4 or PC5 (glutamate drugs):")
    pc45_results = {}
    for stem in EXPECTED_PC4_OR_5:
        stem_norm = _search_key(stem)
        hits = merged[canon_compact.str.contains(re.escape(stem_norm), na=False, regex=True)]
        if len(hits) == 0:
            print(f"  ! {stem:12s}  NOT IN PIPELINE LIST")
            pc45_results[stem] = {"found": False}
            continue
        clusters = hits.dropna(subset=["cluster"])["cluster"].astype(int).tolist()
        rows = hits[["drug_name", "cluster"]].to_dict("records")
        ok = any(c in (4, 5) for c in clusters)
        flag = "✓" if ok else "✗"
        print(f"  {flag} {stem:12s}  {len(hits)} pipeline rows  clusters={clusters}")
        for r in rows:
            print(f"     - {r['drug_name']!r:40s}  cluster={r['cluster']}")
        pc45_results[stem] = {"found": True, "in_pc4_or_5": ok, "rows": rows}

    all_pc7_ok = all(v.get("in_pc7", False) for v in pc7_results.values() if v.get("found"))
    all_pc45_ok = all(v.get("in_pc4_or_5", False) for v in pc45_results.values() if v.get("found"))
    verdict = "JOIN VALIDATED" if (all_pc7_ok and all_pc45_ok) else "JOIN HAS ISSUES -- INSPECT"
    print(f"\n[sanity] verdict: {verdict}")
    return {
        "pc7_drugs": pc7_results,
        "pc4_or_5_drugs": pc45_results,
        "all_pc7_ok": all_pc7_ok,
        "all_pc45_ok": all_pc45_ok,
        "verdict": verdict,
    }


def report_table6_structure(pheno: pd.DataFrame, pipeline: pd.DataFrame) -> dict:
    print(f"\n[probe] {Path(MOESM9).name} / 'Supplementary Table 6'")
    wb = openpyxl.load_workbook(MOESM9, read_only=True, data_only=True)
    ws = wb["Supplementary Table 6"]

    rows_iter = ws.iter_rows(values_only=True)
    header = list(next(rows_iter))
    drugs_t6 = []
    for r in rows_iter:
        if r and r[0] is not None:
            drugs_t6.append(strip_quotes(r[0]))
    wb.close()

    feature_cols = [strip_quotes(c) for c in header[1:]]
    print(f"  drugs (rows) : {len(drugs_t6)}")
    print(f"  features     : {len(feature_cols)} continuous columns")
    print(f"  first column : {strip_quotes(header[0])!r}  (drug name, single-quote-wrapped in raw cells)")
    print(f"  feature cols (first 12): {feature_cols[:12]}")
    print(f"  feature cols (last  4): {feature_cols[-4:]}")

    # join compatibility check: how many of the Table 6 drug names map onto the
    # pipeline list using the SAME normalize_drug_name key?
    t6_norm = [norm_key(d) for d in drugs_t6]
    pipe_norm = set(pipeline["norm"])
    pheno_norm = set(pheno["norm"])
    n_match_pipeline = sum(1 for k in t6_norm if k in pipe_norm)
    n_match_pheno = sum(1 for k in t6_norm if k in pheno_norm)
    print(f"\n  join check (canonical-name key, same normalizer):")
    print(f"    Table 6 → pipeline drug list : {n_match_pipeline}/{len(drugs_t6)}")
    print(f"    Table 6 → MOESM5 drug list   : {n_match_pheno}/{len(drugs_t6)}")
    overlap_pct = n_match_pipeline / len(drugs_t6) * 100 if drugs_t6 else 0
    print(f"  → continuous activity is {'AVAILABLE' if overlap_pct > 90 else 'PARTIALLY available'} "
          f"for pipeline drugs ({overlap_pct:.1f}% of Table 6 rows joinable)")

    return {
        "n_drugs": len(drugs_t6),
        "n_features": len(feature_cols),
        "drug_column_name": strip_quotes(header[0]),
        "drug_names_quoted_in_source": True,
        "feature_columns": feature_cols,
        "join_compatibility": {
            "table6_to_pipeline_master": f"{n_match_pipeline}/{len(drugs_t6)} ({overlap_pct:.1f}%)",
            "table6_to_moesm5_pheno":   f"{n_match_pheno}/{len(drugs_t6)} "
                                        f"({n_match_pheno/len(drugs_t6)*100:.1f}%)",
            "uses_same_normalizer": True,
        },
        "note": "DMSO control row is included (n=1) -- excluded if computing only on perturbations.",
    }


def main():
    print("=" * 70)
    print("PhenoCluster JOIN + VALIDATION  (Phase 5 step 1, no API)")
    print("=" * 70)
    pheno = load_phenocluster_table()
    pipeline = load_pipeline_drugs()
    join = join_and_report(pheno, pipeline)
    merged = join.pop("merged_df")
    sanity = sanity_check(merged)
    table6 = report_table6_structure(pheno, pipeline)

    report = {
        "join": join,
        "sanity_check": sanity,
        "table6_structure": table6,
        "outputs": {
            "drug_phenocluster_map_csv": str(OUT_CSV.relative_to(ROOT)),
            "report_json": str(REPORT_JSON.relative_to(ROOT)),
        },
        "policy": {
            "v3_artifacts_modified": False,
            "v4_locked_ranking_modified": False,
            "score_integration": "DEFERRED -- separate step",
        },
    }
    REPORT_JSON.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n[report] {REPORT_JSON.relative_to(ROOT)}")
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
