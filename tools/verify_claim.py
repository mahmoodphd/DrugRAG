#!/usr/bin/env python3
"""
verify_claim.py — check a numeric/string claim against on-disk data.

The "don't trust the paste, check the disk" tool. Computes a value from a CSV
(optionally filtered/aggregated) and compares it to what was claimed. Used to
verify any number before trusting it.

Usage:
    # how many TRUE-DARK drugs? (expect 13)
    python tools/verify_claim.py --csv data/phase5/tables/pubmed_verification_v5.csv \
        --filter "verification == 'TRUE-DARK'" --expect 13
    # sum of a column over a filtered slice
    python tools/verify_claim.py --csv data/phase5/drug_repurposing_candidates_v5_final.csv \
        --filter "cluster == 7" --col activity_magnitude --agg mean
    # just show the value (no --expect)
    python tools/verify_claim.py --csv X.csv --filter "is_active_dark_real == True"
"""
import argparse
import sys

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--filter", help="pandas .query() expression")
    ap.add_argument("--col", help="column to aggregate (default: row count)")
    ap.add_argument("--agg", default="count",
                    choices=["count", "sum", "mean", "min", "max", "nunique"])
    ap.add_argument("--expect")
    ap.add_argument("--tol", type=float, default=1e-6)
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    if a.filter:
        df = df.query(a.filter)

    if a.col and a.agg != "count":
        val = getattr(df[a.col], a.agg)()
    elif a.col:
        val = int(df[a.col].count())
    else:
        val = int(len(df))

    if isinstance(val, float):
        val = round(val, 6)
    print(f"value = {val}   (rows after filter: {len(df)})")

    if a.expect is None:
        return
    try:
        ok = abs(float(val) - float(a.expect)) <= a.tol
    except (ValueError, TypeError):
        ok = str(val) == str(a.expect)
    print(f"expected = {a.expect}  ->  {'PASS ✓' if ok else 'FAIL ✗'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
