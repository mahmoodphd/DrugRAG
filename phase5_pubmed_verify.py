"""Phase 5 step 5: real-PubMed corpus-novelty verification (sidecar).

The v5 `novelty` term is corpus-relative: it counts hits in our 8 514-abstract
angiogenesis-anchored corpus. A drug like Suramin can score `novelty = 0.900`
on the strength of having 1 abstract in our corpus while real PubMed lists it
as a textbook anti-angiogenic with hundreds of hits. This sidecar settles
which v5 dark / top-list candidates are *genuinely* understudied vs simply
missed by our corpus build.

For each drug in the verification set:
    angio_query = "{drug}" AND (angiogenesis OR endothelial OR VEGF
                                OR "blood vessel" OR neovascularization
                                OR "tube formation")
    total_query = "{drug}"
Run each through Entrez.esearch (PubMed) with retmax=0 (count only). Try the
salt-stripped canonical form first; if angio=0 also try the raw name and take
the larger. Respect NCBI rate limits via config (0.1 s with API key, 0.4 s
without).

Verification set:
    (a) all 16 dark drugs (corpus n_papers == 0)
    (b) top 25 of the v5 anti-angiogenic list
    (c) top 25 of the v5 pro-angiogenic list
    (deduped, tagged by membership)

Classification:
    TRUE-DARK     = corpus n_papers == 0  AND  real_angio_hits <= 2
    CORPUS-MISSED = corpus n_papers <= 5  AND  real_angio_hits >= 10
    PARTIAL       = everything else

OUTPUTS (sidecar; v5 CSV NOT modified):
    data/phase5/tables/pubmed_verification_v5.csv
    data/phase5/tables/pubmed_verification_v5.md

Read-only on v3, locked v4, v5 ranking. PubMed only -- no OpenRouter / no LLM.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
from Bio import Entrez

ROOT = Path("/home/mshz88/Samiha")
sys.path.insert(0, str(ROOT))
import config  # email, NCBI_API_KEY, PUBMED_RATE_LIMIT_SEC
from drug_normalize import normalize_drug_name_v2  # type: ignore

PHASE5 = ROOT / "data" / "phase5"
TABLES = PHASE5 / "tables"
V5_CSV = PHASE5 / "drug_repurposing_candidates_v5.csv"
OUT_CSV = TABLES / "pubmed_verification_v5.csv"
OUT_MD = TABLES / "pubmed_verification_v5.md"

# Verification thresholds (per user spec)
TRUE_DARK_REAL_ANGIO_MAX = 2          # corpus n_papers==0 and real_angio<=2
CORPUS_MISSED_REAL_ANGIO_MIN = 10     # corpus low and real_angio>=10
CORPUS_MISSED_CORPUS_MAX = 5          # "corpus n_papers low"

ANGIO_TERMS = ('(angiogenesis OR endothelial OR VEGF OR "blood vessel"'
               ' OR neovascularization OR "tube formation")')

TOP_N = 25  # top of each direction list


# ---------------------------------------------------------------------------
def setup_entrez():
    Entrez.email = config.PUBMED_EMAIL
    if config.NCBI_API_KEY:
        Entrez.api_key = config.NCBI_API_KEY
    print(f"[entrez] email={config.PUBMED_EMAIL!r}  "
          f"key={'set' if config.NCBI_API_KEY else 'NONE'}  "
          f"rate_limit={config.PUBMED_RATE_LIMIT_SEC}s")


def pubmed_count(query: str, retries: int = 3, backoff: float = 2.0) -> int:
    """Return the total PubMed hit count for `query`. Uses retmax=0 (count only).

    Returns -1 on persistent error so the row is preserved.
    """
    last_err = None
    for attempt in range(retries):
        try:
            handle = Entrez.esearch(db="pubmed", term=query, retmax=0)
            rec = Entrez.read(handle)
            handle.close()
            return int(rec.get("Count", 0))
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(backoff * (attempt + 1))
    print(f"  ! pubmed_count failed: {query[:60]!r}  err={last_err!s}")
    return -1


def query_drug(raw_name: str, canonical: str) -> dict:
    """Run angio + total queries for a drug. Try canonical first; if angio
    returns 0, also try the raw name and take the larger count for both.
    """
    # canonical = salt-stripped lowercased core (already in v5 CSV)
    # raw_name  = original drug_name (may carry salt / hydrate / stereo)
    raw_q_drug = f'"{raw_name}"'
    can_q_drug = f'"{canonical}"'

    angio_can = pubmed_count(f"{can_q_drug} AND {ANGIO_TERMS}")
    time.sleep(config.PUBMED_RATE_LIMIT_SEC)
    total_can = pubmed_count(can_q_drug)
    time.sleep(config.PUBMED_RATE_LIMIT_SEC)

    used_form = "canonical"
    angio = angio_can
    total = total_can

    # If angio returns exactly 0 with canonical, try raw form as the user
    # specified ("retry the salt-stripped form if 0 hits"). The "salt-stripped"
    # form *is* canonical here, so the meaningful retry is the OTHER form
    # (raw, with the salt suffix included). We take the larger count.
    if angio_can == 0 and raw_name.strip().lower() != canonical.strip().lower():
        angio_raw = pubmed_count(f"{raw_q_drug} AND {ANGIO_TERMS}")
        time.sleep(config.PUBMED_RATE_LIMIT_SEC)
        total_raw = pubmed_count(raw_q_drug)
        time.sleep(config.PUBMED_RATE_LIMIT_SEC)
        if angio_raw > angio_can or total_raw > total_can:
            used_form = "raw_after_retry"
            angio = max(angio_can, angio_raw)
            total = max(total_can, total_raw)
    return {
        "real_angio_pubmed_hits": angio,
        "real_total_pubmed_hits": total,
        "pubmed_query_form": used_form,
    }


def classify(corpus_n_papers: int, real_angio: int) -> str:
    if real_angio < 0:
        return "QUERY-ERROR"
    if corpus_n_papers == 0 and real_angio <= TRUE_DARK_REAL_ANGIO_MAX:
        return "TRUE-DARK"
    if (corpus_n_papers <= CORPUS_MISSED_CORPUS_MAX and
            real_angio >= CORPUS_MISSED_REAL_ANGIO_MIN):
        return "CORPUS-MISSED"
    return "PARTIAL"


# ---------------------------------------------------------------------------
def build_verification_set(v5: pd.DataFrame) -> pd.DataFrame:
    print("\n[set] building verification set:")
    dark = v5[v5["n_papers"] == 0].copy()
    print(f"  (a) dark drugs (n_papers==0)            : {len(dark)}")

    anti = v5[v5["net_direction"] < 0].sort_values(
        "repurposing_score_v5", ascending=False).head(TOP_N).copy()
    pro = v5[v5["net_direction"] > 0].sort_values(
        "repurposing_score_v5", ascending=False).head(TOP_N).copy()
    print(f"  (b) top-{TOP_N} anti-angiogenic              : {len(anti)}")
    print(f"  (c) top-{TOP_N} pro-angiogenic               : {len(pro)}")

    dark["set_dark"] = True
    anti["set_top_anti"] = True
    pro["set_top_pro"] = True

    combined = pd.concat([
        dark.assign(set_top_anti=False, set_top_pro=False),
        anti.assign(set_dark=False, set_top_pro=False),
        pro.assign(set_dark=False, set_top_anti=False),
    ], ignore_index=True)
    # collapse duplicates: OR the membership flags, keep first row data
    agg = (combined.groupby("drug_name", as_index=False)
           .agg({
               "canonical_name": "first",
               "cluster": "first",
               "net_direction": "first",
               "activity_magnitude": "first",
               "n_papers": "first",
               "angio_edges": "first",
               "novelty": "first",
               "kg_plausibility": "first",
               "repurposing_score_v5": "first",
               "rank": "first",
               "is_active_dark": "first",
               "set_dark": "max",
               "set_top_anti": "max",
               "set_top_pro": "max",
           }))
    print(f"  -- deduped union                            : {len(agg)} drugs to query")
    return agg


def run_pubmed_verification(verif: pd.DataFrame) -> pd.DataFrame:
    print(f"\n[query] running PubMed lookups (~{2 * len(verif)} core queries; "
          f"+ retries on angio==0)")
    print(f"        rate limit per query: {config.PUBMED_RATE_LIMIT_SEC}s")
    rows = []
    t0 = time.time()
    for i, r in verif.iterrows():
        result = query_drug(r["drug_name"], r["canonical_name"])
        cls = classify(int(r["n_papers"]), result["real_angio_pubmed_hits"])
        marker = "*" if (r["n_papers"] == 0 and cls == "CORPUS-MISSED") else \
                 "!" if cls == "QUERY-ERROR" else " "
        print(f"  [{i + 1:>3}/{len(verif)}] {marker} "
              f"{r['drug_name']:46s}  "
              f"angio={result['real_angio_pubmed_hits']:>6}  "
              f"total={result['real_total_pubmed_hits']:>7}  "
              f"corpus_n={int(r['n_papers']):>3}  "
              f"-> {cls}")
        rows.append({**r.to_dict(), **result, "verification": cls})
    elapsed = time.time() - t0
    print(f"\n  done in {elapsed:.1f}s  "
          f"(avg {elapsed / max(1, len(verif)):.2f}s/drug)")
    return pd.DataFrame(rows)


def write_outputs(df: pd.DataFrame) -> dict:
    cols = [
        "drug_name", "canonical_name", "cluster",
        "set_dark", "set_top_anti", "set_top_pro",
        "n_papers", "angio_edges", "novelty", "kg_plausibility",
        "activity_magnitude", "net_direction",
        "real_angio_pubmed_hits", "real_total_pubmed_hits",
        "pubmed_query_form", "verification",
        "repurposing_score_v5", "rank", "is_active_dark",
    ]
    df_out = df.reindex(columns=cols)
    df_out = df_out.sort_values(
        ["verification", "real_angio_pubmed_hits", "rank"],
        ascending=[True, True, True]).reset_index(drop=True)
    df_out.to_csv(OUT_CSV, index=False)
    print(f"\n[write] {OUT_CSV.relative_to(ROOT)}  ({len(df_out)} rows)")

    # markdown summary
    n_total = len(df_out)
    by_cls = df_out["verification"].value_counts().to_dict()
    true_dark = df_out[df_out["verification"] == "TRUE-DARK"].copy()
    corpus_missed = df_out[df_out["verification"] == "CORPUS-MISSED"].copy()
    partial = df_out[df_out["verification"] == "PARTIAL"].copy()
    err = df_out[df_out["verification"] == "QUERY-ERROR"].copy()

    def _fmt(sub: pd.DataFrame, header_cols: list) -> str:
        if len(sub) == 0:
            return "_(none)_\n"
        h = "| " + " | ".join(header_cols) + " |\n"
        h += "|" + "|".join(["---"] * len(header_cols)) + "|\n"
        for _, r in sub.iterrows():
            row = []
            for c in header_cols:
                v = r[c]
                if isinstance(v, float):
                    row.append(f"{v:.4f}" if not pd.isna(v) else "")
                elif pd.isna(v):
                    row.append("")
                else:
                    row.append(str(v))
            h += "| " + " | ".join(row) + " |\n"
        return h

    show = ["drug_name", "cluster", "set_dark", "set_top_anti", "set_top_pro",
            "n_papers", "real_angio_pubmed_hits", "real_total_pubmed_hits",
            "novelty", "activity_magnitude", "rank"]

    md = []
    md.append("# Real-PubMed corpus-novelty verification (v5 sidecar)\n")
    md.append(
        "Our v5 `novelty` is **corpus-relative** — it counts hits in our "
        "8 514-abstract angiogenesis-anchored corpus. This sidecar checks "
        "every dark / top-list candidate against real, uncapped PubMed "
        "(Entrez `esearch`, count-only) to separate genuinely understudied "
        "drugs from drugs our corpus simply missed.\n")
    md.append("**Queries (per drug):**")
    md.append("- `angio  = \"{drug}\" AND (angiogenesis OR endothelial OR "
              "VEGF OR \"blood vessel\" OR neovascularization OR "
              "\"tube formation\")`")
    md.append("- `total  = \"{drug}\"`\n")
    md.append("**Classification:**")
    md.append(f"- `TRUE-DARK`     = `n_papers == 0` AND "
              f"`real_angio_hits <= {TRUE_DARK_REAL_ANGIO_MAX}`")
    md.append(f"- `CORPUS-MISSED` = `n_papers <= {CORPUS_MISSED_CORPUS_MAX}` "
              f"AND `real_angio_hits >= {CORPUS_MISSED_REAL_ANGIO_MIN}`")
    md.append(f"- `PARTIAL`       = anything else\n")
    md.append(f"**Counts:** total queried = **{n_total}**, "
              f"TRUE-DARK = **{by_cls.get('TRUE-DARK', 0)}**, "
              f"CORPUS-MISSED = **{by_cls.get('CORPUS-MISSED', 0)}**, "
              f"PARTIAL = **{by_cls.get('PARTIAL', 0)}**, "
              f"QUERY-ERROR = **{by_cls.get('QUERY-ERROR', 0)}**.\n")

    md.append("## TRUE-DARK (confirmed real discoveries)\n")
    md.append(_fmt(true_dark[show], show))

    md.append("## CORPUS-MISSED (corpus-novel but real-PubMed evidence exists)\n")
    md.append(_fmt(corpus_missed[show], show))

    md.append("## PARTIAL\n")
    md.append(_fmt(partial[show], show))

    if len(err):
        md.append("## QUERY-ERROR\n")
        md.append(_fmt(err[show], show))

    OUT_MD.write_text("\n".join(md))
    print(f"[write] {OUT_MD.relative_to(ROOT)}")

    return {
        "n_total": n_total,
        "n_true_dark": int(by_cls.get("TRUE-DARK", 0)),
        "n_corpus_missed": int(by_cls.get("CORPUS-MISSED", 0)),
        "n_partial": int(by_cls.get("PARTIAL", 0)),
        "n_error": int(by_cls.get("QUERY-ERROR", 0)),
        "true_dark_drugs": true_dark["drug_name"].tolist(),
        "corpus_missed_drugs": corpus_missed["drug_name"].tolist(),
        "false_novelty_flags": [
            {"drug_name": r["drug_name"],
             "corpus_n_papers": int(r["n_papers"]),
             "real_angio_pubmed_hits": int(r["real_angio_pubmed_hits"]),
             "novelty": float(r["novelty"]),
             "rank": int(r["rank"]),
             "in_top_anti": bool(r["set_top_anti"]),
             "in_top_pro": bool(r["set_top_pro"])}
            for _, r in corpus_missed.iterrows()
        ],
    }


def main():
    setup_entrez()
    print(f"[load] {V5_CSV.relative_to(ROOT)}")
    v5 = pd.read_csv(V5_CSV)
    verif = build_verification_set(v5)
    queried = run_pubmed_verification(verif)
    summary = write_outputs(queried)

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"  total queried   : {summary['n_total']}")
    print(f"  TRUE-DARK       : {summary['n_true_dark']}  "
          f"(confirmed novel)")
    print(f"  CORPUS-MISSED   : {summary['n_corpus_missed']}  "
          f"(novelty contradicted by real PubMed)")
    print(f"  PARTIAL         : {summary['n_partial']}")
    if summary["n_error"]:
        print(f"  QUERY-ERROR     : {summary['n_error']}")
    print()
    print(f"  TRUE-DARK drugs:")
    for d in summary["true_dark_drugs"]:
        print(f"    - {d}")
    print(f"\n  CORPUS-MISSED drugs (false-novelty flags):")
    for d in summary["corpus_missed_drugs"]:
        print(f"    - {d}")

    # JSON sidecar (machine-readable summary alongside CSV/MD)
    (TABLES / "pubmed_verification_v5_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))
    print(f"\n[write] {(TABLES / 'pubmed_verification_v5_summary.json').relative_to(ROOT)}")
    print("\nDONE")


if __name__ == "__main__":
    main()
