"""Phase 5 step 6: real-PubMed novelty for the full library + v5_final ranking.

Replaces v5's corpus-relative `novelty` term with `novelty_real`, computed from
real, uncapped PubMed angiogenesis hit-counts for *every* drug in the v5 file
(~1 278 drugs). Same `max(0.1, 1 - log10(x + 1) / 3)` log shape, but with
`real_angio_pubmed_hits` instead of corpus `n_papers`, so well-studied drugs
(Suramin, Y-27632, Sunitinib, ...) are demoted to `novelty ≈ 0.1` automatically
and the corpus-coverage gap stops driving rank.

Pipeline:
    1. Load v5 file + previously-queried 65-drug sidecar (reuse those hits).
    2. PubMed-query the remaining ~1 213 drugs (angio + total, salt-strip retry).
    3. Compute `novelty_real = max(0.1, 1 - log10(real_angio + 1) / 3)`.
    4. Compute `score_v5_final = 0.40·activity + 0.35·novelty_real + 0.25·kg`.
    5. Classify `novelty_status` ∈ {TRUE-DARK, well-studied, CORPUS-MISSED, PARTIAL}.
    6. Re-run validation gate. Emit two-tier deliverables.

OUTPUTS (all in `data/phase5/`, never touches v3 / v4 / original v5):
    drug_repurposing_candidates_v5_final.csv
    tables/v5_final_top20_anti_angiogenic.{csv,md}
    tables/v5_final_top20_pro_angiogenic.{csv,md}
    tables/v5_final_validation_anti.{csv,md}      # known anti-angiogenics
    tables/v5_final_discovery_anti.{csv,md}       # genuinely understudied
    tables/v5_final_discovery_pro.{csv,md}
    tables/v5_final_validation_gate.{json,md}
    tables/v5_final_full_pubmed_counts.csv         # raw counts cache

Local + PubMed only. No OpenRouter, no LLM.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import Entrez

ROOT = Path("/home/mshz88/Samiha")
sys.path.insert(0, str(ROOT))
import config  # noqa: E402

PHASE5 = ROOT / "data" / "phase5"
TABLES = PHASE5 / "tables"
V5_CSV = PHASE5 / "drug_repurposing_candidates_v5.csv"
SIDECAR_CSV = TABLES / "pubmed_verification_v5.csv"
PUBMED_CACHE = TABLES / "v5_final_full_pubmed_counts.csv"
V5_FINAL_CSV = PHASE5 / "drug_repurposing_candidates_v5_final.csv"

# Score weights (carried verbatim from v5)
W_ACTIVITY = 0.40
W_NOVELTY = 0.35
W_KG = 0.25

# Classification thresholds
WELL_STUDIED_MIN = 100   # real_angio >= 100
TRUE_DARK_MAX = 2        # real_angio <= 2 AND corpus n_papers == 0
CORPUS_MISSED_LOW = 10   # 10 <= real_angio < WELL_STUDIED_MIN
CORPUS_MISSED_CORPUS_MAX = 5   # corpus n_papers <= 5

ANGIO_TERMS = ('(angiogenesis OR endothelial OR VEGF OR "blood vessel"'
               ' OR neovascularization OR "tube formation")')

# Known anti-angiogenics for the validation table (drug-substring match,
# case-insensitive). Used only to *display*; doesn't influence the score.
KNOWN_ANTI_ANGIOGENICS = [
    "Sunitinib", "Suramin", "Colchicine", "Vinblastine", "Vincristine",
    "2-methoxyestradiol", "Resveratrol", "Tamoxifen", "Topotecan",
    "Sorafenib", "Bevacizumab", "Idarubicin", "Doxorubicin", "Paclitaxel",
    "Indomethacin", "Forskolin", "Daunorubicin", "Mitomycin",
]

PC7_CONTROLS = ["Sunitinib", "SU5416", "SU 5416", "Suramin", "DMH4"]


# ---------------------------------------------------------------------------
def setup_entrez():
    Entrez.email = config.PUBMED_EMAIL
    if config.NCBI_API_KEY:
        Entrez.api_key = config.NCBI_API_KEY
    has_key = bool(config.NCBI_API_KEY)
    rate = config.PUBMED_RATE_LIMIT_SEC
    print(f"[entrez] email={config.PUBMED_EMAIL!r}  "
          f"key={'set' if has_key else 'NONE'}  rate_limit={rate}s")
    return rate


def pubmed_count(query: str, retries: int = 3, backoff: float = 2.5) -> int:
    """Return PubMed hit count for `query` (retmax=0). -1 on persistent error."""
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
    print(f"\n  ! pubmed_count failed: {query[:70]!r}  err={last_err!s}")
    return -1


def query_drug(raw_name: str, canonical: str, rate: float) -> dict:
    """Two queries (angio + total). Try canonical; if angio==0 retry raw."""
    raw_q = f'"{raw_name}"'
    can_q = f'"{canonical}"'

    angio_can = pubmed_count(f"{can_q} AND {ANGIO_TERMS}")
    time.sleep(rate)
    total_can = pubmed_count(can_q)
    time.sleep(rate)

    used_form = "canonical"
    angio = angio_can
    total = total_can

    if angio_can == 0 and raw_name.strip().lower() != canonical.strip().lower():
        angio_raw = pubmed_count(f"{raw_q} AND {ANGIO_TERMS}")
        time.sleep(rate)
        total_raw = pubmed_count(raw_q)
        time.sleep(rate)
        if angio_raw > angio_can or total_raw > total_can:
            used_form = "raw_after_retry"
            angio = max(angio_can, angio_raw)
            total = max(total_can, total_raw)
    return {
        "real_angio_pubmed_hits": angio,
        "real_total_pubmed_hits": total,
        "pubmed_query_form": used_form,
    }


# ---------------------------------------------------------------------------
def load_v5_and_sidecar() -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"[load] {V5_CSV.relative_to(ROOT)}")
    v5 = pd.read_csv(V5_CSV)
    print(f"  v5 rows: {len(v5)}")
    if SIDECAR_CSV.exists():
        side = pd.read_csv(SIDECAR_CSV)
        print(f"[load] {SIDECAR_CSV.relative_to(ROOT)}  (reuse {len(side)} hits)")
    else:
        side = pd.DataFrame(columns=["drug_name", "real_angio_pubmed_hits",
                                     "real_total_pubmed_hits",
                                     "pubmed_query_form"])
    return v5, side


def get_or_query_all(v5: pd.DataFrame, side: pd.DataFrame, rate: float
                     ) -> pd.DataFrame:
    """Resume-aware merge: re-use existing hits, query the remainder. Persists
    the cache after every 25 queries so a kill is recoverable."""

    # If a previous full sweep was interrupted, resume from its cache:
    cache_seed = side.set_index("drug_name")[
        ["real_angio_pubmed_hits", "real_total_pubmed_hits",
         "pubmed_query_form"]]
    if PUBMED_CACHE.exists():
        prior = pd.read_csv(PUBMED_CACHE)
        prior_idx = prior.set_index("drug_name")[
            ["real_angio_pubmed_hits", "real_total_pubmed_hits",
             "pubmed_query_form"]]
        # Prefer cache (more recent / fuller) over sidecar
        cache_seed = pd.concat([cache_seed[~cache_seed.index.isin(prior_idx.index)],
                                prior_idx])
        print(f"[resume] {PUBMED_CACHE.relative_to(ROOT)} "
              f"({len(prior_idx)} cached drugs)")

    cache = cache_seed.to_dict("index")
    todo = [r for r in v5.itertuples(index=False)
            if r.drug_name not in cache]
    print(f"[plan] reuse: {len(cache)}  todo: {len(todo)}  "
          f"(est ~{len(todo) * 2 * rate / 60:.1f} min @ {rate}s/req baseline)")

    t0 = time.time()
    for i, r in enumerate(todo, start=1):
        result = query_drug(r.drug_name, r.canonical_name, rate)
        cache[r.drug_name] = result
        # Progress line: print every 25 OR if angio==0 (interesting hits)
        if i <= 5 or i % 25 == 0 or i == len(todo) or result["real_angio_pubmed_hits"] == 0:
            print(f"  [{i:>4}/{len(todo)}]  "
                  f"{r.drug_name[:46]:46s}  "
                  f"angio={result['real_angio_pubmed_hits']:>6}  "
                  f"total={result['real_total_pubmed_hits']:>7}")
        # Persist every 25 queries for kill-resume safety
        if i % 25 == 0 or i == len(todo):
            _persist_cache(cache)
    elapsed = time.time() - t0
    print(f"\n[done] queried {len(todo)} drugs in {elapsed:.1f}s "
          f"({elapsed / max(1, len(todo)):.2f}s/drug avg, includes retries)")
    _persist_cache(cache)

    cache_df = (pd.DataFrame.from_dict(cache, orient="index")
                .reset_index().rename(columns={"index": "drug_name"}))
    return cache_df


def _persist_cache(cache: dict):
    df = (pd.DataFrame.from_dict(cache, orient="index")
          .reset_index().rename(columns={"index": "drug_name"}))
    df.to_csv(PUBMED_CACHE, index=False)


# ---------------------------------------------------------------------------
def compute_novelty_real(real_angio: float) -> float:
    """Same log shape as v4/v5 novelty, but on uncapped real PubMed counts."""
    if real_angio is None or (isinstance(real_angio, float) and math.isnan(real_angio)):
        return float("nan")
    real_angio = max(0, int(real_angio))
    return max(0.1, 1.0 - math.log10(real_angio + 1) / 3.0)


def classify_novelty(corpus_n_papers: int, real_angio: int) -> str:
    if real_angio < 0:
        return "QUERY-ERROR"
    if real_angio >= WELL_STUDIED_MIN:
        return "well-studied"
    if corpus_n_papers == 0 and real_angio <= TRUE_DARK_MAX:
        return "TRUE-DARK"
    if (corpus_n_papers <= CORPUS_MISSED_CORPUS_MAX and
            real_angio >= CORPUS_MISSED_LOW):
        return "CORPUS-MISSED"
    return "PARTIAL"


def build_v5_final(v5: pd.DataFrame, hits: pd.DataFrame) -> pd.DataFrame:
    df = v5.merge(hits, on="drug_name", how="left")
    n_missing = df["real_angio_pubmed_hits"].isna().sum()
    if n_missing:
        print(f"  WARN: {n_missing} drugs missing pubmed counts (treating as -1)")
        df["real_angio_pubmed_hits"] = df["real_angio_pubmed_hits"].fillna(-1).astype(int)
        df["real_total_pubmed_hits"] = df["real_total_pubmed_hits"].fillna(-1).astype(int)
    else:
        df["real_angio_pubmed_hits"] = df["real_angio_pubmed_hits"].astype(int)
        df["real_total_pubmed_hits"] = df["real_total_pubmed_hits"].astype(int)

    # Rename auxiliary corpus column for clarity
    df = df.rename(columns={"n_papers": "corpus_n_papers"})

    df["novelty_real"] = df["real_angio_pubmed_hits"].apply(compute_novelty_real)
    df["novelty_status"] = df.apply(
        lambda r: classify_novelty(int(r["corpus_n_papers"]),
                                   int(r["real_angio_pubmed_hits"])), axis=1)

    df["score_v5_final"] = (W_ACTIVITY * df["activity_magnitude"]
                            + W_NOVELTY * df["novelty_real"]
                            + W_KG * df["kg_plausibility"]).round(4)

    df = df.sort_values(
        ["score_v5_final", "activity_magnitude", "novelty_real", "drug_name"],
        ascending=[False, False, False, True]).reset_index(drop=True)
    df["rank_final"] = range(1, len(df) + 1)

    # Active-dark on real PubMed: real_angio <= 2 AND activity at 75th-percentile
    act_p75 = df["activity_magnitude"].quantile(0.75)
    df["is_active_dark_real"] = ((df["real_angio_pubmed_hits"] <= 2)
                                  & (df["activity_magnitude"] >= act_p75))

    out_cols = [
        "rank_final", "drug_name", "canonical_name", "cluster", "net_direction",
        "activity_magnitude", "real_angio_pubmed_hits", "real_total_pubmed_hits",
        "novelty_real", "novelty_status", "kg_plausibility", "score_v5_final",
        "corpus_n_papers", "angio_edges",
        "novelty",  # v5's corpus-relative novelty (kept for audit / comparison)
        "repurposing_score_v5", "rank",  # original v5 score + rank (audit trail)
        "is_active_dark", "is_active_dark_real",
        "pubmed_query_form",
    ]
    return df[out_cols]


# ---------------------------------------------------------------------------
def emit_table(df: pd.DataFrame, path_csv: Path, path_md: Path,
               title: str, intro: str, cols: list[str]):
    df_out = df.reindex(columns=cols)
    df_out.to_csv(path_csv, index=False)
    md = [f"# {title}\n", intro, ""]
    md.append("| " + " | ".join(cols) + " |")
    md.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, r in df_out.iterrows():
        row = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                row.append(f"{v:.4f}" if not pd.isna(v) else "")
            elif pd.isna(v):
                row.append("")
            else:
                row.append(str(v))
        md.append("| " + " | ".join(row) + " |")
    path_md.write_text("\n".join(md) + "\n")
    print(f"[write] {path_csv.relative_to(ROOT)}  +  {path_md.relative_to(ROOT)}  "
          f"({len(df_out)} rows)")


def emit_deliverables(df: pd.DataFrame) -> dict:
    show = ["rank_final", "drug_name", "cluster", "net_direction",
            "activity_magnitude", "real_angio_pubmed_hits",
            "novelty_real", "novelty_status",
            "kg_plausibility", "score_v5_final",
            "corpus_n_papers"]

    anti = df[df["net_direction"] < 0].copy()
    pro = df[df["net_direction"] > 0].copy()

    # Top-20 anti / pro by score_v5_final (within-direction rank)
    top_anti = anti.head(20).copy()
    top_pro = pro.head(20).copy()
    emit_table(top_anti, TABLES / "v5_final_top20_anti_angiogenic.csv",
               TABLES / "v5_final_top20_anti_angiogenic.md",
               "v5_final — Top-20 anti-angiogenic candidates",
               "Sorted by `score_v5_final = 0.40·activity_magnitude "
               "+ 0.35·novelty_real + 0.25·kg_plausibility`. "
               "`novelty_real` is computed from real PubMed angio-anchored "
               "hits, NOT from our corpus.", show)
    emit_table(top_pro, TABLES / "v5_final_top20_pro_angiogenic.csv",
               TABLES / "v5_final_top20_pro_angiogenic.md",
               "v5_final — Top-20 pro-angiogenic candidates",
               "Sorted by `score_v5_final`. Pro = `net_direction > 0`.",
               show)

    # Validation table: known anti-angiogenics (well-studied + anti)
    pat = "|".join(KNOWN_ANTI_ANGIOGENICS)
    known_mask = anti["drug_name"].str.contains(pat, case=False, regex=True, na=False)
    known_anti = anti[known_mask].sort_values("score_v5_final", ascending=False).head(25)
    emit_table(known_anti,
               TABLES / "v5_final_validation_anti.csv",
               TABLES / "v5_final_validation_anti.md",
               "v5_final — Validation: recovered known anti-angiogenics",
               "Drugs whose name matches a known anti-angiogenic (Sunitinib, "
               "Colchicine, Vinblastine, 2-Methoxyestradiol, Suramin, "
               "Resveratrol, Tamoxifen, Topotecan, ...). These should appear "
               "with `net_direction < 0` and high `kg_plausibility`/"
               "`activity_magnitude`, demonstrating that the method recovers "
               "real biology even after the novelty term is moved to "
               "real PubMed (which floors their `novelty_real` to ≈0.1).",
               show)

    # Discovery: genuinely understudied (real_angio <= 2) per direction
    disc_anti = (anti[anti["real_angio_pubmed_hits"] <= 2]
                 .sort_values("score_v5_final", ascending=False).head(25))
    disc_pro = (pro[pro["real_angio_pubmed_hits"] <= 2]
                .sort_values("score_v5_final", ascending=False).head(25))
    emit_table(disc_anti,
               TABLES / "v5_final_discovery_anti.csv",
               TABLES / "v5_final_discovery_anti.md",
               "v5_final — Discovery: top genuinely understudied anti-angiogenic",
               "Filter: `real_angio_pubmed_hits ≤ 2` (uncapped real PubMed). "
               "These are the genuine repurposing candidates: high measured "
               "activity, anti-angiogenic direction, AND scarcely any prior "
               "angiogenesis literature. The TRUE-DARK 13 from the sidecar "
               "(BNTX, Ziprasidone, PHA 767491, ...) are the centerpiece.",
               show)
    emit_table(disc_pro,
               TABLES / "v5_final_discovery_pro.csv",
               TABLES / "v5_final_discovery_pro.md",
               "v5_final — Discovery: top genuinely understudied pro-angiogenic",
               "Filter: `real_angio_pubmed_hits ≤ 2`, `net_direction > 0`.",
               show)

    return {
        "top_anti_top1": top_anti.iloc[0].to_dict() if len(top_anti) else None,
        "top_pro_top1": top_pro.iloc[0].to_dict() if len(top_pro) else None,
        "n_validation_anti": len(known_anti),
        "n_discovery_anti": len(disc_anti),
        "n_discovery_pro": len(disc_pro),
    }


def validation_gate(df: pd.DataFrame) -> dict:
    """Re-run the 3-gate validation on v5_final."""
    gates = {}

    # Gate A: PC7 controls anti-direction; should now NOT top the list
    ctl_mask = df["drug_name"].str.contains(
        "|".join([f"^{c}" for c in PC7_CONTROLS]), case=False, regex=True, na=False)
    ctl = df[ctl_mask].copy()
    gates["A_pc7_controls"] = {
        "drugs": [
            {"drug_name": r["drug_name"],
             "rank_final": int(r["rank_final"]),
             "net_direction": round(float(r["net_direction"]), 3),
             "real_angio_pubmed_hits": int(r["real_angio_pubmed_hits"]),
             "novelty_real": round(float(r["novelty_real"]), 3),
             "score_v5_final": round(float(r["score_v5_final"]), 4),
             "novelty_status": r["novelty_status"]}
            for _, r in ctl.iterrows()],
        "all_anti": bool((ctl["net_direction"] < 0).all()) if len(ctl) else False,
        "median_rank_final": int(ctl["rank_final"].median()) if len(ctl) else -1,
        "any_top_5_anti": bool((ctl[ctl["net_direction"] < 0]["rank_final"] <= 5).any()),
    }

    # Gate B: top-5 of each direction list — composition check
    top5_anti = df[df["net_direction"] < 0].head(5)
    top5_pro = df[df["net_direction"] > 0].head(5)
    gates["B_top5_composition"] = {
        "anti_top5_status_breakdown": top5_anti["novelty_status"].value_counts().to_dict(),
        "pro_top5_status_breakdown": top5_pro["novelty_status"].value_counts().to_dict(),
        "anti_top5_median_real_angio": float(top5_anti["real_angio_pubmed_hits"].median()),
        "pro_top5_median_real_angio": float(top5_pro["real_angio_pubmed_hits"].median()),
    }

    # Gate C: discovery set is dominated by TRUE-DARK
    disc = df[df["real_angio_pubmed_hits"] <= 2].copy()
    gates["C_discovery_set"] = {
        "n_discovery": int(len(disc)),
        "n_true_dark": int((disc["novelty_status"] == "TRUE-DARK").sum()),
        "n_active_dark_real": int(disc["is_active_dark_real"].sum()),
        "top10_discovery": [
            {"drug_name": r["drug_name"],
             "rank_final": int(r["rank_final"]),
             "novelty_status": r["novelty_status"],
             "score_v5_final": round(float(r["score_v5_final"]), 4),
             "activity_magnitude": round(float(r["activity_magnitude"]), 4),
             "net_direction": round(float(r["net_direction"]), 3)}
            for _, r in disc.head(10).iterrows()],
    }

    # Suramin / Y-27632 demotion check (the headline reality-check assertion)
    flagged = ["Suramin sodium salt", "Y-27632 dihydrochloride",
               "Atorvastatin calcium salt trihydrate",
               "Serotonin hydrochloride", "Sildenafil citrate salt"]
    sm = df[df["drug_name"].isin(flagged)]
    gates["D_corpus_missed_demotions"] = [
        {"drug_name": r["drug_name"],
         "rank_v5_original": int(r["rank"]),
         "rank_final": int(r["rank_final"]),
         "delta_rank": int(r["rank_final"] - r["rank"]),
         "real_angio_pubmed_hits": int(r["real_angio_pubmed_hits"]),
         "novelty_real": round(float(r["novelty_real"]), 4),
         "corpus_novelty_v5": round(float(r["novelty"]), 4)}
        for _, r in sm.iterrows()]

    return gates


# ---------------------------------------------------------------------------
def main():
    rate = setup_entrez()

    v5, sidecar = load_v5_and_sidecar()
    print(f"\n[step1] full PubMed sweep")
    hits = get_or_query_all(v5, sidecar, rate)
    print(f"  cache: {len(hits)} drugs with real PubMed counts")

    print(f"\n[step2] build v5_final")
    final = build_v5_final(v5, hits)
    final.to_csv(V5_FINAL_CSV, index=False)
    print(f"[write] {V5_FINAL_CSV.relative_to(ROOT)}  ({len(final)} rows)")

    # status summary
    status_counts = final["novelty_status"].value_counts().to_dict()
    print(f"  novelty_status: {status_counts}")

    print(f"\n[step3] emit two-tier deliverables")
    emit_deliverables(final)

    print(f"\n[step4] validation gate")
    gates = validation_gate(final)
    (TABLES / "v5_final_validation_gate.json").write_text(
        json.dumps(gates, indent=2, default=str))
    print(f"[write] {(TABLES / 'v5_final_validation_gate.json').relative_to(ROOT)}")

    # Markdown summary of the gate
    md = ["# v5_final — validation gate\n"]
    md.append("## Gate A — PC7 controls (well-studied; should NOT top the list)\n")
    md.append("| drug | rank_final | net_direction | real_angio | novelty_real "
              "| score_v5_final | novelty_status |")
    md.append("|---|---|---|---|---|---|---|")
    for d in gates["A_pc7_controls"]["drugs"]:
        md.append(f"| {d['drug_name']} | {d['rank_final']} | "
                  f"{d['net_direction']} | {d['real_angio_pubmed_hits']} | "
                  f"{d['novelty_real']} | {d['score_v5_final']} | "
                  f"{d['novelty_status']} |")
    md.append(f"\n**all_anti**: {gates['A_pc7_controls']['all_anti']}  "
              f"**median rank_final**: {gates['A_pc7_controls']['median_rank_final']}  "
              f"**any in top-5 anti**: {gates['A_pc7_controls']['any_top_5_anti']}\n")

    md.append("## Gate B — top-5 composition by direction\n")
    md.append("```")
    md.append(json.dumps(gates["B_top5_composition"], indent=2))
    md.append("```\n")

    md.append("## Gate C — discovery set composition\n")
    md.append(f"- n_discovery (real_angio ≤ 2): "
              f"**{gates['C_discovery_set']['n_discovery']}**")
    md.append(f"- n_TRUE-DARK in discovery: "
              f"**{gates['C_discovery_set']['n_true_dark']}**")
    md.append(f"- n_active_dark_real (real_angio ≤ 2 ∧ activity ≥ p75): "
              f"**{gates['C_discovery_set']['n_active_dark_real']}**\n")
    md.append("### Top-10 of discovery set\n")
    md.append("| drug | rank_final | novelty_status | net_direction | "
              "activity | score_v5_final |")
    md.append("|---|---|---|---|---|---|")
    for d in gates["C_discovery_set"]["top10_discovery"]:
        md.append(f"| {d['drug_name']} | {d['rank_final']} | "
                  f"{d['novelty_status']} | {d['net_direction']} | "
                  f"{d['activity_magnitude']} | {d['score_v5_final']} |")

    md.append("\n## Gate D — Suramin / Y-27632 demotion (the headline assertion)\n")
    md.append("| drug | rank (v5) | rank_final | Δ rank | real_angio | "
              "novelty_real | corpus_novelty (v5) |")
    md.append("|---|---|---|---|---|---|---|")
    for d in gates["D_corpus_missed_demotions"]:
        sign = "+" if d['delta_rank'] >= 0 else ""
        md.append(f"| {d['drug_name']} | {d['rank_v5_original']} | "
                  f"{d['rank_final']} | {sign}{d['delta_rank']} | "
                  f"{d['real_angio_pubmed_hits']} | {d['novelty_real']} | "
                  f"{d['corpus_novelty_v5']} |")
    (TABLES / "v5_final_validation_gate.md").write_text("\n".join(md) + "\n")
    print(f"[write] {(TABLES / 'v5_final_validation_gate.md').relative_to(ROOT)}")

    # Console banner with the headline numbers
    print("\n" + "=" * 72)
    print("v5_FINAL SUMMARY")
    print("=" * 72)
    print(f"  total drugs ranked            : {len(final)}")
    print(f"  novelty_status counts         : {status_counts}")
    print()
    print("  Gate A — PC7 controls:")
    for d in gates["A_pc7_controls"]["drugs"]:
        print(f"    {d['drug_name']:46s}  rank={d['rank_final']:>4}  "
              f"score={d['score_v5_final']:.4f}  {d['novelty_status']}")
    print()
    print("  Gate D — corpus-missed demotion:")
    for d in gates["D_corpus_missed_demotions"]:
        sign = "+" if d['delta_rank'] >= 0 else ""
        print(f"    {d['drug_name']:46s}  v5={d['rank_v5_original']:>4} -> "
              f"final={d['rank_final']:>4}  ({sign}{d['delta_rank']})")

    print("\nDONE")


if __name__ == "__main__":
    main()
