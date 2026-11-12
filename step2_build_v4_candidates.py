"""
Build interim drug_repurposing_candidates_v4.csv

Score (interim, literature-gap x graph-connectivity prior):
  novelty         = max(0.1, 1 - log10(n_papers + 1) / 3)
  kg_plausibility = min(1.0, log10(edges + 1) / log10(11))   # ~10 edges -> 1.0
  class_bonus     = 0.10 if drug in PC4 union PC7 else 0.0   # PLACEHOLDER
  repurposing_score = 0.60 * novelty + 0.30 * kg_plausibility + class_bonus

Edges: alias-union over all KG entities whose normalized name equals the
drug's normalized name; dedup by relationship identity (each record in
vdb_relationships.json contributes at most once even when multiple aliases
of the same drug appear at both endpoints).

Exclusion: drugs with n_papers == 0 AND edges == 0 are excluded from the
ranked list and saved separately as the literature-dark set.

Auditable artifacts:
  data/phase4/drug_repurposing_candidates_v4.csv      (ranked, with aliases_used)
  data/phase4/drug_repurposing_literature_dark_v4.csv (n_papers==0 AND edges==0)
  data/phase4/alias_to_canonical_v4.json              (resolved alias -> canonical map)
  data/phase4/v4_candidate_build_summary.json         (build stats + verification gate)

NOTE: v3 (drug_repurposing_candidates_v3.csv) is NOT modified.
"""
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
import data_loader

# Reuse the normalizer from step1
sys.path.insert(0, str(Path(__file__).resolve().parent))
from step1_normalize_remeasure import normalize_drug_name


ROOT = Path(__file__).resolve().parent
KG_DIR = config.LIGHTRAG_DIR
PHASE4 = config.PHASE4_DIR

CAND_CSV = PHASE4 / "drug_repurposing_candidates_v4.csv"
DARK_CSV = PHASE4 / "drug_repurposing_literature_dark_v4.csv"
ALIAS_JSON = PHASE4 / "alias_to_canonical_v4.json"
SUMMARY_JSON = PHASE4 / "v4_candidate_build_summary.json"
LEXICON_JSON = PHASE4 / "angio_lexicon_v4.json"

INTERIM_HEADER_COMMENT = (
    "# Interim ranking - literature-gap + topical-graph-connectivity prior. "
    "kg_plausibility uses angio_edges (lexicon-filtered); edges_all kept for audit. "
    "Experimental-activity (PhenoCluster) term pending lab data; "
    "not the final repurposing score."
)


def _flatten_lexicon(lex):
    terms = set()
    for cat, lst in lex.get("categories", {}).items():
        for t in lst:
            t = str(t).strip()
            if t:
                terms.add(t.lower())
    return terms


def _is_topical(name_lower, lex_terms):
    """Case-insensitive substring match against any lexicon term."""
    if not name_lower:
        return False
    for t in lex_terms:
        if t in name_lower:
            return True
    return False


def main():
    # ------------------------------------------------------------------
    # Load KG
    # ------------------------------------------------------------------
    print(f"[load] KG dir: {KG_DIR}")
    with open(KG_DIR / "vdb_entities.json") as f:
        vdb_ent = json.load(f).get("data", [])
    with open(KG_DIR / "vdb_relationships.json") as f:
        vdb_rel = json.load(f).get("data", [])
    print(f"  entities: {len(vdb_ent):,}   relationships: {len(vdb_rel):,}")

    # Build: normalized_name_lower -> list of canonical entity names
    norm_to_entities = defaultdict(list)
    ent_canonical_lookup = {}  # lower(entity_name) -> entity_name
    for e in vdb_ent:
        nm = str(e.get("entity_name", "") or "").strip()
        if not nm:
            continue
        ent_canonical_lookup[nm.lower()] = nm
        nm_norm = normalize_drug_name(nm).lower()
        if nm_norm:
            norm_to_entities[nm_norm].append(nm)

    # Build per-entity-lowercase edge index, with the SET of relationship
    # record indices touching each endpoint.
    edges_by_endpoint = defaultdict(set)  # endpoint_lower -> {record_index}
    edge_endpoints = []  # i -> (src_lower, tgt_lower) for fast topical check
    for i, r in enumerate(vdb_rel):
        s = str(r.get("src_id", "") or "").strip().lower()
        t = str(r.get("tgt_id", "") or "").strip().lower()
        edge_endpoints.append((s, t))
        if s:
            edges_by_endpoint[s].add(i)
        if t and t != s:
            edges_by_endpoint[t].add(i)

    # Load curated angiogenesis lexicon
    print(f"[load] lexicon: {LEXICON_JSON}")
    with open(LEXICON_JSON) as f:
        lexicon = json.load(f)
    lex_terms = _flatten_lexicon(lexicon)
    print(f"  lexicon terms: {len(lex_terms)}  (categories: {len(lexicon.get('categories', {}))})")

    # ------------------------------------------------------------------
    # Load LOPAC drugs and corpus paper counts
    # ------------------------------------------------------------------
    raw = data_loader.load_raw_dataset()
    clean = data_loader.clean_text_columns(raw)
    all_drugs = data_loader.extract_unique_drugs(clean)
    pc4 = set(d.lower() for d in
              data_loader.get_subset(all_drugs, "pc4")["compound_clean"].tolist())
    pc7 = set(d.lower() for d in
              data_loader.get_subset(all_drugs, "pc7")["compound_clean"].tolist())
    placebo_drugs = pc4 | pc7
    print(f"  PC4 union PC7 (class bonus pool): {len(placebo_drugs)}")

    drug_paper_counts = {}
    with open(config.PHASE2_DIR / "corpus_index.csv") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            d = row["drug_name"]
            drug_paper_counts[d] = drug_paper_counts.get(d, 0) + 1

    drug_meta = {}
    for _, row in all_drugs.iterrows():
        nm = str(row["compound_clean"]).strip()
        if nm and nm != "nan":
            drug_meta[nm] = {
                "drug_class": str(row.get("Class", "Unknown")),
                "action": str(row.get("Action", "Unknown")),
            }
    drug_names = sorted(drug_meta.keys())
    print(f"  LOPAC drugs: {len(drug_names)}")

    # ------------------------------------------------------------------
    # For each LOPAC drug: alias resolution + alias-union edge count
    # ------------------------------------------------------------------
    records = []
    alias_map_artifact = {}  # drug_name -> {canonical, aliases_used, edges, normalized}

    for drug in drug_names:
        d_norm = normalize_drug_name(drug).lower()
        d_lower = drug.lower()

        # Aliases: any KG entity whose normalized name matches this drug's
        # normalized name. Also include direct exact match (just in case).
        aliases = set()
        if d_lower in ent_canonical_lookup:
            aliases.add(ent_canonical_lookup[d_lower])
        if d_norm and d_norm in norm_to_entities:
            for nm in norm_to_entities[d_norm]:
                aliases.add(nm)

        # Alias-union edges with dedup
        edge_record_ids = set()
        for nm in aliases:
            edge_record_ids |= edges_by_endpoint.get(nm.lower(), set())
        n_edges_all = len(edge_record_ids)

        # Topical-edge filter: edge counts as angio if the NON-DRUG endpoint
        # contains any lexicon term. If both endpoints are drug aliases, skip.
        alias_lower = {a.lower() for a in aliases}
        n_angio_edges = 0
        topical_endpoints = []  # for audit
        for i in edge_record_ids:
            s, t = edge_endpoints[i]
            if s in alias_lower and t not in alias_lower:
                other = t
            elif t in alias_lower and s not in alias_lower:
                other = s
            else:
                # both in aliases or neither (defensive); use whichever endpoint
                # is not a drug alias
                other = t if s in alias_lower else s
            if _is_topical(other, lex_terms):
                n_angio_edges += 1
                if len(topical_endpoints) < 5:
                    topical_endpoints.append(other)

        # Canonical: prefer exact-name entity if present, else shortest alias
        canonical = None
        if d_lower in ent_canonical_lookup:
            canonical = ent_canonical_lookup[d_lower]
        elif aliases:
            canonical = sorted(aliases, key=lambda x: (len(x), x))[0]

        n_papers = drug_paper_counts.get(drug, 0)

        # Score components -- kg_plausibility uses angio_edges (lexicon-filtered)
        novelty = max(0.1, 1.0 - math.log10(n_papers + 1) / 3.0)
        novelty = round(novelty, 4)

        if n_angio_edges <= 0:
            kg_plausibility = 0.0
        else:
            kg_plausibility = min(1.0, math.log10(n_angio_edges + 1) / math.log10(11))
        kg_plausibility = round(kg_plausibility, 4)

        class_bonus = 0.10 if d_lower in placebo_drugs else 0.0

        repurposing_score = round(
            0.60 * novelty + 0.30 * kg_plausibility + class_bonus, 4
        )

        records.append({
            "drug_name": drug,
            "normalized_name": normalize_drug_name(drug),
            "canonical_name": canonical or "",
            "aliases_used": "|".join(sorted(aliases)),
            "n_aliases": len(aliases),
            "drug_class": drug_meta[drug]["drug_class"],
            "action": drug_meta[drug]["action"],
            "in_pc4_pc7": d_lower in placebo_drugs,
            "n_papers": n_papers,
            "edges_all": n_edges_all,
            "angio_edges": n_angio_edges,
            "evidence_only_kg": (n_papers == 0 and n_edges_all > 0),
            "sample_topical_endpoints": "|".join(topical_endpoints),
            "novelty": novelty,
            "kg_plausibility": kg_plausibility,
            "class_bonus": class_bonus,
            "repurposing_score": repurposing_score,
        })
        alias_map_artifact[drug] = {
            "normalized_name": normalize_drug_name(drug),
            "canonical_name": canonical,
            "aliases_used": sorted(aliases),
            "edges_all": n_edges_all,
            "angio_edges": n_angio_edges,
            "n_papers": n_papers,
        }

    df = pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Exclusion: n_papers == 0 AND edges_all == 0 -> literature-dark, separate file
    # NOTE: exclusion still uses edges_all (any KG signal). evidence_only_kg drugs
    # (n_papers==0 AND edges_all>0 AND angio_edges==0) STAY in the ranked list but
    # collapse on score because kg_plausibility=0.
    # ------------------------------------------------------------------
    dark_mask = (df["n_papers"] == 0) & (df["edges_all"] == 0)
    dark_df = df[dark_mask].copy().sort_values("drug_name").reset_index(drop=True)
    ranked_df = df[~dark_mask].copy()

    # Rank: descending score, tiebreak by angio_edges desc, then edges_all desc,
    # then n_papers asc, then name
    ranked_df = ranked_df.sort_values(
        by=["repurposing_score", "angio_edges", "edges_all", "n_papers", "drug_name"],
        ascending=[False, False, False, True, True],
    ).reset_index(drop=True)
    ranked_df["rank"] = ranked_df.index + 1

    # ------------------------------------------------------------------
    # Write outputs (with a leading comment line on the candidate CSV).
    # ------------------------------------------------------------------
    with open(CAND_CSV, "w", newline="") as f:
        f.write(INTERIM_HEADER_COMMENT + "\n")
        ranked_df.to_csv(f, index=False)
    print(f"\n[save] {CAND_CSV}  ({len(ranked_df):,} rows)")

    with open(DARK_CSV, "w", newline="") as f:
        f.write("# Literature-dark drugs: n_papers == 0 AND edges == 0. "
                "Separated from the ranked list per the v4 spec.\n")
        dark_df.to_csv(f, index=False)
    print(f"[save] {DARK_CSV}  ({len(dark_df):,} rows)")

    ALIAS_JSON.write_text(json.dumps(alias_map_artifact, indent=2))
    print(f"[save] {ALIAS_JSON}")

    # ------------------------------------------------------------------
    # Manual synonym groups -- record-only housekeeping. These compounds
    # are NOT in LOPAC1280 (they appear in config.PC7_VEGF_INHIBITORS as
    # foundational-paper anchor names). Persist a separate sidecar map
    # so KG entity-resolution gaps are auditable without affecting ranks.
    # ------------------------------------------------------------------
    SYNONYM_SIDECAR = PHASE4 / "manual_synonyms_v4.json"
    manual_synonyms = {
        "SU5416_group": {
            "canonical": "SU5416",
            "synonyms": ["SU5416", "SU 5416", "Semaxanib"],
            "note": (
                "Same molecule, different names. KG indexer created entity "
                "'SU5416' only; 'Semaxanib' has 0 KG edges as a separate "
                "alias-resolution gap. Not in LOPAC1280; documented for "
                "reproducibility, no LOPAC rank impact."
            ),
        },
        "SU4312_group": {
            "canonical": "SU4312",
            "synonyms": ["SU4312", "SU 4312"],
            "note": "Spacing-only variant; not in LOPAC1280.",
        },
    }
    SYNONYM_SIDECAR.write_text(json.dumps(manual_synonyms, indent=2))
    print(f"[save] {SYNONYM_SIDECAR}")

    # ------------------------------------------------------------------
    # Verification gate
    # ------------------------------------------------------------------
    n_ranked = len(ranked_df)
    print()
    print("=" * 78)
    print(f"VERIFICATION GATE  (n_ranked = {n_ranked})")
    print("=" * 78)

    target_results = {}
    for token in ["Sunitinib", "Thalidomide"]:
        hits = ranked_df[
            ranked_df["drug_name"].str.contains(token, case=False, na=False)
        ]
        if hits.empty:
            print(f"  [{token:12s}] NOT IN RANKED LIST")
            target_results[token] = {"in_ranked": False}
            continue
        for _, r in hits.iterrows():
            pct = r["rank"] / n_ranked * 100
            verdict = "PASS" if pct <= 15 else ("WARN" if pct <= 20 else "FAIL")
            print(f"  [{token:12s}] match='{r['drug_name']}'  "
                  f"rank={int(r['rank'])}/{n_ranked} ({pct:.1f}%)  "
                  f"score={r['repurposing_score']:.3f}  "
                  f"angio={int(r['angio_edges'])}/{int(r['edges_all'])}  "
                  f"n_papers={int(r['n_papers'])}  "
                  f"novelty={r['novelty']:.3f}  "
                  f"kg_plaus={r['kg_plausibility']:.3f}  "
                  f"in_pc4_pc7={r['in_pc4_pc7']}   [{verdict} top-15%]")
            target_results.setdefault(token, []).append({
                "drug_name": r["drug_name"],
                "rank": int(r["rank"]),
                "rank_pct": round(pct, 2),
                "in_top_15pct": pct <= 15,
                "score": float(r["repurposing_score"]),
                "angio_edges": int(r["angio_edges"]),
                "edges_all": int(r["edges_all"]),
                "n_papers": int(r["n_papers"]),
                "novelty": float(r["novelty"]),
                "kg_plausibility": float(r["kg_plausibility"]),
                "in_pc4_pc7": bool(r["in_pc4_pc7"]),
            })

    print()
    print("=" * 78)
    print("TOP 20")
    print("=" * 78)
    cols = ["rank", "drug_name", "drug_class", "n_papers",
            "edges_all", "angio_edges",
            "novelty", "kg_plausibility", "class_bonus", "repurposing_score",
            "in_pc4_pc7"]
    print(ranked_df[cols].head(20).to_string(index=False))

    # Sanity check: any zero-evidence noise leaking into top 20?
    top20 = ranked_df.head(20)
    zero_noise = top20[(top20["n_papers"] == 0) & (top20["edges_all"] == 0)]

    # Where did the 4 evidence_only_kg drugs land?
    eok_targets = [
        "Valganciclovir hydrochloride hydrate",
        "Doxepin hydrochloride",
        "BNTX maleate salt hydrate",
        "Benzamil hydrochloride",
    ]
    print()
    print("  evidence_only_kg drugs after lexicon filter:")
    for nm in eok_targets:
        hit = ranked_df[ranked_df["drug_name"] == nm]
        if hit.empty:
            print(f"    [{nm}] NOT IN RANKED LIST")
        else:
            r = hit.iloc[0]
            pct = r["rank"] / n_ranked * 100
            print(f"    [{nm:42s}] rank={int(r['rank']):>4}/{n_ranked} ({pct:5.1f}%)  "
                  f"score={r['repurposing_score']:.3f}  "
                  f"angio={int(r['angio_edges'])}/{int(r['edges_all'])}")
    print()
    if zero_noise.empty:
        print("  [PASS] no zero-evidence (n_papers==0 AND edges==0) drugs in top 20.")
    else:
        print(f"  [FAIL] {len(zero_noise)} zero-evidence drugs in top 20:")
        print(zero_noise[["rank", "drug_name"]].to_string(index=False))

    # Distribution sanity
    print()
    print(f"  edges_all  : min={ranked_df['edges_all'].min()}  median={ranked_df['edges_all'].median()}  "
          f"max={ranked_df['edges_all'].max()}  mean={ranked_df['edges_all'].mean():.1f}")
    print(f"  angio_edges: min={ranked_df['angio_edges'].min()}  median={ranked_df['angio_edges'].median()}  "
          f"max={ranked_df['angio_edges'].max()}  mean={ranked_df['angio_edges'].mean():.1f}")
    print(f"  topical%   : median {(ranked_df['angio_edges']/ranked_df['edges_all'].clip(lower=1)).median():.2f}")
    print(f"  novelty: min={ranked_df['novelty'].min():.3f}  "
          f"median={ranked_df['novelty'].median():.3f}  "
          f"max={ranked_df['novelty'].max():.3f}")
    print(f"  kg_plau: min={ranked_df['kg_plausibility'].min():.3f}  "
          f"median={ranked_df['kg_plausibility'].median():.3f}  "
          f"max={ranked_df['kg_plausibility'].max():.3f}")
    print(f"  score  : min={ranked_df['repurposing_score'].min():.3f}  "
          f"median={ranked_df['repurposing_score'].median():.3f}  "
          f"max={ranked_df['repurposing_score'].max():.3f}")

    # Persist build summary
    n_evidence_only_kg = int(((ranked_df['n_papers'] == 0) & (ranked_df['edges_all'] > 0)).sum())
    print(f"\n  evidence_only_kg bucket (n_papers==0 AND edges_all>0): {n_evidence_only_kg} drugs")
    if n_evidence_only_kg:
        eok = ranked_df[(ranked_df['n_papers'] == 0) & (ranked_df['edges_all'] > 0)]
        print("    top 10 by rank:")
        print(eok[['rank', 'drug_name', 'edges_all', 'angio_edges', 'repurposing_score']].head(10).to_string(index=False))

    summary = {
        "version": "v4_interim_topical",
        "label": INTERIM_HEADER_COMMENT.lstrip("# "),
        "formula": {
            "novelty": "max(0.1, 1 - log10(n_papers + 1) / 3)",
            "angio_edges": "alias-union edges where the non-drug endpoint matches the curated angiogenesis lexicon (case-insensitive substring)",
            "kg_plausibility": "min(1.0, log10(angio_edges + 1) / log10(11))   # uses angio_edges, NOT edges_all",
            "edges_all": "alias-union edge count, no lexicon filter (audit column only)",
            "class_bonus": "0.10 if drug in PC4 union PC7 (47 drugs) else 0.0",
            "repurposing_score": "0.60 * novelty + 0.30 * kg_plausibility + class_bonus",
        },
        "lexicon_artifact": str(LEXICON_JSON),
        "lexicon_terms": len(lex_terms),
        "weights": {"novelty": 0.60, "kg_plausibility": 0.30, "class_bonus": 0.10},
        "exclusion_rule": "drop n_papers==0 AND edges==0 from ranked list",
        "totals": {
            "lopac_drugs": len(drug_names),
            "ranked": int(len(ranked_df)),
            "literature_dark_excluded": int(len(dark_df)),
        },
        "edges_all_distribution": {
            "min": int(ranked_df["edges_all"].min()),
            "median": float(ranked_df["edges_all"].median()),
            "max": int(ranked_df["edges_all"].max()),
            "mean": float(round(ranked_df["edges_all"].mean(), 2)),
        },
        "angio_edges_distribution": {
            "min": int(ranked_df["angio_edges"].min()),
            "median": float(ranked_df["angio_edges"].median()),
            "max": int(ranked_df["angio_edges"].max()),
            "mean": float(round(ranked_df["angio_edges"].mean(), 2)),
        },
        "verification_gate": target_results,
        "zero_evidence_in_top20": int(len(zero_noise)),
        "evidence_only_kg_count": n_evidence_only_kg,
        "evidence_only_kg_definition": "n_papers == 0 AND edges > 0 (literature-dark but graph-plausible; kept in ranked list, flagged)",
        "novelty_semantics": "novelty = scarcity in the angiogenesis-filtered corpus (PubMed search capped at 20 papers per drug), NOT global literature fame",
        "kg_dir": str(KG_DIR),
        "outputs": {
            "candidates_csv": str(CAND_CSV),
            "literature_dark_csv": str(DARK_CSV),
            "alias_map_json": str(ALIAS_JSON),
        },
        "note": (
            "These weights (0.60 / 0.30 / 0.10) are a tunable methodological "
            "choice, not a derived constant. Activity term is parked as the "
            "PC4/PC7 placeholder bonus; on arrival of lab PhenoCluster (1-7) "
            "labels, the class_bonus is REMOVED and replaced by the "
            "continuous activity term -- not stacked."
        ),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2))
    print(f"\n[save] {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
