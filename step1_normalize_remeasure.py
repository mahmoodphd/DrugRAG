"""
Step 1 re-measurement: name normalization + correct edge counting.
NO scoring changes. NO v4 candidate CSV. Report-only.

Compares OLD scorer logic vs. fixed logic:
  OLD:  exact case-insensitive name match against entity_name set,
        then count entity-content mentions (this is the bug).
  NEW:  normalize LOPAC drug_name (strip salts/hydrates/stereo prefixes),
        resolve to KG canonical entity name via exact-after-normalization,
        then count edges in vdb_relationships where src_id or tgt_id matches
        (matching the spot-check semantics that gave Sunitinib = 52).

Outputs:
  - stdout report
  - data/phase4/normalization_remeasure.json  (numbers, not a candidate file)
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
import data_loader

ROOT = Path(__file__).resolve().parent
KG_DIR = config.LIGHTRAG_DIR
OUT_JSON = ROOT / "data/phase4/normalization_remeasure.json"


# ---------------------------------------------------------------------------
# Name normalization
# ---------------------------------------------------------------------------
# Strip well-known salt / hydrate / form suffixes. Order matters: longest first.
SUFFIX_PATTERNS = [
    r"\btrihydrochloride\s+hydrate\b",
    r"\bdihydrochloride\s+hydrate\b",
    r"\bhydrochloride\s+monohydrate\b",
    r"\bhydrochloride\s+hydrate\b",
    r"\bmesylate\s+hydrate\b",
    r"\bmaleate\s+salt\s+hydrate\b",
    r"\bhemifumarate\s+salt\b",
    r"\bhemisulfate\s+salt\b",
    r"\bhydrobromide\s+hydrate\b",
    r"\bdihydrobromide\b",
    r"\btrihydrochloride\b",
    r"\bdihydrochloride\b",
    r"\bhydrochloride\b",
    r"\bhydrobromide\b",
    r"\bhemifumarate\b",
    r"\bhemisulfate\b",
    r"\bbisulfate\b",
    r"\bdiphosphate\b",
    r"\btriphosphate\b",
    r"\btetrasodium\b",
    r"\btrisodium\b",
    r"\bdisodium\b",
    r"\bmonosodium\b",
    r"\bsodium\s+salt\b",
    r"\bpotassium\s+salt\b",
    r"\bcalcium\s+salt\b",
    r"\bammonium\s+salt\b",
    r"\bmaleate\b",
    r"\bfumarate\b",
    r"\bcitrate\b",
    r"\btartrate\b",
    r"\bsuccinate\b",
    r"\boxalate\b",
    r"\bphosphate\b",
    r"\bsulfate\b",
    r"\bsulphate\b",
    r"\bnitrate\b",
    r"\bbromide\b",
    r"\bchloride\b",
    r"\biodide\b",
    r"\bacetate\b",
    r"\bbenzoate\b",
    r"\bmesylate\b",
    r"\btosylate\b",
    r"\bbesylate\b",
    r"\boxalate\b",
    r"\bmalate\b",
    r"\bgluconate\b",
    r"\boleate\b",
    r"\bstearate\b",
    r"\bpalmitate\b",
    r"\blactate\b",
    r"\baspartate\b",
    r"\bglutamate\b",
    r"\bdecahydrate\b",
    r"\bheptahydrate\b",
    r"\bhexahydrate\b",
    r"\bpentahydrate\b",
    r"\btetrahydrate\b",
    r"\btrihydrate\b",
    r"\bdihydrate\b",
    r"\bmonohydrate\b",
    r"\banhydrous\b",
    r"\bhydrate\b",
    r"\bsodium\b",
    r"\bpotassium\b",
    r"\bsalt\b",
    r"\bfree\s+base\b",
    r"\bfree\s+acid\b",
]

# Stereo / parenthesized prefixes: (±), (+), (-), (R), (S), (R,S), (S,R), (E), (Z)
PREFIX_PATTERNS = [
    r"^\s*\(\s*[\u00b1+\u2212-]\s*\)[\s-]*",
    r"^\s*\(\s*[RS]\s*\)[\s-]*",
    r"^\s*\(\s*[RS]\s*,\s*[RS]\s*\)[\s-]*",
    r"^\s*\(\s*[EZ]\s*\)[\s-]*",
    r"^\s*[\u00b1]\s*[\s-]+",
    r"^\s*DL[\s-]+",
    r"^\s*L[\s-]+",
    r"^\s*D[\s-]+",
]

SUFFIX_RE = re.compile("|".join(SUFFIX_PATTERNS), flags=re.IGNORECASE)
PREFIX_RES = [re.compile(p, flags=re.IGNORECASE) for p in PREFIX_PATTERNS]


def normalize_drug_name(name: str) -> str:
    """Return a normalized 'core' drug name suitable for KG entity matching."""
    if not name:
        return ""
    s = name.strip()
    # Strip stereo / parenthesized prefixes (just the leading ones)
    for pat in PREFIX_RES:
        s = pat.sub("", s)
    # Iteratively strip trailing salt/hydrate tokens
    prev = None
    while prev != s:
        prev = s
        s = SUFFIX_RE.sub("", s).strip(" ,;-")
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


if __name__ == "__main__":
    # ---------------------------------------------------------------------------
    # Load KG
    # ---------------------------------------------------------------------------
    print(f"[load] KG dir: {KG_DIR}")
    with open(KG_DIR / "vdb_entities.json") as f:
        vdb_ent = json.load(f).get("data", [])
    with open(KG_DIR / "vdb_relationships.json") as f:
        vdb_rel = json.load(f).get("data", [])
    print(f"  entities (vdb): {len(vdb_ent):,}   relationships (vdb): {len(vdb_rel):,}")

    # Build name index: lowercase entity_name -> canonical entity_name
    ent_name_to_canonical = {}
    ent_names_lower = set()
    for e in vdb_ent:
        nm = str(e.get("entity_name", "") or "").strip()
        if nm:
            ent_names_lower.add(nm.lower())
            ent_name_to_canonical.setdefault(nm.lower(), nm)

    # Build edge index by lowercased endpoint name
    edges_by_endpoint = Counter()
    for r in vdb_rel:
        src = str(r.get("src_id", "") or "").strip().lower()
        tgt = str(r.get("tgt_id", "") or "").strip().lower()
        if src:
            edges_by_endpoint[src] += 1
        if tgt and tgt != src:
            edges_by_endpoint[tgt] += 1


    def count_edges_exact(entity_name_lower: str) -> int:
        """Count edges where src_id or tgt_id == entity_name (case-insensitive exact)."""
        return edges_by_endpoint.get(entity_name_lower, 0)


    def count_edges_substring(token_lower: str) -> int:
        """Spot-check semantics: substring on src_id/tgt_id (matches the 52 number)."""
        n = 0
        for r in vdb_rel:
            src = str(r.get("src_id", "") or "").lower()
            tgt = str(r.get("tgt_id", "") or "").lower()
            if token_lower in src or token_lower in tgt:
                n += 1
        return n


    # ---------------------------------------------------------------------------
    # Load LOPAC drug list and corpus paper counts
    # ---------------------------------------------------------------------------
    raw_df = data_loader.load_raw_dataset()
    clean_df = data_loader.clean_text_columns(raw_df)
    all_drugs_df = data_loader.extract_unique_drugs(clean_df)
    drug_names = sorted(
        {str(r["compound_clean"]).strip()
         for _, r in all_drugs_df.iterrows()
         if str(r["compound_clean"]).strip() and str(r["compound_clean"]) != "nan"}
    )
    print(f"  LOPAC drugs: {len(drug_names):,}")

    drug_paper_counts = Counter()
    with open(config.PHASE2_DIR / "corpus_index.csv") as f:
        import csv
        for row in csv.DictReader(f):
            drug_paper_counts[row["drug_name"]] += 1

    # ---------------------------------------------------------------------------
    # Re-measure: per drug, OLD vs NEW edge counts
    # ---------------------------------------------------------------------------
    records = []
    for drug in drug_names:
        d_lower = drug.lower()
        n_papers = drug_paper_counts.get(drug, 0)

        # OLD logic: matched ONLY if d_lower in entity_names_lower (exact full name)
        old_matched = d_lower in ent_names_lower
        # OLD "n_kg_connections" = entity-content mentions when matched, else 0
        # (we recompute it the same way for parity)
        old_n_kg = 0
        if old_matched:
            for e in vdb_ent:
                content = (e.get("content", "") or "").lower()
                if d_lower in content:
                    old_n_kg += 1

        # NEW logic: normalize, then exact match against entity names
        norm = normalize_drug_name(drug)
        norm_lower = norm.lower()
        new_match_kind = "none"
        canonical = None
        if d_lower in ent_names_lower:
            canonical = ent_name_to_canonical[d_lower]
            new_match_kind = "exact"
        elif norm_lower and norm_lower in ent_names_lower:
            canonical = ent_name_to_canonical[norm_lower]
            new_match_kind = "normalized"

        new_edges_exact = count_edges_exact(canonical.lower()) if canonical else 0

        records.append({
            "drug_name": drug,
            "normalized": norm,
            "kg_canonical": canonical,
            "match_kind": new_match_kind,
            "n_papers": n_papers,
            "old_matched": old_matched,
            "old_n_kg": old_n_kg,
            "new_edges_exact": new_edges_exact,
        })

    df = pd.DataFrame(records)

    # ---------------------------------------------------------------------------
    # Aggregate stats
    # ---------------------------------------------------------------------------
    n_total = len(df)
    old_zero = df[df["old_n_kg"] == 0]
    new_zero = df[df["new_edges_exact"] == 0]
    recovered = df[(df["old_n_kg"] == 0) & (df["new_edges_exact"] > 0)]
    recovered_with_papers = recovered[recovered["n_papers"] > 0]
    still_zero_with_papers = df[(df["new_edges_exact"] == 0) & (df["n_papers"] > 0)]
    still_zero_no_papers = df[(df["new_edges_exact"] == 0) & (df["n_papers"] == 0)]

    print()
    print("=" * 72)
    print("RE-MEASUREMENT RESULTS")
    print("=" * 72)
    print(f"  LOPAC drugs total                              : {n_total:,}")
    print(f"  OLD logic, n_kg == 0  (the '642' figure)       : {len(old_zero):,}")
    print(f"  NEW logic (normalized + exact-edge), edges == 0: {len(new_zero):,}")
    print(f"  RECOVERED by normalization (was 0, now > 0)    : {len(recovered):,}")
    print(f"    of which had n_papers > 0 (was the '627')    : {len(recovered_with_papers):,}")
    print()
    print("  Remaining zero-edge breakdown after fix:")
    print(f"    still 0 edges, n_papers == 0  (true literature-dark) : {len(still_zero_no_papers):,}")
    print(f"    still 0 edges, n_papers >  0  (real extraction gaps) : {len(still_zero_with_papers):,}")
    print()

    # Match-kind distribution
    print("  Match-kind distribution under NEW logic:")
    print(df["match_kind"].value_counts().to_string())
    print()

    # ---------------------------------------------------------------------------
    # Target rank check (uses ranking by NEW edge count alone, just to see where
    # Sunitinib / Thalidomide land on raw KG signal).
    # ---------------------------------------------------------------------------
    df_sorted = df.sort_values("new_edges_exact", ascending=False).reset_index(drop=True)
    df_sorted["edge_rank"] = df_sorted.index + 1

    print("=" * 72)
    print("TARGET-DRUG EDGE COUNTS  (NEW vs OLD vs spot-check substring)")
    print("=" * 72)
    print(f"  {'Drug match':32s} {'norm':20s} {'old_n_kg':>9s} {'new_edges':>10s} "
          f"{'sub_str':>8s} {'edge_rank':>10s}")
    for token in ["Sunitinib", "Thalidomide", "SU5416", "Y-27632", "ROCK"]:
        hits = df_sorted[df_sorted["drug_name"].str.contains(token, case=False, na=False)]
        if hits.empty:
            sub = count_edges_substring(token.lower())
            print(f"  {token + ' (NOT IN LOPAC)':32s} "
                  f"{'':20s} {'-':>9s} {'-':>10s} {sub:>8d} {'-':>10s}")
            continue
        for _, r in hits.iterrows():
            sub = count_edges_substring(token.lower())
            print(f"  {r['drug_name'][:32]:32s} {str(r['normalized'])[:20]:20s} "
                  f"{r['old_n_kg']:>9d} {r['new_edges_exact']:>10d} "
                  f"{sub:>8d} {r['edge_rank']:>10d}/{n_total}")

    # Top-20 by NEW edge count (no scoring change, just KG signal)
    print()
    print("=" * 72)
    print("TOP-20 BY NEW EDGE COUNT (no scoring change, raw KG signal only)")
    print("=" * 72)
    print(df_sorted.head(20)[
        ["edge_rank", "drug_name", "normalized", "kg_canonical",
         "match_kind", "n_papers", "old_n_kg", "new_edges_exact"]
    ].to_string(index=False))

    # ---------------------------------------------------------------------------
    # Sample lists for the two zero-edge populations after the fix
    # ---------------------------------------------------------------------------
    print()
    print("=" * 72)
    print("SAMPLES: still-zero-edges drugs after normalization")
    print("=" * 72)
    print("\n  [a] true literature-dark (n_papers == 0):")
    for _, r in still_zero_no_papers.head(20).iterrows():
        print(f"      {r['drug_name']}")
    print(f"      ... total: {len(still_zero_no_papers)}")

    print("\n  [b] real extraction failures (n_papers > 0 but no KG entity):")
    for _, r in still_zero_with_papers.sort_values('n_papers', ascending=False).head(20).iterrows():
        print(f"      {r['drug_name']:50s} (n_papers={r['n_papers']}, "
              f"normalized='{r['normalized']}')")
    print(f"      ... total: {len(still_zero_with_papers)}")

    # ---------------------------------------------------------------------------
    # Persist measurement JSON (no candidate CSV)
    # ---------------------------------------------------------------------------
    payload = {
        "kg_dir": str(KG_DIR),
        "vdb_entities": len(vdb_ent),
        "vdb_relationships": len(vdb_rel),
        "lopac_total_drugs": n_total,
        "old_logic_zero_n_kg": int(len(old_zero)),
        "new_logic_zero_edges": int(len(new_zero)),
        "recovered_by_normalization_total": int(len(recovered)),
        "recovered_with_papers": int(len(recovered_with_papers)),
        "remaining_zero_edges": {
            "literature_dark_n_papers_0": int(len(still_zero_no_papers)),
            "extraction_gap_n_papers_gt_0": int(len(still_zero_with_papers)),
        },
        "match_kind_counts": df["match_kind"].value_counts().to_dict(),
        "targets": {},
    }
    for token in ["Sunitinib", "Thalidomide", "SU5416", "Y-27632", "ROCK"]:
        sub = count_edges_substring(token.lower())
        hits = df_sorted[df_sorted["drug_name"].str.contains(token, case=False, na=False)]
        payload["targets"][token] = {
            "substring_edge_count_in_kg": int(sub),
            "lopac_matches": [
                {
                    "drug_name": r["drug_name"],
                    "normalized": r["normalized"],
                    "kg_canonical": r["kg_canonical"],
                    "match_kind": r["match_kind"],
                    "old_n_kg": int(r["old_n_kg"]),
                    "new_edges_exact": int(r["new_edges_exact"]),
                    "edge_rank": int(r["edge_rank"]),
                }
                for _, r in hits.iterrows()
            ],
        }
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    print(f"\n[save] measurement payload: {OUT_JSON}")
