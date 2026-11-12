"""Spot-check the edges for the evidence_only_kg drugs.

For each drug in the bucket, pull every relationship record where one
endpoint matches an alias of the drug, and print src -> tgt with the
relationship description. The user inspects: substantively about the
drug's vascular/angiogenesis role, or incidental co-mention?
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from step1_normalize_remeasure import normalize_drug_name


KG_DIR = config.LIGHTRAG_DIR
TARGETS = [
    "Valganciclovir hydrochloride hydrate",
    "Doxepin hydrochloride",
    "BNTX maleate salt hydrate",
    "Benzamil hydrochloride",
]


def main():
    print(f"[load] {KG_DIR}")
    with open(KG_DIR / "vdb_entities.json") as f:
        ents = json.load(f).get("data", [])
    with open(KG_DIR / "vdb_relationships.json") as f:
        rels = json.load(f).get("data", [])
    # Need full relations with descriptions
    with open(KG_DIR / "kv_store_full_relations.json") as f:
        full_rels = json.load(f)

    # Build normalized -> entity_name map
    from collections import defaultdict
    norm_to_ents = defaultdict(list)
    for e in ents:
        nm = str(e.get("entity_name", "") or "").strip()
        if not nm:
            continue
        nn = normalize_drug_name(nm).lower()
        if nn:
            norm_to_ents[nn].append(nm)

    def descr_for(src, tgt):
        # full_relations is keyed by some ID; values have src_id,tgt_id,description
        for k, v in full_rels.items():
            if not isinstance(v, dict):
                continue
            if (str(v.get("src_id", "")).lower() == src.lower()
                    and str(v.get("tgt_id", "")).lower() == tgt.lower()) or \
               (str(v.get("src_id", "")).lower() == tgt.lower()
                    and str(v.get("tgt_id", "")).lower() == src.lower()):
                return str(v.get("description", "") or "")[:300]
        return ""

    for drug in TARGETS:
        d_norm = normalize_drug_name(drug).lower()
        aliases = set(norm_to_ents.get(d_norm, []))
        # also exact
        for e in ents:
            nm = str(e.get("entity_name", "") or "").strip()
            if nm.lower() == drug.lower():
                aliases.add(nm)

        print()
        print("=" * 92)
        print(f"DRUG: {drug}")
        print(f"  normalized: {normalize_drug_name(drug)}")
        print(f"  KG aliases ({len(aliases)}): {sorted(aliases)}")
        print("-" * 92)

        seen = set()
        edges = []
        for r in rels:
            s = str(r.get("src_id", "") or "")
            t = str(r.get("tgt_id", "") or "")
            if s in aliases or t in aliases:
                key = (s.lower(), t.lower())
                rev = (t.lower(), s.lower())
                if key in seen or rev in seen:
                    continue
                seen.add(key)
                edges.append((s, t))

        print(f"  unique edges (alias-union, dedup): {len(edges)}")
        for i, (s, t) in enumerate(edges, 1):
            d = descr_for(s, t)
            other = t if s in aliases else s
            print(f"\n  [{i}] {s}  -->  {t}")
            if d:
                # wrap manually
                print(f"      desc: {d}")


if __name__ == "__main__":
    main()
