#!/usr/bin/env python3
"""
Build the unique paper set with full drug-attribution list preserved.

Dedup rules:
  - 8,514 corpus_index rows -> 6,608 unique PMIDs (some PMIDs have multiple file paths)
  - For each unique PMID, collect ALL drug attributions and file paths
  - Pick the FIRST file path as the canonical one (content may differ between paths)
  - Content-hash dedup catches papers with identical text
  - Filter out empty/short content (< 50 chars)

Output: data/phase3/unique_papers.json
"""

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

CORPUS_CSV = Path("data/phase2/corpus_index.csv")
OUTPUT = Path("data/phase3/unique_papers.json")

# Step 1: Load corpus
papers = []
with open(CORPUS_CSV) as f:
    reader = csv.DictReader(f)
    for row in reader:
        papers.append(row)

print(f"Corpus rows: {len(papers)}")

# Step 2: Group by PMID (PMID is the unique paper key)
by_pmid = defaultdict(list)
for row in papers:
    by_pmid[row["pmid"]].append(row)

print(f"Unique PMIDs: {len(by_pmid)}")
multi_drug = sum(1 for rows in by_pmid.values() if len(set(r["drug_name"] for r in rows)) > 1)
print(f"PMIDs with >1 drug attribution: {multi_drug}")

# Step 3: Build unique set, preserving drug-attribution list per paper
# Pick first non-empty file path as canonical
content_hashes = {}
empty_skipped = 0
canonical_count = 0

for pmid, rows in by_pmid.items():
    drugs = list(dict.fromkeys(r["drug_name"] for r in rows))  # preserve order, dedup
    paths = [r["file_path"] for r in rows]

    # Find a non-empty file path
    canonical_path = None
    canonical_text = None
    for fp in paths:
        p = Path(fp)
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8", errors="ignore").strip()
        if len(text) >= 50:
            canonical_path = fp
            canonical_text = text
            break

    if canonical_path is None:
        empty_skipped += 1
        continue

    canonical_count += 1
    h = hashlib.md5(canonical_text.encode("utf-8")).hexdigest()

    if h in content_hashes:
        # Merge drug attributions from the duplicate
        existing = content_hashes[h]
        for d in drugs:
            if d not in existing["drugs"]:
                existing["drugs"].append(d)
        for fp in paths:
            if fp not in existing["all_paths"]:
                existing["all_paths"].append(fp)
    else:
        content_hashes[h] = {
            "pmid": pmid,
            "content_hash": h,
            "file_path": canonical_path,
            "all_paths": paths,  # All file paths for this PMID
            "drugs": drugs,  # PRESERVE full drug-attribution list
            "text_length": len(canonical_text),
        }

print(f"Empty/missing all paths: {empty_skipped}")
print(f"Unique papers (content-hash dedup): {len(content_hashes)}")
print(f"  (canonical_count={canonical_count}, dups merged={canonical_count - len(content_hashes)})")

# Step 4: Verify Sunitinib and Thalidomide
for drug_search in ["sunitinib", "thalidomide"]:
    matching = [
        u for u in content_hashes.values()
        if any(drug_search.lower() in d.lower() for d in u["drugs"])
    ]
    print(f"\n{drug_search}: {len(matching)} unique papers")
    for m in matching[:3]:
        print(f"  PMID {m['pmid']}: drugs={m['drugs']}")

# Step 5: Save
out = {
    "label": "unique_papers_for_reindex",
    "dedup_rules": [
        "Group by PMID (same paper, multiple drug attributions)",
        "Pick first non-empty file path as canonical",
        "Content-hash dedup catches papers with identical text under different PMIDs",
        "Drug-attribution list preserved per paper (all rows merged)",
    ],
    "count": len(content_hashes),
    "papers": list(content_hashes.values()),
}

with open(OUTPUT, "w") as f:
    json.dump(out, f, indent=2)

print(f"\nSaved {len(content_hashes)} unique papers to {OUTPUT}")

# Drug attribution distribution
from collections import Counter
attribution_dist = Counter(len(p["drugs"]) for p in content_hashes.values())
print(f"\nDrug attribution distribution:")
for n_drugs, count in sorted(attribution_dist.items())[:10]:
    print(f"  {n_drugs} drug(s): {count} papers")
