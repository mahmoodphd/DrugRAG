# tools/ — shared utilities (Claude + Qoder)

Run from repo root with the venv active (`source .venv/bin/activate`). See `../CLAUDE.md`
for the conventions these enforce.

- **`refsheet.py`** — single source of truth for labels. `python tools/refsheet.py --json`
  writes `data/phase5/reference_labels.json` (cluster meanings, 13 TRUE-DARK drugs + 3
  centerpieces, KG coverage + case-study drugs, PubMed false-match outliers). The dashboard
  should read this JSON, not hardcode labels.
- **`verify_claim.py`** — `--csv P [--filter Q] [--col C --agg sum|mean|...] [--expect V]`.
  Check a number against on-disk data. Exit 1 on mismatch.
- **`verify_citations.py`** — `--file P | --text S | (stdin)` `[--pubmed]`. Extract PMIDs/DOIs →
  IN/NOT-IN the local corpus. `--pubmed` separates fabricated from just-not-in-corpus.
- **`lint_thesis.py`** — `[files...]` (default `dashboard/app.py`). Flags internal tokens on
  reader surfaces (version tags, decision IDs, raw column names, "search"-as-label,
  "misconfigured"). Skips internal scaffolding; ignores `.py` column access / paths. Exit 1 on hits.

Roadmap (not yet built): `dashboard_smoketest.py` (per-tab data wiring vs ground truth),
`doc_consistency.py` (canonical numbers match across docs+figures).
