# Verification and reproduction tools

Run from the repository root with the virtual environment active.

## Checking claims and citations

- **`refsheet.py`** writes `data/phase5/reference_labels.json`: PhenoCluster meanings, the 13
  understudied compounds and the three highlighted candidates, knowledge-graph coverage, and the
  PubMed false-match outliers. The dashboard reads labels from this file.
- **`verify_claim.py`** `--csv P [--filter Q] [--col C --agg sum|mean|...] [--expect V]` checks a
  number against the data on disk and exits 1 on a mismatch.
- **`verify_citations.py`** `--file P | --text S | (stdin)` `[--pubmed]` extracts PMIDs and DOIs
  and reports whether each is in the local corpus index. `--pubmed` also asks PubMed whether an
  identifier that is not in the corpus exists at all.
- **`verify_bib.py`** `refs.bib [--online] [--meta]` checks that every BibTeX entry carries a
  resolvable identifier and, with `--meta`, that its authors and title match the real record.
- **`lint_thesis.py`** `[files...]` flags internal tokens (version tags, raw column names and
  similar) on reader-facing text such as the dashboard.
- **`dashboard_smoketest.py`** checks that each dashboard tab is wired to the right data.

## Scripts behind the paper's tables

- **`run_grounded_eval.py`** runs the 20-question grounded QA comparison (Table III counts and
  latency) and writes `data/phase4/eval_grounded_qa.csv` and `eval_grounded_qa_summary.json`.
- **`rerun_cold_faithfulness.py`** re-runs DrugRAG on the 20 questions with an empty response
  cache to measure cold-start latency, and updates `eval_grounded_qa_summary.json`.
- **`rebuttal_citation_relevance.py`** grades every cited PMID on four levels (exists in PubMed,
  vascular topic, mentions the queried entity, traceable to the corpus) and writes
  `data/phase4/rebuttal_citation_relevance.json` (Table III).
- **`rebuttal_external_pubmed_check.py`** checks the same citations against live PubMed records.
- **`rebuttal_ablation_stats.py`** computes the Wilson intervals in Table II, the Fisher test on
  question match, the knowledge-graph ablation and the weight-sensitivity sweep, and writes
  `data/phase5/tables/rebuttal_ablation_sensitivity.json`.
- **`rebuttal_kg_edge_provenance.py`** traces each edge of the NMDA receptor to endothelial cell
  path back to the abstract it came from (`data/phase3/rebuttal_kg_edge_provenance.json`).
- **`count_kg_surface_forms.py`** counts graph nodes whose names contain "VEGF" or
  "endothelial cell" (the 162 and 314 in the limitations section).
- **`export_kg_figure.py`** draws the knowledge-graph neighbourhood figure;
  **`capture_chat.py`** takes the dashboard screenshot.

The `rebuttal_` prefix marks analyses added during the paper's review round.
