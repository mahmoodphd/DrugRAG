#!/usr/bin/env python3
"""
dashboard_smoketest.py — verify the DrugRAG dashboard's data wiring & invariants.

Checks the things that have actually broken before, without needing a browser:
reference labels, leaderboard data, gap-map outlier exclusion, KG legend de-dup,
KG case-study drugs resolve, the never-empty KG path, the chat citation verifier,
and that the UI text has no banned tokens. Run locally for a green baseline and
again after deploy as the acceptance gate.

Usage:
    python tools/dashboard_smoketest.py
Exit 0 = all pass; exit 1 = at least one FAIL.
"""
import json
import pathlib
import subprocess
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "dashboard"))
REF = ROOT / "data" / "phase5" / "reference_labels.json"
V5 = ROOT / "data" / "phase5" / "drug_repurposing_candidates_v5_final.csv"

results = []


def check(name, fn):
    try:
        ok, detail = fn()
        results.append((name, "PASS" if ok else "FAIL", detail))
    except Exception as e:  # noqa: BLE001
        results.append((name, "FAIL", f"{type(e).__name__}: {e}"))


# 1) reference_labels.json structure
def c_ref():
    r = json.loads(REF.read_text())
    n_cl, n_dark, n_cp, n_out = (
        len(r["clusters"]), len(r["true_dark"]),
        len(r["active_dark_centerpieces"]), len(r["pubmed_false_match_outliers"]),
    )
    ok = n_cl == 7 and n_dark == 13 and n_cp == 3 and n_out >= 1
    return ok, f"clusters={n_cl} dark={n_dark} centerpieces={n_cp} outliers={n_out}"


# 2) leaderboard data loads with required columns
def c_v5():
    df = pd.read_csv(V5)
    need = {"drug_name", "cluster", "net_direction", "activity_magnitude",
            "score_v5_final", "real_angio_pubmed_hits", "real_total_pubmed_hits",
            "is_active_dark_real", "angio_edges"}
    miss = need - set(df.columns)
    return (len(df) == 1278 and not miss), f"rows={len(df)} missing={miss or 'none'}"


# 3) gap-map: the 305k-style outliers are excluded → axis can't be flattened
def c_outliers():
    df = pd.read_csv(V5)
    keep = df[df["real_angio_pubmed_hits"] <= df["real_total_pubmed_hits"]]
    mx = int(keep["real_angio_pubmed_hits"].max())
    return mx < 100000, f"max angio hits after excluding false-matches = {mx:,}"


# 4) KG legend: canonical types de-dupe (no 'Signaling Pathway' vs 'Signalingpathway')
def c_legend():
    import kg_utils
    labels = set(kg_utils.CANONICAL_TYPE.values())
    norm = {}
    for lbl in labels:
        key = lbl.lower().replace(" ", "").replace("/", "")
        norm.setdefault(key, []).append(lbl)
    dups = {k: v for k, v in norm.items() if len(v) > 1}
    return not dups, (f"{len(labels)} display labels, no near-duplicates"
                      if not dups else f"DUPLICATES: {dups}")


# 5) KG showcase builds a non-empty graph
def c_showcase():
    import kg_utils
    H = kg_utils.build_glutamate_showcase()
    return H.number_of_nodes() > 0, f"showcase nodes={H.number_of_nodes()} edges={H.number_of_edges()}"


# 6) known KG-rich drugs resolve to a node (chip clicks must not dead-end)
def c_resolve():
    import kg_utils
    must = ["Memantine", "Adenosine", "Sunitinib", "Histamine"]
    unresolved = [d for d in must if kg_utils.find_drug_node(d) is None]
    return not unresolved, ("all resolve" if not unresolved else f"UNRESOLVED: {unresolved}")


# 7) never-empty path: a true-dark drug has no node, and the placeholder exists
def c_never_empty():
    import kg_utils
    node = kg_utils.find_drug_node("Ziprasidone hydrochloride monohydrate")
    fig = kg_utils.empty_placeholder_figure("test")
    has_fig = hasattr(fig, "to_dict")
    return (node is None and has_fig), f"dark-drug node={node!r}, placeholder_figure={has_fig}"


# 8) chat citation verifier: in-corpus True, not-in-corpus False
def c_verify():
    import rag_chat
    v = rag_chat.verify_pmids(["28472639", "12441250"])
    return (v.get("28472639") is True and v.get("12441250") is False), str(v)


# 9) UI text has no banned tokens
def c_lint():
    p = subprocess.run([sys.executable, str(ROOT / "tools" / "lint_thesis.py"),
                        "dashboard/app.py"], cwd=ROOT, capture_output=True, text=True)
    return p.returncode == 0, p.stdout.strip().splitlines()[-1] if p.stdout else "ran"


for name, fn in [
    ("reference_labels structure", c_ref),
    ("leaderboard data + columns", c_v5),
    ("gap-map outliers excluded", c_outliers),
    ("KG legend de-duplicated", c_legend),
    ("KG showcase non-empty", c_showcase),
    ("KG case-study drugs resolve", c_resolve),
    ("KG never-empty for dark drug", c_never_empty),
    ("chat citation verifier", c_verify),
    ("UI text lint clean", c_lint),
]:
    check(name, fn)

print("DrugRAG dashboard smoke test\n" + "=" * 50)
for name, status, detail in results:
    mark = "✓" if status == "PASS" else "✗"
    print(f"  [{mark}] {name:<32} {detail}")
n_fail = sum(1 for _, s, _ in results if s == "FAIL")
print("=" * 50)
print(f"{len(results) - n_fail}/{len(results)} passed"
      + (f"  —  {n_fail} FAILED" if n_fail else "  —  all green"))
sys.exit(1 if n_fail else 0)
