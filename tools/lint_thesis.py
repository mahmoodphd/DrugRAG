#!/usr/bin/env python3
"""
lint_thesis.py — flag internal/jargon tokens on READER-FACING surfaces.

Scans the surfaces a reader actually sees — the dashboard UI strings and any
thesis/Space prose — for tokens that should never reach them: internal version
tags, decision IDs, raw column names, "search" used as a condition label, and
"misconfigured" on a presentation surface.

Internal scaffolding (DECISIONS.md, THESIS_ASSETS.md, MANIFEST.md,
master_project_plan.md) is NOT a reader surface — filenames, decision IDs and
formula column-names are legitimate there — so those are skipped entirely.

For .py, only DISPLAY strings are scanned (file paths and bare column keys like
df["score_v5_final"] are ignored), so column access isn't flagged.

Usage:
    python tools/lint_thesis.py                       # default: dashboard/app.py
    python tools/lint_thesis.py thesis_chapter4.md dashboard/rag_chat.py
"""
import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_TARGETS = ["dashboard/app.py"]
SKIP_FILES = {"DECISIONS.md", "THESIS_ASSETS.md", "MANIFEST.md", "master_project_plan.md"}

BANNED = [
    # v3/v4/v5 are the pipeline version tags; v2 is excluded (collides with infra
    # identifiers like MiniLM-L6-v2 / lightrag_working_v2).
    (re.compile(r"\bv[3-5]\b|\bv[3-5]_final\b", re.I), "internal version tag"),
    (re.compile(r"\bD\d{1,2}\b"), "internal decision ID"),
    (re.compile(r"\b(deep|standard|web)[ _-]?search\b", re.I), "'search' as a condition label"),
    (re.compile(r"misconfigur", re.I), "implementation detail on a reader surface"),
    (re.compile(r"score_v5_final|net_direction|activity_magnitude|kg_plausibility"
                r"|real_angio_pubmed_hits|is_active_dark\w*|novelty_status"), "raw column name"),
]

QUOTED = re.compile(r'"([^"]*)"|\'([^\']*)\'')
PATHISH = re.compile(r"\.(png|csv|json|txt|py|md|graphml|xlsx|pdf)\b", re.I)


def scannable(line, is_py):
    """For .py, return only display-string text (skip paths + bare column keys)."""
    if not is_py:
        return line
    segs = []
    for m in QUOTED.finditer(line):
        s = next(g for g in m.groups() if g is not None)
        if "/" in s or PATHISH.search(s):
            continue
        if re.fullmatch(r"[A-Za-z0-9_]+", s.strip()):  # bare identifier / column key
            continue
        segs.append(s)
    return " ".join(segs)


def lint(path):
    is_py = path.suffix == ".py"
    hits = []
    for i, line in enumerate(path.read_text(errors="ignore").splitlines(), 1):
        text = scannable(line, is_py)
        if not text:
            continue
        for pat, reason in BANNED:
            m = pat.search(text)
            if m:
                hits.append((str(path.relative_to(ROOT)), i, m.group(0), reason, line.strip()[:90]))
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    a = ap.parse_args()
    targets = [ROOT / f for f in (a.files or DEFAULT_TARGETS)]

    all_hits = []
    for t in targets:
        if t.name in SKIP_FILES:
            print(f"(skipping internal scaffolding: {t.name})")
            continue
        if t.exists():
            all_hits += lint(t)

    if not all_hits:
        print("clean — no banned tokens on reader-facing surfaces.")
        return
    for rel, i, tok, reason, ctx in all_hits:
        print(f"{rel}:{i}: '{tok}' — {reason}\n    {ctx}")
    print(f"\n{len(all_hits)} issue(s).")
    sys.exit(1)


if __name__ == "__main__":
    main()
