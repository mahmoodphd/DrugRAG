#!/usr/bin/env python3
"""
verify_bib.py — the no-fake-citation gate for the BHI 2026 paper.

Every reference must carry a RESOLVABLE identifier (DOI, arXiv id, PMID, or a URL).
Offline mode (default) flags entries that have none. Online mode (--online) actually
pings doi.org / arxiv.org / PubMed and flags identifiers that do not resolve — i.e.
catches invented DOIs and made-up arXiv ids before they reach the paper.

Usage:
    python tools/verify_bib.py BHI_2026/references.bib            # structure check
    python tools/verify_bib.py BHI_2026/references.bib --online   # + confirm each resolves

Exit 0 = every entry has (and, with --online, resolves) an identifier; exit 1 otherwise.
This does NOT confirm a source supports a given sentence — that judgement stays with the
human/Claude review (playbook G1).
"""
import argparse
import json
import re
import sys
import time
import urllib.request
import urllib.error
import urllib.parse

HEAD = re.compile(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", re.DOTALL)
FIELD = re.compile(r"(\w+)\s*=\s*(?:\{(.*?)\}|\"(.*?)\")\s*,?", re.DOTALL)
ARXIV = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?")


def parse_bib(text):
    """Brace-counting parser tolerant of real-world .bib formatting."""
    out = []
    for m in HEAD.finditer(text):
        etype, key = m.group(1), m.group(2).strip()
        # capture the entry body by counting braces from just after the '{'
        start = text.index("{", m.start())
        depth, i = 0, start
        while i < len(text):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        body = text[start + 1:i]
        fields = {}
        for fm in FIELD.finditer(body):
            val = fm.group(2) if fm.group(2) is not None else (fm.group(3) or "")
            fields[fm.group(1).lower()] = val.strip()
        out.append((etype, key, fields))
    return out


def identifier(fields):
    """Return (kind, value) for the best resolvable id, or (None, None)."""
    if fields.get("doi"):
        return "doi", fields["doi"].replace("https://doi.org/", "").strip()
    if fields.get("pmid"):
        return "pmid", fields["pmid"].strip()
    if fields.get("eprint") or fields.get("archiveprefix", "").lower() == "arxiv":
        m = ARXIV.search(fields.get("eprint", "") or fields.get("note", ""))
        if m:
            return "arxiv", m.group(1)
    url = fields.get("url", "")
    m = ARXIV.search(url)
    if "arxiv.org" in url and m:
        return "arxiv", m.group(1)
    if fields.get("url"):
        return "url", fields["url"].strip()
    return None, None


def _get(url, timeout=12):
    req = urllib.request.Request(url, headers={"User-Agent": "verify-bib/1.0 (mailto:user@example.com)"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read(2000).decode("utf-8", "ignore")
    except urllib.error.HTTPError as e:
        return e.code, ""
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}"


def resolves(kind, value):
    """Online check that an identifier actually exists. Returns (ok, detail)."""
    if kind == "doi":
        st, _ = _get(f"https://api.crossref.org/works/{urllib.parse.quote(value)}")
        return (st == 200), f"crossref:{st}"
    if kind == "arxiv":
        st, body = _get(f"https://export.arxiv.org/abs/{value}")
        return (st == 200 and "no longer" not in body.lower()), f"arxiv:{st}"
    if kind == "pmid":
        st, body = _get(
            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={value}&retmode=json")
        ok = (st == 200 and f'"{value}"' in body and "error" not in body.lower())
        return ok, f"pubmed:{st}"
    if kind == "url":
        st, _ = _get(value)
        return (st in (200, 301, 302)), f"url:{st}"
    return False, "no-id"


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def _fetch_full(url):
    req = urllib.request.Request(url, headers={"User-Agent": "verify-bib/1.0 (mailto:user@example.com)"})
    try:
        with urllib.request.urlopen(req, timeout=25) as r:
            return r.read().decode("utf-8", "ignore")
    except Exception:  # noqa: BLE001
        return ""


def canonical_meta(kind, value):
    """Fetch (title, first-author-surname) from the authoritative record."""
    if kind == "doi":
        try:
            m = json.loads(_fetch_full(f"https://api.crossref.org/works/{urllib.parse.quote(value)}"))["message"]
            return (m.get("title") or [""])[0], (m.get("author") or [{}])[0].get("family", "")
        except Exception:  # noqa: BLE001
            return "", ""
    if kind == "pmid":
        try:
            d = json.loads(_fetch_full(
                f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={value}&retmode=json"))["result"][value]
            sfa = d.get("sortfirstauthor", "")
            return d.get("title", ""), (sfa.split()[0] if sfa else "")
        except Exception:  # noqa: BLE001
            return "", ""
    if kind == "arxiv":
        body = _fetch_full(f"http://export.arxiv.org/api/query?id_list={value}")
        ts = re.findall(r"<title>(.*?)</title>", body, re.S)
        am = re.search(r"<name>(.*?)</name>", body, re.S)
        return (ts[1].strip() if len(ts) > 1 else ""), (am.group(1).split()[-1] if am else "")
    return "", ""


def meta_matches(fields, kind, value):
    """True if the bib's first-author surname and title match the real record.
    Returns (ok, detail). Skips url-only entries (institutional authors)."""
    rt, rsur = canonical_meta(kind, value)
    if not rt and not rsur:
        return True, "no-meta-source"
    bibauth = fields.get("author", "").split(" and ")[0]
    bibsur = (bibauth.split(",")[0] if "," in bibauth else (bibauth.split()[-1] if bibauth.split() else ""))
    tt, rtn = _norm(fields.get("title", "")), _norm(rt)
    title_ok = bool(rt) and (tt[:40] in rtn or rtn[:40] in tt)
    auth_ok = (not rsur) or _norm(bibsur) == _norm(rsur) or (_norm(bibsur) and _norm(bibsur) in _norm(rsur))
    ok = title_ok and auth_ok
    return ok, f"bibAuth={bibsur} realAuth={rsur} titleOK={title_ok}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bib")
    ap.add_argument("--online", action="store_true", help="ping doi/arxiv/pubmed to confirm resolution")
    ap.add_argument("--meta", action="store_true",
                    help="also check author/title match the real record (catches fabricated metadata); implies --online")
    a = ap.parse_args()
    if a.meta:
        a.online = True

    text = open(a.bib, encoding="utf-8", errors="ignore").read()
    entries = parse_bib(text)
    if not entries:
        print("No bib entries found — wrong file?")
        sys.exit(1)

    fails = []
    print(f"Checking {len(entries)} entries in {a.bib}"
          + ("  (online)" if a.online else "  (structure only)") + "\n" + "=" * 60)
    for etype, key, fields in entries:
        kind, val = identifier(fields)
        if kind is None:
            print(f"  [✗] {key:<28} NO resolvable id (need doi/arxiv/pmid/url)")
            fails.append(key)
            continue
        if a.online:
            ok, detail = resolves(kind, val)
            mark = "✓" if ok else "✗"
            print(f"  [{mark}] {key:<28} {kind}:{val}  ({detail})")
            if not ok:
                fails.append(key)
            elif a.meta:
                m_ok, m_detail = meta_matches(fields, kind, val)
                if not m_ok:
                    print(f"        ↳ ✗ META MISMATCH — {m_detail} (fabricated author/title?)")
                    fails.append(key + " (meta)")
            time.sleep(0.34)  # be polite to the APIs
        else:
            print(f"  [✓] {key:<28} {kind}:{val}")

    print("=" * 60)
    if fails:
        print(f"{len(fails)} entry(ies) FAILED: {', '.join(fails)}")
        print("Fix or DELETE them — never invent an identifier (playbook G1).")
        sys.exit(1)
    print(f"All {len(entries)} entries carry"
          + (" a resolvable" if a.online else " an") + " identifier.")


if __name__ == "__main__":
    main()
