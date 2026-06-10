"""Standalone, side-effect-free drug-name normalizer.

This module exists so other phases can import the canonical drug-name
normalization without triggering any heavy step1 work (step1 has top-level
KG-loading and writes data/phase4/normalization_remeasure.json on import,
which we want to keep out of phase 5).

Two functions:
    normalize_drug_name(name)         -- v1 (parity with step1)
    normalize_drug_name_v2(name)      -- v2 (adds ribose-position prime-stripping
                                            for nucleotide / nucleoside analogs)

Both return the case-preserved canonical core; lower-case it at the call site
when using as a join key.
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Salt / hydrate suffixes (parity with step1_normalize_remeasure.py)
# ---------------------------------------------------------------------------
SUFFIX_PATTERNS = [
    r"\bhydrochloride\b", r"\bdihydrochloride\b", r"\btrihydrochloride\b",
    r"\bhydrobromide\b", r"\bdihydrobromide\b",
    r"\bhydroiodide\b", r"\bdihydroiodide\b",
    r"\bdisodium\b", r"\bmonosodium\b",
    r"\bsodium\s+salt\b", r"\bpotassium\s+salt\b", r"\bcalcium\s+salt\b", r"\bammonium\s+salt\b",
    r"\bmaleate\b", r"\bfumarate\b", r"\bcitrate\b", r"\btartrate\b",
    r"\bsuccinate\b", r"\boxalate\b", r"\bphosphate\b",
    r"\bsulfate\b", r"\bsulphate\b", r"\bnitrate\b",
    r"\bbromide\b", r"\bchloride\b", r"\biodide\b",
    r"\bacetate\b", r"\bbenzoate\b",
    r"\bmesylate\b", r"\btosylate\b", r"\bbesylate\b",
    r"\boxalate\b", r"\bmalate\b", r"\bgluconate\b",
    r"\boleate\b", r"\bstearate\b", r"\bpalmitate\b",
    r"\blactate\b", r"\baspartate\b", r"\bglutamate\b",
    r"\bdecahydrate\b", r"\bheptahydrate\b", r"\bhexahydrate\b",
    r"\bpentahydrate\b", r"\btetrahydrate\b", r"\btrihydrate\b",
    r"\bdihydrate\b", r"\bmonohydrate\b", r"\banhydrous\b", r"\bhydrate\b",
    r"\bsodium\b", r"\bpotassium\b", r"\bsalt\b",
    r"\bfree\s+base\b", r"\bfree\s+acid\b",
]

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

# ---------------------------------------------------------------------------
# v2 addition: ribose-position prime stripping
# ---------------------------------------------------------------------------
# CRITICAL: MOESM5 / MOESM9 already pre-stripped the apostrophe character
# from prime-position labels but KEPT the digit (e.g. "5-Bromo-2-deoxyuridine"
# instead of "5-Bromo-2'-deoxyuridine"). The symmetric, lossless way to align
# both sides is therefore to simply remove the prime character (' U+0027 and
# ′ U+2032), not the surrounding digit. This preserves legitimate digits
# and only collapses an orthographic difference between the two sources.
PRIME_CHARS_RE = re.compile(r"[\'\u2032]")


def _common_strip(s: str) -> str:
    """Apply the v1 prefix + suffix stripping pipeline."""
    if not s:
        return ""
    s = s.strip()
    for pat in PREFIX_RES:
        s = pat.sub("", s)
    prev = None
    while prev != s:
        prev = s
        s = SUFFIX_RE.sub("", s).strip(" ,;-")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_drug_name(name: str) -> str:
    """v1 normalizer (parity with step1_normalize_remeasure.normalize_drug_name)."""
    return _common_strip(name)


def normalize_drug_name_v2(name: str) -> str:
    """v2 normalizer: v1 plus prime-character stripping for nucleotide /
    nucleoside analogs. Removes only the apostrophe / prime characters
    (U+0027 and U+2032) so that pipeline-side "5-Bromo-2'-deoxyuridine"
    aligns with MOESM5/MOESM9-side "5-Bromo-2-deoxyuridine". Symmetric,
    digit-preserving, no risk to non-nucleotide names.

    Examples:
        "5-Bromo-2'-deoxyuridine"             -> "5-Bromo-2-deoxyuridine"
        "3'-Azido-3'-deoxythymidine"          -> "3-Azido-3-deoxythymidine"
        "Adenosine 3',5'-cyclic monophosphate"-> "Adenosine 3,5-cyclic monophosphate"
        "2',3'-dideoxycytidine"               -> "2,3-dideoxycytidine"
        "P1,P4-Di(adenosine-5')tetraphosphate triammonium"
            -> "P1,P4-Di(adenosine-5)tetraphosphate triammonium"
    """
    if not name:
        return ""
    s = PRIME_CHARS_RE.sub("", name)
    return _common_strip(s)


__all__ = ["normalize_drug_name", "normalize_drug_name_v2"]
