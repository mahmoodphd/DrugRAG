"""
Phase 2: Literature Corpus Construction.

Automated PubMed + Semantic Scholar search for all 1,280 LOPAC1280 drugs.
Downloads abstracts and full texts where available. Maintains a corpus index
and flags "dark" drugs with no literature in angiogenesis/metastasis.

Resume-safe: checks corpus_index.csv before processing each drug.
Class-level fallback: when a drug has 0 individual results, searches by
its pharmacological Class + Action to find related literature.
"""

import csv
import json
import os
import time
from pathlib import Path
from xml.etree import ElementTree as ET

import requests
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    wait_random,
    retry_if_exception_type,
)

import config
import data_loader


# ---------------------------------------------------------------------------
# Entrez configuration helper
# ---------------------------------------------------------------------------
def _configure_entrez():
    """Set Entrez email and API key from config."""
    from Bio import Entrez
    Entrez.email = config.PUBMED_EMAIL
    if config.NCBI_API_KEY:
        Entrez.api_key = config.NCBI_API_KEY


# ---------------------------------------------------------------------------
# Retry policy
# ---------------------------------------------------------------------------
RETRY_POLICY = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30) + wait_random(0, 1),
    retry=retry_if_exception_type((requests.RequestException, ConnectionError)),
    reraise=True,
)

# ---------------------------------------------------------------------------
# Corpus index management
# ---------------------------------------------------------------------------
CORPUS_INDEX_FIELDS = [
    "drug_name", "pmid", "title", "doi", "source", "has_fulltext",
    "has_abstract", "retrieval_date", "file_path",
]


def load_corpus_index() -> list:
    """Load existing corpus index entries."""
    path = config.PHASE2_DIR / "corpus_index.csv"
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def get_indexed_drugs(corpus_index: list) -> set:
    """Return set of drug names already in the corpus index."""
    return {entry["drug_name"] for entry in corpus_index}


def append_to_corpus_index(entries: list):
    """Append entries to the corpus index CSV."""
    path = config.PHASE2_DIR / "corpus_index.csv"
    file_exists = path.exists()

    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CORPUS_INDEX_FIELDS)
        if not file_exists:
            writer.writeheader()
        for entry in entries:
            writer.writerow(entry)


# ---------------------------------------------------------------------------
# PubMed search
# ---------------------------------------------------------------------------
@RETRY_POLICY
def search_pubmed(drug_name: str) -> list:
    """
    Search PubMed for papers about a drug in angiogenesis/metastasis context.

    Returns list of PMIDs found across all query templates.
    """
    from Bio import Entrez
    _configure_entrez()

    all_pmids = set()

    for template in config.PUBMED_QUERY_TEMPLATES:
        query = template.format(drug=drug_name)
        try:
            handle = Entrez.esearch(
                db="pubmed", term=query,
                retmax=config.PUBMED_MAX_RESULTS_PER_QUERY,
            )
            record = Entrez.read(handle)
            handle.close()
            all_pmids.update(record["IdList"])
        except Exception:
            pass  # Skip failed queries, continue with next template

        time.sleep(config.PUBMED_RATE_LIMIT_SEC)

    return list(all_pmids)


@RETRY_POLICY
def search_pubmed_by_class(drug_class: str, drug_action: str) -> list:
    """
    Class-level fallback: search PubMed by pharmacological class + action.

    Used when an individual drug returns 0 results. Finds papers about
    the drug class in angiogenesis context, which can provide indirect
    evidence for the specific drug.

    Returns list of PMIDs.
    """
    from Bio import Entrez
    _configure_entrez()

    if not drug_class or drug_class == "Unknown":
        return []

    query = config.PUBMED_CLASS_QUERY_TEMPLATE.format(
        drug_class=drug_class,
        drug_action=drug_action if drug_action else "",
    )

    all_pmids = set()
    try:
        handle = Entrez.esearch(
            db="pubmed", term=query,
            retmax=config.PUBMED_MAX_RESULTS_PER_QUERY,
        )
        record = Entrez.read(handle)
        handle.close()
        all_pmids.update(record["IdList"])
    except Exception:
        pass

    time.sleep(config.PUBMED_RATE_LIMIT_SEC)
    return list(all_pmids)


@RETRY_POLICY
def fetch_pubmed_details(pmids: list) -> list:
    """Fetch title, abstract, DOI for a list of PMIDs."""
    from Bio import Entrez
    _configure_entrez()

    if not pmids:
        return []

    results = []
    # Batch fetch up to 10 at a time
    batch_size = 10
    for i in range(0, len(pmids), batch_size):
        batch = pmids[i:i + batch_size]
        try:
            handle = Entrez.efetch(
                db="pubmed", id=",".join(batch),
                rettype="xml", retmode="xml",
            )
            xml_data = handle.read()
            handle.close()

            root = ET.fromstring(xml_data)
            for article in root.findall(".//PubmedArticle"):
                pmid_el = article.find(".//PMID")
                title_el = article.find(".//ArticleTitle")
                abstract_el = article.find(".//AbstractText")
                doi_el = article.find('.//ArticleId[@IdType="doi"]')

                pmid = (pmid_el.text or "") if pmid_el is not None else ""
                title = (title_el.text or "") if title_el is not None else ""
                abstract = (abstract_el.text or "") if abstract_el is not None else ""
                doi = (doi_el.text or "") if doi_el is not None else ""

                results.append({
                    "pmid": pmid,
                    "title": title,
                    "abstract": abstract,
                    "doi": doi,
                })
        except Exception:
            pass

        time.sleep(config.PUBMED_RATE_LIMIT_SEC)

    return results


# ---------------------------------------------------------------------------
# Semantic Scholar search
# ---------------------------------------------------------------------------
@RETRY_POLICY
def search_semantic_scholar(drug_name: str) -> list:
    """
    Search Semantic Scholar for papers about a drug.

    Returns list of paper dicts with title, abstract, pmid, doi.
    """
    query = f"{drug_name} angiogenesis cancer"
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": 5,
        "fields": "title,abstract,externalIds,openAccessPdf",
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []

    time.sleep(config.SEMANTIC_SCHOLAR_RATE_LIMIT_SEC)

    papers = []
    for paper in data.get("data", []):
        ext_ids = paper.get("externalIds") or {}
        papers.append({
            "pmid": ext_ids.get("PubMed", ""),
            "title": paper.get("title", ""),
            "abstract": paper.get("abstract", "") or "",
            "doi": ext_ids.get("DOI", ""),
            "open_access_url": (paper.get("openAccessPdf") or {}).get("url", ""),
        })

    return papers


# ---------------------------------------------------------------------------
# Full-text retrieval (PMC Open Access)
# ---------------------------------------------------------------------------
@RETRY_POLICY
def fetch_pmc_fulltext(pmid: str) -> str:
    """
    Attempt to download full text from PMC Open Access.

    Returns the full text as a string, or empty string if unavailable.
    """
    # First check if the paper is in PMC
    from Bio import Entrez
    _configure_entrez()

    try:
        # Convert PMID to PMC ID
        handle = Entrez.elink(
            dbfrom="pubmed", db="pmc", id=pmid,
            linkname="pubmed_pmc",
        )
        links = Entrez.read(handle)
        handle.close()

        time.sleep(config.PUBMED_RATE_LIMIT_SEC)

        pmc_id = None
        for linkset in links:
            for linksetdb in linkset.get("LinkSetDb", []):
                for link in linksetdb.get("Link", []):
                    pmc_id = link.get("Id")
                    break

        if not pmc_id:
            return ""

        # Fetch full text XML from PMC
        url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_id}/xml/"
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200:
            return ""

        # Parse XML to extract text
        root = ET.fromstring(resp.content)
        paragraphs = []
        for p in root.iter("p"):
            text = "".join(p.itertext()).strip()
            if text:
                paragraphs.append(text)

        return "\n\n".join(paragraphs)

    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Paper saving
# ---------------------------------------------------------------------------
def save_paper_text(drug_name: str, pmid: str, text: str,
                    is_fulltext: bool = False) -> str:
    """Save paper text to the appropriate directory. Returns file path."""
    safe_name = drug_name.replace("/", "_").replace("\\", "_")[:40]
    filename = f"pmid_{pmid}_{safe_name}.txt"

    if is_fulltext:
        dir_path = config.FULLTEXT_DIR
    else:
        dir_path = config.ABSTRACT_DIR

    filepath = dir_path / filename
    filepath.write_text(text, encoding="utf-8")
    return str(filepath)


# ---------------------------------------------------------------------------
# Dark drug detection
# ---------------------------------------------------------------------------
def flag_dark_drugs(drug_list: list, corpus_index: list) -> list:
    """
    Identify drugs with zero literature in angiogenesis/metastasis context.

    Returns list of dark drug dicts with name, class, action.
    """
    indexed_drugs = get_indexed_drugs(corpus_index)
    dark_drugs = []

    for drug in drug_list:
        name = drug["compound_clean"]
        if name not in indexed_drugs:
            dark_drugs.append({
                "drug_name": name,
                "class": drug.get("Class", "Unknown"),
                "action": drug.get("Action", "Unknown"),
                "selectivity": drug.get("Selectivity", "Unknown"),
                "description": drug.get("Description", ""),
            })

    # Save dark drugs report
    dark_path = config.PHASE2_DIR / "dark_drugs.csv"
    if dark_drugs:
        import pandas as pd
        pd.DataFrame(dark_drugs).to_csv(dark_path, index=False)

    return dark_drugs


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------
def build_corpus(subset_name: str = "all", limit: int = None,
                 sources: str = "both"):
    """
    Build the literature corpus for all drugs.

    Args:
        subset_name: Drug subset to search (default: all)
        limit: Max drugs to process (None = all)
        sources: Search sources - "pubmed", "scholar", or "both"
    """
    # Load data
    df = data_loader.load_raw_dataset()
    df = data_loader.clean_text_columns(df)
    unique_drugs = data_loader.extract_unique_drugs(df)
    drugs = data_loader.get_subset(unique_drugs, subset_name)

    if limit:
        drugs = drugs.head(limit)

    # Load existing index for resume safety
    corpus_index = load_corpus_index()
    already_indexed = get_indexed_drugs(corpus_index)

    # Filter out already-processed drugs
    remaining = drugs[~drugs["compound_clean"].isin(already_indexed)]
    print(f"\nPhase 2: Building literature corpus")
    print(f"  Total drugs: {len(drugs)}")
    print(f"  Already indexed: {len(already_indexed)}")
    print(f"  Remaining: {len(remaining)}")
    print(f"  Sources: {sources}")
    ncbi_status = "with API key (10 req/sec)" if config.NCBI_API_KEY else "no API key (3 req/sec)"
    print(f"  NCBI access: {ncbi_status}")

    total_papers = 0
    total_errors = 0
    class_fallback_count = 0

    for idx, (_, row) in enumerate(remaining.iterrows()):
        drug_name = row["compound_clean"]
        print(f"\n  [{idx + 1}/{len(remaining)}] {drug_name}")

        all_pmids = set()

        # --- PubMed search ---
        if sources in ("pubmed", "both"):
            try:
                pmids = search_pubmed(drug_name)
                all_pmids.update(pmids)
                print(f"    PubMed: {len(pmids)} PMIDs")
            except Exception as e:
                print(f"    PubMed ERROR: {str(e)[:60]}")
                total_errors += 1

        # --- Semantic Scholar search ---
        if sources in ("scholar", "both"):
            try:
                s2_papers = search_semantic_scholar(drug_name)
                for p in s2_papers:
                    if p["pmid"]:
                        all_pmids.add(p["pmid"])
                print(f"    Semantic Scholar: {len(s2_papers)} papers")
            except Exception as e:
                print(f"    S2 ERROR: {str(e)[:60]}")

        # --- Class-level fallback (when individual search returns 0) ---
        if not all_pmids and sources in ("pubmed", "both"):
            drug_class = str(row.get("Class", ""))
            drug_action = str(row.get("Action", ""))
            if drug_class and drug_class != "Unknown":
                try:
                    class_pmids = search_pubmed_by_class(drug_class, drug_action)
                    if class_pmids:
                        all_pmids.update(class_pmids)
                        class_fallback_count += 1
                        print(f"    Class fallback ({drug_class}/{drug_action}): "
                              f"{len(class_pmids)} PMIDs")
                except Exception as e:
                    print(f"    Class fallback ERROR: {str(e)[:60]}")

        if not all_pmids:
            print(f"    No papers found (dark drug)")
            continue

        # --- Fetch paper details ---
        try:
            details = fetch_pubmed_details(list(all_pmids))
        except Exception as e:
            print(f"    Fetch error: {str(e)[:60]}")
            details = []

        # --- Download and save each paper ---
        new_entries = []
        for paper in details:
            pmid = paper["pmid"]
            has_fulltext = False
            has_abstract = bool(paper.get("abstract"))
            file_path = ""

            # Try full text first
            fulltext = fetch_pmc_fulltext(pmid)
            if fulltext and len(fulltext) > 200:
                file_path = save_paper_text(drug_name, pmid, fulltext,
                                            is_fulltext=True)
                has_fulltext = True
                print(f"    PMC fulltext: {pmid}")
            elif has_abstract:
                file_path = save_paper_text(drug_name, pmid, paper["abstract"],
                                            is_fulltext=False)
                print(f"    Abstract only: {pmid}")
            else:
                print(f"    No text available: {pmid}")
                continue

            new_entries.append({
                "drug_name": drug_name,
                "pmid": pmid,
                "title": (paper.get("title") or "")[:200],
                "doi": paper.get("doi") or "",
                "source": "pubmed",
                "has_fulltext": str(has_fulltext),
                "has_abstract": str(has_abstract),
                "retrieval_date": time.strftime("%Y-%m-%d"),
                "file_path": file_path,
            })

        # Save batch to index
        if new_entries:
            append_to_corpus_index(new_entries)
            total_papers += len(new_entries)
            print(f"    Saved {len(new_entries)} papers")

    # --- Final summary ---
    print(f"\n{'=' * 50}")
    print(f"Phase 2 Complete")
    print(f"  New papers added: {total_papers}")
    print(f"  Class-level fallback hits: {class_fallback_count}")
    print(f"  Errors: {total_errors}")

    # Reload full index and flag dark drugs
    full_index = load_corpus_index()
    drug_list = drugs.to_dict("records")
    dark = flag_dark_drugs(drug_list, full_index)
    print(f"  Dark drugs (no literature): {len(dark)}")
    print(f"  Total corpus entries: {len(full_index)}")
    print(f"{'=' * 50}")

    return full_index


if __name__ == "__main__":
    build_corpus(limit=10)
