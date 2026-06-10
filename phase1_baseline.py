"""
Phase 1: LLM Baseline Benchmarking & Hallucination Audit.

Queries LLMs via OpenRouter in three modes (parametric, standard_search,
deep_search) for each drug in a subset. Extracts and verifies citations
against PubMed. Produces structured JSON responses and a hallucination report.

All outputs are saved incrementally for resume safety.
"""

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

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
# Retry policy for API calls
# ---------------------------------------------------------------------------
RETRY_POLICY = retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=60) + wait_random(0, 2),
    retry=retry_if_exception_type((TimeoutError, ConnectionError)),
    reraise=True,
)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
def build_prompt(drug_name: str, metadata: dict = None) -> str:
    """Build the standardized query prompt for a drug."""
    prompt = config.DRUG_QUERY_TEMPLATE.format(drug=drug_name)
    if metadata:
        context_parts = []
        if metadata.get("class") and metadata["class"] != "Unknown":
            context_parts.append(f"Drug class: {metadata['class']}")
        if metadata.get("action") and metadata["action"] != "Unknown":
            context_parts.append(f"Action: {metadata['action']}")
        if metadata.get("selectivity") and metadata["selectivity"] != "Unknown":
            context_parts.append(f"Target: {metadata['selectivity']}")
        if context_parts:
            prompt += "\n\nContext: " + "; ".join(context_parts)
    return prompt


# ---------------------------------------------------------------------------
# LLM querying
# ---------------------------------------------------------------------------
@RETRY_POLICY
def query_llm(client, drug_name: str, mode: str, model: str = None,
              metadata: dict = None) -> dict:
    """
    Query a single LLM for a drug in a given mode.

    For standard_search and deep_search modes, enables OpenRouter's server-side
    web search tool. For parametric mode, uses only the model's training knowledge.

    Returns a structured dict with response text, token usage, timing, and
    web search metadata.
    """
    model = model or config.DEFAULT_MODEL
    system_prompt = config.SYSTEM_PROMPTS[mode]
    user_prompt = build_prompt(drug_name, metadata)

    # Build API call kwargs
    api_kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": config.EVAL_TEMPERATURE,
        "max_tokens": config.PHASE1_MAX_TOKENS,
    }

    # Enable web search for standard_search and deep_search modes
    web_search_enabled = mode in config.WEB_SEARCH_MODES
    if web_search_enabled:
        if mode == "deep_search":
            api_kwargs["tools"] = [config.WEB_SEARCH_TOOL_DEEP]
        else:
            api_kwargs["tools"] = [config.WEB_SEARCH_TOOL_STANDARD]

    start_time = time.time()
    response = client.chat.completions.create(**api_kwargs)
    latency = time.time() - start_time

    # Extract response content
    choice = response.choices[0]
    response_text = choice.message.content or ""
    finish_reason = choice.finish_reason or "unknown"

    # Extract web search annotations (URL citations from search results)
    annotations_raw = getattr(choice.message, "annotations", None)
    annotations = []
    if annotations_raw:
        for ann in annotations_raw:
            ann_dict = {}
            if hasattr(ann, "url"):
                ann_dict["url"] = ann.url
            if hasattr(ann, "title"):
                ann_dict["title"] = ann.title
            if ann_dict:
                annotations.append(ann_dict)

    # Warn if response is empty despite successful API call
    if not response_text:
        tokens_used = response.usage.completion_tokens if response.usage else 0
        print(f"\n    WARNING: Empty response for {drug_name} ({mode}). "
              f"Completion tokens: {tokens_used}, finish_reason: {finish_reason}")

    result = {
        "drug": drug_name,
        "mode": mode,
        "model": model,
        "response_text": response_text,
        "tokens": {
            "prompt": response.usage.prompt_tokens if response.usage else 0,
            "completion": response.usage.completion_tokens if response.usage else 0,
            "total": response.usage.total_tokens if response.usage else 0,
        },
        "latency_sec": round(latency, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "web_search_enabled": web_search_enabled,
        "annotations": annotations if annotations else None,
        "n_web_sources": len(annotations),
        "finish_reason": finish_reason,
    }

    # Attach metadata if provided
    if metadata:
        result["drug_class"] = metadata.get("class", "Unknown")
        result["drug_action"] = metadata.get("action", "Unknown")
        result["drug_selectivity"] = metadata.get("selectivity", "Unknown")

    return result


# ---------------------------------------------------------------------------
# Citation extraction and verification
# ---------------------------------------------------------------------------
def extract_citations(text: str) -> list:
    """
    Extract paper citations from LLM response text using regex patterns.

    Looks for DOIs, paper titles in quotes, and author-year patterns.
    """
    citations = []

    # DOI pattern: 10.XXXX/...
    doi_pattern = r"10\.\d{4,9}/[^\s,;\"')\]]+"
    dois = re.findall(doi_pattern, text)
    for doi in dois:
        doi = doi.rstrip(".")  # Remove trailing period
        citations.append({"type": "doi", "value": doi})

    # Quoted title pattern: "Title of Paper"
    title_pattern = r'"([A-Z][^"]{10,200})"'
    titles = re.findall(title_pattern, text)
    for title in titles:
        # Filter out obvious non-titles (section headers, etc.)
        if not any(kw in title.lower() for kw in ["figure", "table", "section"]):
            citations.append({"type": "title", "value": title})

    return citations


def verify_citation_pubmed(citation: dict) -> dict:
    """
    Verify a citation against PubMed.

    Returns the citation dict augmented with verification status.
    """
    try:
        from Bio import Entrez
        Entrez.email = config.PUBMED_EMAIL

        if citation["type"] == "doi":
            query = f'{citation["value"]} [doi]'
        else:
            # Use first 80 chars of title for search
            query = f'"{citation["value"][:80]}"'

        handle = Entrez.esearch(db="pubmed", term=query, retmax=1)
        record = Entrez.read(handle)
        handle.close()

        time.sleep(config.PUBMED_RATE_LIMIT_SEC)

        if record["IdList"]:
            citation["verified"] = True
            citation["pmid"] = record["IdList"][0]
            citation["status"] = "verified_correct"
        else:
            citation["verified"] = False
            citation["status"] = "not_found"

    except Exception as e:
        citation["verified"] = None
        citation["status"] = f"verification_error: {str(e)[:100]}"

    return citation


# ---------------------------------------------------------------------------
# Hallucination audit
# ---------------------------------------------------------------------------
def run_hallucination_audit(responses: list) -> dict:
    """
    Audit LLM responses for hallucinations by verifying citations.

    Categorizes each response's citations into:
      - verified_correct: Citation exists in PubMed
      - not_found: Citation could not be verified (possible fabrication)
      - no_citations: Response contains no citable references

    Returns an audit summary dict.
    """
    audit = {
        "total_responses": len(responses),
        "by_mode": {},
        "hallucination_rate": {},
        "details": [],
    }

    for resp in responses:
        citations = extract_citations(resp["response_text"])

        # Verify each citation
        verified_citations = []
        for cit in citations:
            verified = verify_citation_pubmed(cit)
            verified_citations.append(verified)

        # Classify response
        n_total = len(verified_citations)
        n_verified = sum(1 for c in verified_citations if c.get("verified"))
        n_not_found = sum(1 for c in verified_citations if c.get("status") == "not_found")

        if n_total == 0:
            classification = "no_citations"
        elif n_not_found > 0:
            classification = "potential_hallucination"
        else:
            classification = "all_verified"

        detail = {
            "drug": resp["drug"],
            "mode": resp["mode"],
            "model": resp.get("model", ""),
            "n_citations": n_total,
            "n_verified": n_verified,
            "n_not_found": n_not_found,
            "classification": classification,
            "citations": verified_citations,
            "web_search_enabled": resp.get("web_search_enabled", False),
            "n_web_sources": resp.get("n_web_sources", 0),
            "annotations": resp.get("annotations"),
        }
        audit["details"].append(detail)

        # Accumulate per-mode stats
        mode = resp["mode"]
        if mode not in audit["by_mode"]:
            audit["by_mode"][mode] = {
                "total": 0, "verified": 0, "not_found": 0, "no_citations": 0,
                "web_search_used": 0, "total_web_sources": 0,
            }
        audit["by_mode"][mode]["total"] += 1
        audit["by_mode"][mode]["verified"] += n_verified
        audit["by_mode"][mode]["not_found"] += n_not_found
        if n_total == 0:
            audit["by_mode"][mode]["no_citations"] += 1
        if resp.get("web_search_enabled"):
            audit["by_mode"][mode]["web_search_used"] += 1
            audit["by_mode"][mode]["total_web_sources"] += resp.get("n_web_sources", 0)

    # Compute hallucination rates per mode
    for mode, stats in audit["by_mode"].items():
        total_with_citations = stats["total"] - stats["no_citations"]
        if total_with_citations > 0:
            audit["hallucination_rate"][mode] = round(
                stats["not_found"] / max(stats["verified"] + stats["not_found"], 1), 3
            )
        else:
            audit["hallucination_rate"][mode] = None
        # Average web sources for modes that use web search
        if stats.get("web_search_used", 0) > 0:
            stats["avg_web_sources"] = round(
                stats["total_web_sources"] / stats["web_search_used"], 1
            )

    # Compute web search effectiveness comparison
    ws_responses = [d for d in audit["details"] if d.get("web_search_enabled")]
    non_ws_responses = [d for d in audit["details"] if not d.get("web_search_enabled")]
    audit["web_search_effectiveness"] = {
        "web_search_responses": len(ws_responses),
        "non_web_search_responses": len(non_ws_responses),
        "web_search_avg_citations": (
            round(sum(d["n_citations"] for d in ws_responses) / max(len(ws_responses), 1), 1)
        ),
        "non_web_search_avg_citations": (
            round(sum(d["n_citations"] for d in non_ws_responses) / max(len(non_ws_responses), 1), 1)
        ),
    }

    return audit


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------
def load_existing_responses(path: Path) -> dict:
    """Load existing responses for resume safety. Returns {drug_mode: response}."""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        return {f"{r['drug']}|{r['mode']}|{r.get('model', '')}": r for r in data}
    except (json.JSONDecodeError, KeyError):
        return {}


def save_responses(responses: list, path: Path):
    """Save responses list to JSON."""
    path.write_text(json.dumps(responses, indent=2, ensure_ascii=False))


def run_baseline(subset_name: str = "pilot30", modes: list = None,
                 model: str = None, limit: int = None):
    """
    Run the Phase 1 baseline benchmarking.

    Args:
        subset_name: Drug subset to query (pilot30, pc4, pc7, pc2, all)
        modes: List of query modes (default: all three)
        model: OpenRouter model ID (default: from config)
        limit: Max drugs to process (None = all in subset)
    """
    modes = modes or config.EVAL_MODES
    model = model or config.DEFAULT_MODEL

    # Load data
    df = data_loader.load_raw_dataset()
    df = data_loader.clean_text_columns(df)
    unique_drugs = data_loader.extract_unique_drugs(df)
    drugs = data_loader.get_subset(unique_drugs, subset_name)

    if limit:
        drugs = drugs.head(limit)

    print(f"\nPhase 1: Querying {len(drugs)} drugs x {len(modes)} modes")
    print(f"  Model: {model}")
    print(f"  Total API calls: {len(drugs) * len(modes)}")
    ws_modes = [m for m in modes if m in config.WEB_SEARCH_MODES]
    if ws_modes:
        print(f"  Web search: ENABLED for {', '.join(ws_modes)}")
    else:
        print(f"  Web search: disabled")

    # Setup output
    output_path = config.PHASE1_DIR / "all_responses.json"
    existing = load_existing_responses(output_path)
    responses = list(existing.values())

    client = config.get_openrouter_client()

    # Process each drug
    processed = 0
    errors = 0
    for idx, row in drugs.iterrows():
        drug_name = row["compound_clean"]
        metadata = data_loader.get_drug_metadata_row(row)

        for mode in modes:
            key = f"{drug_name}|{mode}|{model}"
            if key in existing:
                continue

            try:
                ws_tag = "[WS] " if mode in config.WEB_SEARCH_MODES else ""
                print(f"  [{processed + 1}] {ws_tag}{drug_name} ({mode})...", end=" ")
                result = query_llm(client, drug_name, mode, model, metadata)
                responses.append(result)
                existing[key] = result
                processed += 1
                sources = result.get("n_web_sources", 0)
                sources_tag = f", {sources} web sources" if sources > 0 else ""
                print(f"OK ({result['tokens']['total']} tokens{sources_tag})")

                # Incremental save after each response
                save_responses(responses, output_path)

            except Exception as e:
                errors += 1
                print(f"ERROR: {str(e)[:80]}")
                # Save what we have so far
                save_responses(responses, output_path)

            # Rate limiting between calls
            time.sleep(0.5)

    print(f"\nPhase 1 complete: {processed} new responses, {errors} errors")
    print(f"Total responses saved: {len(responses)}")

    # Web search usage summary
    ws_count = sum(1 for r in responses if r.get("web_search_enabled"))
    ws_sources = sum(r.get("n_web_sources", 0) for r in responses)
    if ws_count > 0:
        print(f"Web search used in {ws_count} responses, {ws_sources} total web sources")

    # Run hallucination audit on new responses
    if responses:
        print("\nRunning hallucination audit...")
        audit = run_hallucination_audit(responses)
        audit_path = config.PHASE1_DIR / "hallucination_log.json"
        audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False))
        print(f"Audit saved to {audit_path}")

        # Print summary
        print("\nHallucination Audit Summary:")
        for mode, rate in audit["hallucination_rate"].items():
            rate_str = f"{rate:.1%}" if rate is not None else "N/A"
            stats = audit["by_mode"][mode]
            print(f"  {mode:20s}: hallucination rate = {rate_str} "
                  f"({stats['not_found']} unverified / "
                  f"{stats['verified']} verified, "
                  f"{stats['no_citations']} no citations)")

    return responses


if __name__ == "__main__":
    run_baseline()
