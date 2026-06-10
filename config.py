"""
Configuration hub for the Drug Repurposing LLM Pipeline.

Loads secrets from .env, defines all path constants, model lists,
rate limits, and query templates. Auto-creates directory structure on import.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
PUBMED_EMAIL = os.getenv("PUBMED_EMAIL", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")

# ---------------------------------------------------------------------------
# Directory structure (auto-created)
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_DIR / "data"
PHASE1_DIR = DATA_DIR / "phase1"
PHASE2_DIR = DATA_DIR / "phase2"
PHASE3_DIR = DATA_DIR / "phase3"
PHASE4_DIR = DATA_DIR / "phase4"

PAPERS_DIR = PHASE2_DIR / "papers"
FULLTEXT_DIR = PAPERS_DIR / "fulltext"
ABSTRACT_DIR = PAPERS_DIR / "abstract"
LIGHTRAG_DIR = PHASE3_DIR / "lightrag_working_v2"
FIGURES_DIR = PHASE4_DIR / "figures"
GROUND_TRUTH_PATH = PHASE4_DIR / "ground_truth_v2.json"

for _dir in [PHASE1_DIR, PHASE2_DIR, PHASE3_DIR, PHASE4_DIR,
             FULLTEXT_DIR, ABSTRACT_DIR, LIGHTRAG_DIR, FIGURES_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Dataset paths
# ---------------------------------------------------------------------------
DATASET_CSV = PROJECT_DIR / "complete_drug_dataset_with_paths_test.csv"

# ---------------------------------------------------------------------------
# OpenRouter configuration -- 5 paid + 5 free SOTA models
# ---------------------------------------------------------------------------
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# --- Paid models (selected for cost/performance across $50 budget) ---
PAID_MODELS = {
    # Workhorse: bulk Phase 1-3 queries (~70% of spend)
    "workhorse": {
        "id": "deepseek/deepseek-v4-pro",
        "input_per_M": 0.435,
        "output_per_M": 0.87,
        "context": 131072,
        "note": "Best cost/performance ratio for bulk generation",
    },
    # Evaluator: Phase 4 RAG evaluation (~20% of spend)
    "evaluator": {
        "id": "google/gemini-3.5-flash",
        "input_per_M": 1.50,
        "output_per_M": 9.00,
        "context": 1048576,
        "note": "SOTA evaluation judge, 1M context for long comparisons",
    },
    # Budget: fallback for high-volume low-complexity tasks
    "budget": {
        "id": "qwen/qwen3.6-flash",
        "input_per_M": 0.19,
        "output_per_M": 1.13,
        "context": 262144,
        "note": "Cheapest paid option for simple classification/summarization",
    },
    # Gold: selective validation of key findings (~10% of spend)
    "gold": {
        "id": "openai/gpt-5.4",
        "input_per_M": 2.50,
        "output_per_M": 15.00,
        "context": 192000,
        "note": "Gold standard for final validation of top discoveries",
    },
    # Validator: reserved for critical hallucination checks only
    "validator": {
        "id": "anthropic/claude-sonnet-4.6",
        "input_per_M": 3.00,
        "output_per_M": 15.00,
        "context": 1000000,
        "note": "Selective use only -- high quality validator with 1M context",
    },
}

# --- Free models (for development, testing, and fallback) ---
FREE_MODELS = {
    "free_large": {
        "id": "deepseek/deepseek-v4-flash:free",
        "context": 1000000,
        "note": "Largest free model, 1M context window",
    },
    "free_moe": {
        "id": "nvidia/nemotron-3-super-120b-a12b:free",
        "context": 1000000,
        "note": "120B MoE model, 1M context",
    },
    "free_reasoning": {
        "id": "moonshotai/kimi-k2.6:free",
        "context": 262144,
        "note": "Strong reasoning capabilities",
    },
    "free_google": {
        "id": "google/gemma-4-31b-it:free",
        "context": 262144,
        "note": "Latest Google open model, instruction-tuned",
    },
    "free_nano_reasoning": {
        "id": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
        "context": 256000,
        "note": "Reasoning-specialized 30B model",
    },
}

# Convenience lists
ALL_PAID_MODEL_IDS = [v["id"] for v in PAID_MODELS.values()]
ALL_FREE_MODEL_IDS = [v["id"] for v in FREE_MODELS.values()]

# Free-model RAGAS judge configuration (no API cost)
FREE_RAGAS_JUDGE = FREE_MODELS["free_moe"]["id"]        # nemotron-120B MoE
FREE_RAGAS_JUDGE_FALLBACK = FREE_MODELS["free_large"]["id"]  # deepseek-v4-flash

# Local embedding model (no API cost, runs on CPU)
LOCAL_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# RAGAS evaluation tuning (free models have stricter rate limits)
RAGAS_RATE_LIMIT_SEC = 2.0
RAGAS_TIMEOUT_SEC = 180

# Default model assignments per phase (free models to avoid paid credits)
DEFAULT_MODEL = FREE_MODELS["free_moe"]["id"]
EVAL_JUDGE_MODEL = FREE_RAGAS_JUDGE
VALIDATION_MODEL = FREE_RAGAS_JUDGE
DEV_MODEL = FREE_MODELS["free_moe"]["id"]  # For dry runs / testing

EVAL_MODELS = [
    FREE_MODELS["free_moe"]["id"],            # nemotron-120B MoE
    FREE_MODELS["free_reasoning"]["id"],      # kimi-k2.6
    FREE_MODELS["free_google"]["id"],         # gemma-4-31b-it
    FREE_MODELS["free_nano_reasoning"]["id"], # nemotron-nano-30b
]

# Budget allocation strategy ($50 total)
BUDGET_TOTAL_USD = 50.0
BUDGET_ALLOCATION = {
    "phase1_baseline": 10.0,   # ~1,278 drugs x 3 modes x workhorse
    "phase2_corpus": 0.0,      # No LLM cost (PubMed/S2 APIs only)
    "phase3_lightrag": 15.0,   # Indexing + batch queries
    "phase4_evaluation": 20.0,  # Multi-model RAGAS evaluation
    "validation_reserve": 5.0,  # Claude Opus for final checks
}

# ---------------------------------------------------------------------------
# PubMed / Semantic Scholar rate limits
# ---------------------------------------------------------------------------
# NCBI allows 3 req/sec without API key, 10 req/sec with key
PUBMED_RATE_LIMIT_SEC = 0.1 if NCBI_API_KEY else 0.4
PUBMED_MAX_RESULTS_PER_QUERY = 5
SEMANTIC_SCHOLAR_RATE_LIMIT_SEC = 3.0  # Free tier: 100 requests per 5 min

# ---------------------------------------------------------------------------
# PubMed query templates (used for each drug)
# ---------------------------------------------------------------------------
PUBMED_QUERY_TEMPLATES = [
    '"{drug}" AND (angiogenesis OR "blood vessel" OR "endothelial")',
    '"{drug}" AND ("cancer metastasis" OR metastatic OR "tumor spread")',
    '"{drug}" AND (VEGF OR "vascular endothelial" OR "tube formation")',
]

# Class-level fallback: used when individual drug search returns 0 results
PUBMED_CLASS_QUERY_TEMPLATE = (
    '"{drug_class}" AND "{drug_action}" AND '
    '(angiogenesis OR "blood vessel" OR endothelial)'
)

# ---------------------------------------------------------------------------
# LightRAG configuration
# ---------------------------------------------------------------------------
LIGHTRAG_INDEXING_MODEL = FREE_MODELS["free_google"]["id"]
LIGHTRAG_QUERY_MODEL = FREE_MODELS["free_moe"]["id"]
LIGHTRAG_EMBEDDING_MODEL = "text-embedding-3-small"  # Already indexed; keep for compat

ENTITY_TYPES = [
    "Drug",
    "Gene_Receptor",
    "Signaling_Pathway",
    "Disease",
    "Biological_Process",
    "Cell_Type",
    "Phenotype",
    "Experimental_Model",
]

RELATION_TYPES = [
    "INHIBITS",
    "ACTIVATES",
    "ASSOCIATED_WITH",
    "TREATS",
    "CAUSES",
    "EXPRESSED_IN",
    "MODULATES",
]

# ---------------------------------------------------------------------------
# Evaluation configuration
# ---------------------------------------------------------------------------
EVAL_SEED = 42
EVAL_TEMPERATURE = 0.3
EVAL_MODES = ["parametric", "standard_search", "deep_search"]

# Web search tool configurations for OpenRouter (server-side)
WEB_SEARCH_TOOL_STANDARD = {
    "type": "openrouter:web_search",
    "parameters": {
        "engine": "auto",
        "max_results": 5,
    },
}

WEB_SEARCH_TOOL_DEEP = {
    "type": "openrouter:web_search",
    "parameters": {
        "engine": "auto",
        "max_results": 10,
        "max_total_results": 25,
    },
}

WEB_SEARCH_MODES = ["standard_search", "deep_search"]
PHASE1_MAX_TOKENS = 4096  # Raised from 2048 to fix empty response bug

# Three-tier evaluation question categories
EVAL_TIERS = {
    "tier1_factual": {
        "description": "Specific entity-level facts (local retrieval)",
        "count": 20,
        "lightrag_mode": "local",
    },
    "tier2_mechanistic": {
        "description": "Multi-hop mechanistic reasoning (hybrid retrieval)",
        "count": 20,
        "lightrag_mode": "hybrid",
    },
    "tier3_discovery": {
        "description": "Cross-corpus synthesis and gap analysis (global retrieval)",
        "count": 20,
        "lightrag_mode": "global",
    },
}

# ---------------------------------------------------------------------------
# LLM system prompts for the three query modes
# ---------------------------------------------------------------------------
SYSTEM_PROMPTS = {
    "parametric": (
        "You are a biomedical research assistant. Answer using ONLY your "
        "training knowledge. Do not search the web or use external tools. "
        "Cite specific peer-reviewed papers with title, authors, journal, "
        "year, and DOI when possible."
    ),
    "standard_search": (
        "You are a biomedical research assistant with access to web search "
        "results. Use the provided search results to supplement your training "
        "knowledge. Cite specific sources with URLs and DOIs where available. "
        "Focus on the drug's documented role in angiogenesis and cancer metastasis."
    ),
    "deep_search": (
        "You are a biomedical research assistant performing an exhaustive "
        "literature review using web search. Thoroughly analyze all provided "
        "search results. Cross-reference multiple sources before synthesizing "
        "your answer. Cite all papers and web sources with full bibliographic "
        "details including DOIs and URLs."
    ),
}

# ---------------------------------------------------------------------------
# Query template for drug benchmarking
# ---------------------------------------------------------------------------
DRUG_QUERY_TEMPLATE = (
    "What is the documented role of the drug '{drug}' in angiogenesis "
    "or cancer metastasis? Provide key mechanistic details and cite specific "
    "peer-reviewed papers (include title, authors, journal, year, DOI). "
    "If no role is documented, state this explicitly."
)

# ---------------------------------------------------------------------------
# VEGF inhibitors for PhenoCluster 7 (from the foundational paper)
# ---------------------------------------------------------------------------
PC7_VEGF_INHIBITORS = [
    "SU5416", "SU 5416", "Semaxanib",
    "SU4312", "SU 4312",
    "DMH4",
    "Sunitinib", "Sunitinib malate",
    "Suramin",
]

# ---------------------------------------------------------------------------
# Spend tracking
# ---------------------------------------------------------------------------
SPEND_LIMIT_USD = BUDGET_TOTAL_USD
SPEND_LOG_PATH = PHASE1_DIR / "spend_log.json"


def get_openrouter_client():
    """Return a configured OpenAI client pointing at OpenRouter."""
    from openai import OpenAI
    return OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
    )


def print_config_summary():
    """Print a summary of the current configuration."""
    print("=" * 60)
    print("  Drug Repurposing Pipeline - Configuration Summary")
    print("=" * 60)
    print(f"  Project directory : {PROJECT_DIR}")
    print(f"  Dataset           : {DATASET_CSV}")
    print(f"  Data output       : {DATA_DIR}")
    key_preview = "***" + OPENROUTER_API_KEY[-8:] if OPENROUTER_API_KEY else "NOT SET"
    print(f"  OpenRouter key    : {key_preview}")
    print(f"  PubMed email      : {PUBMED_EMAIL or 'NOT SET'}")
    ncbi_status = "SET (10 req/sec)" if NCBI_API_KEY else "NOT SET (3 req/sec)"
    print(f"  NCBI API key      : {ncbi_status}")
    oa_status = "SET" if OPENAI_API_KEY else "NOT SET (embeddings will use fallback)"
    print(f"  OpenAI key        : {oa_status}")
    print(f"  Default model     : {DEFAULT_MODEL}")
    print(f"  Budget            : ${BUDGET_TOTAL_USD}")
    print(f"  Paid models       : {len(PAID_MODELS)} models")
    for role, info in PAID_MODELS.items():
        pricing = f"${info['input_per_M']}/${info['output_per_M']} per M"
        print(f"    [{role:10s}] {info['id']}  ({pricing})")
    print(f"  Free models       : {len(FREE_MODELS)} models")
    for role, info in FREE_MODELS.items():
        print(f"    [{role:20s}] {info['id']}")
    print(f"  Entity types      : {len(ENTITY_TYPES)}")
    print("=" * 60)
