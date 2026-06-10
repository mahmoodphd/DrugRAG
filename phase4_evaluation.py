"""
Phase 4: Multi-LLM Evaluation Framework.

Compares 5 configurations across multiple LLM models:
  A: Baseline LLM (no retrieval)
  B: Naive RAG (TF-IDF retrieval)
  C: LightRAG local mode
  D: LightRAG global mode
  E: LightRAG hybrid mode

Uses RAGAS metrics (faithfulness, answer_relevancy, context_recall,
context_precision) and generates comparison reports.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import config
import data_loader


# ---------------------------------------------------------------------------
# Evaluation questions
# ---------------------------------------------------------------------------
DEFAULT_QUESTIONS = [
    # Tier 1 - Factual (local retrieval)
    {"id": 1, "tier": "tier1_factual", "target_drug": "SU 5416",
     "question": "What receptor does SU5416 inhibit and what is its role in angiogenesis?"},
    {"id": 2, "tier": "tier1_factual", "target_drug": "Sunitinib malate",
     "question": "Which VEGF receptor subtypes does Sunitinib target?"},
    {"id": 3, "tier": "tier1_factual", "target_drug": "DMH4",
     "question": "What is the mechanism of DMH4 as a VEGF inhibitor?"},
    {"id": 4, "tier": "tier1_factual", "target_drug": "Memantine",
     "question": "What type of glutamate receptor does Memantine antagonize?"},
    {"id": 5, "tier": "tier1_factual", "target_drug": "Y-27632",
     "question": "What kinase does Y-27632 inhibit and what is its effect on cell morphology?"},
    {"id": 6, "tier": "tier1_factual", "target_drug": "Suramin",
     "question": "What is the mechanism of action of Suramin as an anti-angiogenic agent?"},
    {"id": 7, "tier": "tier1_factual", "target_drug": "Thalidomide",
     "question": "How does Thalidomide affect blood vessel formation?"},
    {"id": 8, "tier": "tier1_factual", "target_drug": "2-methoxyestradiol",
     "question": "What is the anti-angiogenic mechanism of 2-methoxyestradiol?"},
    {"id": 9, "tier": "tier1_factual", "target_drug": "Minocycline",
     "question": "Does Minocycline have anti-angiogenic properties? What is the evidence?"},
    {"id": 10, "tier": "tier1_factual", "target_drug": "Roscovitine",
     "question": "What kinases does Roscovitine inhibit?"},
    # Tier 2 - Mechanistic (hybrid retrieval)
    {"id": 11, "tier": "tier2_mechanistic", "target_drug": "Glutamate antagonists",
     "question": "How does glutamate receptor inhibition affect endothelial tube formation?"},
    {"id": 12, "tier": "tier2_mechanistic", "target_drug": "VEGF inhibitors",
     "question": "What is the relationship between VEGF signaling and endothelial cell network formation?"},
    {"id": 13, "tier": "tier2_mechanistic", "target_drug": "NMDA receptors",
     "question": "How do NMDA receptor antagonists influence angiogenesis in vitro?"},
    {"id": 14, "tier": "tier2_mechanistic", "target_drug": "ROCK inhibitors",
     "question": "What is the role of ROCK signaling in endothelial tube morphogenesis?"},
    {"id": 15, "tier": "tier2_mechanistic", "target_drug": "Dopamine antagonists",
     "question": "How do dopamine receptor antagonists affect endothelial cell migration and branching?"},
    {"id": 16, "tier": "tier2_mechanistic", "target_drug": "GABA modulators",
     "question": "What is the connection between GABA signaling and vascular biology?"},
    {"id": 17, "tier": "tier2_mechanistic", "target_drug": "Serotonin antagonists",
     "question": "How do serotonin receptor antagonists influence angiogenic processes?"},
    {"id": 18, "tier": "tier2_mechanistic", "target_drug": "GRM5",
     "question": "What is the role of the GRM5 glutamate receptor in endothelial cells?"},
    {"id": 19, "tier": "tier2_mechanistic", "target_drug": "GRIN3A",
     "question": "How does GRIN3A expression relate to pro-angiogenic gene expression?"},
    {"id": 20, "tier": "tier2_mechanistic", "target_drug": "Cytoskeletal drugs",
     "question": "How do cytoskeletal inhibitors like ML-7 and ML-9 affect tube formation?"},
    # Tier 3 - Discovery (global retrieval)
    {"id": 21, "tier": "tier3_discovery", "target_drug": "LOPAC1280",
     "question": "Which LOPAC1280 drug classes modulate angiogenesis through neurotransmitter pathways?"},
    {"id": 22, "tier": "tier3_discovery", "target_drug": "LOPAC1280",
     "question": "Which drugs in the LOPAC library have phenotypic effects but lack angiogenesis literature?"},
    {"id": 23, "tier": "tier3_discovery", "target_drug": "Glutamate",
     "question": "What is the overall evidence that glutamate signaling plays a role in vascular biology?"},
    {"id": 24, "tier": "tier3_discovery", "target_drug": "LOPAC1280",
     "question": "Which PhenoCluster drugs show the strongest discrepancy between experimental effects and literature coverage?"},
    {"id": 25, "tier": "tier3_discovery", "target_drug": "Drug repurposing",
     "question": "What are the most promising drug repurposing candidates for anti-angiogenic therapy based on LOPAC1280 screening?"},
    {"id": 26, "tier": "tier3_discovery", "target_drug": "Alzheimer",
     "question": "How do glutamate receptor genes relate to both angiogenesis and Alzheimer's disease?"},
    {"id": 27, "tier": "tier3_discovery", "target_drug": "LOPAC1280",
     "question": "Which drug classes in LOPAC1280 are most underrepresented in angiogenesis research?"},
    {"id": 28, "tier": "tier3_discovery", "target_drug": "PhenoCluster 4",
     "question": "What biological pathways are shared by drugs in PhenoCluster 4 that enhance network connectivity?"},
    {"id": 29, "tier": "tier3_discovery", "target_drug": "PhenoCluster 7",
     "question": "What common mechanisms do PhenoCluster 7 drugs share that cause complete disruption of tube formation?"},
    {"id": 30, "tier": "tier3_discovery", "target_drug": "Endothelial",
     "question": "How does the Matrigel tube formation assay capture different aspects of endothelial cell behavior?"},
]


def load_eval_questions(questions_file: str = None) -> list:
    """Load evaluation questions from file or use defaults."""
    if questions_file:
        path = Path(questions_file)
        if path.suffix == ".json":
            return json.loads(path.read_text())
        # JSONL format
        questions = []
        for line in path.read_text().strip().split("\n"):
            if line.strip():
                questions.append(json.loads(line))
        return questions

    return DEFAULT_QUESTIONS


# ---------------------------------------------------------------------------
# Naive RAG baseline (TF-IDF)
# ---------------------------------------------------------------------------
class NaiveRAGRetriever:
    """Simple TF-IDF based retriever for baseline comparison."""

    def __init__(self, corpus_dir: Path = None):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words="english",
            ngram_range=(1, 2),
        )
        self.cosine_similarity = cosine_similarity
        self.documents = []
        self.tfidf_matrix = None
        self._load_corpus(corpus_dir or config.PAPERS_DIR)

    def _load_corpus(self, corpus_dir: Path):
        """Load all text files from corpus directory."""
        for subdir in [config.FULLTEXT_DIR, config.ABSTRACT_DIR]:
            if subdir.exists():
                for f in sorted(subdir.glob("*.txt")):
                    text = f.read_text(encoding="utf-8").strip()
                    if len(text) > 50:
                        self.documents.append(text)

        if self.documents:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)
            print(f"  Naive RAG: loaded {len(self.documents)} documents")
        else:
            print("  Naive RAG: no corpus documents found")

    def retrieve(self, query: str, top_k: int = 5) -> list:
        """Retrieve top-k most relevant document chunks."""
        if not self.documents or self.tfidf_matrix is None:
            return []

        query_vec = self.vectorizer.transform([query])
        similarities = self.cosine_similarity(query_vec, self.tfidf_matrix)[0]
        top_indices = similarities.argsort()[-top_k:][::-1]

        return [
            {"text": self.documents[i], "score": float(similarities[i])}
            for i in top_indices if similarities[i] > 0
        ]


# ---------------------------------------------------------------------------
# Answer generation across configurations
# ---------------------------------------------------------------------------
def generate_answer_baseline(client, question: str, model: str) -> dict:
    """Config A: Baseline LLM with no retrieval."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": config.SYSTEM_PROMPTS["parametric"]},
            {"role": "user", "content": question},
        ],
        temperature=config.EVAL_TEMPERATURE,
        max_tokens=2048,
    )
    return {
        "answer": response.choices[0].message.content or "",
        "contexts": [],
        "tokens": response.usage.total_tokens if response.usage else 0,
    }


def generate_answer_naive_rag(client, question: str, model: str,
                              retriever: NaiveRAGRetriever) -> dict:
    """Config B: Naive RAG with TF-IDF retrieval."""
    retrieved = retriever.retrieve(question, top_k=5)
    context = "\n\n---\n\n".join([r["text"][:1000] for r in retrieved])

    if not context:
        return generate_answer_baseline(client, question, model)

    prompt = (
        f"Based on the following literature excerpts, answer the question.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        f"Cite specific papers where possible."
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": config.SYSTEM_PROMPTS["standard_search"]},
            {"role": "user", "content": prompt},
        ],
        temperature=config.EVAL_TEMPERATURE,
        max_tokens=2048,
    )
    return {
        "answer": response.choices[0].message.content or "",
        "contexts": [r["text"][:500] for r in retrieved],
        "tokens": response.usage.total_tokens if response.usage else 0,
    }


def generate_answer_lightrag(question: str, mode: str, top_k: int = 10) -> dict:
    """Config C/D/E: LightRAG with keyword-based retrieval.

    Uses keyword-based KG retrieval (bypasses paid embedding API) to get
    relevant contexts, then synthesizes an answer using a free LLM.
    Contexts are the actual retrieved evidence (not the generated answer).

    Args:
        question: The evaluation question.
        mode: LightRAG query mode (local, global, hybrid).
        top_k: Number of contexts to retrieve (default 10, use 3 for RAGAS
               context_precision to avoid timeout with per-context LLM calls).
    """
    import phase3_knowledge_graph as kg

    # Step 1: Retrieve relevant contexts via keyword matching
    result = kg.query_kg_keyword(question, mode=mode, top_k=top_k)
    contexts = result.get("contexts", [])

    # Step 2: Generate answer from retrieved contexts using free LLM
    if contexts:
        answer = kg.generate_answer_from_contexts(question, contexts)
    else:
        answer = ("Based on the available knowledge graph, no relevant evidence "
                  "was found for this question.")

    return {
        "answer": answer,
        "contexts": contexts if contexts else ["NO_KG_RETRIEVAL"],
        "tokens": 0,
        "n_contexts": result.get("n_chunks", 0),
    }


# ---------------------------------------------------------------------------
# Full evaluation run
# ---------------------------------------------------------------------------
ALL_CONFIGS = ["baseline", "naive_rag", "lightrag_local",
               "lightrag_global", "lightrag_hybrid"]


def run_evaluation(questions: list = None, models: list = None,
                   questions_file: str = None, configs: list = None):
    """
    Run the full multi-config evaluation.

    Evaluates each question across specified configurations and multiple models.
    If configs is None, runs all 5 configurations.
    If configs is specified, only runs those configurations (e.g., LightRAG only).
    """
    questions = questions or load_eval_questions(questions_file)
    models = models or config.EVAL_MODELS
    configs = configs or ALL_CONFIGS
    client = config.get_openrouter_client()

    # Initialize naive RAG retriever only if needed
    retriever = None
    if "naive_rag" in configs:
        retriever = NaiveRAGRetriever()

    results = []

    total_calls = len(questions) * len(models) * len(configs)
    print(f"\nPhase 4: Multi-LLM Evaluation")
    print(f"  Questions: {len(questions)}")
    print(f"  Models: {len(models)} ({', '.join(models)})")
    print(f"  Configs: {len(configs)}")
    print(f"  Total LLM calls: {total_calls}")

    call_count = 0
    for q in questions:
        question = q["question"]
        q_id = q["id"]
        tier = q.get("tier", "unknown")

        print(f"\n  [Q{q_id}] {question[:70]}...")

        for model in models:
            for cfg in configs:
                call_count += 1
                try:
                    if cfg == "baseline":
                        ans = generate_answer_baseline(client, question, model)
                    elif cfg == "naive_rag":
                        ans = generate_answer_naive_rag(client, question, model,
                                                       retriever)
                    elif cfg == "lightrag_local":
                        ans = generate_answer_lightrag(question, "local")
                    elif cfg == "lightrag_global":
                        ans = generate_answer_lightrag(question, "global")
                    elif cfg == "lightrag_hybrid":
                        ans = generate_answer_lightrag(question, "hybrid")
                    else:
                        continue

                    results.append({
                        "question_id": q_id,
                        "question": question,
                        "tier": tier,
                        "model": model,
                        "config": cfg,
                        "answer": ans["answer"],
                        "contexts": ans.get("contexts", []),
                        "tokens": ans.get("tokens", 0),
                    })
                    print(f"    [{call_count}/{total_calls}] {model} / {cfg}: OK")

                except Exception as e:
                    print(f"    [{call_count}/{total_calls}] {model} / {cfg}: "
                          f"ERROR - {str(e)[:60]}")
                    results.append({
                        "question_id": q_id,
                        "question": question,
                        "tier": tier,
                        "model": model,
                        "config": cfg,
                        "answer": f"ERROR: {str(e)}",
                        "contexts": [],
                        "tokens": 0,
                    })

                time.sleep(0.3)

    # Save raw results
    results_path = config.PHASE4_DIR / "eval_results.json"
    results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nRaw results saved to {results_path}")

    return results


# ---------------------------------------------------------------------------
# Ground truth generation
# ---------------------------------------------------------------------------
def generate_ground_truth(questions: list = None, output_path: Path = None):
    """
    Generate expert-quality reference answers for evaluation questions.

    Uses the gold model (GPT-5.4) with corpus context and KG context
    to produce well-cited reference answers for RAGAS evaluation.

    Args:
        questions: List of question dicts (default: DEFAULT_QUESTIONS)
        output_path: Output file path (default: config.GROUND_TRUTH_PATH)
    """
    questions = questions or DEFAULT_QUESTIONS
    output_path = output_path or config.GROUND_TRUTH_PATH

    client = config.get_openrouter_client()
    retriever = NaiveRAGRetriever()

    # Decontaminated: NO KG context to prevent circular bias in RAGAS evaluation.
    # Ground truth uses ONLY corpus (TF-IDF) evidence, independent of KG retrieval.

    gt_prompt = (
        "You are an expert biomedical researcher writing reference answers "
        "for an evaluation benchmark. Given a research question and supporting "
        "evidence from the literature corpus, write a "
        "comprehensive 200-500 word reference answer.\n\n"
        "Requirements:\n"
        "- Be factual and precise, citing specific drugs, receptors, pathways\n"
        "- Include PMID numbers and DOIs where available from the evidence\n"
        "- If evidence is limited, state what is known and what is uncertain\n"
        "- Structure: direct answer first, then supporting evidence, then limitations\n\n"
    )

    answers = []
    for i, q in enumerate(questions):
        question = q["question"]
        q_id = q["id"]
        tier = q.get("tier", "unknown")

        print(f"\n  [{i + 1}/{len(questions)}] Q{q_id}: {question[:60]}...")

        # Gather evidence from corpus (TF-IDF)
        retrieved = retriever.retrieve(question, top_k=10)
        corpus_context = ""
        if retrieved:
            snippets = []
            for r in retrieved[:5]:
                text = r["text"][:600]
                snippets.append(f"[Score: {r['score']:.3f}] {text}")
            corpus_context = "\n\n".join(snippets)

        # NO KG context - decontaminated to prevent circular bias in RAGAS
        # (audit found 22/30 original answers used KG context that was then evaluated)

        # Build prompt with corpus-only evidence
        evidence_parts = []
        if corpus_context:
            evidence_parts.append(f"CORPUS EVIDENCE:\n{corpus_context}")
        if not evidence_parts:
            evidence_parts.append("No specific evidence found in the literature corpus.")

        user_content = (
            f"QUESTION: {question}\n\n"
            f"EVIDENCE:\n" + "\n\n".join(evidence_parts) + "\n\n"
            f"Write a comprehensive reference answer (200-500 words)."
        )

        try:
            response = client.chat.completions.create(
                model=config.VALIDATION_MODEL,
                messages=[
                    {"role": "system", "content": gt_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.3,
                max_tokens=1024,
            )
            answer_text = response.choices[0].message.content or ""
            print(f"    Generated {len(answer_text)} chars")

            answers.append({
                "question_id": q_id,
                "question": question,
                "tier": tier,
                "reference_answer": answer_text,
                "evidence_sources": len(retrieved),
                "kg_context_used": False,
            })

        except Exception as e:
            print(f"    ERROR: {str(e)[:80]}")
            answers.append({
                "question_id": q_id,
                "question": question,
                "tier": tier,
                "reference_answer": f"ERROR: {str(e)}",
                "evidence_sources": 0,
                "kg_context_used": False,
            })

        time.sleep(0.5)

    # Save ground truth
    gt_data = {
        "version": "1.0",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "method": "llm-assisted-expert",
        "model": config.VALIDATION_MODEL,
        "answers": answers,
    }
    output_path.write_text(json.dumps(gt_data, indent=2, ensure_ascii=False))
    print(f"\nGround truth saved to {output_path}")
    print(f"  {len(answers)} answers generated")
    errors = sum(1 for a in answers if a["reference_answer"].startswith("ERROR"))
    if errors:
        print(f"  WARNING: {errors} answers had errors")

    return gt_data


# ---------------------------------------------------------------------------
# RAGAS evaluation
# ---------------------------------------------------------------------------
def run_ragas_evaluation(results: list = None,
                        ground_truth_file: str = None) -> pd.DataFrame:
    """
    Run RAGAS evaluation metrics on the results.

    Computes: faithfulness, answer_relevancy, context_recall, context_precision.

    Args:
        results: List of evaluation result dicts (loads from file if None)
        ground_truth_file: Path to ground truth JSON with expert reference answers.
                          If None, falls back to question-as-proxy (deprecated).
    """
    if results is None:
        results_path = config.PHASE4_DIR / "eval_results.json"
        results = json.loads(results_path.read_text())

    # Filter out error and empty results (fix NaN faithfulness issue)
    total_results = len(results)
    valid = [r for r in results
             if not r["answer"].startswith("ERROR")
             and len(r.get("answer", "").strip()) >= 50]
    excluded_errors = sum(1 for r in results if r["answer"].startswith("ERROR"))
    excluded_short = sum(1 for r in results
                        if not r["answer"].startswith("ERROR")
                        and len(r.get("answer", "").strip()) < 50)

    if not valid:
        print("No valid results to evaluate.")
        return pd.DataFrame()

    # NaN Audit
    print(f"\nNaN Audit:")
    print(f"  Total results: {total_results}")
    print(f"  Excluded (ERROR): {excluded_errors}")
    print(f"  Excluded (too short <50 chars): {excluded_short}")
    print(f"  Remaining for RAGAS: {len(valid)}")

    # Load ground truth
    gt_lookup = {}
    if ground_truth_file:
        gt_path = Path(ground_truth_file)
        if gt_path.exists():
            gt_data = json.loads(gt_path.read_text())
            for a in gt_data.get("answers", []):
                if not a.get("reference_answer", "").startswith("ERROR"):
                    gt_lookup[a["question_id"]] = a["reference_answer"]
            print(f"  Ground truth loaded: {len(gt_lookup)} reference answers")
        else:
            print(f"  WARNING: Ground truth file not found: {ground_truth_file}")
    elif config.GROUND_TRUTH_PATH.exists():
        # Auto-load from default location
        gt_data = json.loads(config.GROUND_TRUTH_PATH.read_text())
        for a in gt_data.get("answers", []):
            if not a.get("reference_answer", "").startswith("ERROR"):
                gt_lookup[a["question_id"]] = a["reference_answer"]
        print(f"  Ground truth auto-loaded: {len(gt_lookup)} reference answers")

    if not gt_lookup:
        print("  WARNING: Using question-as-proxy for ground truth (deprecated)")

    print(f"\nRunning RAGAS evaluation on {len(valid)} results...")

    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        )
        from ragas.llms import LangchainLLMWrapper
        from langchain_openai import ChatOpenAI

        # Prepare RAGAS dataset
        eval_data = {
            "question": [],
            "answer": [],
            "contexts": [],
            "reference": [],
        }

        for r in valid:
            eval_data["question"].append(r["question"])
            eval_data["answer"].append(r["answer"])
            eval_data["contexts"].append(
                r["contexts"] if r["contexts"] else ["No context available"]
            )
            # Use real ground truth if available, otherwise fall back to question
            q_id = r.get("question_id", 0)
            if q_id in gt_lookup:
                eval_data["reference"].append(gt_lookup[q_id])
            else:
                eval_data["reference"].append(r["question"])

        from datasets import Dataset
        dataset = Dataset.from_dict(eval_data)

        # Use free model as RAGAS judge LLM
        evaluator_llm = LangchainLLMWrapper(ChatOpenAI(
            model=config.VALIDATION_MODEL,
            base_url=config.OPENROUTER_BASE_URL,
            api_key=config.OPENROUTER_API_KEY,
            temperature=0.3,
            timeout=config.RAGAS_TIMEOUT_SEC,
        ))

        # Use local HuggingFace embeddings (no API cost, runs on CPU)
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_community.embeddings import HuggingFaceEmbeddings
        _lc_embeddings = HuggingFaceEmbeddings(
            model_name=config.LOCAL_EMBEDDING_MODEL
        )
        evaluator_embeddings = LangchainEmbeddingsWrapper(_lc_embeddings)

        scores = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_recall,
                     context_precision],
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
        )

        scores_df = scores.to_pandas()

        # Post-process NaN values: fill faithfulness NaN with 0.0 for short answers
        nan_faithfulness = scores_df["faithfulness"].isna().sum()
        if nan_faithfulness > 0:
            print(f"  Filling {nan_faithfulness} NaN faithfulness values with 0.0")
            scores_df["faithfulness"] = scores_df["faithfulness"].fillna(0.0)

        # Add metadata columns for dashboard compatibility
        scores_df["question_id"] = [r.get("question_id", 0) for r in valid]
        scores_df["config"] = [r.get("config", "unknown") for r in valid]
        scores_df["model"] = [r.get("model", "unknown") for r in valid]
        scores_df["tier"] = [r.get("tier", "unknown") for r in valid]

        scores_path = config.PHASE4_DIR / "ragas_scores_v3.csv"
        scores_df.to_csv(scores_path, index=False)
        print(f"RAGAS scores saved to {scores_path}")

        # Print summary statistics
        print(f"\nRAGAS Score Summary:")
        for metric in ["faithfulness", "answer_relevancy",
                        "context_recall", "context_precision"]:
            if metric in scores_df.columns:
                vals = pd.to_numeric(scores_df[metric], errors="coerce")
                print(f"  {metric}: mean={vals.mean():.3f}, "
                      f"median={vals.median():.3f}, "
                      f"NaN={vals.isna().sum()}")

        return scores_df

    except Exception as e:
        print(f"RAGAS evaluation error: {str(e)}")
        print("  This may require an OpenAI API key for the evaluator LLM.")
        print("  You can run RAGAS evaluation manually later.")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def generate_comparison_report(results: list = None):
    """
    Generate comparison reports: CSV summary, HTML table, and figures.
    """
    if results is None:
        results_path = config.PHASE4_DIR / "eval_results.json"
        results = json.loads(results_path.read_text())

    df = pd.DataFrame(results)

    # --- Summary statistics ---
    print("\nGenerating comparison report...")

    # Token usage summary
    if "tokens" in df.columns:
        token_summary = df.groupby(["model", "config"])["tokens"].agg(
            ["mean", "sum", "count"]
        ).round(1)
        token_summary.to_csv(config.PHASE4_DIR / "token_usage.csv")

    # Answer length analysis
    df["answer_length"] = df["answer"].str.len()
    length_summary = df.groupby(["model", "config"])["answer_length"].agg(
        ["mean", "median"]
    ).round(0)

    # --- HTML report ---
    report_html = _build_html_report(df, length_summary)
    report_path = config.PHASE4_DIR / "comparison_report.html"
    report_path.write_text(report_html, encoding="utf-8")
    print(f"HTML report saved to {report_path}")

    # --- Figures ---
    _generate_figures(df)

    return df


def _build_html_report(df: pd.DataFrame, length_summary: pd.DataFrame) -> str:
    """Build an HTML comparison report."""
    html = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Phase 4 Evaluation Report</title>
<style>
body { font-family: Arial, sans-serif; margin: 40px; }
h1 { color: #2c3e50; }
h2 { color: #34495e; }
table { border-collapse: collapse; width: 100%; margin: 20px 0; }
th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
th { background-color: #3498db; color: white; }
tr:nth-child(even) { background-color: #f2f2f2; }
.metric { font-weight: bold; color: #2c3e50; }
</style></head><body>
<h1>Phase 4: Multi-LLM Evaluation Report</h1>
"""
    html += f"<p>Generated: {time.strftime('%Y-%m-%d %H:%M')}</p>"
    html += f"<p>Total evaluations: {len(df)}</p>"

    # Configuration comparison table
    html += "<h2>Configuration Comparison</h2>"
    html += length_summary.to_html(classes="table")

    # Error summary
    errors = df[df["answer"].str.startswith("ERROR", na=False)]
    if len(errors) > 0:
        html += f"<h2>Errors ({len(errors)})</h2>"
        html += errors[["model", "config", "answer"]].head(20).to_html(
            classes="table"
        )

    html += "</body></html>"
    return html


def _generate_figures(df: pd.DataFrame):
    """Generate comparison figures using matplotlib/seaborn."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Filter out errors for visualization
        valid = df[~df["answer"].str.startswith("ERROR", na=False)].copy()
        valid["answer_length"] = valid["answer"].str.len()

        if len(valid) == 0:
            return

        # Figure 1: Answer length by config
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=valid, x="config", y="answer_length", ax=ax)
        ax.set_title("Answer Length by Configuration")
        ax.set_xlabel("Configuration")
        ax.set_ylabel("Answer Length (characters)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(config.FIGURES_DIR / "answer_length_by_config.png", dpi=150)
        plt.close()

        # Figure 2: Answer length by model
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=valid, x="model", y="answer_length", ax=ax)
        ax.set_title("Answer Length by Model")
        ax.set_xlabel("Model")
        ax.set_ylabel("Answer Length (characters)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(config.FIGURES_DIR / "answer_length_by_model.png", dpi=150)
        plt.close()

        # Figure 3: Heatmap of config x model (mean answer length)
        if len(valid) > 0:
            pivot = valid.pivot_table(
                values="answer_length",
                index="config", columns="model", aggfunc="mean",
            )
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax)
            ax.set_title("Mean Answer Length: Config x Model")
            plt.tight_layout()
            plt.savefig(config.FIGURES_DIR / "heatmap_config_model.png", dpi=150)
            plt.close()

        print(f"Figures saved to {config.FIGURES_DIR}")

    except Exception as e:
        print(f"Figure generation error: {str(e)}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    results = run_evaluation()
    generate_comparison_report(results)
