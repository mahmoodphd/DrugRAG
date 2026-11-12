#!/usr/bin/env python3
"""
V4 Re-run: Entity-boosted LightRAG evaluation (corrected plan).

Fixes from audit:
  1. Entity-name boost in keyword retrieval (in phase3_knowledge_graph.py)
  2. RAGAS: ALL 4 metrics via paid gpt-4o-mini judge (removes same-family bias)
  3. LightRAG top_k=3 (fixes context_precision 100% NaN from v3)
  4. Regenerate stale companion files (token_usage.csv, comparison_report.html)
  5. Save all outputs as v4

Reporting rule: All artifacts labeled 'intermediate'.
v3 and v4 results are NOT comparable as free-vs-paid.

Usage:
    .venv/bin/python rerun_v4_entity_boost.py
    .venv/bin/python rerun_v4_entity_boost.py --skip-eval
    .venv/bin/python rerun_v4_entity_boost.py --skip-ragas
    .venv/bin/python rerun_v4_entity_boost.py --only-analysis
"""

import json
import sys
import time
import traceback
from pathlib import Path

# Unbuffered stdout for monitoring background runs
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd

import config
from phase4_evaluation import (
    DEFAULT_QUESTIONS,
    ALL_CONFIGS,
    NaiveRAGRetriever,
    generate_answer_baseline,
    generate_answer_naive_rag,
    generate_answer_lightrag,
    generate_comparison_report,
)


# ---------------------------------------------------------------------------
# Step 1: Full evaluation with entity-boosted retrieval
# ---------------------------------------------------------------------------
def run_v4_evaluation():
    """Run 600-row evaluation (30Q x 4 models x 5 configs)."""
    questions = DEFAULT_QUESTIONS
    models = config.EVAL_MODELS
    configs = ALL_CONFIGS
    client = config.get_openrouter_client()

    # Initialize naive RAG retriever (for Config B)
    retriever = NaiveRAGRetriever()

    results = []
    checkpoint_path = config.PHASE4_DIR / "eval_results_v4_checkpoint.json"
    final_path = config.PHASE4_DIR / "eval_results_v4.json"

    total_calls = len(questions) * len(models) * len(configs)
    print(f"\n{'='*70}")
    print(f"V4 EVALUATION: Entity-Boosted LightRAG")
    print(f"  Questions : {len(questions)}")
    print(f"  Models    : {len(models)}")
    print(f"  Configs   : {len(configs)}")
    print(f"  Total LLM calls: {total_calls}")
    print(f"{'='*70}")
    # Wait for clean rate limit window (no API calls in last 60s)
    print("\n  Waiting 70s for clean rate limit window...")
    time.sleep(70)

    call_count = 0
    errors = 0
    start_time = time.time()

    for q in questions:
        question = q["question"]
        q_id = q["id"]
        tier = q.get("tier", "unknown")
        print(f"\n  [Q{q_id}] {question[:70]}...")

        for model in models:
            for cfg in configs:
                call_count += 1
                model_short = model.split("/")[-1].split(":")[0][:20]

                # Retry with exponential backoff for rate limits
                ans = None
                last_error = None
                for attempt in range(4):  # up to 4 attempts
                    try:
                        if cfg == "baseline":
                            ans = generate_answer_baseline(client, question, model)
                        elif cfg == "naive_rag":
                            ans = generate_answer_naive_rag(
                                client, question, model, retriever
                            )
                        elif cfg == "lightrag_local":
                            ans = generate_answer_lightrag(question, "local", top_k=3)
                        elif cfg == "lightrag_global":
                            ans = generate_answer_lightrag(question, "global", top_k=3)
                        elif cfg == "lightrag_hybrid":
                            ans = generate_answer_lightrag(question, "hybrid", top_k=3)
                        else:
                            break
                        break  # success, exit retry loop

                    except Exception as e:
                        last_error = e
                        err_str = str(e)
                        if "429" in err_str and attempt < 3:
                            # Rate limit: wait for reset (typically 60s)
                            wait = 70
                            print(f"    [{call_count}/{total_calls}] "
                                  f"429 rate limit, waiting {wait}s "
                                  f"(attempt {attempt+1}/4)...")
                            time.sleep(wait)
                        else:
                            break  # non-rate-limit error, don't retry

                if ans is not None:
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
                    status = "OK"
                    if "no relevant evidence" in ans["answer"].lower():
                        status = "NO-INFO"
                    print(f"    [{call_count}/{total_calls}] "
                          f"{model_short:20s} / {cfg:18s}: {status}")
                else:
                    errors += 1
                    err_msg = str(last_error)[:80] if last_error else "unknown"
                    print(f"    [{call_count}/{total_calls}] "
                          f"{model_short:20s} / {cfg:18s}: ERROR - {err_msg}")
                    results.append({
                        "question_id": q_id,
                        "question": question,
                        "tier": tier,
                        "model": model,
                        "config": cfg,
                        "answer": f"ERROR: {str(last_error)}",
                        "contexts": [],
                        "tokens": 0,
                    })

                # Rate limiting: 16 req/min → 5s = ~12 req/min
                time.sleep(7.0)

        # Checkpoint after each question (30 checkpoints total)
        checkpoint_path.write_text(
            json.dumps(results, indent=2, ensure_ascii=False)
        )

    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"Evaluation complete in {elapsed/60:.1f} minutes")
    print(f"  Total rows: {len(results)}")
    print(f"  Errors: {errors} ({errors/max(len(results),1)*100:.1f}%)")
    print(f"{'='*70}")

    # Save final results
    final_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Results saved to {final_path}")

    return results


# ---------------------------------------------------------------------------
# Step 2: RAGAS evaluation (ALL 4 metrics via paid gpt-4o-mini judge)
# ---------------------------------------------------------------------------
PAID_JUDGE_MODEL = "openai/gpt-4o-mini"

def run_v4_ragas(results=None):
    """Run RAGAS with ALL 4 metrics via paid gpt-4o-mini judge.

    Fixes from audit:
      - context_precision: top_k=3 (was 10, caused 100% NaN in v3)
      - context_recall: kept (was 65% NaN in v3 due to timeouts)
      - Paid judge removes same-family bias (free models judging free models)
      - gpt-4o-mini via OpenRouter: ~$0.95 for ~4500 calls
    """
    if results is None:
        results_path = config.PHASE4_DIR / "eval_results_v4.json"
        results = json.loads(results_path.read_text())

    # Filter out error and short results
    total_results = len(results)
    valid = [
        r for r in results
        if not r["answer"].startswith("ERROR")
        and len(r.get("answer", "").strip()) >= 50
    ]
    excluded_errors = sum(1 for r in results if r["answer"].startswith("ERROR"))
    excluded_short = total_results - len(valid) - excluded_errors

    print(f"\n{'='*70}")
    print(f"V4 RAGAS EVALUATION (all 4 metrics, paid judge)")
    print(f"{'='*70}")
    print(f"  Total results       : {total_results}")
    print(f"  Excluded (ERROR)    : {excluded_errors}")
    print(f"  Excluded (short)    : {excluded_short}")
    print(f"  Remaining for RAGAS : {len(valid)}")
    print(f"  Judge model         : {PAID_JUDGE_MODEL} (paid, independent)")
    print(f"  Metrics             : faithfulness, answer_relevancy,")
    print(f"                        context_precision (top_k=3), context_recall")

    if not valid:
        print("No valid results to evaluate.")
        return pd.DataFrame()

    # Load ground truth
    gt_lookup = {}
    if config.GROUND_TRUTH_PATH.exists():
        gt_data = json.loads(config.GROUND_TRUTH_PATH.read_text())
        for a in gt_data.get("answers", []):
            if not a.get("reference_answer", "").startswith("ERROR"):
                gt_lookup[a["question_id"]] = a["reference_answer"]
        print(f"  Ground truth loaded : {len(gt_lookup)} reference answers")

    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness, answer_relevancy,
            context_precision, context_recall,
        )
        from ragas.llms import LangchainLLMWrapper
        from langchain_openai import ChatOpenAI
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from datasets import Dataset

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
            q_id = r.get("question_id", 0)
            if q_id in gt_lookup:
                eval_data["reference"].append(gt_lookup[q_id])
            else:
                eval_data["reference"].append(r["question"])

        dataset = Dataset.from_dict(eval_data)

        # Paid model as RAGAS judge LLM (removes same-family bias)
        evaluator_llm = LangchainLLMWrapper(ChatOpenAI(
            model=PAID_JUDGE_MODEL,
            base_url=config.OPENROUTER_BASE_URL,
            api_key=config.OPENROUTER_API_KEY,
            temperature=0.3,
            timeout=300,  # longer timeout for paid model
        ))

        # Local HuggingFace embeddings (no API cost)
        _lc_embeddings = HuggingFaceEmbeddings(
            model_name=config.LOCAL_EMBEDDING_MODEL
        )
        evaluator_embeddings = LangchainEmbeddingsWrapper(_lc_embeddings)

        print(f"\nRunning RAGAS on {len(valid)} rows "
              f"(4 metrics: faithfulness, answer_relevancy, "
              f"context_precision, context_recall)...")
        print(f"  Judge model: {PAID_JUDGE_MODEL} (paid, independent)")
        print(f"  Embeddings: {config.LOCAL_EMBEDDING_MODEL} (local)")

        scores = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy,
                     context_precision, context_recall],
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
        )

        scores_df = scores.to_pandas()

        # Post-process NaN faithfulness
        nan_f = scores_df["faithfulness"].isna().sum()
        if nan_f > 0:
            print(f"  Filling {nan_f} NaN faithfulness values with 0.0")
            scores_df["faithfulness"] = scores_df["faithfulness"].fillna(0.0)

        # Add metadata columns
        scores_df["question_id"] = [r.get("question_id", 0) for r in valid]
        scores_df["config"] = [r.get("config", "unknown") for r in valid]
        scores_df["model"] = [r.get("model", "unknown") for r in valid]
        scores_df["tier"] = [r.get("tier", "unknown") for r in valid]

        scores_path = config.PHASE4_DIR / "ragas_scores_v4.csv"
        scores_df.to_csv(scores_path, index=False)
        print(f"\nRAGAS scores saved to {scores_path}")

        # Summary
        print(f"\nRAGAS Score Summary:")
        for metric in ["faithfulness", "answer_relevancy",
                        "context_precision", "context_recall"]:
            if metric in scores_df.columns:
                vals = pd.to_numeric(scores_df[metric], errors="coerce")
                print(f"  {metric}: mean={vals.mean():.3f}, "
                      f"median={vals.median():.3f}, "
                      f"NaN={vals.isna().sum()}/{len(vals)}")

        return scores_df

    except Exception as e:
        print(f"RAGAS evaluation failed: {e}")
        traceback.print_exc()
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Step 3: Regenerate companion files
# ---------------------------------------------------------------------------
def regenerate_companions(results=None):
    """Regenerate token_usage.csv and comparison_report.html from v4 data."""
    if results is None:
        results_path = config.PHASE4_DIR / "eval_results_v4.json"
        results = json.loads(results_path.read_text())

    df = pd.DataFrame(results)

    print(f"\n{'='*70}")
    print(f"REGENERATING COMPANION FILES")
    print(f"{'='*70}")

    # Token usage summary
    if "tokens" in df.columns:
        token_summary = df.groupby(["model", "config"])["tokens"].agg(
            ["mean", "sum", "count"]
        ).round(1)
        token_path = config.PHASE4_DIR / "token_usage.csv"
        token_summary.to_csv(token_path)
        print(f"  Token usage saved to {token_path}")

    # HTML report via existing function
    generate_comparison_report(results)
    print(f"  comparison_report.html regenerated")


# ---------------------------------------------------------------------------
# Step 4: Analysis + figures
# ---------------------------------------------------------------------------
def run_v4_analysis():
    """Run analysis on v4 results with bootstrap CIs."""
    results_path = config.PHASE4_DIR / "eval_results_v4.json"
    if not results_path.exists():
        print("No v4 results to analyze.")
        return None

    data = json.loads(results_path.read_text())
    df = pd.DataFrame(data)
    df["answer_length"] = df["answer"].str.len()
    df["is_error"] = df["answer"].str.startswith("ERROR", na=False)
    df["is_no_info"] = df["answer"].str.contains(
        "no relevant evidence|insufficient|does not contain|"
        "not possible to determine|no specific|not available|"
        "not contain any information",
        case=False, na=False,
    )

    valid = df[~df["is_error"]]
    print(f"\n{'='*70}")
    print(f"V4 ANALYSIS")
    print(f"{'='*70}")
    print(f"  Total rows   : {len(df)}")
    print(f"  Errors       : {df['is_error'].sum()}")
    print(f"  No-info      : {df['is_no_info'].sum()}")

    # Bootstrap CI helper
    def bootstrap_ci(values, n_boot=1000, ci=0.95, seed=42):
        arr = np.array(values, dtype=float)
        arr = arr[~np.isnan(arr)]
        if len(arr) < 2:
            m = arr.mean() if len(arr) else np.nan
            return m, m, m, len(arr)
        rng = np.random.RandomState(seed)
        means = [rng.choice(arr, size=len(arr), replace=True).mean()
                 for _ in range(n_boot)]
        alpha = (1 - ci) / 2
        return (arr.mean(),
                np.percentile(means, 100 * alpha),
                np.percentile(means, 100 * (1 - alpha)),
                len(arr))

    # Per-config summary
    configs = ["baseline", "naive_rag", "lightrag_local",
               "lightrag_global", "lightrag_hybrid"]
    config_labels = {
        "baseline": "A: Baseline (no retrieval)",
        "naive_rag": "B: Naive RAG (TF-IDF)",
        "lightrag_local": "C: LightRAG local (keyword)",
        "lightrag_global": "D: LightRAG global (keyword)",
        "lightrag_hybrid": "E: LightRAG hybrid (keyword)",
    }

    summary_rows = []
    for cfg in configs:
        cfg_data = valid[valid["config"] == cfg]
        if cfg_data.empty:
            continue
        n = len(cfg_data)
        n_errors = df[(df["config"] == cfg) & df["is_error"]].shape[0]
        n_no_info = cfg_data[cfg_data["is_no_info"]].shape[0]

        mean_len, ci_lo, ci_hi, _ = bootstrap_ci(cfg_data["answer_length"])
        no_info_flags = cfg_data["is_no_info"].astype(float).values
        ni_mean, ni_lo, ni_hi, _ = bootstrap_ci(no_info_flags)

        label = config_labels.get(cfg, cfg)
        print(f"\n  {label}")
        print(f"    N: {n} valid / {n_errors} errors")
        print(f"    Answer length: {mean_len:.0f} chars "
              f"(95% CI: {ci_lo:.0f}-{ci_hi:.0f})")
        print(f"    No-info rate: {ni_mean:.1%} "
              f"(95% CI: {ni_lo:.1%}-{ni_hi:.1%})")

        summary_rows.append({
            "config": cfg,
            "label": label,
            "n_valid": n,
            "n_errors": n_errors,
            "mean_answer_length": round(mean_len),
            "ci_length_lower": round(ci_lo),
            "ci_length_upper": round(ci_hi),
            "no_info_rate": round(ni_mean, 3),
            "ci_no_info_lower": round(ni_lo, 3),
            "ci_no_info_upper": round(ni_hi, 3),
        })

    # Context quality audit
    print(f"\n  Context Quality Audit:")
    for cfg in ["lightrag_local", "lightrag_global", "lightrag_hybrid"]:
        cfg_data = valid[valid["config"] == cfg]
        ctx_bug = 0
        no_retrieval = 0
        has_contexts = 0
        for _, row in cfg_data.iterrows():
            contexts = row.get("contexts", [])
            if isinstance(contexts, list) and len(contexts) > 0:
                if contexts[0] == row["answer"]:
                    ctx_bug += 1
                elif contexts == ["NO_KG_RETRIEVAL"]:
                    no_retrieval += 1
                else:
                    has_contexts += 1
        print(f"    {cfg}: {len(cfg_data)} rows, "
              f"ctx==answer BUG: {ctx_bug}, "
              f"no retrieval: {no_retrieval}, "
              f"proper contexts: {has_contexts}")

    # RAGAS integration
    ragas_metrics = ["faithfulness", "answer_relevancy",
                     "context_precision", "context_recall"]
    ragas_path = config.PHASE4_DIR / "ragas_scores_v4.csv"
    ragas_summary_rows = []
    if ragas_path.exists():
        ragas_df = pd.read_csv(ragas_path)
        print(f"\n  RAGAS v4: {len(ragas_df)} rows")
        for m in ragas_metrics:
            if m in ragas_df.columns:
                vals = pd.to_numeric(ragas_df[m], errors="coerce")
                print(f"    {m}: mean={vals.mean():.3f}, "
                      f"NaN={vals.isna().sum()}/{len(vals)}")

        # Per-config RAGAS with CIs
        print(f"\n  Per-Config RAGAS Scores:")
        for cfg in configs:
            cfg_ragas = ragas_df[ragas_df["config"] == cfg]
            if cfg_ragas.empty:
                continue
            label = config_labels.get(cfg, cfg)
            row_data = {"config": cfg, "label": label,
                        "n_ragas": len(cfg_ragas)}
            print(f"    {label} (n={len(cfg_ragas)})")
            for m in ragas_metrics:
                if m in cfg_ragas.columns:
                    vals = pd.to_numeric(cfg_ragas[m], errors="coerce")
                    mean, lo, hi, n_valid = bootstrap_ci(vals)
                    print(f"      {m}: {mean:.3f} "
                          f"(95% CI: {lo:.3f}-{hi:.3f}, n_valid={n_valid})")
                    row_data[f"{m}_mean"] = round(mean, 4)
                    row_data[f"{m}_ci_lower"] = round(lo, 4)
                    row_data[f"{m}_ci_upper"] = round(hi, 4)
                    row_data[f"{m}_n_valid"] = n_valid
            ragas_summary_rows.append(row_data)

        if ragas_summary_rows:
            ragas_summary_df = pd.DataFrame(ragas_summary_rows)
            ragas_summary_df.to_csv(
                config.PHASE4_DIR / "ragas_summary_v4.csv", index=False
            )

        # Caveats
        print(f"\n  CAVEATS:")
        print(f"    - Paid judge ({PAID_JUDGE_MODEL}) provides independent evaluation")
        print(f"    - context_precision uses top_k=3 (v3 used top_k=10, caused 100% NaN)")
        print(f"    - KG covers only 8.2% of papers; high no-info rate expected")
        print(f"    - v4 results are INTERMEDIATE (not final)")
        print(f"    - v3 and v4 are NOT comparable (different GT, bugs, methodology)")

        # Merge RAGAS into summary
        for srow in summary_rows:
            cfg = srow["config"]
            cfg_ragas = [r for r in ragas_summary_rows if r["config"] == cfg]
            if cfg_ragas:
                for m in ragas_metrics:
                    key = f"{m}_mean"
                    if key in cfg_ragas[0]:
                        srow[key] = cfg_ragas[0][key]

    # Save summary
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        config.PHASE4_DIR / "analysis_summary_v4.csv", index=False
    )
    print(f"\n  V4 summary saved to analysis_summary_v4.csv")

    return summary_df


def generate_v4_figures():
    """Generate figures from v4 results."""
    results_path = config.PHASE4_DIR / "eval_results_v4.json"
    if not results_path.exists():
        print("No v4 results for figures.")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams["figure.dpi"] = 150

    data = json.loads(results_path.read_text())
    df = pd.DataFrame(data)
    df["answer_length"] = df["answer"].str.len()
    df["is_error"] = df["answer"].str.startswith("ERROR", na=False)
    df["is_no_info"] = df["answer"].str.contains(
        "no relevant evidence|insufficient|does not contain|"
        "not possible to determine|no specific|not available|"
        "not contain any information",
        case=False, na=False,
    )
    valid = df[~df["is_error"]]

    config_order = ["baseline", "naive_rag", "lightrag_local",
                    "lightrag_global", "lightrag_hybrid"]
    config_labels = {
        "baseline": "A: Baseline",
        "naive_rag": "B: Naive RAG",
        "lightrag_local": "C: LR Local",
        "lightrag_global": "D: LR Global",
        "lightrag_hybrid": "E: LR Hybrid",
    }
    fig_dir = config.FIGURES_DIR

    # Figure 1: Answer length by config
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_data = valid[valid["config"].isin(config_order)].copy()
    plot_data["config_label"] = plot_data["config"].map(config_labels)
    order = [config_labels[c] for c in config_order]
    sns.boxplot(data=plot_data, x="config_label", y="answer_length",
                order=order, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Answer Length (chars)")
    ax.set_title("V4: Answer Length by Configuration")
    plt.tight_layout()
    fig.savefig(fig_dir / "v4_answer_length_by_config.png")
    plt.close(fig)
    print(f"  Saved v4_answer_length_by_config.png")

    # Figure 2: No-info rate by config
    fig, ax = plt.subplots(figsize=(10, 6))
    no_info_rates = []
    for cfg in config_order:
        cfg_data = valid[valid["config"] == cfg]
        if len(cfg_data) > 0:
            no_info_rates.append({
                "config": config_labels[cfg],
                "no_info_rate": cfg_data["is_no_info"].mean() * 100,
            })
    ni_df = pd.DataFrame(no_info_rates)
    if not ni_df.empty:
        sns.barplot(data=ni_df, x="config", y="no_info_rate", ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("No-Info Rate (%)")
    ax.set_title("V4: No-Info Rate by Configuration")
    ax.set_ylim(0, 100)
    plt.tight_layout()
    fig.savefig(fig_dir / "v4_no_info_rate_by_config.png")
    plt.close(fig)
    print(f"  Saved v4_no_info_rate_by_config.png")

    # Figure 3: Heatmap (config x model)
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot = valid.groupby(["config", "model"])["answer_length"].mean().reset_index()
    pivot["model_short"] = pivot["model"].apply(
        lambda x: x.split("/")[-1].split(":")[0][:15]
    )
    pivot["config_label"] = pivot["config"].map(config_labels)
    heatmap_data = pivot.pivot_table(
        index="config_label", columns="model_short",
        values="answer_length", fill_value=0
    )
    sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax)
    ax.set_title("V4: Mean Answer Length (Config x Model)")
    plt.tight_layout()
    fig.savefig(fig_dir / "v4_heatmap_config_model.png")
    plt.close(fig)
    print(f"  Saved v4_heatmap_config_model.png")

    # Figure 4: Tier breakdown
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    tier_labels = {
        "tier1_factual": "Tier 1: Factual",
        "tier2_mechanistic": "Tier 2: Mechanistic",
        "tier3_discovery": "Tier 3: Discovery",
    }
    for i, tier in enumerate(["tier1_factual", "tier2_mechanistic",
                               "tier3_discovery"]):
        tier_data = valid[valid["tier"] == tier].copy()
        if tier_data.empty:
            continue
        tier_data["config_label"] = tier_data["config"].map(config_labels)
        sns.boxplot(
            data=tier_data, x="config_label", y="answer_length",
            order=order, ax=axes[i]
        )
        axes[i].set_title(tier_labels.get(tier, tier))
        axes[i].set_xlabel("")
        axes[i].set_ylabel("Answer Length" if i == 0 else "")
    plt.suptitle("V4: Answer Length by Tier and Config", y=1.02)
    plt.tight_layout()
    fig.savefig(fig_dir / "v4_tier_breakdown.png")
    plt.close(fig)
    print(f"  Saved v4_tier_breakdown.png")

    # Figure 5: Context count distribution (LightRAG only)
    fig, ax = plt.subplots(figsize=(10, 6))
    lightrag_data = valid[valid["config"].str.startswith("lightrag")]
    if not lightrag_data.empty:
        ctx_counts = []
        for _, row in lightrag_data.iterrows():
            ctxs = row.get("contexts", [])
            if isinstance(ctxs, list) and ctxs != ["NO_KG_RETRIEVAL"]:
                ctx_counts.append({
                    "config": config_labels.get(row["config"], row["config"]),
                    "n_contexts": len(ctxs),
                })
        if ctx_counts:
            ctx_df = pd.DataFrame(ctx_counts)
            sns.boxplot(data=ctx_df, x="config", y="n_contexts", ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Number of Contexts")
    ax.set_title("V4: Context Count Distribution (LightRAG)")
    plt.tight_layout()
    fig.savefig(fig_dir / "v4_context_count_distribution.png")
    plt.close(fig)
    print(f"  Saved v4_context_count_distribution.png")

    # Figure 6: RAGAS scores (if available)
    ragas_path = config.PHASE4_DIR / "ragas_scores_v4.csv"
    all_ragas_metrics = ["faithfulness", "answer_relevancy",
                         "context_precision", "context_recall"]
    if ragas_path.exists():
        ragas_df = pd.read_csv(ragas_path)
        ragas_df["config_label"] = ragas_df["config"].map(config_labels)

        # Box plots (2x2 for 4 metrics)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes_flat = axes.flatten()
        for i, metric in enumerate(all_ragas_metrics):
            if metric in ragas_df.columns:
                plot_df = ragas_df.dropna(subset=[metric])
                if not plot_df.empty:
                    sns.boxplot(
                        data=plot_df, x="config_label", y=metric,
                        order=order, ax=axes_flat[i]
                    )
            axes_flat[i].set_title(f"V4: {metric.replace('_', ' ').title()}")
            axes_flat[i].set_xlabel("")
            axes_flat[i].set_ylim(0, 1.05)
        plt.tight_layout()
        fig.savefig(fig_dir / "v4_ragas_scores_by_config.png")
        plt.close(fig)
        print(f"  Saved v4_ragas_scores_by_config.png")

        # RAGAS heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        metrics_available = [m for m in all_ragas_metrics
                            if m in ragas_df.columns]
        if metrics_available:
            heatmap_data = ragas_df.groupby("config_label")[metrics_available].mean()
            heatmap_data = heatmap_data.reindex(
                [config_labels[c] for c in config_order]
            )
            heatmap_data.columns = [m.replace("_", " ").title()
                                   for m in metrics_available]
            sns.heatmap(heatmap_data, annot=True, fmt=".3f",
                       cmap="RdYlGn", vmin=0, vmax=1, ax=ax)
        ax.set_title("V4: RAGAS Scores Heatmap (Config x Metric)")
        plt.tight_layout()
        fig.savefig(fig_dir / "v4_ragas_heatmap.png")
        plt.close(fig)
        print(f"  Saved v4_ragas_heatmap.png")

    print(f"\nAll v4 figures saved to {fig_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="V4 re-run with entity boost")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip evaluation (use existing v4 results)")
    parser.add_argument("--skip-ragas", action="store_true",
                        help="Skip RAGAS evaluation")
    parser.add_argument("--only-analysis", action="store_true",
                        help="Only run analysis + figures (no eval/RAGAS)")
    args = parser.parse_args()

    print("=" * 70)
    print("  V4 RE-RUN: Entity-Boosted LightRAG Evaluation")
    print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    if args.only_analysis:
        run_v4_analysis()
        generate_v4_figures()
    else:
        # Step 1: Evaluation
        if not args.skip_eval:
            results = run_v4_evaluation()
        else:
            results_path = config.PHASE4_DIR / "eval_results_v4.json"
            results = json.loads(results_path.read_text())
            print(f"Loaded {len(results)} existing v4 results")

        # Step 2: Regenerate companion files
        regenerate_companions(results)

        # Step 3: RAGAS
        if not args.skip_ragas:
            run_v4_ragas(results)

        # Step 4: Analysis + figures
        run_v4_analysis()
        generate_v4_figures()

    print(f"\n{'='*70}")
    print("  V4 PIPELINE COMPLETE")
    print(f"{'='*70}")
