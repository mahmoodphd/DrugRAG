"""
Regenerate ALL 16 canonical figures — local-only, no API calls.

Global style policy:
- No internal version tags, filenames, dirs, or source-path watermarks on figures.
- Axis/legend/colorbar labels: Title Case, no underscores, no code formulas.
- No text collisions: rotate ticks, declutter, use legends.
- Typo fixes: "Opiods"→"Opioids", "kinase phosphotase"→"Kinase/Phosphatase".

Outputs to data/phase4/figures/:
  Phase 1 (2):
    phase1_hallucination_audit.png
    phase1_response_quality.png
  Phase 2 (3):
    phase2_coverage_breakdown.png
    phase2_papers_per_drug.png
    phase2_dark_drugs_by_class.png
  Phase 3 (1):
    phase3_entity_distribution.png
  Phase 4 — v4 RAGAS (5):
    v4_answer_length_by_config.png
    v4_faithfulness_heatmap_by_tier.png
    v4_no_info_rate_by_config.png
    v4_ragas_heatmap.png
    v4_ragas_scores_by_config.png
  Phase 5 — v5_final (3):
    v5_final_gap_map.png
    v5_final_novelty_vs_kg_scatter.png
    v5_final_score_distribution.png
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT = Path(__file__).resolve().parent
FIG = PROJECT / "data" / "phase4" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

PUB_DPI = 200

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------

TYPO_MAP = {
    "Opiods": "Opioids",
    "kinase phosphotase": "Kinase/Phosphatase",
    "Kinase Phosphotase": "Kinase/Phosphatase",
    "Immune Cell Signaling": "Immune Cell Signalling",
}

def fix_typos(s: str) -> str:
    for k, v in TYPO_MAP.items():
        s = s.replace(k, v)
    return s


def apply_style():
    """Apply global matplotlib/seaborn style policy."""
    sns.set_theme(style="whitegrid", font_scale=1.0)
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": PUB_DPI,
        "figure.figsize": (10, 6),
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
    })


apply_style()

CONFIG_ORDER = ["baseline", "naive_rag", "lightrag_local",
                "lightrag_global", "lightrag_hybrid"]
CONFIG_LABELS = {
    "baseline": "Baseline",
    "naive_rag": "Naive RAG",
    "lightrag_local": "LightRAG Local",
    "lightrag_global": "LightRAG Global",
    "lightrag_hybrid": "LightRAG Hybrid",
}
TIER_LABELS = {
    "tier1_factual": "Tier 1: Factual",
    "tier2_mechanistic": "Tier 2: Mechanistic",
    "tier3_discovery": "Tier 3: Discovery",
}

PC7_CONTROLS = {
    "Sunitinib malate": "Sunitinib",
    "SU 5416": "SU 5416",
    "Suramin sodium": "Suramin",
    "DMH4": "DMH4",
    "Y-27632 dihydrochloride": "Y-27632",
}


# ============================================================================
# PHASE 1 — Baseline hallucination audit (D15 honest reconciliation)
# ============================================================================

def fig_phase1_hallucination_audit():
    h = json.loads((PROJECT / "data" / "phase1" / "hallucination_log.json").read_text())
    rates = h["hallucination_rate"]
    by_mode = h["by_mode"]

    # Single model (DeepSeek v4 Pro) under THREE PROMPT CONDITIONS that differ only in
    # how strongly the system prompt asks the model to cite sources. Web search was
    # misconfigured and never executed (by_mode[*]["total_web_sources"] == 0), so these
    # are citation-instruction variants, NOT retrieval depths. Order: increasing citation ask.
    modes = ["parametric", "standard_search", "deep_search"]
    labels = ["Parametric\n(closed-book)", "Cite-sources\nprompt", "Cite-sources prompt\n(intensive)"]
    rate_vals = [rates[m] for m in modes]
    verified = [by_mode[m]["verified"] for m in modes]
    not_found = [by_mode[m]["not_found"] for m in modes]
    cites = [v + nf for v, nf in zip(verified, not_found)]  # citation-level denominator

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 5.4))

    colors = ["#d6604d", "#92c5de", "#4393c3"]
    bars = ax1.bar(labels, rate_vals, color=colors, edgecolor="black", linewidth=0.5)
    for b, v, c in zip(bars, rate_vals, cites):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.012,
                 f"{v:.1%}\n({c} citations)", ha="center", va="bottom", fontsize=9.5)
    ax1.set_ylim(0, max(rate_vals) * 1.45)
    ax1.set_ylabel("Hallucination Rate\n(unverifiable / total citations)")
    ax1.set_title("Citation Hallucination Rate by Prompt Condition")
    ax1.set_xlabel("")

    # Right: stacked verified / not_found counts
    x = np.arange(len(modes))
    width = 0.55
    ax2.bar(x, verified, width, color="#5aae61", label="Verified (PubMed)", edgecolor="black", linewidth=0.4)
    ax2.bar(x, not_found, width, bottom=verified, color="#d6604d", label="Not Found", edgecolor="black", linewidth=0.4)
    for i, (v, nf) in enumerate(zip(verified, not_found)):
        ax2.text(i, v + nf + max(cites) * 0.02,
                 f"{v}/{v+nf}", ha="center", va="bottom", fontsize=9.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Number of Citations Checked")
    ax2.set_title("Citation Verification Against PubMed")
    ax2.legend(loc="upper left", framealpha=0.92)

    fig.suptitle(f"Baseline LLM Hallucination Audit  (DeepSeek v4 Pro, n = {h['total_responses']} responses)",
                 fontsize=12, y=0.99)
    fig.text(0.5, 0.02,
             "Single model, three prompt conditions, no live retrieval. Instructing the model to cite inflates "
             "citations (33 → 506)\nwhile ~20–24% fail PubMed verification — motivating grounded RAG. "
             "Closed-book cited in only 9 of 27 responses (small denominator).",
             ha="center", va="bottom", fontsize=7.6, style="italic", color="#444444")
    out = FIG / "phase1_hallucination_audit.png"
    plt.tight_layout(rect=[0, 0.08, 1, 0.94])
    plt.savefig(out, dpi=PUB_DPI, bbox_inches="tight")
    plt.close()
    print(f"[save] {out}")


def fig_phase1_response_quality():
    resp = json.loads((PROJECT / "data" / "phase1" / "all_responses.json").read_text())
    df = pd.DataFrame(resp)
    df["Completion Tokens"] = df["tokens"].apply(lambda d: d.get("completion", 0) if isinstance(d, dict) else 0)
    df["Latency (seconds)"] = df["latency_sec"].astype(float)
    mode_map = {"parametric": "Parametric\n(closed-book)", "standard_search": "Cite-sources\nprompt",
                "deep_search": "Cite-sources prompt\n(intensive)"}
    df["Mode"] = df["mode"].map(mode_map)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    order = ["Parametric\n(closed-book)", "Cite-sources\nprompt", "Cite-sources prompt\n(intensive)"]
    palette = {"Parametric\n(closed-book)": "#d6604d", "Cite-sources\nprompt": "#92c5de",
               "Cite-sources prompt\n(intensive)": "#4393c3"}

    sns.boxplot(data=df, x="Mode", y="Completion Tokens", order=order, hue="Mode",
                palette=palette, legend=False, ax=axes[0], width=0.55, fliersize=2.5, linewidth=0.8)
    axes[0].set_title("Response Length by Prompt Condition")
    axes[0].set_xlabel("")

    sns.boxplot(data=df, x="Mode", y="Latency (seconds)", order=order, hue="Mode",
                palette=palette, legend=False, ax=axes[1], width=0.55, fliersize=2.5, linewidth=0.8)
    axes[1].set_title("Latency by Prompt Condition")
    axes[1].set_xlabel("")
    axes[1].set_yscale("log")

    fig.suptitle(f"Phase 1 Response Quality  (DeepSeek v4 Pro, n = {len(df)} responses)",
                 fontsize=12, y=0.99)
    fig.text(0.5, 0.015,
             "One model, three prompt conditions, no live retrieval.",
             ha="center", va="bottom", fontsize=7.6, style="italic", color="#444444")
    out = FIG / "phase1_response_quality.png"
    plt.tight_layout(rect=[0, 0.06, 1, 0.94])
    plt.savefig(out, dpi=PUB_DPI, bbox_inches="tight")
    plt.close()
    print(f"[save] {out}")


# ============================================================================
# PHASE 2
# ============================================================================

def fig_phase2_coverage_breakdown():
    corpus = pd.read_csv(PROJECT / "data" / "phase2" / "corpus_index.csv")
    master = pd.read_csv(PROJECT / "data" / "phase1" / "drug_master_list.csv")
    # Join class from master via drug_name == compound_clean
    corpus = corpus.merge(master[["compound_clean", "Class"]].rename(columns={"compound_clean": "drug_name"}),
                          on="drug_name", how="left")
    corpus["Class"] = corpus["Class"].fillna("(Unknown)").apply(fix_typos)
    # Group by class
    by_class = corpus.groupby("Class").agg(
        n_drugs=("drug_name", "nunique"),
        n_papers=("pmid", "nunique"),
    ).sort_values("n_papers", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(by_class))
    w = 0.4
    ax.bar(x - w / 2, by_class["n_drugs"], w, label="Drugs", color="#4393c3", edgecolor="black", linewidth=0.4)
    ax.bar(x + w / 2, by_class["n_papers"], w, label="Papers", color="#92c5de", edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(by_class.index, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Count")
    ax.set_title("Corpus Coverage: Top 15 Drug Classes by Paper Count")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")

    out = FIG / "phase2_coverage_breakdown.png"
    plt.tight_layout()
    plt.savefig(out, dpi=PUB_DPI, bbox_inches="tight")
    plt.close()
    print(f"[save] {out}")


def fig_phase2_papers_per_drug():
    corpus = pd.read_csv(PROJECT / "data" / "phase2" / "corpus_index.csv")
    master = pd.read_csv(PROJECT / "data" / "phase1" / "drug_master_list.csv")
    # Papers per drug for the full 1,278-drug library (include 0-paper drugs)
    per_drug_corpus = corpus.groupby("drug_name")["pmid"].nunique()
    all_drugs = master["compound_clean"].unique()
    per_drug = pd.Series(0, index=all_drugs)
    per_drug.update(per_drug_corpus)
    per_drug = per_drug.sort_values(ascending=False)

    n_zero = (per_drug == 0).sum()
    n_total = len(per_drug)

    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.arange(-0.5, per_drug.max() + 1.5, 1)
    ax.hist(per_drug.values, bins=bins, color="#4c78a8", edgecolor="white", linewidth=0.3, alpha=0.92)
    med = per_drug[per_drug > 0].median()
    ax.axvline(med, color="black", ls=":", lw=1.2, alpha=0.7)
    ax.text(med + 0.5, ax.get_ylim()[1] * 0.85 if ax.get_ylim()[1] > 0 else 50,
            f"  Median (non-zero) = {med:.0f}", fontsize=9.5)

    # Annotate the 0-bin
    ax.annotate(
        f"{n_zero} literature-dark\n(0 papers in corpus)",
        xy=(0, n_zero), xytext=(3, n_zero + 15),
        fontsize=9, color="crimson", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="crimson", lw=1.2),
    )

    ax.set_xlabel("Number of Papers per Drug")
    ax.set_ylabel("Number of Drugs")
    ax.set_title(f"Literature Coverage Distribution  (full {n_total}-drug library)")

    # Caption note about retrieval cap
    ax.text(0.97, 0.95,
            "Note: spike at 5 papers reflects\nPubMed retrieval cap per drug",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
            style="italic", alpha=0.75,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))

    out = FIG / "phase2_papers_per_drug.png"
    plt.tight_layout()
    plt.savefig(out, dpi=PUB_DPI, bbox_inches="tight")
    plt.close()
    print(f"[save] {out}")


def fig_phase2_dark_drugs_by_class():
    v5f = pd.read_csv(PROJECT / "data" / "phase5" / "drug_repurposing_candidates_v5_final.csv")
    master = pd.read_csv(PROJECT / "data" / "phase1" / "drug_master_list.csv")

    td = v5f[v5f["novelty_status"] == "TRUE-DARK"].copy()
    j = td.merge(master[["compound_clean", "Class"]],
                 left_on="drug_name", right_on="compound_clean", how="left")
    if j["Class"].isna().any():
        miss = j["Class"].isna()
        j2 = j[miss].drop(columns=["Class", "compound_clean"]).merge(
            master[["compound_clean", "Class"]],
            left_on="canonical_name", right_on="compound_clean", how="left")
        j.loc[miss, "Class"] = j2["Class"].values
    j["Class"] = j["Class"].fillna("(Unmatched)").apply(fix_typos)

    counts = j["Class"].value_counts().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(9, 4.2))
    bars = ax.barh(counts.index, counts.values, color="#3b6791",
                   edgecolor="black", linewidth=0.4)
    for b, v in zip(bars, counts.values):
        ax.text(v + 0.05, b.get_y() + b.get_height() / 2,
                f"  {int(v)}", va="center", fontsize=10)
    ax.set_xlim(0, counts.max() + 1.2)
    ax.set_xlabel("Number of TRUE-DARK Drugs")
    ax.set_title("TRUE-DARK Drugs by Pharmacological Class  (n = 13)")

    out = FIG / "phase2_dark_drugs_by_class.png"
    plt.tight_layout()
    plt.savefig(out, dpi=PUB_DPI, bbox_inches="tight")
    plt.close()
    print(f"[save] {out}")


# ============================================================================
# PHASE 3 — KG entity distribution
# ============================================================================

def fig_phase3_entity_distribution():
    graphml = PROJECT / "data" / "phase3" / "lightrag_working_v2" / "graph_chunk_entity_relation.graphml"
    tree = ET.parse(graphml)
    ns = {"g": "http://graphml.graphdrawing.org/xmlns"}
    types = Counter()
    n_nodes = 0
    for n in tree.findall(".//g:node", ns):
        n_nodes += 1
        for d in n.findall("g:data", ns):
            if d.attrib.get("key") == "d1":
                raw_type = (d.text or "unknown").strip()
                types[raw_type] += 1
                break
    n_edges = len(tree.findall(".//g:edge", ns))

    # Separate "other" and "UNKNOWN" from named types → merge into one bar
    other_keys = [k for k in types if k.lower() in ("other", "unknown")]
    other_total = sum(types[k] for k in other_keys)
    named_types = Counter({k: v for k, v in types.items() if k.lower() not in ("other", "unknown")})

    # Top 11 named types + 1 combined "Other / Unclassified" = 12 bars
    top_named = named_types.most_common(11)
    # Add remaining named types to the "other" bucket
    remaining_named = sum(c for _, c in named_types.most_common()[11:])
    other_final = other_total + remaining_named

    # Build final list sorted descending
    all_bars = top_named + [("Other / Unclassified", other_final)]
    all_bars.sort(key=lambda x: x[1], reverse=True)

    # Clean labels: Title Case, replace underscores
    labels = [fix_typos(t.replace("_", " ").title()) if t != "Other / Unclassified" else t
              for t, _ in all_bars]
    vals = [c for _, c in all_bars]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    y_pos = range(len(labels))
    bars = ax.barh(y_pos, vals, color="#2c7fb8", edgecolor="black", linewidth=0.4)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.invert_yaxis()
    for b, v in zip(bars, vals):
        pct = v / n_nodes * 100
        ax.text(v + n_nodes * 0.005, b.get_y() + b.get_height() / 2,
                f"{v:,}  ({pct:.1f}%)", va="center", fontsize=9)
    ax.set_xlim(0, max(vals) * 1.18)
    ax.set_xlabel("Number of Entities")
    ax.set_title(f"Knowledge-Graph Entity Types  ({n_nodes:,} entities; {n_edges:,} relationships)")

    out = FIG / "phase3_entity_distribution.png"
    plt.tight_layout()
    plt.savefig(out, dpi=PUB_DPI, bbox_inches="tight")
    plt.close()
    print(f"[save] {out}  (types={len(types)})")


# ============================================================================
# PHASE 4 — v4 RAGAS figures
# ============================================================================

def _load_v4_eval():
    return json.loads((PROJECT / "data" / "phase4" / "eval_results_v4.json").read_text())


def _load_v4_ragas():
    return pd.read_csv(PROJECT / "data" / "phase4" / "ragas_scores_v4.csv")


def fig_v4_answer_length_by_config():
    data = _load_v4_eval()
    rows = []
    for entry in data:
        rows.append({
            "Config": CONFIG_LABELS.get(entry.get("config", ""), entry.get("config", "")),
            "Answer Length (tokens)": len(entry.get("answer", "").split()),
        })
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9, 5))
    order = [CONFIG_LABELS[c] for c in CONFIG_ORDER if c in CONFIG_LABELS]
    sns.boxplot(data=df, x="Config", y="Answer Length (tokens)", order=order,
                hue="Config", palette="Blues_d", legend=False, ax=ax, width=0.55)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.set_xlabel("")
    ax.set_title("Answer Length by Configuration")

    out = FIG / "v4_answer_length_by_config.png"
    plt.tight_layout()
    plt.savefig(out, dpi=PUB_DPI, bbox_inches="tight")
    plt.close()
    print(f"[save] {out}")


def fig_v4_context_count_distribution():
    data = _load_v4_eval()
    rows = []
    for entry in data:
        ctx = entry.get("contexts", entry.get("retrieved_contexts", []))
        rows.append({
            "Config": CONFIG_LABELS.get(entry.get("config", ""), entry.get("config", "")),
            "Context Count": len(ctx) if isinstance(ctx, list) else 0,
        })
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9, 5))
    order = [CONFIG_LABELS[c] for c in CONFIG_ORDER if c in CONFIG_LABELS]
    sns.boxplot(data=df, x="Config", y="Context Count", order=order,
                hue="Config", palette="Greens_d", legend=False, ax=ax, width=0.55)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.set_xlabel("")
    ax.set_title("Retrieved Context Count by Configuration")

    out = FIG / "v4_context_count_distribution.png"
    plt.tight_layout()
    plt.savefig(out, dpi=PUB_DPI, bbox_inches="tight")
    plt.close()
    print(f"[save] {out}")


def fig_v4_faithfulness_heatmap_by_tier():
    ragas = _load_v4_ragas()
    # Pivot: config × tier → mean faithfulness
    if "tier" not in ragas.columns:
        print("[skip] v4_faithfulness_heatmap_by_tier: 'tier' column not in ragas_scores_v4.csv")
        return
    pivot = ragas.pivot_table(index="config", columns="tier", values="faithfulness", aggfunc="mean")
    pivot = pivot.reindex(index=[c for c in CONFIG_ORDER if c in pivot.index])
    pivot.columns = [TIER_LABELS.get(c, c) for c in pivot.columns]
    pivot.index = [CONFIG_LABELS.get(c, c) for c in pivot.index]

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu", ax=ax,
                linewidths=0.5, cbar_kws={"label": "Mean Faithfulness"})
    ax.set_xlabel("Question Tier")
    ax.set_ylabel("")
    ax.set_title("Faithfulness: Configuration × Question Tier")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    out = FIG / "v4_faithfulness_heatmap_by_tier.png"
    plt.tight_layout()
    plt.savefig(out, dpi=PUB_DPI, bbox_inches="tight")
    plt.close()
    print(f"[save] {out}")


def fig_v4_no_info_rate_by_config():
    data = _load_v4_eval()
    from collections import defaultdict
    counts = defaultdict(lambda: {"total": 0, "no_info": 0})
    NO_INFO_MARKERS = ["no information", "i don't have", "cannot find", "no relevant"]
    for entry in data:
        cfg = entry.get("config", "")
        counts[cfg]["total"] += 1
        ans = entry.get("answer", "").lower()
        if any(m in ans for m in NO_INFO_MARKERS):
            counts[cfg]["no_info"] += 1

    cfgs = [c for c in CONFIG_ORDER if c in counts]
    rates = [counts[c]["no_info"] / counts[c]["total"] if counts[c]["total"] else 0 for c in cfgs]
    labels_cfg = [CONFIG_LABELS.get(c, c) for c in cfgs]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(labels_cfg, rates, color="#f4a582", edgecolor="black", linewidth=0.5)
    for b, v in zip(bars, rates):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005,
                f"{v:.1%}", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, max(rates) * 1.3 if rates else 0.5)
    ax.set_ylabel("\"No Information\" Rate")
    ax.set_title("Proportion of Responses Declining to Answer")
    ax.set_xticklabels(labels_cfg, rotation=30, ha="right")
    ax.set_xlabel("")

    out = FIG / "v4_no_info_rate_by_config.png"
    plt.tight_layout()
    plt.savefig(out, dpi=PUB_DPI, bbox_inches="tight")
    plt.close()
    print(f"[save] {out}")


def fig_v4_ragas_heatmap():
    ragas = _load_v4_ragas()
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    metric_labels = ["Faithfulness", "Answer Relevancy", "Context Precision", "Context Recall"]
    avail_metrics = [m for m in metrics if m in ragas.columns]
    avail_labels = [metric_labels[i] for i, m in enumerate(metrics) if m in ragas.columns]

    pivot = ragas.groupby("config")[avail_metrics].mean()
    pivot = pivot.reindex([c for c in CONFIG_ORDER if c in pivot.index])
    pivot.index = [CONFIG_LABELS.get(c, c) for c in pivot.index]
    pivot.columns = avail_labels

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu", ax=ax,
                linewidths=0.5, vmin=0, vmax=1,
                cbar_kws={"label": "Score (0\u20131)"})
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("RAGAS Metrics by Configuration")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    out = FIG / "v4_ragas_heatmap.png"
    plt.tight_layout()
    plt.savefig(out, dpi=PUB_DPI, bbox_inches="tight")
    plt.close()
    print(f"[save] {out}")


def fig_v4_ragas_scores_by_config():
    ragas = _load_v4_ragas()
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    metric_labels = {"faithfulness": "Faithfulness", "answer_relevancy": "Answer Relevancy",
                     "context_precision": "Context Precision", "context_recall": "Context Recall"}
    avail = [m for m in metrics if m in ragas.columns]

    fig, axes = plt.subplots(1, len(avail), figsize=(4 * len(avail), 5), sharey=True)
    if len(avail) == 1:
        axes = [axes]
    order = [c for c in CONFIG_ORDER if c in ragas["config"].unique()]
    palette = sns.color_palette("Blues_d", len(order))

    for i, (metric, ax) in enumerate(zip(avail, axes)):
        sns.boxplot(data=ragas[ragas["config"].isin(order)],
                    x="config", y=metric, order=order, hue="config",
                    palette=palette, legend=False, ax=ax, width=0.6)
        ax.set_xticklabels([CONFIG_LABELS.get(c, c) for c in order], rotation=40, ha="right", fontsize=8.5)
        ax.set_xlabel("")
        ax.set_ylabel(metric_labels[metric] if i == 0 else "")
        ax.set_title(metric_labels[metric])
        ax.set_ylim(-0.05, 1.05)

    fig.suptitle("RAGAS Scores by Configuration", fontsize=12, y=0.99)
    out = FIG / "v4_ragas_scores_by_config.png"
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out, dpi=PUB_DPI, bbox_inches="tight")
    plt.close()
    print(f"[save] {out}")


def fig_v4_tier_breakdown():
    ragas = _load_v4_ragas()
    if "tier" not in ragas.columns:
        print("[skip] v4_tier_breakdown: 'tier' column missing")
        return
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    avail = [m for m in metrics if m in ragas.columns]
    metric_labels = {"faithfulness": "Faithfulness", "answer_relevancy": "Answer Relevancy",
                     "context_precision": "Context Precision", "context_recall": "Context Recall"}

    # Mean score per config × tier for first available metric (faithfulness)
    pivot = ragas.pivot_table(index="tier", columns="config", values=avail[0], aggfunc="mean")
    tier_order = [t for t in ["tier1_factual", "tier2_mechanistic", "tier3_discovery"] if t in pivot.index]
    cfg_order = [c for c in CONFIG_ORDER if c in pivot.columns]
    pivot = pivot.loc[tier_order, cfg_order]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(cfg_order))
    width = 0.25
    colors = ["#4393c3", "#92c5de", "#f4a582"]

    for i, tier in enumerate(tier_order):
        vals = pivot.loc[tier].values
        bars = ax.bar(x + i * width - width, vals, width,
                      label=TIER_LABELS.get(tier, tier), color=colors[i],
                      edgecolor="black", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels([CONFIG_LABELS.get(c, c) for c in cfg_order], rotation=35, ha="right")
    ax.set_ylabel(metric_labels[avail[0]])
    ax.set_title(f"{metric_labels[avail[0]]} by Configuration and Question Tier")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left", framealpha=0.92)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.set_xlabel("")

    out = FIG / "v4_tier_breakdown.png"
    plt.tight_layout()
    plt.savefig(out, dpi=PUB_DPI, bbox_inches="tight")
    plt.close()
    print(f"[save] {out}")


# ============================================================================
# PHASE 5 — v5_final figures
# ============================================================================

def _load_v5f():
    return pd.read_csv(PROJECT / "data" / "phase5" / "drug_repurposing_candidates_v5_final.csv")


def fig_v5_final_gap_map():
    df = _load_v5f()

    fig, ax = plt.subplots(figsize=(10.5, 6.5))

    sc = ax.scatter(
        df["real_angio_pubmed_hits"] + 1,
        df["activity_magnitude"],
        c=df["score_v5_final"], cmap="viridis", alpha=0.50,
        s=14, edgecolors="none", zorder=2,
    )

    # TRUE-DARK 13 — red diamonds
    td = df[df["novelty_status"] == "TRUE-DARK"]
    ax.scatter(td["real_angio_pubmed_hits"] + 1, td["activity_magnitude"],
               marker="D", facecolors="none", edgecolors="crimson",
               linewidths=1.6, s=120, label=f"TRUE-DARK (n = {len(td)})", zorder=4)

    # PC7 controls — gold stars
    ctl = df[df["drug_name"].isin(PC7_CONTROLS.keys())].copy()
    ax.scatter(ctl["real_angio_pubmed_hits"] + 1, ctl["activity_magnitude"],
               marker="*", facecolors="gold", edgecolors="black",
               linewidths=0.7, s=240, label="PC7 Controls", zorder=5)

    for _, r in ctl.iterrows():
        ax.annotate(
            PC7_CONTROLS[r["drug_name"]],
            (r["real_angio_pubmed_hits"] + 1, r["activity_magnitude"]),
            xytext=(8, 5), textcoords="offset points",
            fontsize=8.5, fontweight="bold", color="black",
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
        )

    # Label only top-3 highest-activity TRUE-DARK (avoid clutter)
    td_top = td.nlargest(3, "activity_magnitude")
    for _, r in td_top.iterrows():
        name = r["drug_name"].split(" ")[0]  # first word only
        ax.annotate(
            name,
            (r["real_angio_pubmed_hits"] + 1, r["activity_magnitude"]),
            xytext=(8, -8), textcoords="offset points",
            fontsize=8, color="crimson",
            arrowprops=dict(arrowstyle="-", color="crimson", lw=0.4),
        )

    ax.set_xscale("log")
    ax.set_xlabel("Real PubMed Angiogenesis Hits  (uncapped, all-time)")
    ax.set_ylabel("Experimental Activity  (endothelial phenotype magnitude)")
    ax.set_title("Discovery Gap Map: Activity vs. Literature Coverage")
    ax.grid(True, alpha=0.25, linestyle="--")
    cbar = plt.colorbar(sc, ax=ax, pad=0.015)
    cbar.set_label("Repurposing Score", rotation=270, labelpad=14)
    ax.legend(loc="upper right", framealpha=0.92)

    out = FIG / "v5_final_gap_map.png"
    plt.tight_layout()
    plt.savefig(out, dpi=PUB_DPI, bbox_inches="tight")
    plt.close()
    print(f"[save] {out}")


def fig_v5_final_novelty_vs_kg_scatter():
    df = _load_v5f()

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    sc = ax.scatter(
        df["novelty_real"], df["kg_plausibility"],
        c=df["score_v5_final"], cmap="viridis", alpha=0.50,
        s=14, edgecolors="none", zorder=2,
    )

    # TRUE-DARK
    td = df[df["novelty_status"] == "TRUE-DARK"]
    ax.scatter(td["novelty_real"], td["kg_plausibility"],
               marker="D", facecolors="none", edgecolors="crimson",
               linewidths=1.4, s=100,
               label=f"TRUE-DARK (n = {len(td)})", zorder=3)

    # Well-studied
    ws = df[df["novelty_status"] == "well-studied"]
    ax.scatter(ws["novelty_real"], ws["kg_plausibility"],
               marker="x", c="#666666", s=30, linewidths=0.8,
               label=f"Well-Studied (n = {len(ws)})", zorder=3)

    # PC7 controls
    ctl = df[df["drug_name"].isin(PC7_CONTROLS.keys())]
    ax.scatter(ctl["novelty_real"], ctl["kg_plausibility"],
               marker="*", facecolors="gold", edgecolors="black",
               linewidths=0.7, s=240, label="PC7 Controls", zorder=5)
    for _, r in ctl.iterrows():
        ax.annotate(
            PC7_CONTROLS[r["drug_name"]],
            (r["novelty_real"], r["kg_plausibility"]),
            xytext=(7, 4), textcoords="offset points",
            fontsize=8.5, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
        )

    ax.set_xlabel("Literature Novelty (0\u20131)")
    ax.set_ylabel("Knowledge-Graph Plausibility (0\u20131)")
    ax.set_title("Candidate Landscape: Novelty vs. KG Plausibility")
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.04, 1.04)
    ax.grid(True, alpha=0.25, linestyle="--")
    cbar = plt.colorbar(sc, ax=ax, pad=0.015)
    cbar.set_label("Repurposing Score", rotation=270, labelpad=14)
    ax.legend(loc="lower left", framealpha=0.92, fontsize=9)

    out = FIG / "v5_final_novelty_vs_kg_scatter.png"
    plt.tight_layout()
    plt.savefig(out, dpi=PUB_DPI, bbox_inches="tight")
    plt.close()
    print(f"[save] {out}")


def fig_v5_final_score_distribution():
    df = _load_v5f()
    scores = df["score_v5_final"].dropna().values

    fig, ax = plt.subplots(figsize=(9.5, 5))
    bins = np.linspace(scores.min(), scores.max(), 50)
    counts, _, _ = ax.hist(
        scores, bins=bins, color="#4c78a8", edgecolor="white", linewidth=0.4, alpha=0.92,
    )
    ymax = counts.max() if len(counts) else 1

    # PC7 controls — vertical lines
    ctl = df[df["drug_name"].isin(PC7_CONTROLS.keys())]
    for _, r in ctl.iterrows():
        ax.axvline(r["score_v5_final"], color="goldenrod", lw=1.6, alpha=0.9, zorder=3)

    # TRUE-DARK 13 — dashed verticals (no labels, use legend)
    td = df[df["novelty_status"] == "TRUE-DARK"]
    for _, r in td.iterrows():
        ax.axvline(r["score_v5_final"], color="crimson", lw=0.9, alpha=0.5, ls="--", zorder=2)

    med = float(np.median(scores))
    ax.axvline(med, color="black", lw=1.0, ls=":", alpha=0.7)
    ax.text(med + 0.003, ymax * 0.85, f"Median = {med:.3f}", fontsize=9, alpha=0.85)

    ax.set_xlabel("Repurposing Score")
    ax.set_ylabel("Number of Drugs")
    ax.set_title(f"Score Distribution  (n = {len(df)} ranked drugs)")
    ax.grid(True, axis="y", alpha=0.25, linestyle="--")

    handles = [
        mpatches.Patch(facecolor="#4c78a8", label=f"All Drugs (n = {len(df)})"),
        mpatches.Patch(facecolor="goldenrod", label="PC7 Controls (solid lines)"),
        mpatches.Patch(facecolor="crimson", alpha=0.5, label="TRUE-DARK 13 (dashed lines)"),
    ]
    ax.legend(handles=handles, loc="upper right", framealpha=0.92, fontsize=9)

    out = FIG / "v5_final_score_distribution.png"
    plt.tight_layout()
    plt.savefig(out, dpi=PUB_DPI, bbox_inches="tight")
    plt.close()
    print(f"[save] {out}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("CANONICAL FIGURE REGENERATION — all 16 figures")
    print("=" * 60)

    print("\n== Phase 1 ==")
    fig_phase1_hallucination_audit()
    fig_phase1_response_quality()

    print("\n== Phase 2 ==")
    fig_phase2_coverage_breakdown()
    fig_phase2_papers_per_drug()
    fig_phase2_dark_drugs_by_class()

    print("\n== Phase 3 ==")
    fig_phase3_entity_distribution()

    print("\n== Phase 4 (RAGAS) ==")
    fig_v4_answer_length_by_config()
    # fig_v4_context_count_distribution() — demoted (shows top_k config, not a result)
    fig_v4_faithfulness_heatmap_by_tier()
    fig_v4_no_info_rate_by_config()
    fig_v4_ragas_heatmap()
    fig_v4_ragas_scores_by_config()
    # fig_v4_tier_breakdown() — superseded by faithfulness heatmap

    print("\n== Phase 5 (v5_final) ==")
    fig_v5_final_gap_map()
    fig_v5_final_novelty_vs_kg_scatter()
    fig_v5_final_score_distribution()

    print("\n" + "=" * 60)
    print("ALL 14 CANONICAL FIGURES REGENERATED")
    print(f"Output: {FIG}")
    print("=" * 60)


if __name__ == "__main__":
    main()
