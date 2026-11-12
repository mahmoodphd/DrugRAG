#!/usr/bin/env python
"""Draw the five-stage pipeline overview used in the README and the dashboard About tab.

Every number in the figure is taken from the accepted BHI 2026 paper:
1,278 compounds; 8,514 drug-paper pairs, 6,608 unique abstracts, 6,500 indexed;
36,576 entities and 43,300 relations; 46 DrugRAG citations, all in the corpus,
87% mentioning the queried entity against 0% for the ungrounded model (Table III);
65 leads checked against PubMed, 13 understudied candidates.

Usage:  python tools/make_architecture_figure.py   (writes docs/architecture.png)
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "architecture.png"

BG, INK, MUTED = "#f7f8fb", "#1f2937", "#6b7280"
STAGES = [
    ("#3b82f6", "Phenotypic\nScreen",
     "LOPAC1280 library,\n1,278 compounds,\nHUVEC tube-formation assay",
     "PhenoClusters 1–7 · activity\nmagnitude · pro-/anti-angiogenic"),
    ("#8b5cf6", "Literature\nCorpus",
     "PubMed harvest\n(NCBI E-utilities),\nper-drug queries",
     "8,514 drug–paper pairs →\n6,608 unique abstracts\n(6,500 indexed)"),
    ("#06b6d4", "Knowledge\nGraph",
     "LightRAG entity–relation\nextraction (gpt-4o-mini)\n+ MiniLM embeddings\n(all-MiniLM-L6-v2)",
     "36,576 entities ·\n43,300 relations"),
    ("#10b981", "Grounded QA +\nCitation Check",
     "Gemma 4 (31B) via OpenRouter,\nhybrid retrieval (local + global),\ntop-20 chunks · every cited\nPMID checked against corpus",
     "All 46 cited PMIDs in corpus;\n87% mention the queried entity\n(ungrounded model: 0%)"),
    ("#f59e0b", "Repurposing\nHypotheses",
     "score s = 0.40·activity +\n0.35·verified novelty +\n0.25·KG support",
     "65 leads checked in PubMed →\n13 understudied candidates\n(BNTX, Ziprasidone, PHA 767491)"),
]


def main() -> None:
    fig = plt.figure(figsize=(20, 9.12), dpi=189, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 9.12)
    ax.axis("off")

    ax.text(10, 8.08, "DrugRAG Pipeline Architecture", ha="center", va="center",
            fontsize=24, fontweight="bold", color=INK)
    ax.text(10, 7.70, "From phenotypic screen to repurposing hypotheses", ha="center",
            va="center", fontsize=13.5, style="italic", color=MUTED)

    w, gap, x0 = 3.23, 0.48, 0.57
    top_y, top_h = 2.55, 3.83
    bot_y, bot_h = 0.70, 1.38
    for i, (col, title, body, result) in enumerate(STAGES):
        x = x0 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, top_y), w, top_h, boxstyle="round,pad=0,rounding_size=0.12",
                                    fc=col, ec="white", lw=2.5))
        ax.add_patch(Circle((x + 0.48, top_y + top_h - 0.48), 0.2, fc="white", ec="none"))
        ax.text(x + 0.48, top_y + top_h - 0.48, str(i + 1), ha="center", va="center",
                fontsize=13, fontweight="bold", color=col)
        ax.text(x + w / 2, top_y + top_h - 1.12, title, ha="center", va="center",
                fontsize=13, fontweight="bold", color="white", linespacing=1.15)
        ax.text(x + w / 2, top_y + 1.55, body, ha="center", va="center", fontsize=9.6,
                family="DejaVu Sans Mono", color="white", linespacing=1.45)

        ax.add_patch(FancyBboxPatch((x + 0.03, bot_y), w - 0.06, bot_h,
                                    boxstyle="round,pad=0,rounding_size=0.08",
                                    fc="white", ec=col, lw=1.8))
        ax.text(x + w / 2, bot_y + bot_h / 2, result, ha="center", va="center",
                fontsize=9.6, color=INK, linespacing=1.4)
        ax.add_patch(FancyArrowPatch((x + w / 2, top_y - 0.03), (x + w / 2, bot_y + bot_h + 0.02),
                                     arrowstyle="-|>", mutation_scale=11, color=col, lw=1.6))
        if i < len(STAGES) - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.03, top_y + top_h / 2),
                                         (x + w + gap - 0.03, top_y + top_h / 2),
                                         arrowstyle="-|>", mutation_scale=16, color=MUTED, lw=2.6))

    ax.text(10, 0.35, "Every cited PMID is checked against a fixed 6,500-abstract corpus  ·  "
            "LightRAG hybrid retrieval  ·  Gemma 4 (31B)", ha="center", va="center",
            fontsize=10, style="italic", color="#94a3b8")

    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, facecolor=BG)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
