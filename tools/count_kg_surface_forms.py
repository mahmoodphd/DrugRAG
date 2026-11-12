#!/usr/bin/env python3
"""Count unmerged surface forms in the DrugRAG knowledge graph.

Backs the limitations-section sentence of the BHI 2026 paper: "Surface forms
are not merged, so 162 nodes contain 'VEGF' and 314 contain 'endothelial
cell'". A node counts when the term is a case-insensitive substring of its
node ID (the entity name), i.e. term.lower() in node_id.lower().

The graph file ships with the LightRAG index (Hugging Face dataset
mshz88/DrugRAG), unpacked under data/phase3/lightrag_working_v2/.

Usage:
  python tools/count_kg_surface_forms.py [--graphml PATH]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRAPHML = (ROOT / "data" / "phase3" / "lightrag_working_v2"
                   / "graph_chunk_entity_relation.graphml")
TERMS = ["VEGF", "endothelial cell"]


def count_nodes_containing(node_ids, term: str) -> int:
    t = term.lower()
    return sum(1 for n in node_ids if t in str(n).lower())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--graphml", type=Path, default=DEFAULT_GRAPHML,
                    help="knowledge-graph GraphML file (default: the LightRAG index)")
    args = ap.parse_args()
    if not args.graphml.is_file():
        raise SystemExit(f"Graph file not found: {args.graphml}\n"
                         "Download the LightRAG index from the Hugging Face "
                         "dataset mshz88/DrugRAG first.")

    g = nx.read_graphml(args.graphml)
    node_ids = list(g.nodes())
    print(f"Knowledge-graph nodes: {len(node_ids):,}")
    for term in TERMS:
        n = count_nodes_containing(node_ids, term)
        print(f'Nodes whose name contains "{term}": {n:,}')
    print("(case-insensitive substring match on the node name)")


if __name__ == "__main__":
    main()
