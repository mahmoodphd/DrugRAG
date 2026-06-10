#!/usr/bin/env python3
"""Export the glutamate KG showcase subgraph as a clean PNG for the paper."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dashboard"))
import kg_utils

H = kg_utils.build_glutamate_showcase()
print(f"Subgraph: {H.number_of_nodes()} nodes, {H.number_of_edges()} edges")

fig = kg_utils.render_plotly_graph(H, seed_node="Memantine",
                                    title="Glutamate -> NMDA -> Memantine subgraph",
                                    label_top_n=12)
fig.update_layout(
    width=900, height=700,
    margin=dict(b=10, l=10, r=10, t=40),
    paper_bgcolor="white",
    plot_bgcolor="white",
)
out = os.path.join(os.path.dirname(__file__), "..", "BHI_2026", "figures", "F5_kg.png")
fig.write_image(out, scale=2)
print(f"Saved to {os.path.abspath(out)}")
