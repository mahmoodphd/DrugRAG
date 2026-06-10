"""KG utilities — load graphml once, build drug neighborhoods, render Plotly graph."""

import re
import pathlib
import networkx as nx
import plotly.graph_objects as go

ROOT = pathlib.Path(__file__).resolve().parent.parent
GRAPHML = ROOT / "data" / "phase3" / "lightrag_working_v2" / "graph_chunk_entity_relation.graphml"

# ─── Canonical entity-type mapping ───────────────────────────────────────────
# Many raw types in the graphml → ONE display label + ONE color.
# Anything not in this map falls into "Other (biology)".
CANONICAL_TYPE = {
    "drug": "Drug",
    "compound": "Drug",
    "metabolite": "Drug",
    "gene_receptor": "Gene / Receptor",
    "genereceptor": "Gene / Receptor",
    "gene": "Gene / Receptor",
    "protein": "Gene / Receptor",
    "enzyme": "Gene / Receptor",
    "biological_process": "Biological Process",
    "biologicalprocess": "Biological Process",
    "signaling_pathway": "Signaling Pathway",
    "signalingpathway": "Signaling Pathway",
    "disease": "Disease",
    "phenotype": "Phenotype",
    "cell_type": "Cell Type",
    "celltype": "Cell Type",
    "neurotransmitter": "Neurotransmitter",
    "cytokine": "Cytokine",
    "growth_factor": "Cytokine",
    "angiogenic_factor": "Cytokine",
    "signaling_molecule": "Cytokine",
    "peptide_hormone": "Cytokine",
}

# Distinct qualitative palette — no two categories share a hue.
TYPE_COLORS = {
    "Drug": "#4393c3",
    "Gene / Receptor": "#d6604d",
    "Signaling Pathway": "#66a61e",
    "Biological Process": "#e6ab02",
    "Disease": "#e7298a",
    "Phenotype": "#a6761d",
    "Cell Type": "#7570b3",
    "Neurotransmitter": "#1b9e77",
    "Cytokine": "#666666",
    "Other (biology)": "#bdbdbd",
}

DEFAULT_COLOR = "#bdbdbd"


def _canon_type(node_attrs) -> str:
    """Return the canonical display label for a node's entity type."""
    raw = str(node_attrs.get("entity_type", "unknown")).lower()
    return CANONICAL_TYPE.get(raw, "Other (biology)")

# Biological entity types we keep — everything else (person, method, concept,
# organization, study artifacts, generic "other"/"unknown") is filtered out.
BIO_TYPES = set(CANONICAL_TYPE.keys())


def _is_bio(node_attrs) -> bool:
    et = str(node_attrs.get("entity_type", "unknown")).lower()
    return et in BIO_TYPES


# Cached graph (loaded lazily, then kept)
_G = None


def get_graph() -> nx.Graph:
    """Load (and cache) the LightRAG graphml."""
    global _G
    if _G is None:
        _G = nx.read_graphml(GRAPHML)
    return _G


def find_drug_node(drug_name: str) -> str | None:
    """Map a v5_final drug_name to a node ID in the KG.

    Tries exact, title-cased, and stripped-suffix variants.
    Returns the node id, or None.
    """
    G = get_graph()
    if drug_name in G.nodes():
        return drug_name

    # Title-cased
    tc = drug_name.title()
    if tc in G.nodes():
        return tc

    # Strip common salt/form suffixes
    for suffix in [
        " hydrochloride", " Hydrochloride",
        " sodium salt", " sodium",
        " maleate", " citrate", " sulfate",
        " mesylate", " besylate", " dihydrochloride",
        " calcium salt", " potassium salt",
    ]:
        if drug_name.endswith(suffix):
            stripped = drug_name[: -len(suffix)]
            if stripped in G.nodes():
                return stripped
            if stripped.title() in G.nodes():
                return stripped.title()

    # Substring fuzzy match (case-insensitive)
    low = drug_name.lower()
    for n in G.nodes():
        if low == str(n).lower():
            return n
    # First-word match (e.g. "Memantine Hydrochloride" -> "Memantine")
    first = drug_name.split()[0] if drug_name else ""
    if first in G.nodes():
        return first
    if first.title() in G.nodes():
        return first.title()
    return None


def build_neighborhood(seed_node: str, hops: int = 1, max_nodes: int = 60) -> nx.Graph:
    """Return the k-hop subgraph around seed_node, filtered to biological entity types,
    capped at max_nodes."""
    G = get_graph()
    if seed_node not in G.nodes():
        return nx.Graph()

    nodes = {seed_node}
    frontier = {seed_node}
    for _ in range(hops):
        next_frontier = set()
        for n in frontier:
            for nbr in G.neighbors(n):
                if nbr in nodes:
                    continue
                # Always allow seed-adjacent biology; for further hops also require bio.
                if nbr == seed_node or _is_bio(G.nodes[nbr]):
                    next_frontier.add(nbr)
        nodes.update(next_frontier)
        frontier = next_frontier
        if len(nodes) >= max_nodes:
            break

    # Cap at max_nodes by keeping highest-degree nodes around the seed
    if len(nodes) > max_nodes:
        keepers = {seed_node} | {
            nbr for nbr in G.neighbors(seed_node)
            if _is_bio(G.nodes[nbr])
        }
        candidates = [n for n in nodes if n not in keepers]
        candidates.sort(key=lambda x: G.degree(x), reverse=True)
        room = max_nodes - len(keepers)
        nodes = keepers | set(candidates[: max(0, room)])

    return G.subgraph(nodes).copy()


def build_glutamate_showcase() -> nx.Graph:
    """Pre-built showcase: glutamate / NMDA subgraph centered on receptors + Memantine."""
    G = get_graph()
    seeds = [
        "Memantine",
        "Glutamate",
        "L-Glutamate",
        "NMDA",
        "NMDA Receptors",
        "N-Methyl-D-Aspartate (NMDA) Receptors",
        "Metabotropic Glutamate Receptor 5",
        "Glutamate Receptor Modulators",
    ]
    nodes = set()
    for s in seeds:
        if s in G.nodes():
            nodes.add(s)  # seeds always allowed
            for nbr in G.neighbors(s):
                if _is_bio(G.nodes[nbr]):
                    nodes.add(nbr)
    # Cap to ~50 for readability
    if len(nodes) > 50:
        scores = {}
        for n in nodes:
            scores[n] = sum(1 for s in seeds if s in G.nodes() and (n == s or G.has_edge(n, s)))
        ranked = sorted(nodes, key=lambda x: scores[x], reverse=True)
        nodes = set(ranked[:50])
    return G.subgraph(nodes).copy()


def empty_placeholder_figure(message: str, title: str = "") -> go.Figure:
    """Return a fixed-height (~700px) figure with a centered annotation.
    Used whenever we have no graph data or hit an error — the Plot must never collapse."""
    fig = go.Figure()
    fig.add_annotation(
        x=0.5, y=0.5, xref="paper", yref="paper",
        text=message, showarrow=False,
        font=dict(size=14, color="#444"),
        align="center",
    )
    fig.update_layout(
        title=dict(text=title, x=0.02, font=dict(size=14)),
        template="plotly_white",
        height=700,
        margin=dict(b=10, l=10, r=10, t=40),
        xaxis=dict(visible=False, range=[0, 1]),
        yaxis=dict(visible=False, range=[0, 1]),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


def render_plotly_graph(
    H: nx.Graph,
    seed_node: str | None = None,
    title: str = "",
    label_top_n: int = 14,
) -> go.Figure:
    """Build a Plotly figure for subgraph H. Highlight seed_node with a thick
    dark ring (colored by type, not black). Label only the top `label_top_n`
    nodes by degree (+ seed); all others hover-only."""
    if H.number_of_nodes() == 0:
        return empty_placeholder_figure(
            "No graph data \u2014 this drug is genuinely understudied<br>"
            "(no abstracts \u2192 no extracted entities).",
            title=title,
        )

    # Spring layout \u2014 bigger k and more iterations for dense graphs.
    n_nodes = H.number_of_nodes()
    k_val = 2.2 if n_nodes <= 30 else (3.0 if n_nodes <= 60 else 3.6)
    iters = 200 if n_nodes <= 30 else (300 if n_nodes <= 60 else 400)
    pos = nx.spring_layout(H, seed=42, k=k_val, iterations=iters)

    # Decide which nodes to label: top-N by degree + the seed.
    degrees = dict(H.degree())
    labeled = set(sorted(degrees, key=lambda x: degrees[x], reverse=True)[:label_top_n])
    if seed_node is not None and seed_node in H.nodes():
        labeled.add(seed_node)

    # Edges
    edge_x, edge_y = [], []
    for u, v in H.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.7, color="#bbbbbb"),
        hoverinfo="none",
        mode="lines",
        showlegend=False,
    )

    # Nodes \u2014 use canonical types for color
    node_x, node_y, node_text, node_color, node_size, hover_text = [], [], [], [], [], []
    node_line_width, node_line_color = [], []
    for n in H.nodes():
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)
        attrs = H.nodes[n]
        canon = _canon_type(attrs)
        color = TYPE_COLORS.get(canon, DEFAULT_COLOR)
        # Seed node: same type color but thick dark ring + larger
        if seed_node is not None and n == seed_node:
            size = 30
            node_line_width.append(3.5)
            node_line_color.append("#111111")
        else:
            size = 14 + min(8, H.degree(n))
            node_line_width.append(1.0)
            node_line_color.append("#444444")
        node_color.append(color)
        node_size.append(size)
        # Label only for top-degree nodes + seed; others hover-only.
        if n in labeled:
            label = str(n)
            if len(label) > 22:
                label = label[:20] + "\u2026"
            node_text.append(label)
        else:
            node_text.append("")
        # Hover
        desc = attrs.get("description", "")
        if desc:
            desc = desc.split(".")[0][:200]
        hover_text.append(f"<b>{n}</b><br>type: {canon}<br>{desc}")

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        textfont=dict(size=10, color="#222"),
        hovertext=hover_text,
        hoverinfo="text",
        marker=dict(
            color=node_color,
            size=node_size,
            line=dict(width=node_line_width, color=node_line_color),
        ),
        showlegend=False,
    )

    # Legend traces \u2014 one per canonical type seen (deduplicated)
    seen_types = sorted(set(_canon_type(H.nodes[n]) for n in H.nodes()))
    legend_traces = []
    for t in seen_types:
        legend_traces.append(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(color=TYPE_COLORS.get(t, DEFAULT_COLOR), size=12,
                        line=dict(width=1, color="#444")),
            name=t,
            showlegend=True,
        ))
    # Add legend note for seed node ring
    if seed_node and seed_node in H.nodes():
        legend_traces.append(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(color="white", size=12,
                        line=dict(width=3, color="#111")),
            name="\u25cb outline = selected drug",
            showlegend=True,
        ))

    fig = go.Figure(data=[edge_trace, node_trace] + legend_traces)
    fig.update_layout(
        title=dict(text=title, x=0.02, font=dict(size=14)),
        showlegend=True,
        legend=dict(orientation="v", x=1.02, y=1, font=dict(size=10)),
        hovermode="closest",
        margin=dict(b=10, l=10, r=120, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=700,
    )
    return fig


def neighborhood_summary(H: nx.Graph, seed_node: str) -> str:
    """Markdown summary of the neighborhood (counts by canonical type)."""
    if H.number_of_nodes() == 0:
        return "*No graph data found for this drug.*"

    by_type = {}
    for n in H.nodes():
        if n == seed_node:
            continue
        t = _canon_type(H.nodes[n])
        by_type[t] = by_type.get(t, 0) + 1

    lines = [f"### Connected to **{seed_node}**", ""]
    lines.append(f"- **{H.number_of_nodes() - 1}** connected entities, **{H.number_of_edges()}** relations")
    for t, c in sorted(by_type.items(), key=lambda x: -x[1]):
        if c == 0:
            continue
        lines.append(f"  - {t}: {c}")
    return "\n".join(lines)
