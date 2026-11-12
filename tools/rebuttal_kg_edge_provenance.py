#!/usr/bin/env python
"""Edge-level provenance for the multi-hop knowledge-graph path in the paper.

The paper traces NMDA Receptors -> Kynurenic Acid -> VEGF -> Endothelial Cells
and states which abstract each hop comes from. Every
LightRAG relation carries a source_id pointing at the text chunk it was
extracted from, so each hop can be traced back to a specific PMID and the
sentence that produced it. This script resolves that chain.

Output: data/phase3/rebuttal_kg_edge_provenance.json
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config

WORK = config.PHASE3_DIR / "lightrag_working_v2"
PATH = [("NMDA Receptors", "Kynurenic Acid"),
        ("Kynurenic Acid", "VEGF"),
        ("VEGF", "Endothelial Cells")]


def main():
    graph_file = next((f for f in [WORK / "graph_chunk_entity_relation.graphml"]
                       if f.exists()), None)
    if graph_file is None:
        print("graphml not found; listing working dir:")
        for f in sorted(WORK.iterdir()):
            print("  ", f.name)
        return

    import networkx as nx
    G = nx.read_graphml(graph_file)
    print(f"graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    chunks = json.loads((WORK / "kv_store_text_chunks.json").read_text())

    results = []
    for src, dst in PATH:
        s = src if src in G else None
        d = dst if dst in G else None
        hop = {"from": src, "to": dst, "resolved_from": s, "resolved_to": d}
        if not (s and d and G.has_edge(s, d)):
            # try the reverse orientation; the graph is undirected in practice
            if s and d and G.has_edge(d, s):
                s, d = d, s
                hop["note"] = "edge stored in reverse orientation"
            else:
                hop["found"] = False
                results.append(hop)
                print(f"\n{src} -> {dst}: NOT a direct edge")
                continue
        e = G.get_edge_data(s, d)
        hop["found"] = True
        hop["description"] = (e.get("description") or "")[:400]
        hop["keywords"] = e.get("keywords", "")
        hop["weight"] = e.get("weight")

        src_ids = [c for c in (e.get("source_id") or "").split("<SEP>") if c]
        prov = []
        for cid in src_ids[:6]:
            ch = chunks.get(cid, {})
            fp = ch.get("file_path", "")
            pm = re.search(r"pmid_(\d+)_", fp)
            prov.append({
                "chunk_id": cid,
                "pmid": pm.group(1) if pm else None,
                "file": Path(fp).name if fp else None,
                "snippet": " ".join((ch.get("content") or "").split())[:260],
            })
        hop["n_source_chunks"] = len(src_ids)
        hop["provenance"] = prov
        results.append(hop)

        print(f"\n{src} -> {dst}  ({len(src_ids)} source chunk(s))")
        print(f"   desc: {hop['description'][:180]}")
        for p in prov[:3]:
            print(f"   PMID {p['pmid']}: {p['snippet'][:150]}")

    out = {
        "note": ("Each hop in the Fig. 6 path resolves to LightRAG relation "
                 "source_id values, which point at indexed text chunks and "
                 "therefore at specific corpus PMIDs. 'Traceable' here means "
                 "the edge carries source chunks that can be opened; it is not "
                 "a claim that a domain expert has adjudicated the relation."),
        "path": " -> ".join([PATH[0][0]] + [b for _, b in PATH]),
        "hops": results,
        "all_hops_traceable": all(h.get("found") and h.get("n_source_chunks", 0) > 0
                                  for h in results),
    }
    dest = ROOT / "data" / "phase3" / "rebuttal_kg_edge_provenance.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"\nall hops traceable: {out['all_hops_traceable']}")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
