import json
import networkx as nx
from datetime import datetime

INPUT = "attack_graph.json"
OUT_GRAPH = "entreprise_attack/attack_graph_graphrag.json"
OUT_TEXTS = "entreprise_attack/attack_node_texts.jsonl"

# --- normalize relationship types to a small set ---
REL_MAP = {
    "subtechnique-of": "subtechnique_of",
    "mitigates": "mitigates",
    "uses": "uses",
    "detects": "detects",
    "revoked-by": "revoked_by",
    "deprecated-by": "deprecated_by",
}

KEEP_NODE_TYPES = {
    "technique", "tactic", "mitigation", "group", "software",
    "data_source", "data_component", "campaign"
}

KEEP_EDGE_TYPES = {
    "uses", "mitigates", "subtechnique_of",
    "part_of", "belongs_to", "related_to", "detects"
}

def norm_rel(r: str) -> str:
    if not r:
        return "related_to"
    r = r.strip()
    r = REL_MAP.get(r, r)
    return r.replace("-", "_")

def export_node_id(stix_id: str, data: dict) -> str:
    # Prefer ATT&CK external ID (T1059, TA0001, etc.)
    return data.get("attack_id") or data.get("external_id") or stix_id

def clean_text(x):
    if not x:
        return ""
    return " ".join(str(x).split())

# -----------------------
# LOAD + BUILD GRAPH
# -----------------------
with open(INPUT, "r", encoding="utf-8") as f:
    g = json.load(f)

G = nx.DiGraph()

for n in g.get("nodes", []):
    if n.get("deprecated") or n.get("revoked"):
        continue

    node_type = (n.get("label") or n.get("type") or "").strip().lower()
    # some ATT&CK exports use "label" like "technique"
    if node_type and node_type not in KEEP_NODE_TYPES:
        # you can comment this out if you want every node type
        pass

    G.add_node(n["id"], **n)

for e in g.get("edges", []):
    s, t = e.get("source"), e.get("target")
    if s in G.nodes and t in G.nodes:
        G.add_edge(s, t, **e)

print("Graph loaded.")
print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())

# -----------------------
# EXPORT NODES (once)
# -----------------------
nodes_out = {}
texts_out = []

for stix_id, d in G.nodes(data=True):
    nid = export_node_id(stix_id, d)

    node_type = (d.get("label") or d.get("type") or "unknown").strip().lower()
    name = d.get("name") or nid
    desc = d.get("description") or ""

    node_obj = {
        "id": nid,
        "type": node_type,
        "name": name,
        "description": desc,
        "properties": {
            "stix_id": stix_id
        }
    }
    nodes_out[nid] = node_obj

    # LLM-ready node text for embeddings (GraphRAG retrieval)
    node_text = f"{node_type.upper()} {nid} — {name}. {clean_text(desc)}"
    texts_out.append({
        "id": nid,
        "type": node_type,
        "name": name,
        "text": node_text
    })

# -----------------------
# EXPORT EDGES (IDs only)
# -----------------------
edges_out = []
for u, v, ed in G.edges(data=True):
    su = export_node_id(u, G.nodes[u])
    sv = export_node_id(v, G.nodes[v])

    rel = norm_rel(ed.get("type"))

    # Convert some relationship directions into nicer semantics
    # (optional) If your source graph has technique->tactic edges as "phase-of", map it:
    if rel in ("phase_of", "is_phase_of"):
        rel = "belongs_to"

    # Keep only high-value edge types
    if rel not in KEEP_EDGE_TYPES:
        # keep the rest as related_to (optional)
        rel = "related_to"

    edges_out.append({
        "source": su,
        "relation": rel,
        "target": sv
    })

# -----------------------
# FINAL JSON
# -----------------------
out = {
    "meta": {
        "name": "attack_graph_graphrag",
        "description": "GraphRAG-ready MITRE ATT&CK Enterprise graph (nodes+edges) + separate node_texts for embeddings.",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "total_nodes": len(nodes_out),
        "total_edges": len(edges_out)
    },
    "nodes": list(nodes_out.values()),
    "edges": edges_out
}

with open(OUT_GRAPH, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)

with open(OUT_TEXTS, "w", encoding="utf-8") as f:
    for line in texts_out:
        f.write(json.dumps(line) + "\n")

print("Saved:", OUT_GRAPH)
print("Saved:", OUT_TEXTS)
print("Nodes exported:", len(nodes_out))
print("Edges exported:", len(edges_out))