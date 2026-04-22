import json
import os
from difflib import SequenceMatcher
from datetime import datetime

ATTACK_FILE = "data/graphrag/attacks/attack_graph_graphrag.json"
VULN_FILE = "data/graphrag/vulnerabilities/vuln_graph_graphrag.json"
OUT_BRIDGE = "data/graphrag/bridge_attack_vuln/bridge_graph.json"
OUT_GLOBAL = "data/graphrag/multi_layer/global_graph.json"


def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def normalize_name(text: str) -> str:
    if not text:
        return ""
    return (
        text.lower()
        .replace("_", " ")
        .replace("-", " ")
        .strip()
    )


def similar(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_name(a), normalize_name(b)).ratio()


def add_bridge(edges, seen, source, target, edge_type, method, confidence,
               layer_from, layer_to, evidence=None):
    key = (source, edge_type, target)
    if key in seen:
        return
    seen.add(key)

    edge = {
        "source": source,
        "target": target,
        "type": edge_type,
        "method": method,
        "confidence": round(float(confidence), 3),
        "layer_from": layer_from,
        "layer_to": layer_to,
        "evidence": evidence or []
    }
    edges.append(edge)


with open(ATTACK_FILE, "r", encoding="utf-8") as f:
    attack_graph = json.load(f)

with open(VULN_FILE, "r", encoding="utf-8") as f:
    vuln_graph = json.load(f)

attack_nodes = attack_graph["nodes"]
attack_edges = attack_graph["edges"]

vuln_nodes = vuln_graph["nodes"]
vuln_edges = vuln_graph["edges"]

# -----------------------
# Build ATT&CK lookups
# -----------------------
software_nodes = {}
software_to_techniques = {}

for node in attack_nodes:
    if node.get("type") == "software":
        software_nodes[node["id"]] = node.get("name", node["id"])

for edge in attack_edges:
    if edge.get("type") == "uses":
        src = edge.get("source")
        tgt = edge.get("target")

        if src in software_nodes:
            software_to_techniques.setdefault(src, []).append(tgt)

# -----------------------
# Build vuln lookups
# CVE -> product names
# -----------------------
node_by_id = {n["id"]: n for n in vuln_nodes}

cpe_to_products = {}
for edge in vuln_edges:
    if edge.get("type") == "has_product":
        cpe_id = edge["source"]
        product_id = edge["target"]

        product_node = node_by_id.get(product_id)
        if product_node:
            cpe_to_products.setdefault(cpe_id, []).append(product_node.get("name", product_id))

cve_to_products = {}
for edge in vuln_edges:
    if edge.get("type") == "affects":
        cve_id = edge["source"]
        cpe_id = edge["target"]

        for product_name in cpe_to_products.get(cpe_id, []):
            cve_to_products.setdefault(cve_id, []).append(product_name)

# -----------------------
# Build bridge edges
# -----------------------
bridge_edges = []
bridge_seen = set()

for cve_id, product_names in cve_to_products.items():
    for product_name in product_names:
        for software_id, software_name in software_nodes.items():
            score = similar(product_name, software_name)

            if score >= 0.86:
                add_bridge(
                    bridge_edges,
                    bridge_seen,
                    cve_id,
                    software_id,
                    "related_software",
                    "name_match",
                    score,
                    "vulnerability",
                    "attack",
                    [f"product={product_name}", f"software={software_name}"]
                )

                for technique_id in software_to_techniques.get(software_id, []):
                    add_bridge(
                        bridge_edges,
                        bridge_seen,
                        cve_id,
                        technique_id,
                        "exploited_via",
                        "software_to_technique",
                        min(score, 0.80),
                        "vulnerability",
                        "attack",
                        [f"matched_product={product_name}", f"software_id={software_id}"]
                    )

# -----------------------
# Save bridge graph
# -----------------------
bridge_graph = {
    "meta": {
        "name": "bridge_graph",
        "description": "Cross-layer bridge edges between vulnerability and ATT&CK graphs.",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "total_edges": len(bridge_edges)
    },
    "nodes": [],
    "edges": bridge_edges
}

ensure_parent_dir(OUT_BRIDGE)
with open(OUT_BRIDGE, "w", encoding="utf-8") as f:
    json.dump(bridge_graph, f, indent=2, ensure_ascii=False)

# -----------------------
# Save global graph
# -----------------------
global_graph = {
    "meta": {
        "name": "global_graph",
        "description": "Merged ATT&CK + vulnerability + bridge graph.",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "total_nodes": len(attack_nodes) + len(vuln_nodes),
        "total_edges": len(attack_edges) + len(vuln_edges) + len(bridge_edges)
    },
    "nodes": attack_nodes + vuln_nodes,
    "edges": attack_edges + vuln_edges + bridge_edges
}

with open(OUT_GLOBAL, "w", encoding="utf-8") as f:
    json.dump(global_graph, f, indent=2, ensure_ascii=False)

print("Saved:", OUT_BRIDGE)
print("Bridge edges:", len(bridge_edges))
print("Saved:", OUT_GLOBAL)
print("Global nodes:", len(global_graph["nodes"]))
print("Global edges:", len(global_graph["edges"]))