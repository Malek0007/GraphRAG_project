import json
import os
from difflib import SequenceMatcher
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

ATTACK_FILE = "data/graphrag/attacks/attack_graph_graphrag.json"
VULN_FILE = "data/graphrag/vulnerabilities/vuln_graph_graphrag.json"
OUT_BRIDGE = "data/graphrag/bridge_attack_vuln/bridge_graph.json"
OUT_GLOBAL = "data/graphrag/multi_layer/global_graph.json"

MODEL_NAME = "all-MiniLM-L6-v2"


def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def normalize_name(text: str) -> str:
    if not text:
        return ""
    return text.lower().replace("_", " ").replace("-", " ").strip()


def similar(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_name(a), normalize_name(b)).ratio()


def cosine(a, b):
    return float(cosine_similarity([a], [b])[0][0])


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

attack_node_by_id = {n["id"]: n for n in attack_nodes}
vuln_node_by_id = {n["id"]: n for n in vuln_nodes}

# -----------------------
# Build ATT&CK lookups
# -----------------------
software_nodes = {}
software_to_techniques = {}

for node in attack_nodes:
    if node.get("type") == "software":
        software_nodes[node["id"]] = node

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
cpe_to_products = {}
for edge in vuln_edges:
    if edge.get("type") == "has_product":
        cpe_id = edge["source"]
        product_id = edge["target"]
        product_node = vuln_node_by_id.get(product_id)
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
# Prepare NLP texts
# -----------------------
def node_text(node):
    return " ".join([
        node.get("id", ""),
        node.get("name", ""),
        node.get("description", "")
    ]).strip()

model = SentenceTransformer(MODEL_NAME)

software_texts = {}
software_embeddings = {}

for sid, node in software_nodes.items():
    text = node_text(node)
    software_texts[sid] = text
    software_embeddings[sid] = model.encode(text)

technique_nodes = {
    n["id"]: n for n in attack_nodes
    if n.get("type") in {"technique", "sub-technique"}
}

technique_texts = {}
technique_embeddings = {}

for tid, node in technique_nodes.items():
    text = node_text(node)
    technique_texts[tid] = text
    technique_embeddings[tid] = model.encode(text)

# -----------------------
# Build bridge edges
# -----------------------
bridge_edges = []
bridge_seen = set()

for cve_id, product_names in cve_to_products.items():
    cve_node = vuln_node_by_id.get(cve_id)
    if not cve_node:
        continue

    cve_text = node_text(cve_node)
    cve_emb = model.encode(cve_text)

    # ---- CVE -> software
    matched_softwares = []

    for product_name in product_names:
        for software_id, software_node in software_nodes.items():
            software_name = software_node.get("name", software_id)

            name_score = similar(product_name, software_name)
            desc_score = cosine(cve_emb, software_embeddings[software_id])

            final_score = 0.4 * name_score + 0.6 * desc_score

            if final_score >= 0.65:
                add_bridge(
                    bridge_edges,
                    bridge_seen,
                    cve_id,
                    software_id,
                    "related_software",
                    "hybrid_name_embedding",
                    final_score,
                    "vulnerability",
                    "attack",
                    [
                        f"product={product_name}",
                        f"software={software_name}",
                        f"name_score={round(name_score, 3)}",
                        f"desc_score={round(desc_score, 3)}"
                    ]
                )
                matched_softwares.append((software_id, final_score, product_name))

    # ---- CVE -> technique
    for software_id, software_score, product_name in matched_softwares:
        for technique_id in software_to_techniques.get(software_id, []):
            tech_emb = technique_embeddings.get(technique_id)
            if tech_emb is None:
                continue

            tech_desc_score = cosine(cve_emb, tech_emb)
            final_score = 0.5 * software_score + 0.5 * tech_desc_score

            if final_score >= 0.60:
                add_bridge(
                    bridge_edges,
                    bridge_seen,
                    cve_id,
                    technique_id,
                    "related_technique",
                    "software_plus_embedding",
                    final_score,
                    "vulnerability",
                    "attack",
                    [
                        f"matched_product={product_name}",
                        f"software_id={software_id}",
                        f"technique_id={technique_id}",
                        f"software_score={round(software_score, 3)}",
                        f"tech_desc_score={round(tech_desc_score, 3)}"
                    ]
                )

# -----------------------
# Save bridge graph
# -----------------------
bridge_graph = {
    "meta": {
        "name": "bridge_graph",
        "description": "Cross-layer bridge edges between vulnerability and ATT&CK graphs using hybrid name + embedding similarity.",
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