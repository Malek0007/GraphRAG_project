import json
import os
import numpy as np
from openai import embeddings
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

GRAPH_PATH = "data/graphrag/bridge_attack_vuln/global_graph.json"
EMBEDDINGS_PATH = "data/graphrag/bridge_attack_vuln/embeddings.npy"
METADATA_PATH = "data/graphrag/bridge_attack_vuln/embeddings_metadata.json"



def flatten_value(value):
    if isinstance(value, list):
        return " ; ".join(str(v) for v in value)
    if isinstance(value, dict):
        return " ; ".join(f"{k}={v}" for k, v in value.items())
    return str(value)


def build_node_text(node):
    parts = [
        f"node_type: {node.get('type', '')}",
        f"id: {node.get('id', '')}",
        f"name: {node.get('name', '')}",
        f"description: {node.get('description', '')}",
    ]

    if "properties" in node and node["properties"]:
        parts.append(f"properties: {flatten_value(node['properties'])}")

    return " | ".join(parts)


def build_edge_text(edge):
    parts = [
        f"edge_type: {edge.get('type', '')}",
        f"source: {edge.get('source', '')}",
        f"target: {edge.get('target', '')}",
    ]

    for field in ["method", "confidence", "layer_from", "layer_to", "evidence"]:
        if field in edge and edge[field] not in (None, "", [], {}):
            parts.append(f"{field}: {flatten_value(edge[field])}")

    if "properties" in edge and edge["properties"]:
        parts.append(f"properties: {flatten_value(edge['properties'])}")

    return " | ".join(parts)


def should_keep_node(node):
    # Prototype-focused filter
    keep_types = {
        "attack", "technique", "sub-technique",
        "cve", "cvss", "cvss_vector"
    }
    return node.get("type") in keep_types or str(node.get("id", "")).startswith("T")


def should_keep_edge(edge):
    keep_edge_types = {
        "exploited_via",
        "has_cvss",
        "subtechnique_of",
        "uses"
    }
    return edge.get("type") in keep_edge_types


def main():
    if not os.path.exists(GRAPH_PATH):
        raise FileNotFoundError(f"Graph file not found: {GRAPH_PATH}")

    with open(GRAPH_PATH, "r", encoding="utf-8") as f:
        graph = json.load(f)

    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    texts = []
    metadata = []

    kept_nodes = 0
    kept_edges = 0

    for node in nodes:
        if should_keep_node(node):
            texts.append(build_node_text(node))
            metadata.append({
                "kind": "node",
                "id": node.get("id"),
                "type": node.get("type"),
                "name": node.get("name", "")
            })
            kept_nodes += 1

    for edge in edges:
        if should_keep_edge(edge):
            texts.append(build_edge_text(edge))
            metadata.append({
                "kind": "edge",
                "source": edge.get("source"),
                "target": edge.get("target"),
                "type": edge.get("type"),
                "raw": edge
            })
            kept_edges += 1

    print(f"Original nodes: {len(nodes)}")
    print(f"Original edges: {len(edges)}")
    print(f"Kept nodes: {kept_nodes}")
    print(f"Kept edges: {kept_edges}")
    print(f"Total items to embed: {len(texts)}")

    print("Embedding locally...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    os.makedirs(os.path.dirname(EMBEDDINGS_PATH), exist_ok=True)

    np.save(EMBEDDINGS_PATH, embeddings)

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Embeddings saved to: {EMBEDDINGS_PATH}")
    print(f"Metadata saved to: {METADATA_PATH}")
    print("Done ✅")


if __name__ == "__main__":
    main()