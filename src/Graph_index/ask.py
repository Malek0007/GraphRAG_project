from retriever import Retriever
from graph_utils import GraphStore
import re

GRAPH_PATH = "data/graphrag/bridge_attack_vuln/global_graph.json"
EMBEDDINGS_PATH = "data/graphrag/bridge_attack_vuln/embeddings.npy"
METADATA_PATH = "data/graphrag/bridge_attack_vuln/embeddings_metadata.json"


# ----------- ID DETECTION (NEW 🔥) -----------
def detect_exact_id(question):
    cve_match = re.findall(r"CVE-\d{4}-\d+", question, re.IGNORECASE)
    if cve_match:
        return cve_match[0].upper()

    t_match = re.findall(r"T\d{4}", question, re.IGNORECASE)
    if t_match:
        return t_match[0].upper()

    return None


# ----------- FORMAT FUNCTIONS -----------
def format_node(node: dict | None) -> str:
    if not node:
        return "None"

    parts = [
        f"ID: {node.get('id', '')}",
        f"Type: {node.get('type', '')}",
    ]

    if node.get("name"):
        parts.append(f"Name: {node.get('name')}")

    if node.get("description"):
        desc = str(node.get("description"))[:300]
        parts.append(f"Description: {desc}")

    return " | ".join(parts)


def format_edge(edge: dict | None) -> str:
    if not edge:
        return "None"

    parts = [
        f"{edge.get('source', '')} --[{edge.get('type', '')}]--> {edge.get('target', '')}"
    ]

    for field in ["method", "confidence", "layer_from", "layer_to"]:
        if edge.get(field) not in (None, "", [], {}):
            parts.append(f"{field}={edge.get(field)}")

    if edge.get("evidence"):
        parts.append(f"evidence={edge.get('evidence')}")

    if edge.get("properties"):
        parts.append(f"properties={edge.get('properties')}")

    return " | ".join(parts)


# ----------- PRINT RESULT -----------
def print_result(expanded: dict, score: float, rank: int) -> None:
    print("\n" + "=" * 90)
    print(f"RESULT {rank} | score={score:.4f}")
    print("=" * 90)

    if expanded["kind"] == "node":
        print("Matched Node:")
        print(format_node(expanded["matched_node"]))

        if expanded["connected_edges"]:
            print("\nConnected Edges:")
            for edge in expanded["connected_edges"][:5]:
                print(f"- {format_edge(edge)}")

        if expanded["neighbor_nodes"]:
            print("\nNeighbor Nodes:")
            seen = set()
            count = 0
            for node in expanded["neighbor_nodes"]:
                node_id = node.get("id")
                if node_id in seen:
                    continue
                seen.add(node_id)
                print(f"- {format_node(node)}")
                count += 1
                if count >= 5:
                    break

    elif expanded["kind"] == "edge":
        print("Matched Edge:")
        print(format_edge(expanded["matched_edge"]))

        print("\nSource Node:")
        print(format_node(expanded["source_node"]))

        print("\nTarget Node:")
        print(format_node(expanded["target_node"]))


# ----------- MAIN PIPELINE -----------
def main():
    question = input("Ask a question: ").strip()
    if not question:
        print("Question cannot be empty.")
        return

    retriever = Retriever(EMBEDDINGS_PATH, METADATA_PATH)
    graph = GraphStore(GRAPH_PATH)

    # 🔥 NEW: Exact ID handling
    exact_id = detect_exact_id(question)

    if exact_id:
        print(f"\nExact match detected: {exact_id}")
        node = graph.get_node(exact_id)

        if node:
            results = [{
                "kind": "node",
                "id": exact_id,
                "score": 1.0
            }]
        else:
            print("ID not found → fallback to semantic search")
            results = retriever.search(question, top_k=5)
    else:
        results = retriever.search(question, top_k=5)

    print("\nTop Retrieved Results")
    for i, result in enumerate(results, start=1):
        expanded = graph.expand_result(result)
        print_result(expanded, result["score"], i)


if __name__ == "__main__":
    main()