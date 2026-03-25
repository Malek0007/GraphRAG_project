import json
from collections import defaultdict
from typing import Any


class GraphStore:
    def __init__(self, graph_path: str):
        with open(graph_path, "r", encoding="utf-8") as f:
            self.graph = json.load(f)

        self.nodes = self.graph.get("nodes", [])
        self.edges = self.graph.get("edges", [])

        self.node_by_id = {
            node["id"]: node
            for node in self.nodes
            if "id" in node
        }

        self.edges_by_node = defaultdict(list)
        for edge in self.edges:
            source = edge.get("source")
            target = edge.get("target")

            if source:
                self.edges_by_node[source].append(edge)
            if target:
                self.edges_by_node[target].append(edge)

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        return self.node_by_id.get(node_id)

    def get_neighbors(self, node_id: str) -> list[dict[str, Any]]:
        return self.edges_by_node.get(node_id, [])

    def expand_result(self, result: dict[str, Any], max_edges: int = 8) -> dict[str, Any]:
        if result["kind"] == "node":
            node_id = result["id"]
            node = self.get_node(node_id)
            connected_edges = self.get_neighbors(node_id)[:max_edges]

            neighbor_nodes = []
            for edge in connected_edges:
                source = edge.get("source")
                target = edge.get("target")

                other_id = None
                if source == node_id:
                    other_id = target
                elif target == node_id:
                    other_id = source

                if other_id and other_id in self.node_by_id:
                    neighbor_nodes.append(self.node_by_id[other_id])

            return {
                "kind": "node",
                "matched_node": node,
                "connected_edges": connected_edges,
                "neighbor_nodes": neighbor_nodes,
            }

        if result["kind"] == "edge":
            edge = result["raw"]
            source_node = self.get_node(edge.get("source"))
            target_node = self.get_node(edge.get("target"))

            return {
                "kind": "edge",
                "matched_edge": edge,
                "source_node": source_node,
                "target_node": target_node,
            }

        return {"kind": "unknown"}