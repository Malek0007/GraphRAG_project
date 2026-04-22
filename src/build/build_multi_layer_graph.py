"""
Merges four graph layers into a single multi-layer global graph:

  Layer 1 — ATT&CK        (attacks/attack_graph_graphrag.json)
  Layer 2 — NVD CVE       (vulnerabilities/vuln_graph_graphrag.json)
  Layer 3 — Atomic Red Team (Telemetry/atomics/Indexes/telemetry_graph.json)
  Layer 4 — Threat Reports  (report_layer/report_graph.json)

Bridge edges
  • L2↔L1: existing CVE↔ATT&CK edges from bridge_graph.json
  • L3↔L1: has_atomic_test edges already embedded in atomic graph
  • L4↔L1: mentions_technique edges already embedded in report graph
  • L4↔L2: mentions_cve edges already embedded in report graph
  • L4↔L1: new report malware → ATT&CK software (name similarity ≥ 0.72)

Output: data/graphrag/multi_layer/global_graph.json
"""
import json
import os
from datetime import datetime
from difflib import SequenceMatcher

ATTACK_FILE = "data/graphrag/attacks/attack_graph_graphrag.json"
VULN_FILE = "data/graphrag/vulnerabilities/vuln_graph_graphrag.json"
ATOMIC_FILE = "data/graphrag/Telemetry/atomics/Indexes/telemetry_graph.json"
REPORT_FILE = "data/graphrag/report_layer/report_graph.json"
BRIDGE_FILE = "data/graphrag/bridge_attack_vuln/bridge_graph.json"

OUTPUT_DIR = "data/graphrag/multi_layer"
OUTPUT_GLOBAL = f"{OUTPUT_DIR}/global_graph.json"

SIMILARITY_THRESHOLD = 0.72
MALWARE_TYPES = {"malware", "threat_actor", "tool"}


def load_graph(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_name(text: str) -> str:
    return (text or "").lower().replace("_", " ").replace("-", " ").strip()


def similar(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_name(a), normalize_name(b)).ratio()


def add_bridge(
    edges: list, seen: set, source, target, edge_type,
    method, confidence, layer_from, layer_to, evidence=None
):
    key = (source, edge_type, target)
    if key in seen:
        return
    seen.add(key)
    edges.append({
        "source": source,
        "target": target,
        "type": edge_type,
        "method": method,
        "confidence": round(float(confidence), 3),
        "layer_from": layer_from,
        "layer_to": layer_to,
        "evidence": evidence or [],
    })


def build_report_software_bridges(
    report_nodes: list[dict],
    attack_nodes: list[dict],
) -> list[dict]:
    """Match report malware/tool/threat_actor nodes to ATT&CK software by name."""
    attack_software = {
        n["id"]: n.get("name", n["id"])
        for n in attack_nodes
        if n.get("type") == "software"
    }

    # Only scoped (non-canonical) report nodes of malware/tool/threat_actor type
    report_entities = [
        n for n in report_nodes
        if n.get("type") in MALWARE_TYPES and "::" in n.get("id", "")
        and not n.get("id", "").startswith("REPORT::")
    ]

    bridges: list[dict] = []
    seen: set[tuple] = set()

    for rm in report_entities:
        rm_name = rm.get("name", rm["id"])
        for sw_id, sw_name in attack_software.items():
            score = similar(rm_name, sw_name)
            if score >= SIMILARITY_THRESHOLD:
                add_bridge(
                    bridges, seen,
                    rm["id"], sw_id,
                    "related_software",
                    "report_name_match",
                    score,
                    "report", "attack",
                    [f"report_entity={rm_name}", f"attack_software={sw_name}"],
                )

    return bridges


def merge_nodes(layer_node_lists: list[list[dict]]) -> list[dict]:
    """Merge node lists; first-seen wins so ATT&CK nodes take priority."""
    seen: set[str] = set()
    merged: list[dict] = []
    for layer_nodes in layer_node_lists:
        for node in layer_nodes:
            nid = node.get("id")
            if nid and nid not in seen:
                seen.add(nid)
                merged.append(node)
    return merged


def merge_edges(layer_edge_lists: list[list[dict]]) -> list[dict]:
    """Merge edge lists, dedup by (source, type, target)."""
    seen: set[tuple] = set()
    merged: list[dict] = []
    for layer_edges in layer_edge_lists:
        for edge in layer_edges:
            key = (edge.get("source"), edge.get("type"), edge.get("target"))
            if key not in seen:
                seen.add(key)
                merged.append(edge)
    return merged


def main():
    print("Loading graphs...")
    attack = load_graph(ATTACK_FILE)
    vuln = load_graph(VULN_FILE)
    atomic = load_graph(ATOMIC_FILE)
    report = load_graph(REPORT_FILE)
    bridge = load_graph(BRIDGE_FILE)

    print(f"  ATT&CK:  {len(attack['nodes']):>6} nodes, {len(attack['edges']):>7} edges")
    print(f"  Vuln:    {len(vuln['nodes']):>6} nodes, {len(vuln['edges']):>7} edges")
    print(f"  Atomic:  {len(atomic['nodes']):>6} nodes, {len(atomic['edges']):>7} edges")
    print(f"  Reports: {len(report['nodes']):>6} nodes, {len(report['edges']):>7} edges")
    print(f"  Bridge:  {len(bridge['edges']):>7} edges (CVE↔ATT&CK)")

    print("\nBuilding report→software bridge edges...")
    rpt_sw_bridges = build_report_software_bridges(report["nodes"], attack["nodes"])
    print(f"  Report→software bridges: {len(rpt_sw_bridges)}")

    print("\nMerging layers (ATT&CK nodes take priority on ID conflicts)...")
    all_nodes = merge_nodes([
        attack["nodes"],
        vuln["nodes"],
        atomic["nodes"],
        report["nodes"],
    ])

    all_edges = merge_edges([
        attack["edges"],
        vuln["edges"],
        bridge["edges"],
        atomic["edges"],
        report["edges"],
        rpt_sw_bridges,
    ])

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    global_graph = {
        "meta": {
            "name": "multi_layer_global_graph",
            "description": (
                "Multi-layer graph: ATT&CK (L1) + NVD CVE (L2) + "
                "Atomic Red Team (L3) + Threat Reports (L4)."
            ),
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "total_nodes": len(all_nodes),
            "total_edges": len(all_edges),
            "layer_stats": {
                "attack_nodes": len(attack["nodes"]),
                "vuln_nodes": len(vuln["nodes"]),
                "atomic_nodes": len(atomic["nodes"]),
                "report_nodes": len(report["nodes"]),
                "attack_vuln_bridge_edges": len(bridge["edges"]),
                "report_software_bridge_edges": len(rpt_sw_bridges),
            },
        },
        "nodes": all_nodes,
        "edges": all_edges,
    }

    with open(OUTPUT_GLOBAL, "w", encoding="utf-8") as f:
        json.dump(global_graph, f, indent=2, ensure_ascii=False)

    print(f"\nSaved: {OUTPUT_GLOBAL}")
    print(f"Total nodes: {len(all_nodes)}")
    print(f"Total edges: {len(all_edges)}")


if __name__ == "__main__":
    main()
