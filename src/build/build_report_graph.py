"""
Normalises the 6 threat-intelligence JSON reports into a single graph
stored at data/graphrag/report_layer/report_graph.json.

Design decisions
- CVE IDs (CVE-YYYY-NNNN) and ATT&CK IDs (TNNNN / TNNNN.NNN) are kept
  as-is so they act as bridge anchors to other layers.
- All other nodes are scoped as  <report_id>::<original_id>  to avoid
  cross-report ID collisions.
- Edge relations are normalised from CAPS to snake_case.
"""
import json
import os
import re
from datetime import datetime
from pathlib import Path

REPORTS_DIR = "data/graphrag/reports"
OUTPUT_DIR = "data/graphrag/report_layer"
OUTPUT_PATH = f"{OUTPUT_DIR}/report_graph.json"

CVE_RE = re.compile(r"^CVE-\d{4}-\d+$", re.IGNORECASE)
TECHNIQUE_RE = re.compile(r"^T\d{4}(\.\d{3})?$")

RELATION_MAP: dict[str, str] = {
    "IMPLEMENTS": "implements_technique",
    "USES": "uses",
    "TARGETS": "targets",
    "EXECUTES": "executes",
    "LAUNCHES": "launches",
    "MODIFIES": "modifies",
    "CONNECTS_TO": "connects_to",
    "DROPS": "drops",
    "SPAWNS": "spawns",
    "INFECTS": "infects",
    "ABUSES_FOR_PERSISTENCE": "achieves_persistence",
    "PERFORMS": "performs",
    "DEPLOYS": "deploys",
    "OPERATES": "operates",
    "FORM": "is_form_of",
    "DOWNLOADS": "downloads",
    "EXTRACTS": "extracts",
    "TRICKS": "tricks",
    "DISGUISES_AS": "disguises_as",
    "DELIVERS": "delivers",
    "EXPLOITS": "exploits_cve",
    "DOWNLOADS_FROM": "downloads_from",
    "BEACONS_TO": "connects_to",
    "EXFILTRATES_VIA": "exfiltrates_via",
    "COMMUNICATES_WITH": "connects_to",
    "ENCRYPTS": "encrypts",
    "DECRYPTS": "decrypts",
    "OBFUSCATES": "obfuscates",
    "DEOBFUSCATES": "deobfuscates",
    "RELATED_TO": "related_to",
    "ASSOCIATED_WITH": "associated_with",
}

TYPE_MAP: dict[str, str] = {
    "threat_actor": "threat_actor",
    "malware": "malware",
    "malware_campaign": "malware",
    "malware_variant": "malware",
    "malware_module": "malware",
    "attack_technique": "technique",
    "vulnerability": "cve",
    "behavior": "behavior",
    "persistence": "persistence_mechanism",
    "persistence_mechanism": "persistence_mechanism",
    "process": "process",
    "artifact": "artifact",
    "tool": "tool",
    "tool_or_method": "tool",
    "infrastructure": "infrastructure",
    "platform": "platform",
    "component": "component",
    "crypto": "crypto_algo",
    "c2": "c2_infra",
    "ipc": "ipc",
    "target": "target_sector",
    "attack_vector": "attack_vector",
    "campaign": "campaign",
    "sector": "sector",
    "region": "region",
    "file_type": "file_type",
    "credential_store": "credential_store",
    "technique": "technique",
}


def is_cve(node_id: str) -> bool:
    return bool(CVE_RE.match(node_id))


def is_technique(node_id: str) -> bool:
    return bool(TECHNIQUE_RE.match(node_id))


def scoped_id(report_id: str, node_id: str) -> str:
    if is_technique(node_id):
        return node_id
    if is_cve(node_id):
        return node_id.upper()
    return f"{report_id}::{node_id}"


def make_node(node_id, node_type, name, description="", properties=None):
    return {
        "id": node_id,
        "type": node_type,
        "name": name,
        "description": description,
        "properties": properties or {},
    }


def make_edge(source, target, edge_type, properties=None):
    return {
        "source": source,
        "target": target,
        "type": edge_type,
        "properties": properties or {},
    }


def parse_report(path: Path) -> tuple[list[dict], list[dict]]:
    with open(path, "r", encoding="utf-8") as f:
        report = json.load(f)

    rid = report["report_id"]
    title = report.get("title", rid)
    source = report.get("source", "unknown")
    summary = report.get("summary", title)
    report_node_id = f"REPORT::{rid}"

    nodes: list[dict] = []
    edges: list[dict] = []

    nodes.append(make_node(
        report_node_id, "threat_report", title,
        description=summary,
        properties={"source": source, "report_id": rid, "layer": "report"},
    ))

    for rn in report.get("nodes", []):
        oid = rn["id"]
        otype = rn.get("type", "unknown")
        nid = scoped_id(rid, oid)
        ntype = TYPE_MAP.get(otype, otype)

        nodes.append(make_node(
            nid, ntype, oid,
            description=rn.get("description", oid),
            properties={
                "source": "threat_report",
                "report_id": rid,
                "original_type": otype,
                "layer": "report",
            },
        ))

        if is_technique(oid):
            edges.append(make_edge(
                report_node_id, nid, "mentions_technique",
                {"confidence": 1.0, "layer_from": "report", "layer_to": "attack"},
            ))
        elif is_cve(oid):
            edges.append(make_edge(
                report_node_id, nid, "mentions_cve",
                {"confidence": 1.0, "layer_from": "report", "layer_to": "vulnerability"},
            ))
        else:
            edges.append(make_edge(report_node_id, nid, "mentions_entity"))

    for re_edge in report.get("edges", []):
        src = scoped_id(rid, re_edge["from"])
        tgt = scoped_id(rid, re_edge["to"])
        relation = re_edge.get("relation", "RELATED_TO")
        etype = RELATION_MAP.get(relation, relation.lower())
        edges.append(make_edge(src, tgt, etype, {"original_relation": relation}))

    for i, obs in enumerate(report.get("observables", [])):
        otype = obs.get("type", "observable")
        oval = (
            obs.get("value") or obs.get("command") or obs.get("description") or ""
        )[:200]
        obs_id = f"{rid}::OBS::{otype}::{i}"
        nodes.append(make_node(
            obs_id, "observable", f"{otype} observable",
            description=oval,
            properties={"source": "threat_report", "obs_type": otype, "layer": "report"},
        ))
        edges.append(make_edge(report_node_id, obs_id, "has_observable"))

    return nodes, edges


def main():
    reports_dir = Path(REPORTS_DIR)
    report_files = sorted(reports_dir.glob("report*.json"))

    if not report_files:
        raise FileNotFoundError(f"No report files found in {REPORTS_DIR}")

    all_nodes: list[dict] = []
    all_edges: list[dict] = []
    seen_nodes: set[str] = set()
    seen_edges: set[tuple] = set()

    for rf in report_files:
        print(f"Parsing {rf.name} ...")
        nodes, edges = parse_report(rf)

        for n in nodes:
            if n["id"] not in seen_nodes:
                seen_nodes.add(n["id"])
                all_nodes.append(n)

        for e in edges:
            key = (e["source"], e["target"], e["type"])
            if key not in seen_edges:
                seen_edges.add(key)
                all_edges.append(e)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    graph = {
        "meta": {
            "name": "report_graph",
            "description": "Threat intelligence report layer (Layer 4).",
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "total_nodes": len(all_nodes),
            "total_edges": len(all_edges),
        },
        "nodes": all_nodes,
        "edges": all_edges,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2, ensure_ascii=False)

    print(f"\nSaved: {OUTPUT_PATH}")
    print(f"Total nodes: {len(all_nodes)}")
    print(f"Total edges: {len(all_edges)}")


if __name__ == "__main__":
    main()
