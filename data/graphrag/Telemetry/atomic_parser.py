import json
import os
import re
from pathlib import Path

import yaml


ATOMICS_ROOT = "data/graphrag/Telemetry/atomics"
OUTPUT_PATH = "data/graphrag/Telemetry/atomics/Indexes/telemetry_graph.json"


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def make_node(node_id: str, node_type: str, name: str, description: str = "", properties=None):
    return {
        "id": node_id,
        "type": node_type,
        "name": name,
        "description": description,
        "properties": properties or {}
    }


def make_edge(source: str, target: str, edge_type: str, properties=None):
    return {
        "source": source,
        "target": target,
        "type": edge_type,
        "properties": properties or {}
    }


def extract_observables(command: str, description: str) -> list[dict]:
    """
    Rule-based observable extraction from command + description.
    Returns a list of observable descriptors.
    """
    text = f"{command}\n{description}".lower()
    observables = []

    rules = [
        {
            "id": "OBS::remote_file_download",
            "name": "Remote File Download",
            "description": "Download of a remote file or payload",
            "patterns": ["invoke-webrequest", "curl ", "wget "]
        },
        {
            "id": "OBS::base64_activity",
            "name": "Base64 Encode/Decode Activity",
            "description": "Base64 encoding or decoding activity",
            "patterns": ["base64", "frombase64string", "base64 -d"]
        },
        {
            "id": "OBS::powershell_module_import",
            "name": "PowerShell Module Import",
            "description": "Importing a PowerShell module",
            "patterns": ["import-module"]
        },
        {
            "id": "OBS::file_content_combination",
            "name": "Byte-level File Combination",
            "description": "Combining file contents or byte streams into a new file",
            "patterns": ["get-content", "set-content", "-encoding byte", "cat "]
        },
        {
            "id": "OBS::archive_creation",
            "name": "Archive Creation",
            "description": "Creation of archive files such as tar",
            "patterns": ["tar -cvf", "tar "]
        },
        {
            "id": "OBS::script_extraction_from_image",
            "name": "Script Extraction from Image",
            "description": "Extracting hidden script content from an image file",
            "patterns": ["extract-invoke-psimage", "strings ", "image"]
        },
        {
            "id": "OBS::decoded_script_execution",
            "name": "Decoded Script Execution",
            "description": "Decoded script content is executed",
            "patterns": [" . \"$home\\textextraction.ps1\"", " | sh", "textExtraction.ps1", "decoded.ps1"]
        },
        {
            "id": "OBS::file_deletion_cleanup",
            "name": "File Deletion Cleanup",
            "description": "Cleanup activity removing generated files",
            "patterns": ["remove-item", "rm "]
        },
        {
            "id": "OBS::execution_policy_bypass",
            "name": "PowerShell Execution Policy Bypass",
            "description": "PowerShell execution policy bypass attempt",
            "patterns": ["set-executionpolicy bypass"]
        },
    ]

    for rule in rules:
        if any(pattern in text for pattern in rule["patterns"]):
            observables.append({
                "id": rule["id"],
                "name": rule["name"],
                "description": rule["description"]
            })

    return observables


def parse_atomic_yaml(yaml_path: Path):
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not data:
        return [], []

    technique_id = data.get("attack_technique")
    technique_name = data.get("display_name", technique_id)

    if not technique_id:
        return [], []

    nodes = []
    edges = []

    # Technique node
    technique_node = make_node(
        node_id=technique_id,
        node_type="technique",
        name=technique_name,
        description=f"Atomic Red Team technique: {technique_name}",
        properties={"source": "atomic_red_team"}
    )
    nodes.append(technique_node)

    atomic_tests = data.get("atomic_tests", [])

    for test in atomic_tests:
        test_guid = test.get("auto_generated_guid", slugify(test.get("name", "unknown_test")))
        atomic_id = f"ATOMIC::{test_guid}"

        test_name = normalize_text(test.get("name", "Unnamed Atomic Test"))
        test_description = normalize_text(test.get("description", ""))

        atomic_node = make_node(
            node_id=atomic_id,
            node_type="atomic_test",
            name=test_name,
            description=test_description,
            properties={
                "source": "atomic_red_team",
                "technique_id": technique_id
            }
        )
        nodes.append(atomic_node)

        edges.append(make_edge(
            source=technique_id,
            target=atomic_id,
            edge_type="has_atomic_test"
        ))

        # Supported platforms
        for platform in test.get("supported_platforms", []):
            platform_name = str(platform).lower().strip()
            platform_id = f"PLATFORM::{platform_name}"

            platform_node = make_node(
                node_id=platform_id,
                node_type="platform",
                name=platform_name,
                description=f"Supported platform: {platform_name}"
            )
            nodes.append(platform_node)

            edges.append(make_edge(
                source=atomic_id,
                target=platform_id,
                edge_type="runs_on"
            ))

        # Executor command
        executor = test.get("executor", {})
        command = normalize_text(executor.get("command", ""))
        cleanup_command = normalize_text(executor.get("cleanup_command", ""))

        if command:
            command_id = f"COMMAND::{test_guid}"
            command_node = make_node(
                node_id=command_id,
                node_type="command",
                name=f"Command for {test_name}",
                description=command,
                properties={
                    "executor_name": executor.get("name", ""),
                    "elevation_required": executor.get("elevation_required", False)
                }
            )
            nodes.append(command_node)

            edges.append(make_edge(
                source=atomic_id,
                target=command_id,
                edge_type="has_command"
            ))

        if cleanup_command:
            cleanup_id = f"CLEANUP::{test_guid}"
            cleanup_node = make_node(
                node_id=cleanup_id,
                node_type="cleanup_command",
                name=f"Cleanup for {test_name}",
                description=cleanup_command
            )
            nodes.append(cleanup_node)

            edges.append(make_edge(
                source=atomic_id,
                target=cleanup_id,
                edge_type="has_cleanup_command"
            ))

        # Observables from description + command
        extracted_observables = extract_observables(command, test_description)

        for obs in extracted_observables:
            obs_node = make_node(
                node_id=obs["id"],
                node_type="observable",
                name=obs["name"],
                description=obs["description"],
                properties={"source": "atomic_red_team"}
            )
            nodes.append(obs_node)

            edges.append(make_edge(
                source=atomic_id,
                target=obs["id"],
                edge_type="produces_observable"
            ))

            edges.append(make_edge(
                source=obs["id"],
                target=technique_id,
                edge_type="indicates"
            ))

    return nodes, edges


def deduplicate_nodes(nodes: list[dict]) -> list[dict]:
    by_id = {}
    for node in nodes:
        by_id[node["id"]] = node
    return list(by_id.values())


def deduplicate_edges(edges: list[dict]) -> list[dict]:
    seen = set()
    unique_edges = []

    for edge in edges:
        key = (
            edge["source"],
            edge["target"],
            edge["type"],
            json.dumps(edge.get("properties", {}), sort_keys=True)
        )
        if key not in seen:
            seen.add(key)
            unique_edges.append(edge)

    return unique_edges


def main():
    root = Path(ATOMICS_ROOT)

    if not root.exists():
        raise FileNotFoundError(f"Atomics root not found: {ATOMICS_ROOT}")

    all_nodes = []
    all_edges = []

    yaml_files = list(root.rglob("*.yaml"))

    print(f"Found {len(yaml_files)} YAML files")

    for yaml_file in yaml_files:
        try:
            nodes, edges = parse_atomic_yaml(yaml_file)
            all_nodes.extend(nodes)
            all_edges.extend(edges)
        except Exception as e:
            print(f"Error parsing {yaml_file}: {e}")

    all_nodes = deduplicate_nodes(all_nodes)
    all_edges = deduplicate_edges(all_edges)

    graph = {
        "meta": {
            "name": "telemetry_atomics_graph",
            "description": "Telemetry/Atomic Red Team graph built from Atomic YAML files",
            "total_nodes": len(all_nodes),
            "total_edges": len(all_edges)
        },
        "nodes": all_nodes,
        "edges": all_edges
    }

    os.makedirs(Path(OUTPUT_PATH).parent, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2)

    print(f"Saved graph to: {OUTPUT_PATH}")
    print(f"Total nodes: {len(all_nodes)}")
    print(f"Total edges: {len(all_edges)}")


if __name__ == "__main__":
    main()