import json
import os
from datetime import datetime
from collections import Counter

INPUT = "data/raw/enterprise-attack.json"
OUT_GRAPH = "data/graphrag/attack_graph_graphrag.json"
OUT_TEXTS = "data/graphrag/attack_node_texts.jsonl"


def clean_text(x):
    if not x:
        return ""
    return " ".join(str(x).split())


def get_attack_external_id(obj):
    for ref in obj.get("external_references", []):
        if ref.get("source_name") == "mitre-attack" and ref.get("external_id"):
            return ref["external_id"]
    return None


def get_attack_url(obj):
    for ref in obj.get("external_references", []):
        if ref.get("source_name") == "mitre-attack" and ref.get("url"):
            return ref["url"]
    return None


def normalize_node_type(obj, attack_id):
    stix_type = obj.get("type", "")

    if stix_type == "x-mitre-tactic":
        return "tactic"

    if stix_type == "attack-pattern":
        if obj.get("x_mitre_is_subtechnique") is True:
            return "sub-technique"
        if attack_id and "." in attack_id:
            return "sub-technique"
        return "technique"

    if stix_type == "course-of-action":
        return "mitigation"

    if stix_type == "intrusion-set":
        return "group"

    if stix_type in {"malware", "tool"}:
        return "software"

    if stix_type == "campaign":
        return "campaign"

    if stix_type == "x-mitre-data-source":
        return "data_source"

    if stix_type == "x-mitre-data-component":
        return "data_component"

    return None


def normalize_rel(rel):
    if not rel:
        return "related_to"

    rel = rel.strip().lower()

    mapping = {
        "uses": "uses",
        "mitigates": "mitigates",
        "detects": "detects",
        "subtechnique-of": "subtechnique_of",
        "revoked-by": "revoked_by",
        "related-to": "related_to",
    }
    return mapping.get(rel, rel.replace("-", "_"))


def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


with open(INPUT, "r", encoding="utf-8") as f:
    bundle = json.load(f)

objects = bundle.get("objects", [])

nodes = []
edges = []
texts = []

# maps STIX id -> exported ATT&CK id
id_map = {}

# exported node id -> node type
type_map = {}

# tactic shortname -> tactic ATT&CK id
tactic_shortname_to_id = {}

# -----------------------
# PASS 1: BUILD NODES
# -----------------------
for obj in objects:
    if obj.get("revoked") is True or obj.get("x_mitre_deprecated") is True:
        continue

    attack_id = get_attack_external_id(obj)
    node_type = normalize_node_type(obj, attack_id)

    if not node_type:
        continue

    stix_id = obj["id"]
    export_id = attack_id if attack_id else stix_id

    node = {
        "id": export_id,
        "type": node_type,
        "name": obj.get("name", export_id),
        "description": obj.get("description", ""),
        "properties": {
            "stix_id": stix_id,
            "stix_type": obj.get("type"),
            "attack_url": get_attack_url(obj),
        }
    }

    if "x_mitre_platforms" in obj:
        node["properties"]["platforms"] = obj["x_mitre_platforms"]
    if "x_mitre_domains" in obj:
        node["properties"]["domains"] = obj["x_mitre_domains"]
    if "x_mitre_permissions_required" in obj:
        node["properties"]["permissions_required"] = obj["x_mitre_permissions_required"]
    if "x_mitre_data_sources" in obj:
        node["properties"]["data_sources"] = obj["x_mitre_data_sources"]

    nodes.append(node)
    id_map[stix_id] = export_id
    type_map[export_id] = node_type

    texts.append({
        "id": export_id,
        "type": node_type,
        "name": obj.get("name", export_id),
        "text": f"{node_type.upper()} {export_id} — {obj.get('name', export_id)}. {clean_text(obj.get('description', ''))}"
    })

    if obj.get("type") == "x-mitre-tactic":
        shortname = obj.get("x_mitre_shortname")
        if shortname:
            tactic_shortname_to_id[shortname] = export_id

# -----------------------
# PASS 2: RELATIONSHIP OBJECTS
# -----------------------
seen_edges = set()

def add_edge(source, edge_type, target, properties=None):
    key = (source, edge_type, target)
    if key in seen_edges:
        return

    seen_edges.add(key)

    edge = {
        "source": source,
        "target": target,
        "type": edge_type,
    }
    if properties:
        edge["properties"] = properties

    edges.append(edge)


for obj in objects:
    if obj.get("type") != "relationship":
        continue
    if obj.get("revoked") is True or obj.get("x_mitre_deprecated") is True:
        continue

    src = id_map.get(obj.get("source_ref"))
    tgt = id_map.get(obj.get("target_ref"))

    if not src or not tgt:
        continue

    rel = normalize_rel(obj.get("relationship_type"))

    add_edge(
        src,
        rel,
        tgt,
        {
            "stix_id": obj.get("id"),
            "description": obj.get("description", "")
        }
    )

# -----------------------
# PASS 3: technique/sub-technique -> tactic
# from kill_chain_phases
# -----------------------
for obj in objects:
    if obj.get("type") != "attack-pattern":
        continue
    if obj.get("revoked") is True or obj.get("x_mitre_deprecated") is True:
        continue

    src = id_map.get(obj["id"])
    if not src:
        continue

    for phase in obj.get("kill_chain_phases", []):
        if phase.get("kill_chain_name") != "mitre-attack":
            continue

        shortname = phase.get("phase_name")
        tactic_id = tactic_shortname_to_id.get(shortname)
        if tactic_id:
            add_edge(
                src,
                "belongs_to",
                tactic_id,
                {
                    "method": "kill_chain_phase"
                }
            )

# -----------------------
# PASS 4: force sub-technique -> parent technique
# -----------------------
node_ids = {n["id"] for n in nodes}

for node in nodes:
    nid = node["id"]
    if node["type"] == "sub-technique" and "." in nid:
        parent = nid.split(".")[0]
        if parent in node_ids:
            add_edge(
                nid,
                "subtechnique_of",
                parent,
                {
                    "method": "attack_id_pattern"
                }
            )

# -----------------------
# OUTPUT
# -----------------------
out = {
    "meta": {
        "name": "attack_graph_graphrag",
        "description": "GraphRAG-ready MITRE ATT&CK Enterprise graph built from data/raw/enterprise-attack.json",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "total_nodes": len(nodes),
        "total_edges": len(edges)
    },
    "nodes": nodes,
    "edges": edges
}

ensure_parent_dir(OUT_GRAPH)
ensure_parent_dir(OUT_TEXTS)

with open(OUT_GRAPH, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)

with open(OUT_TEXTS, "w", encoding="utf-8") as f:
    for row in texts:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print("Saved:", OUT_GRAPH)
print("Saved:", OUT_TEXTS)
print("Nodes:", len(nodes))
print("Edges:", len(edges))

node_counts = Counter(n["type"] for n in nodes)
edge_counts = Counter(e["type"] for e in edges)

print("\nNode types:")
for k, v in sorted(node_counts.items()):
    print(f"  {k}: {v}")

print("\nEdge types:")
for k, v in sorted(edge_counts.items()):
    print(f"  {k}: {v}")