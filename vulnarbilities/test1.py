import json
import hashlib

def ref_id(url: str) -> str:
    return "ref:" + hashlib.md5(url.encode("utf-8")).hexdigest()[:12]

def pick_english_description(descriptions):
    for d in descriptions or []:
        if d.get("lang") == "en":
            return d.get("value")
    return None

def best_cvss_v31(metrics):
    """Return best available CVSS v3.1 (prefer NVD Primary)."""
    arr = (metrics or {}).get("cvssMetricV31") or []
    # prefer NVD Primary if exists
    for m in arr:
        if m.get("source") == "nvd@nist.gov" and m.get("type") == "Primary":
            return m.get("cvssData", {})
    # else take first
    if arr:
        return arr[0].get("cvssData", {})
    return None

def extract_cwes(weaknesses):
    cwes = set()
    for w in weaknesses or []:
        for desc in w.get("description", []) or []:
            val = desc.get("value")
            if isinstance(val, str) and val.startswith("CWE-"):
                cwes.add(val)
    return sorted(cwes)

def extract_cpes(configurations):
    cpes = set()
    for conf in configurations or []:
        for node in conf.get("nodes", []) or []:
            for m in node.get("cpeMatch", []) or []:
                crit = m.get("criteria")
                if m.get("vulnerable") is True and crit:
                    cpes.add(crit)
    return sorted(cpes)

# -------------------------
# LOAD NVD JSON 2.0 FILE
# -------------------------
with open("vulnarbilities/nvd_cves.json", "r", encoding="utf-8") as f:
    nvd = json.load(f)

nodes = {}     # id -> node dict
triples = []   # list of (subject, verb, object) with rich details

def add_node(node_id, node_type, name=None, description=None, props=None):
    if node_id not in nodes:
        nodes[node_id] = {
            "id": node_id,
            "type": node_type,
            "name": name,
            "description": description,
            "properties": props or {}
        }
    else:
        # merge missing fields
        if description and not nodes[node_id].get("description"):
            nodes[node_id]["description"] = description
        if name and not nodes[node_id].get("name"):
            nodes[node_id]["name"] = name

def add_triple(s, verb, o, rel_props=None):
    triples.append({
        "subject": s,
        "verb": verb,
        "object": o,
        "relationship_properties": rel_props or {}
    })

for item in nvd.get("vulnerabilities", []):
    cve = item.get("cve", {})
    cve_id = cve.get("id")
    if not cve_id:
        continue

    desc = pick_english_description(cve.get("descriptions"))
    cvss = best_cvss_v31(cve.get("metrics"))

    # --- CVE NODE ---
    add_node(
        cve_id,
        "cve",
        name=cve_id,
        description=desc,
        props={
            "sourceIdentifier": cve.get("sourceIdentifier"),
            "published": cve.get("published"),
            "lastModified": cve.get("lastModified"),
            "vulnStatus": cve.get("vulnStatus"),
            "cvss_v31_vector": cvss.get("vectorString") if cvss else None,
            "cvss_v31_baseScore": cvss.get("baseScore") if cvss else None,
            "cvss_v31_severity": cvss.get("baseSeverity") if cvss else None,
        }
    )

    # --- CWE NODES + EDGES ---
    for cwe in extract_cwes(cve.get("weaknesses")):
        add_node(cwe, "cwe", name=cwe)
        add_triple(
            {"id": cve_id, "type": "cve", "name": nodes[cve_id]["name"]},
            "has_weakness",
            {"id": cwe, "type": "cwe", "name": nodes[cwe]["name"]},
            {"source": "nvd.weaknesses"}
        )

    # --- CPE NODES + EDGES ---
    for cpe in extract_cpes(cve.get("configurations")):
        add_node(cpe, "cpe", name=cpe)
        add_triple(
            {"id": cve_id, "type": "cve", "name": nodes[cve_id]["name"]},
            "affects",
            {"id": cpe, "type": "cpe", "name": nodes[cpe]["name"]},
            {"source": "nvd.configurations"}
        )

    # --- REFERENCES + EDGES ---
    for r in cve.get("references", []) or []:
        url = r.get("url")
        if not url:
            continue
        rid = ref_id(url)
        add_node(rid, "reference", name=url, props={"url": url, "source": r.get("source"), "tags": r.get("tags")})
        add_triple(
            {"id": cve_id, "type": "cve", "name": nodes[cve_id]["name"]},
            "has_reference",
            {"id": rid, "type": "reference", "name": nodes[rid]["name"]},
            {"source": "nvd.references"}
        )

graph = {
    "meta": {
        "name": "nvd_layer2_graph",
        "description": "NVD CVE 2.0 feed converted to GraphRAG triples with verbs (CVE/CWE/CPE/References).",
        "node_count": len(nodes),
        "triple_count": len(triples)
    },
    "nodes": list(nodes.values()),
    "triples": triples
}

with open("nvd_layer2_graphrag.json", "w", encoding="utf-8") as f:
    json.dump(graph, f, indent=2)

print("Saved: nvd_layer2_graphrag.json")
print("Nodes:", len(nodes), "Triples:", len(triples))