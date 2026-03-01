import json
from datetime import datetime
from collections import defaultdict

# ----------------------------
# INPUT / OUTPUT
# ----------------------------
IN_FILE = "vulnarbilities/nvd_cves.json"  # <-- your uploaded file path/name
OUT_FULL_JSON = "vulnarbilities/vuln_full_graph.json"
OUT_EXAMPLE_JSON = "vulnarbilities/vuln_example_graph.json"
OUT_EXAMPLE_HTML = "vulnarbilities/vuln_example_graph.html"

# How many CVEs to include in the EXAMPLE graph (HTML). Keep small for speed.
EXAMPLE_CVE_LIMIT = 1 

# Toggle optional node types (makes the graph larger)
MAKE_VENDOR_PRODUCT_VERSION_NODES = True
MAKE_SEVERITY_VECTOR_DATE_SOURCE_NODES = True

# ----------------------------
# Helpers
# ----------------------------
def pick_english_desc(desc_list):
    for d in desc_list or []:
        if d.get("lang") == "en" and d.get("value"):
            return d["value"]
    return (desc_list[0]["value"] if desc_list else "")

def safe_id(prefix: str, value: str) -> str:
    # Stable string IDs for nodes
    return f"{prefix}::{value}"

def add_node(nodes, node):
    nid = node["id"]
    if nid not in nodes:
        nodes[nid] = node

def add_edge(edges, src, rel, dst, props=None):
    e = {"source": src, "relation": rel, "target": dst}
    if props:
        e["properties"] = props
    edges.append(e)

def parse_cpe(cpe_str: str):
    # cpe:2.3:a:vendor:product:version:...
    parts = (cpe_str or "").split(":")
    out = {"cpe": cpe_str}
    if len(parts) >= 6:
        out["part"] = parts[2]
        out["vendor"] = parts[3]
        out["product"] = parts[4]
        out["version"] = parts[5]
    return out

def normalize_date(dt_str: str):
    # Keep as-is, but ensure it's a string
    return str(dt_str) if dt_str else None

def to_str(x):
    return str(x) if x is not None else None

# ----------------------------
# Load NVD
# ----------------------------
with open(IN_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

vulns = data.get("vulnerabilities", [])
print("Loaded vulnerabilities:", len(vulns))

# ----------------------------
# Build FULL graph
# ----------------------------
nodes = {}
edges = []

for item in vulns:
    cve = item.get("cve", {})
    cve_id = cve.get("id")
    if not cve_id:
        continue

    # --- CVE node ---
    cve_desc = pick_english_desc(cve.get("descriptions"))
    cve_node_id = cve_id  # keep plain CVE ID as node ID

    add_node(nodes, {
        "id": cve_node_id,
        "type": "cve",
        "name": cve_id,
        "description": cve_desc,
        "properties": {
            "vulnStatus": cve.get("vulnStatus"),
            "sourceIdentifier": cve.get("sourceIdentifier"),
            "published": normalize_date(cve.get("published")),
            "lastModified": normalize_date(cve.get("lastModified"))
        }
    })

    # --- reported_by node (optional) ---
    if MAKE_SEVERITY_VECTOR_DATE_SOURCE_NODES:
        src_ident = cve.get("sourceIdentifier")
        if src_ident:
            sid = safe_id("SOURCE", src_ident)
            add_node(nodes, {"id": sid, "type": "source_identifier", "name": src_ident, "description": "", "properties": {}})
            add_edge(edges, cve_node_id, "reported_by", sid)

    # --- published_on / modified_on nodes (optional) ---
    if MAKE_SEVERITY_VECTOR_DATE_SOURCE_NODES:
        pub = cve.get("published")
        mod = cve.get("lastModified")
        if pub:
            did = safe_id("DATE", pub)
            add_node(nodes, {"id": did, "type": "date", "name": pub, "description": "", "properties": {"kind": "published"}})
            add_edge(edges, cve_node_id, "published_on", did)
        if mod:
            did = safe_id("DATE", mod)
            add_node(nodes, {"id": did, "type": "date", "name": mod, "description": "", "properties": {"kind": "modified"}})
            add_edge(edges, cve_node_id, "modified_on", did)

    # --- CWE: CVE -> has_weakness -> CWE-xxx ---
    for w in cve.get("weaknesses", []) or []:
        for d in w.get("description", []) or []:
            val = (d.get("value") or "").strip()
            if val.startswith("CWE-"):
                add_node(nodes, {"id": val, "type": "cwe", "name": val, "description": "", "properties": {}})
                add_edge(edges, cve_node_id, "has_weakness", val, {
                    "source": w.get("source"),
                    "weakness_type": w.get("type")
                })

    # --- CPE: CVE -> affects -> CPE ---
    for conf in cve.get("configurations", []) or []:
        for n in conf.get("nodes", []) or []:
            for m in n.get("cpeMatch", []) or []:
                cpe_str = m.get("criteria")
                if not cpe_str:
                    continue

                cpe_node_id = safe_id("CPE", cpe_str)
                parsed = parse_cpe(cpe_str)

                add_node(nodes, {
                    "id": cpe_node_id,
                    "type": "cpe",
                    "name": cpe_str,
                    "description": "",
                    "properties": parsed
                })

                add_edge(edges, cve_node_id, "affects", cpe_node_id, {
                    "vulnerable": m.get("vulnerable", True),
                    "matchCriteriaId": m.get("matchCriteriaId"),
                    "versionStartIncluding": m.get("versionStartIncluding"),
                    "versionStartExcluding": m.get("versionStartExcluding"),
                    "versionEndIncluding": m.get("versionEndIncluding"),
                    "versionEndExcluding": m.get("versionEndExcluding")
                })

                # --- vendor/product/version nodes (optional) ---
                if MAKE_VENDOR_PRODUCT_VERSION_NODES:
                    vendor = parsed.get("vendor")
                    product = parsed.get("product")
                    version = parsed.get("version")

                    if vendor:
                        vid = safe_id("VENDOR", vendor)
                        add_node(nodes, {"id": vid, "type": "vendor", "name": vendor, "description": "", "properties": {}})
                        add_edge(edges, cpe_node_id, "has_vendor", vid)

                    if product:
                        pid = safe_id("PRODUCT", product)
                        add_node(nodes, {"id": pid, "type": "product", "name": product, "description": "", "properties": {}})
                        add_edge(edges, cpe_node_id, "has_product", pid)

                    if version:
                        verid = safe_id("VERSION", version)
                        add_node(nodes, {"id": verid, "type": "version", "name": version, "description": "", "properties": {}})
                        add_edge(edges, cpe_node_id, "has_version", verid)

    # --- References: CVE -> has_reference -> URL ---
    for r in cve.get("references", []) or []:
        url = r.get("url")
        if not url:
            continue
        url_node_id = safe_id("URL", url)
        add_node(nodes, {
            "id": url_node_id,
            "type": "url",
            "name": url,
            "description": "",
            "properties": {
                "source": r.get("source"),
                "tags": r.get("tags", [])
            }
        })
        add_edge(edges, cve_node_id, "has_reference", url_node_id, {"tags": r.get("tags", [])})

    # --- CVSS metrics: CVE -> has_cvss -> CVSS node ---
    metrics = cve.get("metrics", {}) or {}
    for key in ["cvssMetricV40", "cvssMetricV31", "cvssMetricV2"]:
        for entry in metrics.get(key, []) or []:
            cvss = entry.get("cvssData", {}) or {}
            vector = cvss.get("vectorString")
            base = cvss.get("baseScore")
            sev = cvss.get("baseSeverity")

            if not (vector or base or sev):
                continue

            cvss_node_id = safe_id("CVSS", f"{cve_id}::{key}::{vector or 'NOVECTOR'}")

            add_node(nodes, {
                "id": cvss_node_id,
                "type": "cvss",
                "name": f"{key} {sev or ''} {base or ''}".strip(),
                "description": "",
                "properties": {
                    "metric_type": key,
                    "vectorString": vector,
                    "baseScore": base,
                    "baseSeverity": sev,
                    "attackVector": cvss.get("attackVector"),
                    "attackComplexity": cvss.get("attackComplexity"),
                    "privilegesRequired": cvss.get("privilegesRequired"),
                    "userInteraction": cvss.get("userInteraction"),
                    "scope": cvss.get("scope"),
                    "confidentialityImpact": cvss.get("confidentialityImpact"),
                    "integrityImpact": cvss.get("integrityImpact"),
                    "availabilityImpact": cvss.get("availabilityImpact"),
                }
            })

            add_edge(edges, cve_node_id, "has_cvss", cvss_node_id, {
                "source": entry.get("source"),
                "type": entry.get("type")
            })

            # --- severity & vector nodes (optional) ---
            if MAKE_SEVERITY_VECTOR_DATE_SOURCE_NODES:
                if sev:
                    sev_id = safe_id("SEVERITY", sev)
                    add_node(nodes, {"id": sev_id, "type": "severity", "name": sev, "description": "", "properties": {}})
                    add_edge(edges, cve_node_id, "has_severity", sev_id, {"metric_type": key})

                if vector:
                    vec_id = safe_id("VECTOR", vector)
                    add_node(nodes, {"id": vec_id, "type": "cvss_vector", "name": vector, "description": "", "properties": {"metric_type": key}})
                    add_edge(edges, cve_node_id, "has_vector", vec_id, {"metric_type": key})

# Save FULL graph
full_graph = {
    "meta": {
        "name": "nvd_vulnerability_full_graph",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "description": "Full NVD-derived vulnerability graph with CVE-CWE-CPE-URL-CVSS and optional vendor/product/version, severity/vector/date/source nodes.",
        "total_nodes": len(nodes),
        "total_edges": len(edges),
        "options": {
            "MAKE_VENDOR_PRODUCT_VERSION_NODES": MAKE_VENDOR_PRODUCT_VERSION_NODES,
            "MAKE_SEVERITY_VECTOR_DATE_SOURCE_NODES": MAKE_SEVERITY_VECTOR_DATE_SOURCE_NODES
        }
    },
    "nodes": list(nodes.values()),
    "edges": edges
}

with open(OUT_FULL_JSON, "w", encoding="utf-8") as f:
    json.dump(full_graph, f, indent=2)

print("Saved FULL graph:", OUT_FULL_JSON)
print("FULL Nodes:", len(nodes), "FULL Edges:", len(edges))

# ----------------------------
# Build EXAMPLE graph (small)
# ----------------------------
# Pick first N CVEs and include their neighbors (1 hop)
example_cves = []
for item in vulns:
    cve_id = (item.get("cve", {}) or {}).get("id")
    if cve_id:
        example_cves.append(cve_id)
    if len(example_cves) >= EXAMPLE_CVE_LIMIT:
        break

example_node_ids = set(example_cves)

# include nodes that are targets from those CVEs, and edges originating from those CVEs
example_edges = []
for e in edges:
    if e["source"] in example_node_ids:
        example_edges.append(e)
        example_node_ids.add(e["target"])

# also include edges among included nodes (like CPE->vendor/product/version)
final_example_edges = []
for e in edges:
    if e["source"] in example_node_ids and e["target"] in example_node_ids:
        final_example_edges.append(e)

example_nodes = [nodes[nid] for nid in example_node_ids if nid in nodes]

example_graph = {
    "meta": {
        "name": "nvd_vulnerability_example_graph",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "description": f"Example subgraph: first {EXAMPLE_CVE_LIMIT} CVEs + 1-hop neighbors.",
        "total_nodes": len(example_nodes),
        "total_edges": len(final_example_edges)
    },
    "nodes": example_nodes,
    "edges": final_example_edges
}

with open(OUT_EXAMPLE_JSON, "w", encoding="utf-8") as f:
    json.dump(example_graph, f, indent=2)

print("Saved EXAMPLE graph:", OUT_EXAMPLE_JSON)

# ----------------------------
# Build HTML (embedded JSON)
# ----------------------------
graph_str = json.dumps(example_graph)

html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>NVD Vulnerability Example Graph</title>
  <script src="https://unpkg.com/cytoscape@3.26.0/dist/cytoscape.min.js"></script>
  <style>
    body {{ margin:0; font-family: system-ui, sans-serif; }}
    #cy {{ width:100vw; height:100vh; }}
    .panel {{
      position:fixed; left:12px; top:12px; z-index:10;
      background:rgba(255,255,255,0.92);
      padding:10px 12px; border-radius:10px;
      box-shadow:0 4px 16px rgba(0,0,0,0.12);
      font-size:13px; max-width:540px;
    }}
    .panel code {{ background:#f1f5f9; padding:1px 5px; border-radius:6px; }}
  </style>
</head>
<body>
<div class="panel">
  <div><b>NVD Vulnerability Example Graph</b></div>
  <div>Showing a small subgraph (so it loads fast).</div>
  <div>Click a node/edge → check Console for data.</div>
  <div style="margin-top:6px;">
    Relations: <code>has_weakness</code>, <code>affects</code>, <code>has_reference</code>, <code>has_cvss</code>,
    <code>has_vendor</code>, <code>has_product</code>, <code>has_version</code>,
    <code>has_severity</code>, <code>has_vector</code>, <code>published_on</code>, <code>modified_on</code>, <code>reported_by</code>.
  </div>
</div>
<div id="cy"></div>

<script>
const graph = {graph_str};
const elements = [];

for (const n of graph.nodes) {{
  elements.push({{
    data: {{
      id: n.id,
      label: (n.type || "node") + "\\n" + (n.name || n.id),
      type: n.type || "node"
    }}
  }});
}}

for (const e of graph.edges) {{
  elements.push({{
    data: {{
      id: e.source + "_" + e.relation + "_" + e.target,
      source: e.source,
      target: e.target,
      label: e.relation,
      relation: e.relation
    }}
  }});
}}

const cy = cytoscape({{
  container: document.getElementById('cy'),
  elements,
  style: [
    {{
      selector: 'node',
      style: {{
        'label': 'data(label)',
        'text-wrap': 'wrap',
        'text-max-width': 160,
        'text-valign': 'center',
        'text-halign': 'center',
        'font-size': 9,
        'background-color': '#2563eb',
        'color': '#ffffff',
        'width': 58,
        'height': 58
      }}
    }},
    {{
      selector: 'edge',
      style: {{
        'label': 'data(label)',
        'curve-style': 'bezier',
        'target-arrow-shape': 'triangle',
        'line-color': '#64748b',
        'target-arrow-color': '#64748b',
        'font-size': 8,
        'text-rotation': 'autorotate'
      }}
    }}
  ],
  layout: {{ name: 'cose', animate: true }}
}});

cy.on('tap', 'node', (evt) => {{
  console.log("NODE:", evt.target.data('id'));
}});

cy.on('tap', 'edge', (evt) => {{
  console.log("EDGE:", evt.target.data());
}});
</script>
</body>
</html>
"""

with open(OUT_EXAMPLE_HTML, "w", encoding="utf-8") as f:
    f.write(html)

print("Saved EXAMPLE HTML:", OUT_EXAMPLE_HTML)
print("Done.")