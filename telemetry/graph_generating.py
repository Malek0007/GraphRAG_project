import csv, json, re
from collections import defaultdict

INPUT_CSV = "telemetry/CTDAPD Dataset.csv"
OUT_JSON  = "telemetry/telemetry_agg_graph.json"
OUT_HTML  = "telemetry/telemetry_agg_graph.html"

# columns
SRC="Source_IP"
DST="Destination_IP"
DPORT="Destination_Port"
PROTO="Protocol_Type"
LABEL="Label"
ANOM="Anomaly_Score"

def safe_id(prefix, value):
    return f"{prefix}_{re.sub(r'[^a-zA-Z0-9_.:-]', '_', str(value))}"

def to_float(x):
    try:
        return float(x)
    except:
        return None

# ---- 1) aggregate flows into edges ----
agg = defaultdict(lambda: {"flow_count":0, "attack_count":0, "anom_sum":0.0, "anom_n":0})
ips = set()

with open(INPUT_CSV, newline="", encoding="utf-8") as f:
    r = csv.DictReader(f)
    for row in r:
        src = row.get(SRC)
        dst = row.get(DST)
        if not src or not dst:
            continue

        ips.add(src); ips.add(dst)

        dport = row.get(DPORT, "")
        proto = row.get(PROTO, "")
        key = (src, dst, dport, proto)

        a = agg[key]
        a["flow_count"] += 1

        if (row.get(LABEL, "") or "").strip().lower() == "attack":
            a["attack_count"] += 1

        an = to_float(row.get(ANOM, ""))
        if an is not None:
            a["anom_sum"] += an
            a["anom_n"] += 1

# ---- 2) build nodes ----
nodes = []
for ip in sorted(ips):
    nodes.append({
        "id": safe_id("ip", ip),
        "type": "ip",
        "label": ip,
        "ip": ip
    })

# ---- 3) build edges ----
edges = []
for (src, dst, dport, proto), a in agg.items():
    avg_anom = (a["anom_sum"] / a["anom_n"]) if a["anom_n"] else None

    # IMPORTANT: make the word ATTACK appear in the label
    attack_tag = f" | ATTACK={a['attack_count']}" if a["attack_count"] > 0 else ""
    edge_label = f"{proto}:{dport} | flows={a['flow_count']}{attack_tag}"

    edges.append({
        "id": safe_id("e", f"{src}_{dst}_{proto}_{dport}"),
        "source": safe_id("ip", src),
        "target": safe_id("ip", dst),
        "type": "communicates",
        "label": edge_label,
        "flow_count": a["flow_count"],
        "attack_count": a["attack_count"],
        "avg_anomaly": avg_anom
    })

graph = {
    "schema_version": "1.0",
    "nodes": nodes,
    "edges": edges,
    "meta": {
        "description": "Aggregated telemetry graph. One edge per (src,dst,proto,dst_port) with counts + avg anomaly.",
        "input_csv": INPUT_CSV,
        "node_count": len(nodes),
        "edge_count": len(edges)
    }
}

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(graph, f, indent=2)
print("Wrote", OUT_JSON)

# ---- 4) write HTML (embedded JSON so file:// works) ----
graph_str = json.dumps(graph)

html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Telemetry Aggregated Graph</title>
  <script src="https://unpkg.com/cytoscape@3.26.0/dist/cytoscape.min.js"></script>
  <style>
    body {{ margin:0; font-family: system-ui, sans-serif; }}
    #cy {{ width:100vw; height:100vh; }}
    .panel {{
      position:fixed; left:12px; top:12px; z-index:10;
      background:rgba(255,255,255,0.92);
      padding:10px 12px; border-radius:10px;
      box-shadow:0 4px 16px rgba(0,0,0,0.12);
      font-size:13px; max-width:520px;
    }}
    .panel code {{ background:#f1f5f9; padding:1px 5px; border-radius:6px; }}
  </style>
</head>
<body>
<div class="panel">
  <div><b>Aggregated Telemetry Graph</b></div>
  <div>Edges show: <code>proto:port | flows=N | ATTACK=M</code></div>
  <div>Edge thickness = flow_count; edge color = has ATTACK</div>
  <div>Click an edge to see full details in console.</div>
</div>
<div id="cy"></div>

<script>
const graph = {graph_str};

const elements = [];
for (const n of graph.nodes) {{
  elements.push({{ data: {{ id:n.id, label:n.label, type:n.type }} }});
}}
for (const e of graph.edges) {{
  elements.push({{
    data: {{
      id:e.id, source:e.source, target:e.target,
      label:e.label,
      flow_count:e.flow_count,
      attack_count:e.attack_count,
      avg_anomaly:e.avg_anomaly
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
        'font-size': 10,
        'text-valign': 'center',
        'text-halign': 'center',
        'width': 34,
        'height': 34,
        'background-color': '#2563eb',
        'color': '#fff'
      }}
    }},
    {{
      selector: 'edge',
      style: {{
        'label': 'data(label)',
        'font-size': 9,
        'curve-style': 'bezier',
        'target-arrow-shape': 'triangle',
        'line-color': '#64748b',
        'target-arrow-color': '#64748b',
        'text-rotation': 'autorotate',
        'width': 'mapData(flow_count, 1, 200, 1, 10)'
      }}
    }}
  ],
  layout: {{ name: 'cose', animate: false, randomize: true }}
}});

// Color edges that have attacks
cy.edges().forEach(e => {{
  const atk = e.data('attack_count') || 0;
  if (atk > 0) {{
    e.style('line-color', '#dc2626');         // red
    e.style('target-arrow-color', '#dc2626'); // red
  }}
}});

cy.on('tap', 'edge', (evt) => {{
  console.log("EDGE:", evt.target.data());
}});
</script>
</body>
</html>
"""

with open(OUT_HTML, "w", encoding="utf-8") as f:
    f.write(html)

print("Wrote", OUT_HTML)
print("Done.")