import json
import hashlib
import networkx as nx
from pyvis.network import Network
from urllib.parse import urlparse

NVD_FILE = "vulnarbilities/nvd_cves.json"
OUT_HTML = "vulnarbilities/nvd_graph.html"
MAX_CVES = 300

def ref_id(url: str) -> str:
    return "ref:" + hashlib.md5(url.encode("utf-8")).hexdigest()[:12]

def pick_en(descs):
    for d in descs or []:
        if d.get("lang") == "en":
            return d.get("value")
    return None

def extract_cwes(weaknesses):
    cwes = set()
    for w in weaknesses or []:
        for d in w.get("description", []) or []:
            val = d.get("value")
            if isinstance(val, str) and val.startswith("CWE-"):
                cwes.add(val)
    return sorted(cwes)

def extract_cpes(configurations):
    cpes = set()
    for conf in configurations or []:
        for node in conf.get("nodes", []) or []:
            for m in node.get("cpeMatch", []) or []:
                if m.get("vulnerable") is True:
                    crit = m.get("criteria")
                    if crit:
                        cpes.add(crit)
    return sorted(cpes)

def cpe_short(cpe: str) -> str:
    """
    cpe:2.3:a:vendor:product:version:...
    => vendor:product:version
    """
    parts = cpe.split(":")
    if len(parts) >= 6:
        vendor = parts[3]
        product = parts[4]
        version = parts[5]
        return f"{vendor}:{product}:{version}"
    return cpe[:30]

def domain_short(url: str) -> str:
    try:
        host = urlparse(url).netloc
        return host.replace("www.", "") or "ref"
    except Exception:
        return "ref"

# -----------------------
# Load NVD
# -----------------------
with open(NVD_FILE, "r", encoding="utf-8") as f:
    nvd = json.load(f)

items = nvd.get("vulnerabilities", [])[:MAX_CVES]

G = nx.DiGraph()

# -----------------------
# Build graph
# -----------------------
for item in items:
    cve = item.get("cve", {})
    cve_id = cve.get("id")
    if not cve_id:
        continue

    cve_desc = pick_en(cve.get("descriptions"))
    G.add_node(cve_id, label="cve", name=cve_id, description=cve_desc)

    # CVE -> CWE
    for cwe in extract_cwes(cve.get("weaknesses")):
        if cwe not in G:
            G.add_node(cwe, label="cwe", name=cwe, description=None)
        G.add_edge(cve_id, cwe, type="has_weakness")

    # CVE -> CPE
    for cpe in extract_cpes(cve.get("configurations")):
        if cpe not in G:
            G.add_node(cpe, label="cpe", name=cpe, description=None)
        G.add_edge(cve_id, cpe, type="affects")

    # CVE -> References
    for r in cve.get("references", []) or []:
        url = r.get("url")
        if not url:
            continue
        rid = ref_id(url)
        if rid not in G:
            G.add_node(rid, label="reference", name=url, description=None)
        G.add_edge(cve_id, rid, type="has_reference")

print("NVD graph built!")
print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())

# -----------------------
# Visualize with PyVis
# -----------------------
net = Network(
    height="800px",
    width="100%",
    directed=True,
    notebook=False,
    cdn_resources="local"
)

net.toggle_physics(False)

def node_label(nid, d):
    t = d.get("label")

    if t == "cve":
        return d.get("name")

    if t == "cwe":
        return d.get("name")

    if t == "cpe":
        return cpe_short(d.get("name", nid))

    if t == "reference":
        return domain_short(d.get("name", ""))

    return d.get("name") or nid


def node_color(t):
    return {
        "cve": "#e74c3c",        # red
        "cwe": "#f39c12",        # orange
        "cpe": "#8e44ad",        # purple
        "reference": "#7f8c8d"   # gray
    }.get(t, "#3498db")


for nid, d in G.nodes(data=True):
    t = d.get("label")
    short = node_label(nid, d)

    # Clean tooltip (no giant raw string explosion)
    title = f"""
    <b>Type:</b> {t}<br>
    <b>Name:</b> {short}
    """

    if t == "cve" and d.get("description"):
        title += f"<br><br><b>Description:</b><br>{d['description'][:500]}"

    net.add_node(
        nid,
        label=short,
        title=title,
        color=node_color(t),
        size=12
    )

for u, v, ed in G.edges(data=True):
    net.add_edge(u, v, title=ed.get("type", ""), color="#95a5a6")

net.write_html(OUT_HTML, open_browser=False)
print("Saved:", OUT_HTML)