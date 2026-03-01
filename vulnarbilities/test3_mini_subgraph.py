import json
import hashlib
import networkx as nx
from pyvis.network import Network
# -----------------------------
# CONFIG
# -----------------------------
NVD_FILE = "vulnarbilities/nvd_cves.json"          # <-- your NVD JSON 2.0 file
CENTER_CVE = "CVE-2026-0544"        # <-- the CVE you want to center the mini graph on
MAX_CVES = 2000                     # build graph from first N CVEs (increase if needed)
RADIUS = 1                          # ego radius (1 = direct neighbors, 2 = neighbors of neighbors)
OUT_HTML = f"mini_{CENTER_CVE}.html"

# -----------------------------
# HELPERS
# -----------------------------
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

def short_label(nid, data):
    lab = data.get("label")
    if lab == "cpe":
        return "CPE"
    if lab == "reference":
        return "REF"
    return data.get("name") or nid

# -----------------------------
# 1) LOAD NVD JSON
# -----------------------------
with open(NVD_FILE, "r", encoding="utf-8") as f:
    nvd = json.load(f)

items = nvd.get("vulnerabilities", [])
if not items:
    raise ValueError("No 'vulnerabilities' key found. Check NVD_FILE is a JSON 2.0 CVE feed.")

items = items[:MAX_CVES]

# -----------------------------
# 2) BUILD FULL NVD GRAPH (G)
# -----------------------------
G = nx.DiGraph()

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

print("Full NVD graph built!")
print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())

# -----------------------------
# 3) BUILD MINI GRAPH AROUND ONE CVE (H)
# -----------------------------
if CENTER_CVE not in G:
    # Give helpful suggestions
    examples = [n for n, d in G.nodes(data=True) if d.get("label") == "cve"][:10]
    raise ValueError(
        f"{CENTER_CVE} not found in the graph built from the first {MAX_CVES} CVEs.\n"
        f"Try increasing MAX_CVES or choose one of these sample CVEs:\n{examples}"
    )

UG = G.to_undirected()
mini_nodes = nx.ego_graph(UG, CENTER_CVE, radius=RADIUS).nodes()
H = G.subgraph(mini_nodes).copy()

print(f"Mini graph around {CENTER_CVE} (radius={RADIUS})")
print("Mini nodes:", H.number_of_nodes())
print("Mini edges:", H.number_of_edges())

# -----------------------------
# 4) VISUALIZE MINI GRAPH WITH PYVIS
# -----------------------------
net = Network(height="800px", width="100%", directed=True, notebook=False, cdn_resources="local")
net.toggle_physics(False)  # keep it fast for the professor demo

for nid, d in H.nodes(data=True):
    title = f"{d.get('label')}<br>{d.get('name')}"
    if d.get("description"):
        title += "<br><br>" + d["description"][:800]
    net.add_node(nid, label=short_label(nid, d), title=title)

for u, v, ed in H.edges(data=True):
    net.add_edge(u, v, title=ed.get("type", ""))

net.show_buttons(filter_=["physics"])
net.write_html(OUT_HTML, open_browser=False)

print("Saved mini graph HTML:", OUT_HTML)