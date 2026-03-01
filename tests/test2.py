import json
import networkx as nx
from pyvis.network import Network

# Load full graph JSON you exported
with open("attack_graph.json", "r", encoding="utf-8") as f:
    g = json.load(f)

G = nx.DiGraph()
for n in g["nodes"]:
    if n.get("deprecated") or n.get("revoked"):
        continue
    G.add_node(n["id"], **n)
for e in g["edges"]:
    if e["source"] in G.nodes and e["target"] in G.nodes:
        G.add_edge(e["source"], e["target"], **e)

# Choose a technique by ATT&CK ID (good demo ones: T1003, T1059, T1110)
center_attack_id = "T1003"

center_id = None
for nid, d in G.nodes(data=True):
    if d.get("attack_id") == center_attack_id:
        center_id = nid
        break
if center_id is None:
    raise ValueError(f"Center node not found for {center_attack_id}")

# Build mini subgraph: 2-hop neighborhood (undirected neighborhood but keep directed edges)
UG = G.to_undirected()
mini_nodes = nx.ego_graph(UG, center_id, radius=2).nodes()
H = G.subgraph(mini_nodes).copy()

print("Mini graph nodes:", H.number_of_nodes(), "edges:", H.number_of_edges())

# Export interactive HTML
net = Network(height="700px", width="100%", directed=True, notebook=False, cdn_resources="local")
net.toggle_physics(False)  # important for fast load

for nid, d in H.nodes(data=True):
    label = d.get("attack_id") or (d.get("name") or nid)
    title = f"{d.get('label')}<br>{d.get('name')}<br>{d.get('attack_id')}"
    net.add_node(nid, label=label, title=title)

for u, v, ed in H.edges(data=True):
    net.add_edge(u, v, title=ed.get("type", ""))

out_file = f"mini_{center_attack_id}.html"
net.write_html(out_file, open_browser=False)
print("Saved:", out_file)