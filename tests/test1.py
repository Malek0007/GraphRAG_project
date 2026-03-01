import json
import networkx as nx
from pyvis.network import Network

# Load the graph you already built (or rebuild from enterprise-attack.json)
with open("attack_graph.json", "r", encoding="utf-8") as f:
    g = json.load(f)

G = nx.DiGraph()
for n in g["nodes"]:
    # Skip deprecated/revoked to reduce noise (optional)
    if n.get("deprecated") or n.get("revoked"):
        continue
    G.add_node(n["id"], **n)

for e in g["edges"]:
    if e["source"] in G.nodes and e["target"] in G.nodes:
        G.add_edge(e["source"], e["target"], **e)

print("Nodes:", G.number_of_nodes(), "Edges:", G.number_of_edges())

# --- IMPORTANT: Visualizing ALL nodes may be heavy.
# If it’s too slow, sample nodes or show largest connected component.

# Largest connected component (undirected view)
UG = G.to_undirected()
largest_cc = max(nx.connected_components(UG), key=len)
H = G.subgraph(largest_cc).copy()

print("Largest CC nodes:", H.number_of_nodes(), "edges:", H.number_of_edges())

# Build interactive network
net = Network(height="800px", width="100%", directed=True, notebook=False)

# Add nodes
for nid, d in H.nodes(data=True):
    label = d.get("attack_id") or d.get("name") or nid
    title = f"{d.get('label')}<br>{d.get('name')}<br>{d.get('attack_id')}"
    net.add_node(nid, label=label, title=title)

# Add edges
for u, v, ed in H.edges(data=True):
    net.add_edge(u, v, title=ed.get("type", ""))

# Physics for nicer layout
net.toggle_physics(True)
net.show_buttons(filter_=["physics"])

# ... after building `net` ...
out_file = "attack_full_graph.html"
net.write_html(out_file, open_browser=False)
print("Saved:", out_file)