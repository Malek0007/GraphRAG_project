
import json
import networkx as nx

with open("attack_graph.json", "r", encoding="utf-8") as f:
    g = json.load(f)

G = nx.DiGraph()

for n in g["nodes"]:
    G.add_node(n["id"], **n)

for e in g["edges"]:
    G.add_edge(e["source"], e["target"], **e)

print("Loaded graph from JSON!")
print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())
import matplotlib.pyplot as plt

technique_name = "Debugger Evasion"

# find technique node id
tech_id = None
for n, d in G.nodes(data=True):
    if d.get("label") == "technique" and d.get("name") == technique_name:
        tech_id = n
        break

if tech_id is None:
    raise ValueError(f"Technique not found: {technique_name}")

# take 1-hop neighborhood
nbrs = set(G.successors(tech_id)) | set(G.predecessors(tech_id))
sub_nodes = list(nbrs | {tech_id})
subG = G.subgraph(sub_nodes).copy()

pos = nx.spring_layout(subG, k=0.8)

plt.figure(figsize=(10, 10))
nx.draw(subG, pos, with_labels=False, node_size=800)

# label nodes with names (short)
labels = {n: (subG.nodes[n].get("attack_id") or subG.nodes[n].get("name", "")[:20]) for n in subG.nodes}
nx.draw_networkx_labels(subG, pos, labels=labels, font_size=8)

plt.title(f"Subgraph around: {technique_name}")
plt.axis("off")
plt.show()