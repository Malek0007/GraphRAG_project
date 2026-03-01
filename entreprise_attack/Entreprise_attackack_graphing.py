import json
import networkx as nx
import matplotlib.pyplot as plt

INPUT = "data/enterprise-attack.json"  # your file

def get_attack_id(obj):
    """Return ATT&CK external id like T1003, TA0006, M1026."""
    for ref in obj.get("external_references", []):
        if ref.get("source_name") == "mitre-attack" and "external_id" in ref:
            return ref["external_id"]
    return None

# 1) Load STIX bundle
with open(INPUT, "r", encoding="utf-8") as f:
    stix = json.load(f)

objects = stix["objects"]

# 2) Index objects by type and id
by_id = {}
techniques = {}
tactics = {}
mitigations = {}
relationships = []

for o in objects:
    oid = o.get("id")
    if oid:
        by_id[oid] = o

    t = o.get("type")
    if t == "attack-pattern":
        techniques[o["id"]] = o
    elif t == "x-mitre-tactic":
        tactics[o["id"]] = o
    elif t == "course-of-action":
        mitigations[o["id"]] = o
    elif t == "relationship":
        relationships.append(o)

# Map tactic shortname -> tactic object id (kill_chain uses shortnames like "credential-access")
short_to_tactic_id = {}
for tid, tac in tactics.items():
    short = tac.get("x_mitre_shortname")
    if short:
        short_to_tactic_id[short] = tid

# Map ATT&CK external id -> STIX object id (for techniques/mitigations/tactics)
attack_id_to_stix = {}
for oid, obj in by_id.items():
    aid = get_attack_id(obj)
    if aid:
        attack_id_to_stix[aid] = oid

# -------------------------
# Choose the technique
# -------------------------
TARGET_ATTACK_ID = "T1003"   # change to any technique id, e.g., "T1059"
tech_stix_id = attack_id_to_stix.get(TARGET_ATTACK_ID)
if not tech_stix_id:
    raise ValueError(f"Technique {TARGET_ATTACK_ID} not found in file. Try another Txxxx.")

tech_obj = by_id[tech_stix_id]
tech_name = tech_obj.get("name", "(no name)")

# -------------------------
# 3) Find tactic(s) for technique via kill_chain_phases
# -------------------------
tactic_ids = []
for kcp in tech_obj.get("kill_chain_phases", []):
    if kcp.get("kill_chain_name") == "mitre-attack":
        short = kcp.get("phase_name")  # e.g., "credential-access"
        tid = short_to_tactic_id.get(short)
        if tid:
            tactic_ids.append(tid)

# Convert to (TAxxxx, name)
tactic_infos = []
for tid in tactic_ids:
    tac = by_id.get(tid, {})
    tactic_infos.append((get_attack_id(tac), tac.get("name")))

# -------------------------
# 4) Find mitigations: Mitigation --mitigates--> Technique
# -------------------------
mitigation_ids = []
for rel in relationships:
    if rel.get("type") != "relationship":
        continue
    if rel.get("relationship_type") != "mitigates":
        continue
    if rel.get("target_ref") == tech_stix_id:
        # source_ref is the mitigation course-of-action
        mitigation_ids.append(rel.get("source_ref"))

mitigation_infos = []
for mid in mitigation_ids:
    m = by_id.get(mid, {})
    mitigation_infos.append((get_attack_id(m), m.get("name")))

# -------------------------
# 5) Find sub-techniques: SubTechnique --subtechnique-of--> Technique
# -------------------------
subtech_ids = []
for rel in relationships:
    if rel.get("type") != "relationship":
        continue
    if rel.get("relationship_type") != "subtechnique-of":
        continue
    if rel.get("target_ref") == tech_stix_id:
        subtech_ids.append(rel.get("source_ref"))

subtech_infos = []
for sid in subtech_ids:
    s = by_id.get(sid, {})
    subtech_infos.append((get_attack_id(s), s.get("name")))

# -------------------------
# 6) Print ASCII mini-graph like your example
# -------------------------
print("\n================= MINI GRAPH =================")
if tactic_infos:
    for ta, taname in tactic_infos:
        print(f"({ta} - {taname})")
        print("          ↑")
        print("          │ belongs-to")
        print("          │")
else:
    print("(No tactic found from kill_chain_phases)")

print(f"({TARGET_ATTACK_ID} - {tech_name})")

if mitigation_infos:
    print("          ↑")
    print("          │ mitigates")
    print("          │")
    # show first 5 mitigations (can be many)
    for mid, mname in mitigation_infos[:5]:
        print(f"({mid} - {mname})")
    if len(mitigation_infos) > 5:
        print(f"... +{len(mitigation_infos)-5} more mitigations")
else:
    print("(No mitigations found linked by 'mitigates')")

if subtech_infos:
    print("          ↑")
    print("          │ subtechnique-of (incoming from sub-techniques)")
    print("          │")
    for sid, sname in subtech_infos[:5]:
        print(f"({sid} - {sname})")
    if len(subtech_infos) > 5:
        print(f"... +{len(subtech_infos)-5} more sub-techniques")

print("================================================\n")

# -------------------------
# 7) Build a small NetworkX subgraph and plot it (readable)
# -------------------------
G = nx.DiGraph()

# Add central technique node
G.add_node(tech_stix_id, label="technique", attack_id=TARGET_ATTACK_ID, name=tech_name)

# Add tactic nodes + edge
for tid in tactic_ids:
    tac = by_id[tid]
    G.add_node(tid, label="tactic", attack_id=get_attack_id(tac), name=tac.get("name"))
    G.add_edge(tech_stix_id, tid, type="belongs-to")

# Add mitigation nodes + edge
for mid in mitigation_ids[:10]:  # limit so plot is not huge
    m = by_id[mid]
    G.add_node(mid, label="mitigation", attack_id=get_attack_id(m), name=m.get("name"))
    G.add_edge(mid, tech_stix_id, type="mitigates")

# Add sub-technique nodes + edge
for sid in subtech_ids[:10]:
    s = by_id[sid]
    G.add_node(sid, label="technique", attack_id=get_attack_id(s), name=s.get("name"))
    G.add_edge(sid, tech_stix_id, type="subtechnique-of")

# Plot
pos = nx.spring_layout(G, k=1.0, seed=7)
plt.figure(figsize=(10, 10))

def node_color(n):
    lab = G.nodes[n].get("label")
    if lab == "tactic": return "salmon"
    if lab == "mitigation": return "lightgreen"
    return "skyblue"

nx.draw_networkx_nodes(G, pos, node_color=[node_color(n) for n in G.nodes], node_size=1200)
nx.draw_networkx_edges(G, pos, alpha=0.5)

labels = {n: (G.nodes[n].get("attack_id") or "") for n in G.nodes}
nx.draw_networkx_labels(G, pos, labels=labels, font_size=9)

plt.title(f"Mini-graph around {TARGET_ATTACK_ID}")
plt.axis("off")
plt.show()