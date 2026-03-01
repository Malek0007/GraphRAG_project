import csv, json, math
from collections import defaultdict
from datetime import datetime

INPUT_CSV = "telemetry/CTDAPD Dataset.csv"
OUT_JSON  = "telemetry/telemetry_insights.json"

# columns
SRC="Source_IP"
DST="Destination_IP"
DPORT="Destination_Port"
PROTO="Protocol_Type"
LABEL="Label"
ANOM="Anomaly_Score"
PATCH="System_Patch_Status"

TOP_K = 20  # how many to keep in each top list

def to_float(x):
    try:
        return float(x)
    except:
        return None

def is_attack(v):
    return (v or "").strip().lower() == "attack"

def norm_patch(v):
    if not v: return "Unknown"
    v = v.strip()
    if v.lower() in ["outdated", "old", "unpatched"]: return "Outdated"
    if v.lower() in ["updated", "up-to-date", "uptodate", "patched"]: return "UpToDate"
    return v

# global counters
row_count = 0
attack_rows = 0
normal_rows = 0

src_ips = set()
dst_ips = set()
pairs = set()

# aggregations
src_stats = defaultdict(lambda: {"flows":0,"attack_flows":0,"anom_sum":0.0,"anom_n":0,"max_anom":None,"outdated_patch_flows":0})
dst_stats = defaultdict(lambda: {"flows":0,"attack_flows":0,"anom_sum":0.0,"anom_n":0})
port_stats = defaultdict(lambda: {"flows":0,"attack_flows":0,"anom_sum":0.0,"anom_n":0})
patch_stats = defaultdict(lambda: {"flows":0,"attack_flows":0})

with open(INPUT_CSV, newline="", encoding="utf-8") as f:
    r = csv.DictReader(f)
    for row in r:
        row_count += 1

        src = row.get(SRC)
        dst = row.get(DST)
        if not src or not dst:
            continue

        src_ips.add(src)
        dst_ips.add(dst)
        pairs.add((src, dst))

        lab_attack = is_attack(row.get(LABEL))
        if lab_attack: attack_rows += 1
        else: normal_rows += 1

        an = to_float(row.get(ANOM))
        patch = norm_patch(row.get(PATCH))
        proto = (row.get(PROTO) or "").strip()
        dport = (row.get(DPORT) or "").strip()

        # patch summary
        patch_stats[patch]["flows"] += 1
        if lab_attack:
            patch_stats[patch]["attack_flows"] += 1

        # source stats
        s = src_stats[src]
        s["flows"] += 1
        if lab_attack:
            s["attack_flows"] += 1
        if patch == "Outdated":
            s["outdated_patch_flows"] += 1
        if an is not None:
            s["anom_sum"] += an
            s["anom_n"] += 1
            s["max_anom"] = an if (s["max_anom"] is None or an > s["max_anom"]) else s["max_anom"]

        # destination stats
        d = dst_stats[dst]
        d["flows"] += 1
        if lab_attack:
            d["attack_flows"] += 1
        if an is not None:
            d["anom_sum"] += an
            d["anom_n"] += 1

        # port/protocol stats
        key = (proto, dport)
        p = port_stats[key]
        p["flows"] += 1
        if lab_attack:
            p["attack_flows"] += 1
        if an is not None:
            p["anom_sum"] += an
            p["anom_n"] += 1

def finalize_avg(obj):
    avg = (obj["anom_sum"]/obj["anom_n"]) if obj.get("anom_n") else None
    return avg

# build top lists
top_risky_sources = []
for ip, s in src_stats.items():
    top_risky_sources.append({
        "source_ip": ip,
        "flows": s["flows"],
        "attack_flows": s["attack_flows"],
        "avg_anomaly": finalize_avg(s),
        "max_anomaly": s["max_anom"],
        "outdated_patch_flows": s["outdated_patch_flows"]
    })

# sort by attack_flows then avg_anomaly
top_risky_sources.sort(key=lambda x: (x["attack_flows"], x["avg_anomaly"] or -1), reverse=True)
top_risky_sources = top_risky_sources[:TOP_K]

top_targets = []
for ip, d in dst_stats.items():
    top_targets.append({
        "destination_ip": ip,
        "flows": d["flows"],
        "attack_flows": d["attack_flows"],
        "avg_anomaly": finalize_avg(d)
    })
top_targets.sort(key=lambda x: (x["attack_flows"], x["flows"]), reverse=True)
top_targets = top_targets[:TOP_K]

top_ports = []
for (proto, dport), p in port_stats.items():
    top_ports.append({
        "protocol": proto,
        "destination_port": int(dport) if dport.isdigit() else dport,
        "flows": p["flows"],
        "attack_flows": p["attack_flows"],
        "avg_anomaly": finalize_avg(p)
    })
top_ports.sort(key=lambda x: (x["attack_flows"], x["flows"]), reverse=True)
top_ports = top_ports[:TOP_K]

insights = {
    "meta": {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "input_csv": INPUT_CSV,
        "row_count": row_count
    },
    "summary": {
        "attack_rows": attack_rows,
        "normal_rows": normal_rows,
        "unique_source_ips": len(src_ips),
        "unique_destination_ips": len(dst_ips),
        "unique_pairs": len(pairs)
    },
    "top_risky_sources": top_risky_sources,
    "top_targets": top_targets,
    "top_ports": top_ports,
    "patch_risk": dict(patch_stats)
}

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(insights, f, indent=2)

print("Wrote", OUT_JSON)