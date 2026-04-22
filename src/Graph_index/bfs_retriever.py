"""
GraphRAG v2 — Bounded Multi-Hop BFS Retrieval Engine

Retrieval flow:
  1. Seed retrieval     — exact-ID fast-path or semantic top-k
  2. BFS expansion      — bounded traversal from seeds (configurable depth + limits)
  3. Node/edge scoring  — edge-type weights × depth decay × bonuses
  4. Pruning            — drop branches below score threshold immediately
  5. Subgraph assembly  — select highest-value evidence nodes and edges
  6. Context formatting — structured, path-annotated text for the LLM

All tunable constants live in retrieval_config.py.
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from graph_utils import GraphStore
from retriever import Retriever
from retrieval_config import (
    MAX_BFS_DEPTH,
    TOP_K_SEEDS,
    MAX_NODES_IN_SUBGRAPH,
    MAX_NODES_MULTI_HOP,
    MAX_EDGES_IN_SUBGRAPH,
    MAX_EDGES_PER_NODE,
    EXACT_MATCH_BONUS,
    CROSS_DOMAIN_SEED_BONUS,
    DEPTH_DECAY,
    EDGE_TYPE_WEIGHTS,
    DEFAULT_EDGE_WEIGHT,
    BRIDGE_EDGE_BONUS,
    PRUNE_SCORE_THRESHOLD,
    SOFT_PRUNE_THRESHOLD,
    SOFT_PRUNE_TYPES,
    NOISE_NODE_TYPES,
    TYPE_CAPS,
    DEFAULT_TYPE_CAP,
    DESCRIPTION_MAX_CHARS,
    MAX_PATHS_IN_CONTEXT,
)


# ---------------------------------------------------------------------------
# ID helpers — single canonical implementation used by all modules
# ---------------------------------------------------------------------------

_ID_PATTERNS: list[str] = [
    r"\bCVE-\d{4}-\d+\b",
    r"\bTA\d{4}\b",
    r"\bT\d{4}(?:\.\d{3})?\b",
    r"\bM\d{4}\b",
    r"\bG\d{4}\b",
    r"\bC\d{4}\b",
    r"\bS\d{4}\b",
    r"\bATOMIC::[0-9a-f-]{36}\b",         # Atomic Red Team test GUIDs
    r"\bREPORT::[a-z0-9_]+\b",            # Threat report root nodes
    r"\bCOMMAND::[0-9a-f-]{36}\b",        # Atomic command nodes
]


def detect_exact_id(question: str) -> Optional[str]:
    """Return the first exact cybersecurity ID found in the text, or None."""
    for pattern in _ID_PATTERNS:
        matches = re.findall(pattern, question, re.IGNORECASE)
        if matches:
            return matches[0].upper()
    return None


def extract_ids(text: str) -> list[str]:
    """Return all unique cybersecurity IDs found in the text (sorted, uppercase)."""
    found: list[str] = []
    for pattern in _ID_PATTERNS:
        found.extend(re.findall(pattern, text, re.IGNORECASE))
    return sorted(set(x.upper() for x in found))


def is_cross_domain_query(question: str) -> bool:
    """
    True when the question spans both CVE/NVD and ATT&CK concepts.
    Used to apply CROSS_DOMAIN_SEED_BONUS and boost bridge edges.
    """
    has_vuln = bool(re.search(
        r"\b(CVE|vulnerability|vulnerabilities|exploit|CVSS|CWE|NVD|patch|advisory)\b",
        question, re.IGNORECASE,
    ))
    has_attack = bool(re.search(
        r"\b(ATT&CK|technique|tactic|TA\d{4}|T\d{4}|malware|threat.actor|group|campaign|TTP)\b",
        question, re.IGNORECASE,
    ))
    return has_vuln and has_attack


def detect_query_intent(question: str) -> dict:
    """
    Classify the query to enable intent-aware BFS edge boosting and depth selection.

    Returns a dict of boolean flags used by bfs_expand and rerank_by_intent.
    """
    q = question.lower()
    return {
        "wants_tactic": bool(re.search(
            r"\b(tactic|tactics|belong|belongs to|part of|ta\d{4})\b", q
        )),
        "wants_mitigation": bool(re.search(
            r"\b(mitigation|mitigations|mitigate|protect|prevent|defend|defense|countermeasure)\b", q
        )),
        "wants_software": bool(re.search(
            r"\b(software|malware|tool|tools|used by|s\d{4})\b", q
        )),
        "wants_group": bool(re.search(
            r"\b(group|actor|apt|threat actor|who uses|g\d{4})\b", q
        )),
        "wants_campaign": bool(re.search(
            r"\b(campaign|c\d{4}|operation)\b", q
        )),
        "is_multi_hop": bool(
            re.search(r"\b(which|what).{0,40}(uses|use|used by|connected|linked|related|associated)\b", q)
            or re.search(r"\b(path|chain|multi.?hop|connect|through)\b", q)
        ),
        "wants_chain": bool(re.search(
            r"\b(through|chain|connect|group.*use|campaign.*use|used in|threat group|reasoning path)\b", q
        )),
        # Negative queries ask about ABSENCE — e.g. "which techniques have NO mitigations"
        "is_negative": bool(re.search(
            r"\b(no |not |without |none |lack |miss|absent|zero)(mitigation|defense|protection|mitigation)?\b", q
        )),
        # Multi-layer (L3/L4) intent flags
        "wants_atomic": bool(re.search(
            r"\b(atomic|red team|test|simulation|emulation|procedure|detection rule)\b", q
        )),
        "wants_report": bool(re.search(
            r"\b(report|intel|intelligence|threat report|observed|campaign report|actor report)\b", q
        )),
        # CVE metadata queries (CVSS score, CWE weakness, severity)
        "wants_cve_metadata": bool(re.search(
            r"\b(severity|cvss|score|rating|critical|high|medium|low|cwe|weakness|base.?score)\b", q
        )),
    }


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ScoredNode:
    """A graph node annotated with its BFS traversal score and the path used to reach it."""
    node_id: str
    score: float
    depth: int
    # IDs from seed node → this node (inclusive on both ends)
    path_node_ids: list[str] = field(default_factory=list)
    # Edges traversed from seed → this node (len == depth)
    path_edges: list[dict] = field(default_factory=list)
    is_seed: bool = False
    is_exact_match: bool = False
    # Forced direct neighbor of an exact-match seed — bypasses type caps in assembly
    is_forced_neighbor: bool = False


# ---------------------------------------------------------------------------
# Edge helpers
# ---------------------------------------------------------------------------

def _edge_weight(edge: dict) -> float:
    """Configured weight for this edge type; falls back to DEFAULT_EDGE_WEIGHT."""
    return EDGE_TYPE_WEIGHTS.get(edge.get("type", ""), DEFAULT_EDGE_WEIGHT)


def _intent_edge_bonus(edge: dict, intent: dict) -> float:
    """Extra multiplier applied when the edge type aligns with the query intent."""
    edge_type = edge.get("type", "")
    bonus = 1.0
    if intent.get("wants_tactic") and edge_type == "belongs_to":
        bonus *= 1.5
    if intent.get("wants_mitigation") and not intent.get("is_negative") and edge_type == "mitigates":
        bonus *= 1.5
    if intent.get("wants_software") and edge_type in ("uses", "related_software"):
        bonus *= 1.3
    if intent.get("wants_group") and edge_type in ("uses", "attributed_to"):
        bonus *= 1.3
    if intent.get("wants_campaign") and edge_type in ("uses", "part_of", "attributed_to"):
        bonus *= 1.4
    if intent.get("wants_chain") and edge_type in ("related_software", "exploited_via", "uses"):
        bonus *= 1.2
    return bonus


def _is_bridge_edge(edge: dict) -> bool:
    """
    True if the edge crosses between CVE/NVD and ATT&CK knowledge layers.
    """
    layer_from = edge.get("layer_from", "")
    layer_to = edge.get("layer_to", "")
    edge_type = edge.get("type", "")
    return (
        edge_type in ("exploited_via", "bridge", "related_software")
        or (bool(layer_from) and bool(layer_to) and layer_from != layer_to)
    )


# ---------------------------------------------------------------------------
# 1. Seed retrieval
# ---------------------------------------------------------------------------

def get_seeds(
    question: str,
    retriever: Retriever,
    graph: GraphStore,
    cross_domain: bool,
) -> list[ScoredNode]:
    """
    Build the initial seed nodes for BFS.

    Exact-ID match  → one seed per detected ID with EXACT_MATCH_BONUS.
    Semantic search → top-k nodes by cosine similarity (score = cosine score).
    Cross-domain flag multiplies seed scores by CROSS_DOMAIN_SEED_BONUS.

    Multiple exact IDs are supported (e.g. multi-hop questions referencing
    two ATT&CK techniques or a CVE alongside a group ID).
    """
    exact_ids = extract_ids(question)
    exact_seeds: list[ScoredNode] = []

    for eid in exact_ids:
        if graph.get_node(eid):
            score = EXACT_MATCH_BONUS * (CROSS_DOMAIN_SEED_BONUS if cross_domain else 1.0)
            exact_seeds.append(ScoredNode(
                node_id=eid,
                score=score,
                depth=0,
                path_node_ids=[eid],
                path_edges=[],
                is_seed=True,
                is_exact_match=True,
            ))
        else:
            print(f"[BFS] ID '{eid}' not found in graph")

    if exact_seeds:
        return exact_seeds

    # No exact IDs found (or none present in graph) — fall back to semantic search
    seeds: list[ScoredNode] = []
    for r in retriever.search(question, top_k=TOP_K_SEEDS):
        node_id = r.get("id")
        if not node_id or not graph.get_node(node_id):
            continue
        score = float(r.get("score", 0.5))
        if cross_domain:
            score *= CROSS_DOMAIN_SEED_BONUS
        seeds.append(ScoredNode(
            node_id=node_id,
            score=score,
            depth=0,
            path_node_ids=[node_id],
            path_edges=[],
            is_seed=True,
            is_exact_match=False,
        ))

    return seeds


# ---------------------------------------------------------------------------
# 2. BFS expansion with scoring and pruning
# ---------------------------------------------------------------------------

def bfs_expand(
    seeds: list[ScoredNode],
    graph: GraphStore,
    max_depth: int = MAX_BFS_DEPTH,
    intent: Optional[dict] = None,
) -> dict[str, ScoredNode]:
    """
    Bounded BFS from seed nodes with per-hop scoring.

    Scoring formula per hop:
        child_score = parent_score × edge_weight × bridge_mult × intent_bonus × DEPTH_DECAY

    Where:
        edge_weight   — from EDGE_TYPE_WEIGHTS (high for uses/mitigates/bridge)
        bridge_mult   — BRIDGE_EDGE_BONUS if edge crosses CVE↔ATT&CK layers
        intent_bonus  — extra multiplier when edge type matches the query intent
        DEPTH_DECAY   — exponential penalty discouraging very long paths

    Keeps the best-scoring path to each node (greedy update).
    Prunes any node whose score falls below PRUNE_SCORE_THRESHOLD.

    Returns mapping node_id → best ScoredNode for all reachable nodes.
    """
    _intent = intent or {}
    best: dict[str, ScoredNode] = {}
    frontier: deque[ScoredNode] = deque()

    for seed in seeds:
        if seed.node_id not in best or seed.score > best[seed.node_id].score:
            best[seed.node_id] = seed
            frontier.append(seed)

    while frontier:
        current = frontier.popleft()

        if current.depth >= max_depth:
            continue

        # Follow strongest edges first so the most informative branches
        # are explored before the node-budget is exhausted.
        # Exact-match seeds process ALL their edges so no direct neighbor
        # (CWE, CVSS, software via related_software) is missed due to the cap.
        edges_raw = sorted(
            graph.get_neighbors(current.node_id),
            key=_edge_weight,
            reverse=True,
        )
        edges = edges_raw if (current.is_seed and current.is_exact_match) else edges_raw[:MAX_EDGES_PER_NODE]

        for edge in edges:
            src = edge.get("source", "")
            tgt = edge.get("target", "")
            neighbor_id = tgt if src == current.node_id else src

            if not neighbor_id:
                continue

            node = graph.get_node(neighbor_id)
            if not node:
                continue

            # Skip pure metadata noise — these add no analytical value
            node_type = (node.get("type") or "").lower()
            if node_type in NOISE_NODE_TYPES:
                continue

            ew = _edge_weight(edge)
            bridge_mult = BRIDGE_EDGE_BONUS if _is_bridge_edge(edge) else 1.0
            intent_bonus = _intent_edge_bonus(edge, _intent)
            new_score = current.score * ew * bridge_mult * intent_bonus * DEPTH_DECAY

            # Soft pruning: important node types survive at a much lower floor
            threshold = (
                SOFT_PRUNE_THRESHOLD if node_type in SOFT_PRUNE_TYPES
                else PRUNE_SCORE_THRESHOLD
            )
            if new_score < threshold:
                continue

            existing = best.get(neighbor_id)
            if existing is not None:
                # NEVER overwrite a seed — this prevents the cycle amplification bug
                # where exploited_via cycles back to the CVE with inflated scores
                if existing.is_seed:
                    continue
                if existing.score >= new_score:
                    continue

            child = ScoredNode(
                node_id=neighbor_id,
                score=new_score,
                depth=current.depth + 1,
                path_node_ids=current.path_node_ids + [neighbor_id],
                path_edges=current.path_edges + [edge],
                is_seed=False,
                is_exact_match=False,
            )
            best[neighbor_id] = child
            frontier.append(child)

    return best


# Type-specific expansion rules for exact-match seeds.
# CVE seeds: ensure software/CWE/CVSS neighbors are never missed.
# Technique seeds: tactic/mitigation/atomic are forced in.
_EXACT_SEED_EXPANSION: dict[str, list[str]] = {
    "cve":          ["software", "cwe", "cvss", "technique", "sub-technique"],
    "technique":    ["tactic", "mitigation", "atomic_test"],
    "sub-technique":["technique", "tactic", "mitigation"],
    "software":     ["technique", "group", "campaign"],
    "group":        ["software", "campaign", "technique"],
    "campaign":     ["software", "group", "technique"],
    "mitigation":   ["technique"],
    "tactic":       ["technique", "sub-technique"],
}

# Minimum score guaranteed to any forced neighbor (above all pruning floors).
_FORCED_NEIGHBOR_SCORE = 0.35


def force_expand_exact_seeds(
    seeds: list[ScoredNode],
    scored_nodes: dict[str, ScoredNode],
    graph: GraphStore,
) -> dict[str, ScoredNode]:
    """
    Safety pass after BFS: for every exact-match seed, inspect ALL direct neighbors
    and force-include those of "important" types (per _EXACT_SEED_EXPANSION) if they
    aren't already in scored_nodes at a meaningful score.

    This guarantees that CWE/CVSS/software/tactic nodes adjacent to the seeded node
    are never accidentally pruned or cut by the MAX_EDGES_PER_NODE cap during BFS
    from other nodes.  Forced neighbors are marked is_forced_neighbor=True so the
    assembly pass can give them priority over type-capped candidates.
    """
    for seed in seeds:
        if not seed.is_exact_match:
            continue
        node = graph.get_node(seed.node_id)
        if not node:
            continue
        seed_type = (node.get("type") or "").lower()
        wanted_types = set(_EXACT_SEED_EXPANSION.get(seed_type, []))
        if not wanted_types:
            continue

        for edge in graph.get_neighbors(seed.node_id):
            src = edge.get("source", "")
            tgt = edge.get("target", "")
            neighbor_id = tgt if src == seed.node_id else src
            if not neighbor_id:
                continue
            neighbor = graph.get_node(neighbor_id)
            if not neighbor:
                continue
            neighbor_type = (neighbor.get("type") or "").lower()
            if neighbor_type not in wanted_types:
                continue

            existing = scored_nodes.get(neighbor_id)
            if existing is None:
                scored_nodes[neighbor_id] = ScoredNode(
                    node_id=neighbor_id,
                    score=_FORCED_NEIGHBOR_SCORE,
                    depth=1,
                    path_node_ids=[seed.node_id, neighbor_id],
                    path_edges=[edge],
                    is_seed=False,
                    is_exact_match=False,
                    is_forced_neighbor=True,
                )
            elif not existing.is_seed and existing.score < _FORCED_NEIGHBOR_SCORE:
                existing.score = _FORCED_NEIGHBOR_SCORE
                existing.is_forced_neighbor = True

    return scored_nodes


_STOP_WORDS: frozenset[str] = frozenset({
    "the", "a", "an", "is", "are", "what", "which", "how", "for",
    "that", "this", "with", "and", "or", "of", "in", "to", "be",
    "have", "has", "do", "does", "used", "use", "uses", "give", "list",
    "find", "show", "get", "tell", "about", "any", "all", "from",
})


def _query_keyword_overlap(query_words: frozenset[str], node: dict) -> float:
    """Fraction of significant query words found in the node's name + description."""
    if not query_words:
        return 0.0
    node_text = f"{node.get('name', '')} {node.get('description', '')}".lower()
    node_words = set(re.findall(r"\b[a-z]{3,}\b", node_text))
    return len(query_words & node_words) / len(query_words)


def rerank_post_bfs(
    question: str,
    scored_nodes: dict[str, ScoredNode],
    graph: GraphStore,
    intent: dict,
) -> dict[str, ScoredNode]:
    """
    Post-BFS score adjustments — applied before subgraph assembly.

    Applied boosts (multiplicative, seeds excluded):
      1. Intent-type boost:   node type matches what the query asks for.
      2. Bridge-path bonus:   node sits on a cross-layer path (CVE↔ATT&CK).
      3. Keyword overlap:     ≥20% of significant query words appear in node text.
      4. Metadata intent:     query mentions severity/CVSS/CWE → boost those types.

    Boosts capped at 1.5× each to stay balanced.
    """
    TYPE_BOOSTS: list[tuple[str, str, float]] = [
        ("wants_tactic",    "tactic",        1.5),
        ("wants_software",  "software",      1.4),
        ("wants_group",     "group",         1.5),
        ("wants_campaign",  "campaign",      1.5),
        ("wants_atomic",    "atomic_test",   1.5),
        ("wants_report",    "threat_report", 1.4),
        ("wants_report",    "malware",       1.3),
        ("wants_report",    "threat_actor",  1.3),
    ]
    if intent.get("wants_mitigation") and not intent.get("is_negative"):
        TYPE_BOOSTS.append(("wants_mitigation", "mitigation", 1.5))
    if intent.get("wants_cve_metadata"):
        TYPE_BOOSTS += [
            ("wants_cve_metadata", "cwe",      1.4),
            ("wants_cve_metadata", "cvss",     1.3),
            ("wants_cve_metadata", "severity", 1.3),
        ]

    query_words = frozenset(re.findall(r"\b[a-z]{3,}\b", question.lower())) - _STOP_WORDS

    for sn in scored_nodes.values():
        if sn.is_seed:
            continue
        node = graph.get_node(sn.node_id)
        if not node:
            continue
        node_type = (node.get("type") or "").lower()

        # 1. Intent-type boost (first matching rule wins)
        for intent_key, target_type, boost in TYPE_BOOSTS:
            if intent.get(intent_key) and node_type == target_type:
                sn.score *= boost
                break

        # 2. Bridge-path bonus: software/technique on a cross-layer path
        if node_type in ("software", "technique", "sub-technique") and any(
            _is_bridge_edge(e) for e in sn.path_edges
        ):
            sn.score *= 1.3

        # 3. Query keyword overlap (boost only when overlap is meaningful)
        kw_overlap = _query_keyword_overlap(query_words, node)
        if kw_overlap >= 0.20:
            sn.score *= 1.0 + min(0.4, kw_overlap)

    return scored_nodes


# ---------------------------------------------------------------------------
# 3. Subgraph assembly
# ---------------------------------------------------------------------------

def assemble_subgraph(
    scored_nodes: dict[str, ScoredNode],
    graph: GraphStore,
    max_nodes: int = MAX_NODES_IN_SUBGRAPH,
    max_edges: int = MAX_EDGES_IN_SUBGRAPH,
) -> tuple[list[ScoredNode], list[dict]]:
    """
    Select top-scoring nodes with diversity enforcement, then collect edges.

    Assembly order:
      1. All seeds (always kept regardless of type cap).
      2. Non-seed nodes in score order, subject to per-type caps from TYPE_CAPS.
         Noise types (cap=0) are excluded entirely.

    Returns (ranked_nodes, subgraph_edges).
    """
    # Sort priority: seeds first, then forced neighbors, then by score descending
    all_sorted = sorted(
        scored_nodes.values(),
        key=lambda n: (n.is_seed, n.is_forced_neighbor, n.score),
        reverse=True,
    )

    type_counts: dict[str, int] = {}
    ranked: list[ScoredNode] = []
    ranked_ids: set[str] = set()

    for sn in all_sorted:
        if sn.is_seed:
            ranked.append(sn)
            ranked_ids.add(sn.node_id)
            continue
        if len(ranked) >= max_nodes:
            break
        node = graph.get_node(sn.node_id)
        node_type = (node.get("type") or "unknown").lower() if node else "unknown"
        cap = TYPE_CAPS.get(node_type, DEFAULT_TYPE_CAP)
        if cap == 0:
            continue  # excluded type (url, date, vendor, etc.)

        # Forced neighbors of exact-match seeds get a relaxed cap (2× normal)
        # so that directly-adjacent CWE/software/tactic nodes are never dropped.
        effective_cap = cap * 2 if sn.is_forced_neighbor else cap
        if type_counts.get(node_type, 0) >= effective_cap:
            continue
        ranked.append(sn)
        ranked_ids.add(sn.node_id)
        type_counts[node_type] = type_counts.get(node_type, 0) + 1

    # --- Diversity floor ---
    # After the main pass, ensure at least one node of each "anchor" type
    # exists in the subgraph if BFS found any candidate of that type.
    # This prevents technique nodes from completely crowding out software/CWE/tactic.
    _DIVERSITY_FLOOR_TYPES: list[str] = [
        "cve", "software", "technique", "tactic", "cwe", "mitigation",
    ]
    for floor_type in _DIVERSITY_FLOOR_TYPES:
        if type_counts.get(floor_type, 0) > 0:
            continue  # already represented
        if len(ranked) >= max_nodes:
            break
        # Find the best unselected candidate of this type
        for sn in all_sorted:
            if sn.node_id in ranked_ids:
                continue
            node = graph.get_node(sn.node_id)
            if not node:
                continue
            node_type = (node.get("type") or "").lower()
            if node_type == floor_type and TYPE_CAPS.get(floor_type, DEFAULT_TYPE_CAP) > 0:
                ranked.append(sn)
                ranked_ids.add(sn.node_id)
                type_counts[floor_type] = 1
                break

    # ranked_ids is already built above; use it for edge collection
    edges_seen: set[tuple[str, str, str]] = set()
    subgraph_edges: list[dict] = []

    for node in ranked:
        for edge in graph.get_neighbors(node.node_id):
            src = edge.get("source", "")
            tgt = edge.get("target", "")
            key = (src, edge.get("type", ""), tgt)
            if key in edges_seen:
                continue
            if src in ranked_ids and tgt in ranked_ids:
                edges_seen.add(key)
                subgraph_edges.append(edge)
                if len(subgraph_edges) >= max_edges:
                    return ranked, subgraph_edges

    return ranked, subgraph_edges


# ---------------------------------------------------------------------------
# 4. Context formatting
# ---------------------------------------------------------------------------

def _fmt_node(node: Optional[dict]) -> str:
    if not node:
        return "None"
    parts = [f"id={node.get('id', '')}", f"type={node.get('type', '')}"]
    if node.get("name"):
        parts.append(f"name={node['name']}")
    desc = str(node.get("description") or "")[:DESCRIPTION_MAX_CHARS]
    if desc:
        parts.append(f"description={desc}")
    return " | ".join(parts)


def _fmt_edge(edge: Optional[dict]) -> str:
    if not edge:
        return "None"
    parts = [
        f"{edge.get('source', '')} --[{edge.get('type', '')}]--> {edge.get('target', '')}"
    ]
    for field_name in ("method", "confidence", "layer_from", "layer_to"):
        v = edge.get(field_name)
        if v not in (None, "", [], {}):
            parts.append(f"{field_name}={v}")
    if edge.get("evidence"):
        parts.append(f"evidence={edge['evidence']}")
    return " | ".join(parts)


def _fmt_path(sn: ScoredNode, graph: GraphStore) -> str:
    """Render the BFS path as a readable arrow chain: A → [edge_type] → B(name) → ..."""
    if not sn.path_edges:
        return sn.node_id
    parts = [sn.path_node_ids[0]]
    for edge, nid in zip(sn.path_edges, sn.path_node_ids[1:]):
        node = graph.get_node(nid)
        label = (node.get("name") or nid) if node else nid
        parts.append(f"→ [{edge.get('type', '')}] → {nid}({label})")
    return " ".join(parts)


def build_context_v2(
    question: str,
    ranked_nodes: list[ScoredNode],
    subgraph_edges: list[dict],
    graph: GraphStore,
    cross_domain: bool,
) -> str:
    """
    Assemble the LLM-facing context string from BFS-scored results.

    Sections:
      1. Header — question + cross-domain flag
      2. Seed Nodes — exact-match or top semantic seeds, with direct edges
      3. Multi-hop Evidence — depth ≥ 1 nodes, each annotated with incoming edge
      4. Reasoning Paths — up to MAX_PATHS_IN_CONTEXT highest-scoring chains
    """
    lines: list[str] = []

    lines.append(f"User question: {question}")
    if cross_domain:
        lines.append("[Cross-domain query: CVE/NVD ↔ ATT&CK — bridge edges boosted]")
    lines.append("")

    seeds = [n for n in ranked_nodes if n.is_seed]
    multi_hop = [n for n in ranked_nodes if not n.is_seed]

    # --- Section 1: Seed nodes ---
    lines.append("=== Seed Nodes ===")
    for i, sn in enumerate(seeds, 1):
        node = graph.get_node(sn.node_id)
        tag = " [EXACT MATCH]" if sn.is_exact_match else ""
        lines.append(f"{i}. [score={sn.score:.3f}]{tag} {_fmt_node(node)}")
        # Show direct edges from this seed for immediate context
        direct_edges = [
            e for e in subgraph_edges
            if e.get("source") == sn.node_id or e.get("target") == sn.node_id
        ][:4]
        for edge in direct_edges:
            lines.append(f"   edge: {_fmt_edge(edge)}")

    # --- Section 2: Multi-hop evidence ---
    if multi_hop:
        lines.append("")
        lines.append("=== Multi-hop Evidence ===")
        for i, sn in enumerate(multi_hop, 1):
            node = graph.get_node(sn.node_id)
            lines.append(
                f"{i}. [score={sn.score:.3f}, depth={sn.depth}] {_fmt_node(node)}"
            )
            if sn.path_edges:
                # The last edge in the path is the one that brought us here
                lines.append(f"   via: {_fmt_edge(sn.path_edges[-1])}")

    # --- Section 3: Reasoning paths ---
    deep_nodes = sorted(
        [n for n in ranked_nodes if n.depth >= 1],
        key=lambda n: n.score,
        reverse=True,
    )[:MAX_PATHS_IN_CONTEXT]
    if deep_nodes:
        lines.append("")
        lines.append("=== Reasoning Paths ===")
        for sn in deep_nodes:
            lines.append(f"[score={sn.score:.3f}] {_fmt_path(sn, graph)}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 5. Top-level pipeline entry point
# ---------------------------------------------------------------------------

def retrieve_and_build_context(
    question: str,
    retriever: Retriever,
    graph: GraphStore,
    max_depth: int = MAX_BFS_DEPTH,
    verbose: bool = False,
) -> tuple[str, list[ScoredNode]]:
    """
    Full GraphRAG v2 pipeline:
        seeds → BFS expand → prune → assemble subgraph → format context

    Args:
        question   — raw user query
        retriever  — Retriever instance (embeddings + sentence-transformer)
        graph      — GraphStore instance (loaded knowledge graph)
        max_depth  — BFS hop limit (default MAX_BFS_DEPTH from config)
        verbose    — print BFS progress to stdout

    Returns:
        context       — formatted string ready to pass to the LLM
        ranked_nodes  — scored nodes for metrics / run logging
    """
    cross_domain = is_cross_domain_query(question)
    intent = detect_query_intent(question)
    if verbose and cross_domain:
        print("[BFS] Cross-domain query detected — bridge edges will be boosted")
    if verbose:
        active = [k for k, v in intent.items() if v]
        if active:
            print(f"[BFS] Query intent: {active}")

    # Chain/multi-hop questions need one extra hop and a bigger node budget
    is_chain = intent.get("wants_chain") or intent.get("wants_campaign")
    effective_depth = max_depth + 1 if (intent.get("is_multi_hop") or is_chain) else max_depth
    assembly_max = MAX_NODES_MULTI_HOP if is_chain else MAX_NODES_IN_SUBGRAPH

    seeds = get_seeds(question, retriever, graph, cross_domain)
    if verbose:
        print(f"[BFS] Seeds ({len(seeds)}): {[s.node_id for s in seeds]}")

    scored_nodes = bfs_expand(seeds, graph, max_depth=effective_depth, intent=intent)
    if verbose:
        print(f"[BFS] Expanded to {len(scored_nodes)} candidate nodes")

    # Force-include typed neighbors of exact-match seeds (bypasses edge-cap cutoff)
    exact_seeds = [s for s in seeds if s.is_exact_match]
    if exact_seeds:
        scored_nodes = force_expand_exact_seeds(exact_seeds, scored_nodes, graph)
        if verbose:
            forced = sum(1 for s in scored_nodes.values() if s.is_forced_neighbor)
            print(f"[BFS] Forced neighbors added: {forced}")

    scored_nodes = rerank_post_bfs(question, scored_nodes, graph, intent)
    if verbose:
        print("[BFS] Post-BFS reranking applied")

    ranked_nodes, subgraph_edges = assemble_subgraph(
        scored_nodes, graph, max_nodes=assembly_max
    )
    if verbose:
        print(
            f"[BFS] Subgraph assembled: {len(ranked_nodes)} nodes, "
            f"{len(subgraph_edges)} edges"
        )

    context = build_context_v2(
        question, ranked_nodes, subgraph_edges, graph, cross_domain
    )
    return context, ranked_nodes
