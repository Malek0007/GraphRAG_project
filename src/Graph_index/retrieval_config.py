# ============================================================
# GraphRAG v2 — Retrieval Configuration
# Tune these constants to control traversal depth, scoring
# weights, and pruning without touching algorithm logic.
# ============================================================

# ----- BFS Traversal Limits -----
MAX_BFS_DEPTH: int = 2           # max hops from seed nodes
TOP_K_SEEDS: int = 5             # seeds from semantic search (no exact ID)
MAX_NODES_IN_SUBGRAPH: int = 14  # tightened 20→14 for precision
MAX_NODES_MULTI_HOP: int = 18    # tightened 30→18 for chain/multi-hop queries
MAX_EDGES_IN_SUBGRAPH: int = 60
MAX_EDGES_PER_NODE: int = 12     # raised from 8 to catch related_software edges

# ----- Seed Scoring -----
EXACT_MATCH_BONUS: float = 2.0
CROSS_DOMAIN_SEED_BONUS: float = 1.2

# ----- Traversal Decay -----
# 0.65 → ~35% penalty per hop; keeps depth-2 nodes, discourages depth-3+.
DEPTH_DECAY: float = 0.65

# ----- Edge-Type Weights -----
EDGE_TYPE_WEIGHTS: dict[str, float] = {
    "exploited_via":        1.9,   # CVE → ATT&CK technique (highest-value bridge)
    "related_software":     1.8,   # CVE/Report → Software (cross-layer bridge)
    "bridge":               1.8,   # generic cross-layer edge
    "uses":                 1.5,   # Group/Campaign/Software → Technique
    "mitigates":            1.7,   # Mitigation → Technique
    "subtechnique_of":      1.4,   # Sub-technique → parent Technique
    "belongs_to":           1.6,   # Technique → Tactic
    "part_of":              1.5,   # Campaign/Software membership
    "attributed_to":        1.5,   # Attribution edges (Group → Campaign)
    "has_atomic_test":      1.7,   # Technique → Atomic test (L1↔L3 bridge)
    "mentions_technique":   1.6,   # Report → ATT&CK technique (L4↔L1 bridge)
    "mentions_cve":         1.6,   # Report → CVE (L4↔L2 bridge)
    "implements_technique": 1.5,   # Report entity → Technique
    "has_cvss":             0.6,   # CVE → CVSS metadata
    "has_weakness":         0.9,   # CVE → CWE
    "related_to":           0.8,
    "associated_with":      0.8,
    "mentions_entity":      0.4,   # generic report mention (low signal)
}
DEFAULT_EDGE_WEIGHT: float = 0.5   # metadata edges should not compete with gold paths

# ----- Bridge-Edge Bonus -----
BRIDGE_EDGE_BONUS: float = 1.3

# ----- Pruning -----
PRUNE_SCORE_THRESHOLD: float = 0.22
# Protected types use a much lower floor so they're never pruned too early
SOFT_PRUNE_THRESHOLD: float = 0.08
SOFT_PRUNE_TYPES: frozenset[str] = frozenset({
    "cve", "software", "group", "campaign", "tactic", "mitigation"
})

# ----- Post-assembly score-ratio floor -----
# After assembling the top-N nodes, drop any non-seed, non-forced node
# whose score is below  SCORE_RATIO_FLOOR × max_non_seed_score.
# This removes low-confidence tail nodes without hurting recall on
# strong-path gold IDs (which score well above this floor).
SCORE_RATIO_FLOOR: float = 0.12

# ----- Noise filtering — excluded from BFS traversal entirely -----
# Adding a type here prevents it from being explored or included at all.
NOISE_NODE_TYPES: frozenset[str] = frozenset({
    # Original metadata noise
    "url", "date", "source_identifier", "version", "vendor",
    # Atomic layer noise (commands are too verbose; platforms too generic)
    "cleanup_command", "platform", "command",
    # Report-layer structural noise (never a gold ID in any benchmark category)
    "behavior", "artifact", "component", "infrastructure",
    "crypto_algo", "c2_infra", "ipc", "target_sector",
    "persistence_mechanism", "file_type", "credential_store",
    "attack_vector",
})

# ----- Diversity caps — max nodes of each type in the assembled subgraph -----
TYPE_CAPS: dict[str, int] = {
    # ATT&CK + CVE layer  (these are gold-ID types — caps are generous)
    "cve":               4,
    "cvss":              1,
    "cvss_vector":       0,   # completely exclude
    "technique":         6,
    "sub-technique":     4,
    "tactic":            4,
    "mitigation":        4,
    "software":          5,
    "group":             4,
    "campaign":          3,
    "cwe":               2,
    # Low-value ATT&CK metadata
    "cpe":               0,
    "product":           0,
    "vendor":            0,
    "url":               0,
    "date":              0,
    "source_identifier": 0,
    "severity":          0,
    "data_component":    0,
    # Atomic Red Team layer
    "atomic_test":       2,
    "observable":        1,
    # Threat Report layer
    "threat_report":     1,
    "malware":           2,
    "threat_actor":      2,
    "process":           1,
    "tool":              1,
}
DEFAULT_TYPE_CAP: int = 1   # lowered from 2 — unknown types are likely noise

# ----- Context Formatting -----
DESCRIPTION_MAX_CHARS: int = 300
MAX_PATHS_IN_CONTEXT: int = 5
