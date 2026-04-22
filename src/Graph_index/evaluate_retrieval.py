"""
Retrieval-Only Evaluator — zero LLM / API calls.

Runs the GraphRAG v2 BFS pipeline and measures only the *retrieval* layer:
  - hit@k       : 1 if any gold ID appears in the retrieved subgraph
  - id_recall   : fraction of gold IDs retrieved
  - id_precision: fraction of retrieved IDs that are gold
  - id_f1       : harmonic mean of the above two
  - depth_cover : deepest depth at which a gold ID was found

Useful for rapid iteration on retrieval config without spending API quota.
Run:
    cd src/Graph_index
    python3 evaluate_retrieval.py
"""

import json
import os
import time
from collections import Counter

from retriever import Retriever
from graph_utils import GraphStore
from bfs_retriever import retrieve_and_build_context, ScoredNode

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_PATH = os.path.join(BASE_DIR, "benchmark_questions.json")
OUTPUT_PATH = os.path.join(BASE_DIR, "retrieval_eval_results.json")

PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
GRAPH_PATH = os.path.join(PROJECT_ROOT, "data/graphrag/multi_layer/global_graph.json")
EMBEDDINGS_PATH = os.path.join(PROJECT_ROOT, "data/graphrag/multi_layer/embeddings.npy")
METADATA_PATH = os.path.join(PROJECT_ROOT, "data/graphrag/multi_layer/embeddings_metadata.json")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _recall_at_k(ranked_nodes: list[ScoredNode], gold_set: set[str], k: int) -> float:
    top_k_ids = {sn.node_id.upper() for sn in ranked_nodes[:k]}
    return len(top_k_ids & gold_set) / len(gold_set)


def compute_retrieval_metrics(
    ranked_nodes: list[ScoredNode],
    gold_ids: list[str],
) -> dict:
    retrieved_ids = {sn.node_id.upper() for sn in ranked_nodes}
    gold_set = {g.upper() for g in gold_ids}

    if not gold_set:
        return {
            "hit_at_k": None,
            "id_precision": None,
            "id_recall": None,
            "id_f1": None,
            "mrr": None,
            "recall_at_1": None,
            "recall_at_5": None,
            "recall_at_10": None,
            "depth_cover": None,
            "retrieved_count": len(retrieved_ids),
            "gold_count": 0,
            "retrieved_gold_ids": [],
            "missing_gold_ids": [],
        }

    tp = retrieved_ids & gold_set
    precision = len(tp) / len(retrieved_ids) if retrieved_ids else 0.0
    recall = len(tp) / len(gold_set)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    hit = 1 if tp else 0

    # MRR: reciprocal rank of the first gold ID in the ranked list
    mrr = 0.0
    for rank, sn in enumerate(ranked_nodes, 1):
        if sn.node_id.upper() in gold_set:
            mrr = 1.0 / rank
            break

    # Recall@k variants
    r_at_1 = _recall_at_k(ranked_nodes, gold_set, 1)
    r_at_5 = _recall_at_k(ranked_nodes, gold_set, 5)
    r_at_10 = _recall_at_k(ranked_nodes, gold_set, 10)

    # Find the maximum depth at which any gold ID was found
    gold_nodes = [sn for sn in ranked_nodes if sn.node_id.upper() in gold_set]
    depth_cover = max((sn.depth for sn in gold_nodes), default=None)

    return {
        "hit_at_k": hit,
        "id_precision": round(precision, 4),
        "id_recall": round(recall, 4),
        "id_f1": round(f1, 4),
        "mrr": round(mrr, 4),
        "recall_at_1": round(r_at_1, 4),
        "recall_at_5": round(r_at_5, 4),
        "recall_at_10": round(r_at_10, 4),
        "depth_cover": depth_cover,
        "retrieved_count": len(retrieved_ids),
        "gold_count": len(gold_set),
        "retrieved_gold_ids": sorted(tp),
        "missing_gold_ids": sorted(gold_set - retrieved_ids),
    }


# ---------------------------------------------------------------------------
# Per-question evaluation
# ---------------------------------------------------------------------------

def evaluate_one(
    item: dict,
    retriever: Retriever,
    graph: GraphStore,
    verbose: bool = False,
) -> dict:
    question = item["question"]
    gold_ids = item.get("gold_ids", [])

    t0 = time.time()
    _, ranked_nodes = retrieve_and_build_context(
        question, retriever, graph, verbose=verbose
    )
    elapsed = time.time() - t0

    metrics = compute_retrieval_metrics(ranked_nodes, gold_ids)
    metrics["latency_seconds"] = round(elapsed, 3)

    return {
        "id": item.get("id", ""),
        "category": item.get("category", ""),
        "difficulty": item.get("difficulty", ""),
        "question": question,
        "gold_ids": gold_ids,
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

def compute_aggregates(results: list[dict]) -> dict:
    # Only include questions with non-null gold_ids
    scored = [r for r in results if r["metrics"]["hit_at_k"] is not None]

    if not scored:
        return {}

    keys = ["hit_at_k", "id_precision", "id_recall", "id_f1", "mrr",
            "recall_at_1", "recall_at_5", "recall_at_10"]
    agg = {}
    for k in keys:
        vals = [r["metrics"][k] for r in scored]
        agg[f"avg_{k}"] = round(sum(vals) / len(vals), 4)

    # By category
    cat_groups: dict[str, list] = {}
    for r in scored:
        cat = r.get("category", "unknown")
        cat_groups.setdefault(cat, []).append(r)

    agg["by_category"] = {}
    for cat, items in cat_groups.items():
        agg["by_category"][cat] = {
            "count": len(items),
            "avg_hit_at_k": round(sum(i["metrics"]["hit_at_k"] for i in items) / len(items), 4),
            "avg_id_recall": round(sum(i["metrics"]["id_recall"] for i in items) / len(items), 4),
            "avg_id_f1": round(sum(i["metrics"]["id_f1"] for i in items) / len(items), 4),
        }

    # By difficulty
    diff_groups: dict[str, list] = {}
    for r in scored:
        d = r.get("difficulty", "unknown")
        diff_groups.setdefault(d, []).append(r)

    agg["by_difficulty"] = {}
    for diff, items in diff_groups.items():
        agg["by_difficulty"][diff] = {
            "count": len(items),
            "avg_hit_at_k": round(sum(i["metrics"]["hit_at_k"] for i in items) / len(items), 4),
            "avg_id_recall": round(sum(i["metrics"]["id_recall"] for i in items) / len(items), 4),
        }

    # Missing-IDs summary: which gold IDs were hardest to retrieve?
    miss_counter: Counter = Counter()
    for r in scored:
        for mid in r["metrics"]["missing_gold_ids"]:
            miss_counter[mid] += 1
    agg["hardest_to_retrieve"] = miss_counter.most_common(10)

    return agg


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(result: dict) -> None:
    m = result["metrics"]
    hit = m["hit_at_k"]
    recall = m["id_recall"]
    f1 = m["id_f1"]
    depth = m["depth_cover"]
    missing = m["missing_gold_ids"]

    status = "PASS" if hit else ("SKIP" if hit is None else "FAIL")
    print(
        f"[{status}] {result['id']:4s} | {result['category']:22s} | "
        f"{result['difficulty']:6s} | hit={hit} | "
        f"recall={recall if recall is not None else '-':5} | "
        f"f1={f1 if f1 is not None else '-':5} | "
        f"depth={depth} | "
        f"missing={missing[:3]}{'…' if len(missing) > 3 else ''}"
    )


def print_summary(agg: dict) -> None:
    print("\n" + "=" * 70)
    print("AGGREGATE RETRIEVAL METRICS")
    print("=" * 70)
    for k in ["avg_hit_at_k", "avg_id_precision", "avg_id_recall", "avg_id_f1",
              "avg_mrr", "avg_recall_at_1", "avg_recall_at_5", "avg_recall_at_10"]:
        print(f"  {k}: {agg.get(k, 'N/A')}")

    print("\nBy category:")
    for cat, vals in agg.get("by_category", {}).items():
        print(
            f"  {cat:22s} n={vals['count']} | "
            f"hit@k={vals['avg_hit_at_k']} | "
            f"recall={vals['avg_id_recall']} | "
            f"f1={vals['avg_id_f1']}"
        )

    print("\nBy difficulty:")
    for diff, vals in agg.get("by_difficulty", {}).items():
        print(
            f"  {diff:8s} n={vals['count']} | "
            f"hit@k={vals['avg_hit_at_k']} | "
            f"recall={vals['avg_id_recall']}"
        )

    hardest = agg.get("hardest_to_retrieve", [])
    if hardest:
        print("\nHardest gold IDs to retrieve:")
        for node_id, count in hardest:
            print(f"  {node_id}: missed in {count} question(s)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading graph and retriever...")
    retriever = Retriever(EMBEDDINGS_PATH, METADATA_PATH)
    graph = GraphStore(GRAPH_PATH)

    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    print(f"Running retrieval eval on {len(benchmark)} questions (no LLM)\n")
    print(
        f"{'':6s} {'id':4s} | {'category':22s} | {'diff':6s} | "
        f"{'hit':3s} | {'recall':6s} | {'f1':5s} | {'depth':5s} | missing IDs"
    )
    print("-" * 90)

    all_results = []
    for item in benchmark:
        result = evaluate_one(item, retriever, graph, verbose=False)
        all_results.append(result)
        print_report(result)

    agg = compute_aggregates(all_results)
    print_summary(agg)

    output = {
        "num_questions": len(all_results),
        "aggregate": agg,
        "results": all_results,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
