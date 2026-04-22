import json
import math
import os
import re
import time
from collections import Counter
from typing import List, Dict, Any

from ask_llm_gemini import (
    configure_gemini,
    get_client,
    ask_gemini,
    retrieve_and_format,
)
from retriever import Retriever
from graph_utils import GraphStore


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

BENCHMARK_PATH = os.path.join(BASE_DIR, "benchmark_questions.json")
OUTPUT_PATH = os.path.join(BASE_DIR, "evaluation_results.json")

GRAPH_PATH = "data/graphrag/multi_layer/global_graph.json"
EMBEDDINGS_PATH = "data/graphrag/multi_layer/embeddings.npy"
METADATA_PATH = "data/graphrag/multi_layer/embeddings_metadata.json"

GEMINI_MODEL = "gemini-2.5-flash"
MAX_RETRIES = 5


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s\-.:/]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize(text: str) -> List[str]:
    return normalize_text(text).split()


def compute_exact_match(prediction: str, reference: str) -> int:
    return int(normalize_text(prediction) == normalize_text(reference))


def compute_precision_recall_f1(prediction: str, reference: str) -> Dict[str, float]:
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)

    if not pred_tokens or not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    pred_counter = Counter(pred_tokens)
    ref_counter = Counter(ref_tokens)
    overlap = pred_counter & ref_counter
    num_same = sum(overlap.values())

    if num_same == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    f1 = 2 * precision * recall / (precision + recall)

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def ngrams(tokens: List[str], n: int) -> List[tuple]:
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def modified_precision(pred_tokens: List[str], ref_tokens: List[str], n: int) -> float:
    pred_ngrams = Counter(ngrams(pred_tokens, n))
    ref_ngrams = Counter(ngrams(ref_tokens, n))

    if not pred_ngrams:
        return 0.0

    overlap = pred_ngrams & ref_ngrams
    return sum(overlap.values()) / sum(pred_ngrams.values())


def compute_bleu(prediction: str, reference: str, max_n: int = 4) -> float:
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)

    if not pred_tokens or not ref_tokens:
        return 0.0

    precisions = []
    for n in range(1, max_n + 1):
        p = modified_precision(pred_tokens, ref_tokens, n)
        precisions.append(max(p, 1e-9))

    log_precision_sum = sum((1 / max_n) * math.log(p) for p in precisions)

    pred_len = len(pred_tokens)
    ref_len = len(ref_tokens)

    if pred_len == 0:
        return 0.0

    bp = 1.0 if pred_len > ref_len else math.exp(1 - ref_len / pred_len)
    bleu = bp * math.exp(log_precision_sum)
    return round(bleu, 4)


def _lcs_length(a: List[str], b: List[str]) -> int:
    """Space-efficient longest common subsequence length."""
    n = len(b)
    prev = [0] * (n + 1)
    for token_a in a:
        curr = [0] * (n + 1)
        for j, token_b in enumerate(b, 1):
            curr[j] = prev[j - 1] + 1 if token_a == token_b else max(curr[j - 1], prev[j])
        prev = curr
    return prev[n]


def compute_rouge_l(prediction: str, reference: str) -> float:
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_length(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)


def extract_ids(text: str) -> List[str]:
    patterns = [
        r"\bCVE-\d{4}-\d+\b",
        r"\bTA\d{4}\b",
        r"\bT\d{4}(?:\.\d{3})?\b",
        r"\bM\d{4}\b",
        r"\bG\d{4}\b",
        r"\bC\d{4}\b",
        r"\bS\d{4}\b",
    ]

    found = []
    for pattern in patterns:
        found.extend(re.findall(pattern, text, re.IGNORECASE))

    return sorted(set(x.upper() for x in found))


def compute_id_match(
    prediction: str,
    gold_ids: List[str],
    retrieved_ids: List[str] | None = None,
) -> Dict[str, Any]:
    pred_ids = set(extract_ids(prediction))
    gold_id_set = set(x.upper() for x in gold_ids)

    tp = len(pred_ids & gold_id_set)
    fp = len(pred_ids - gold_id_set)
    fn = len(gold_id_set - pred_ids)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    # Hallucination: predicted IDs not present in retrieved context
    if retrieved_ids is not None:
        retrieved_set = set(x.upper() for x in retrieved_ids)
        hallucinated = sorted(pred_ids - retrieved_set)
        grounding_score = round(
            len(pred_ids & retrieved_set) / len(pred_ids), 4
        ) if pred_ids else 1.0
    else:
        hallucinated = []
        grounding_score = None

    return {
        "predicted_ids": sorted(pred_ids),
        "gold_ids": sorted(gold_id_set),
        "id_precision": round(precision, 4),
        "id_recall": round(recall, 4),
        "id_f1": round(f1, 4),
        "hallucinated_ids": hallucinated,
        "grounding_score": grounding_score,
    }


def compute_hit_at_k(results: List[dict], gold_ids: List[str]) -> int:
    retrieved_ids = {r.get("id", "").upper() for r in results}
    gold_id_set = {g.upper() for g in gold_ids}
    return int(len(retrieved_ids & gold_id_set) > 0)


def _parse_judge_json(text: str) -> Dict[str, Any]:
    """Parse JSON from judge output, stripping markdown fences if present."""
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group())
    return json.loads(text)


def llm_as_judge(
    question: str,
    reference: str,
    candidate: str,
    retrieved_ids: List[str] | None = None,
) -> Dict[str, Any]:
    ids_line = ""
    if retrieved_ids:
        ids_line = f"\nRetrieved Graph IDs (evidence available to the system): {', '.join(retrieved_ids[:25])}\n"

    prompt = (
        "You are evaluating a cybersecurity GraphRAG system answer. "
        "Score the candidate answer on each dimension from 1 (very poor) to 5 (excellent).\n\n"
        f"Question: {question}\n"
        f"{ids_line}"
        f"Reference Answer: {reference}\n\n"
        f"Candidate Answer: {candidate}\n\n"
        "Dimensions:\n"
        "- correctness: factual accuracy relative to the reference\n"
        "- completeness: covers all key facts from the reference\n"
        "- faithfulness: no invented IDs or facts beyond the provided evidence\n"
        "- clarity: concise, professional, readable\n"
        "- relevance: directly addresses the question without off-topic content\n"
        "- grounding: every ID or claim cited is present in the Retrieved Graph IDs\n"
        "- overall: holistic quality (not a simple average)\n\n"
        "Return ONLY valid JSON with no markdown fences:\n"
        '{"correctness":<1-5>,"completeness":<1-5>,"faithfulness":<1-5>,'
        '"clarity":<1-5>,"relevance":<1-5>,"grounding":<1-5>,'
        '"overall":<1-5>,"justification":"<one sentence>"}'
    )

    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            response = get_client().models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )
            try:
                return _parse_judge_json(response.text)
            except (json.JSONDecodeError, ValueError):
                return {"raw_output": response.text}

        except Exception as e:
            last_error = e
            error_text = str(e)

            if (
                "503" in error_text
                or "UNAVAILABLE" in error_text
                or "429" in error_text
                or "RESOURCE_EXHAUSTED" in error_text
            ) and attempt < MAX_RETRIES - 1:
                wait_time = min(60, 5 * (2 ** attempt))
                print(f"LLM judge temporarily unavailable. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue

            return {"judge_error": str(e)}

    return {"judge_error": str(last_error)}


def evaluate_one(
    item: Dict[str, Any],
    retriever: Retriever,
    graph: GraphStore,
    use_llm_judge: bool = False,
    answer_style: str = "concise",
) -> Dict[str, Any]:
    question = item["question"]
    reference_answer = item["reference_answer"]
    gold_ids = item.get("gold_ids", [])

    context, results = retrieve_and_format(question, retriever, graph)
    retrieved_ids = [r.get("id") for r in results]

    prediction = ask_gemini(context, question, answer_style=answer_style)
    prediction = re.sub(r"\*\*|__", "", prediction).strip()  # strip bold markdown

    prf = compute_precision_recall_f1(prediction, reference_answer)
    bleu = compute_bleu(prediction, reference_answer)
    rouge_l = compute_rouge_l(prediction, reference_answer)
    exact_match = compute_exact_match(prediction, reference_answer)
    id_metrics = compute_id_match(prediction, gold_ids, retrieved_ids=retrieved_ids)
    hit_at_k = compute_hit_at_k(results, gold_ids)

    output = {
        "id": item.get("id", ""),
        "category": item.get("category", ""),
        "difficulty": item.get("difficulty", ""),
        "question": question,
        "reference_answer": reference_answer,
        "prediction": prediction,
        "retrieved_ids": retrieved_ids,
        "non_llm_metrics": {
            "exact_match": exact_match,
            "precision": prf["precision"],
            "recall": prf["recall"],
            "f1": prf["f1"],
            "bleu": bleu,
            "rouge_l": rouge_l,
            "hit_at_k": hit_at_k,
            **id_metrics,
        },
    }

    if use_llm_judge:
        output["llm_judge"] = llm_as_judge(
            question, reference_answer, prediction, retrieved_ids=retrieved_ids
        )

    return output


def print_report(result: Dict[str, Any]) -> None:
    qid = result.get("id", "")
    cat = result.get("category", "")
    print(f"\n{'='*65}")
    print(f"[{qid}] {cat}")
    print(f"{'='*65}")
    print(f"Q : {result['question']}")
    print(f"REF: {result['reference_answer']}")
    print(f"ANS: {result['prediction']}")

    m = result["non_llm_metrics"]
    print(
        f"\n  EM={m['exact_match']} | P={m['precision']} R={m['recall']} F1={m['f1']} "
        f"| BLEU={m['bleu']} ROUGE-L={m['rouge_l']} | hit@k={m['hit_at_k']}"
    )
    print(
        f"  ID: P={m['id_precision']} R={m['id_recall']} F1={m['id_f1']} "
        f"| grounding={m.get('grounding_score')} "
        f"| hallucinated={m.get('hallucinated_ids', [])}"
    )
    print(f"  predicted={m['predicted_ids']}")
    print(f"  gold     ={m['gold_ids']}")

    if "llm_judge" in result:
        j = result["llm_judge"]
        if "judge_error" in j:
            print(f"  Judge error: {j['judge_error']}")
        elif "raw_output" in j:
            print(f"  Judge raw: {j['raw_output'][:120]}")
        else:
            scores = " | ".join(
                f"{k}={j.get(k, '?')}"
                for k in ["correctness", "completeness", "faithfulness",
                           "clarity", "relevance", "grounding", "overall"]
            )
            print(f"  Judge: {scores}")
            print(f"  Justification: {j.get('justification', '')}")


def compute_aggregate_metrics(all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not all_results:
        return {}

    scalar_keys = [
        "exact_match", "precision", "recall", "f1",
        "bleu", "rouge_l", "hit_at_k",
        "id_precision", "id_recall", "id_f1",
    ]
    aggregates: Dict[str, Any] = {}
    for key in scalar_keys:
        values = [r["non_llm_metrics"].get(key, 0.0) for r in all_results]
        aggregates[f"avg_{key}"] = round(sum(values) / len(values), 4)

    # Grounding score — only where it was computed (retrieved_ids available)
    grounding_vals = [
        r["non_llm_metrics"].get("grounding_score")
        for r in all_results
        if r["non_llm_metrics"].get("grounding_score") is not None
    ]
    if grounding_vals:
        aggregates["avg_grounding_score"] = round(sum(grounding_vals) / len(grounding_vals), 4)

    # Hallucination rate: avg number of hallucinated IDs per question
    aggregates["avg_hallucinated_ids"] = round(
        sum(len(r["non_llm_metrics"].get("hallucinated_ids", [])) for r in all_results)
        / len(all_results), 4
    )

    # LLM judge aggregates
    judge_keys = [
        "correctness", "completeness", "faithfulness",
        "clarity", "relevance", "grounding", "overall",
    ]
    available_judges = [
        r["llm_judge"]
        for r in all_results
        if "llm_judge" in r
        and isinstance(r["llm_judge"], dict)
        and "judge_error" not in r["llm_judge"]
        and "raw_output" not in r["llm_judge"]
    ]
    if available_judges:
        for key in judge_keys:
            values = [j[key] for j in available_judges if key in j and isinstance(j[key], (int, float))]
            if values:
                aggregates[f"avg_judge_{key}"] = round(sum(values) / len(values), 4)

    # Per-category breakdown
    cat_groups: Dict[str, list] = {}
    for r in all_results:
        cat = r.get("category", "unknown")
        cat_groups.setdefault(cat, []).append(r)

    aggregates["by_category"] = {}
    for cat, items in cat_groups.items():
        cat_agg: Dict[str, Any] = {"count": len(items)}
        for key in ["precision", "recall", "f1", "bleu", "rouge_l", "hit_at_k"]:
            vals = [i["non_llm_metrics"].get(key, 0.0) for i in items]
            cat_agg[f"avg_{key}"] = round(sum(vals) / len(vals), 4)
        aggregates["by_category"][cat] = cat_agg

    return aggregates


def load_benchmark(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        raise ValueError(f"Benchmark file is empty: {path}")

    data = json.loads(content)

    if not isinstance(data, list):
        raise ValueError("benchmark_questions.json must contain a JSON list.")

    required_keys = {"question", "reference_answer"}
    for i, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Benchmark item #{i} is not a JSON object.")
        missing = required_keys - set(item.keys())
        if missing:
            raise ValueError(f"Benchmark item #{i} is missing required keys: {sorted(missing)}")

    return data


def print_aggregate_summary(agg: Dict[str, Any]) -> None:
    print("\n" + "=" * 65)
    print("AGGREGATE METRICS")
    print("=" * 65)
    top_keys = [
        "avg_exact_match", "avg_precision", "avg_recall", "avg_f1",
        "avg_bleu", "avg_rouge_l", "avg_hit_at_k",
        "avg_id_precision", "avg_id_recall", "avg_id_f1",
        "avg_grounding_score", "avg_hallucinated_ids",
    ]
    for k in top_keys:
        if k in agg:
            print(f"  {k}: {agg[k]}")

    judge_keys = [k for k in agg if k.startswith("avg_judge_")]
    if judge_keys:
        print("\n  LLM Judge averages:")
        for k in judge_keys:
            print(f"    {k}: {agg[k]}")

    if "by_category" in agg:
        print("\n  By category:")
        for cat, vals in agg["by_category"].items():
            print(
                f"    {cat:22s} n={vals['count']} | "
                f"F1={vals['avg_f1']} BLEU={vals['avg_bleu']} "
                f"ROUGE-L={vals['avg_rouge_l']} hit@k={vals['avg_hit_at_k']}"
            )


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="GraphRAG LLM evaluation pipeline")
    parser.add_argument("--style", default="concise",
                        choices=["concise", "detailed", "bullet", "analyst"],
                        help="Answer generation style")
    parser.add_argument("--judge", action="store_true",
                        help="Enable LLM-as-judge scoring")
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only first N questions (for quick tests)")
    args = parser.parse_args()

    configure_gemini()

    retriever = Retriever(EMBEDDINGS_PATH, METADATA_PATH)
    graph = GraphStore(GRAPH_PATH)

    benchmark = load_benchmark(BENCHMARK_PATH)
    if args.limit:
        benchmark = benchmark[: args.limit]

    print(f"Evaluating {len(benchmark)} questions | style={args.style} | judge={args.judge}")

    all_results = []
    for item in benchmark:
        result = evaluate_one(
            item, retriever, graph,
            use_llm_judge=args.judge,
            answer_style=args.style,
        )
        all_results.append(result)
        print_report(result)

    aggregate_metrics = compute_aggregate_metrics(all_results)
    print_aggregate_summary(aggregate_metrics)

    final_output = {
        "benchmark_path": BENCHMARK_PATH,
        "answer_style": args.style,
        "num_questions": len(all_results),
        "aggregate_metrics": aggregate_metrics,
        "results": all_results,
    }

    print(f"\nSaving evaluation results to: {OUTPUT_PATH}")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()