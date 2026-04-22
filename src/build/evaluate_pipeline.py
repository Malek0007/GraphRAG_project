import json
import math
import re
from collections import Counter
from typing import List, Dict, Any, Optional

from google import genai

from retriever import Retriever
from graph_utils import GraphStore
from ask_graph import (
    configure_gemini,
    build_context,
    ask_gemini,
    retrieve_results,
)

GRAPH_PATH = "data/graphrag/multi_layer/global_graph.json"
EMBEDDINGS_PATH = "data/graphrag/multi_layer/embeddings.npy"
METADATA_PATH = "data/graphrag/multi_layer/embeddings_metadata.json"
BENCHMARK_PATH = "benchmark_questions.json"
OUTPUT_PATH = "evaluation_results.json"
GEMINI_MODEL = "gemini-2.5-flash"


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
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


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


def compute_id_match(prediction: str, gold_ids: List[str]) -> Dict[str, Any]:
    pred_ids = set(extract_ids(prediction))
    gold_ids = set(x.upper() for x in gold_ids)

    tp = len(pred_ids & gold_ids)
    fp = len(pred_ids - gold_ids)
    fn = len(gold_ids - pred_ids)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "predicted_ids": sorted(pred_ids),
        "gold_ids": sorted(gold_ids),
        "id_precision": round(precision, 4),
        "id_recall": round(recall, 4),
        "id_f1": round(f1, 4),
    }


def compute_hit_at_k(results: List[dict], gold_ids: List[str]) -> int:
    retrieved_ids = {r.get("id", "").upper() for r in results}
    gold_ids = {g.upper() for g in gold_ids}
    return int(len(retrieved_ids & gold_ids) > 0)


def llm_as_judge(question: str, reference: str, candidate: str) -> Dict[str, Any]:
    prompt = f"""
You are evaluating a cybersecurity GraphRAG answer.

Question:
{question}

Reference Answer:
{reference}

Candidate Answer:
{candidate}

Score from 1 to 5:
- correctness
- completeness
- faithfulness
- clarity
- relevance

Return ONLY valid JSON:
{{
  "correctness": 0,
  "completeness": 0,
  "faithfulness": 0,
  "clarity": 0,
  "relevance": 0,
  "overall": 0,
  "justification": "short explanation"
}}
"""
    client = genai.Client(api_key=None)  # not used directly if configure_gemini already done
    from ask_graph import get_client
    response = get_client().models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )
    try:
        return json.loads(response.text)
    except json.JSONDecodeError:
        return {"raw_output": response.text}


def evaluate_one(
    item: Dict[str, Any],
    retriever: Retriever,
    graph: GraphStore,
    use_llm_judge: bool = True,
) -> Dict[str, Any]:
    question = item["question"]
    reference_answer = item["reference_answer"]
    gold_ids = item.get("gold_ids", [])

    results = retrieve_results(question, retriever, graph)
    context = build_context(question, results, graph)
    prediction = ask_gemini(context, question).replace("**", "").strip()

    prf = compute_precision_recall_f1(prediction, reference_answer)
    bleu = compute_bleu(prediction, reference_answer)
    exact_match = compute_exact_match(prediction, reference_answer)
    id_metrics = compute_id_match(prediction, gold_ids)
    hit_at_k = compute_hit_at_k(results, gold_ids)

    output = {
        "question": question,
        "reference_answer": reference_answer,
        "prediction": prediction,
        "non_llm_metrics": {
            "exact_match": exact_match,
            "precision": prf["precision"],
            "recall": prf["recall"],
            "f1": prf["f1"],
            "bleu": bleu,
            "hit_at_k": hit_at_k,
            **id_metrics,
        },
    }

    if use_llm_judge:
        output["llm_judge"] = llm_as_judge(question, reference_answer, prediction)

    return output


def print_report(result: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)
    print(f"Question: {result['question']}")
    print(f"Prediction: {result['prediction']}")
    print(f"Reference: {result['reference_answer']}")

    metrics = result["non_llm_metrics"]
    print("\nNon-LLM Metrics")
    print(f"- Exact Match: {metrics['exact_match']}")
    print(f"- Precision: {metrics['precision']}")
    print(f"- Recall: {metrics['recall']}")
    print(f"- F1: {metrics['f1']}")
    print(f"- BLEU: {metrics['bleu']}")
    print(f"- Hit@K: {metrics['hit_at_k']}")
    print(f"- ID Precision: {metrics['id_precision']}")
    print(f"- ID Recall: {metrics['id_recall']}")
    print(f"- ID F1: {metrics['id_f1']}")

    if "llm_judge" in result:
        judge = result["llm_judge"]
        print("\nLLM Judge")
        print(f"- Correctness: {judge.get('correctness')}")
        print(f"- Completeness: {judge.get('completeness')}")
        print(f"- Faithfulness: {judge.get('faithfulness')}")
        print(f"- Clarity: {judge.get('clarity')}")
        print(f"- Relevance: {judge.get('relevance')}")
        print(f"- Overall: {judge.get('overall')}")
        print(f"- Justification: {judge.get('justification')}")


def main() -> None:
    configure_gemini()

    retriever = Retriever(EMBEDDINGS_PATH, METADATA_PATH)
    graph = GraphStore(GRAPH_PATH)

    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    all_results = []
    for item in benchmark:
        result = evaluate_one(item, retriever, graph, use_llm_judge=True)
        all_results.append(result)
        print_report(result)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved evaluation results to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()