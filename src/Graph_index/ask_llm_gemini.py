import os
import re
import json
import time
from datetime import datetime
from typing import Optional

from google import genai

from retriever import Retriever
from graph_utils import GraphStore
from bfs_retriever import retrieve_and_build_context, extract_ids


GRAPH_PATH = "data/graphrag/multi_layer/global_graph.json"
EMBEDDINGS_PATH = "data/graphrag/multi_layer/embeddings.npy"
METADATA_PATH = "data/graphrag/multi_layer/embeddings_metadata.json"

GEMINI_MODEL = "gemini-2.5-flash"
MAX_RETRIES = 5

# ---------------------------------------------------------------------------
# Answer styles — controls how the LLM formats its response
# ---------------------------------------------------------------------------

ANSWER_STYLES = {
    "concise": (
        "Answer in 1–3 sentences (50–120 words maximum). "
        "State the key fact directly. Use exact IDs (CVE-XXXX, TXXXX, MXXXX, etc.) "
        "as they appear in the evidence. No preamble, no bullet points, no markdown."
    ),
    "detailed": (
        "Write a professional paragraph covering all relevant nodes, edges, and "
        "relationships from the evidence. Include exact IDs inline. "
        "2–5 sentences, no bullet points."
    ),
    "bullet": (
        "Summarise findings as a compact bullet list (3–6 bullets). "
        "Each bullet: one fact with the relevant ID. No introductory sentence."
    ),
    "analyst": (
        "Write a concise analyst report: one topic sentence, then 2–4 sentences "
        "of supporting detail citing exact IDs and edge types (e.g. 'uses', "
        "'exploited_via'). Close with one-sentence assessment of risk/impact."
    ),
}

_SYSTEM_PROMPT = (
    "You are a cybersecurity intelligence analyst. "
    "Answer ONLY from the provided graph evidence. "
    "Do NOT invent facts, CVEs, technique IDs, or relationships not present in the evidence. "
    "If the evidence is insufficient, respond exactly: "
    "\"Not found in current graph evidence.\""
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
QUESTION_LOG_PATH = os.path.join(BASE_DIR, "question_logs.json")

_client: Optional[genai.Client] = None


def configure_gemini() -> genai.Client:
    global _client
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set.")
    _client = genai.Client(api_key=api_key)
    return _client


def get_client() -> genai.Client:
    if _client is None:
        raise RuntimeError("Gemini client is not configured. Call configure_gemini() first.")
    return _client


def _compress_context(context: str) -> str:
    """
    Pre-process BFS context before sending to the LLM:
    - Strip long CVSS vector strings (leave only the score label).
    - Remove pure metadata lines (url, date, source_identifier nodes).
    - Collapse repeated whitespace.
    """
    _NOISE_PREFIXES = ("url=URL::", "id=SOURCE::", "id=DATE::", "id=VECTOR::",
                       "id=SEVERITY::", "id=CVSS::")
    _CVSS_VECTOR_RE = re.compile(r"CVSS:[A-Z0-9./+:]+", re.IGNORECASE)

    lines = []
    for line in context.split("\n"):
        stripped = line.strip()
        # Drop pure noise lines
        if any(stripped.startswith(p) for p in _NOISE_PREFIXES):
            continue
        if stripped.startswith("id=URL::") or stripped.startswith("id=CPE::"):
            continue
        # Compress CVSS vectors to a short placeholder
        if "CVSS:" in line:
            line = _CVSS_VECTOR_RE.sub("[CVSS-vector]", line)
        lines.append(line)
    return "\n".join(lines)


def _build_prompt(context: str, question: str, answer_style: str = "concise") -> str:
    """Compose the full prompt sent to the LLM."""
    style_instruction = ANSWER_STYLES.get(answer_style, ANSWER_STYLES["concise"])
    clean_context = _compress_context(context)
    return (
        f"{_SYSTEM_PROMPT}\n\n"
        f"Graph Evidence:\n{clean_context}\n\n"
        f"Question: {question}\n\n"
        f"Output format: {style_instruction}"
    )


def ask_gemini(
    context: str,
    question: str,
    answer_style: str = "concise",
) -> str:
    """
    Call Gemini with the assembled graph evidence.

    Args:
        context      — formatted evidence string from retrieve_and_build_context
        question     — raw user question
        answer_style — one of 'concise', 'detailed', 'bullet', 'analyst'
    """
    prompt = _build_prompt(context, question, answer_style)
    client = get_client()
    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            print(f"Using Gemini model: {GEMINI_MODEL}")
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )
            return response.text

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
                print(f"Gemini temporarily unavailable. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue

            if "404" in error_text or "NOT_FOUND" in error_text:
                raise RuntimeError(
                    f"Model '{GEMINI_MODEL}' is not available for this account or project."
                ) from e

            raise RuntimeError(f"Gemini request failed: {e}") from e

    raise RuntimeError(f"Gemini request failed after retries: {last_error}")


def retrieve_and_format(
    question: str,
    retriever: Retriever,
    graph: GraphStore,
    verbose: bool = False,
) -> tuple[str, list[dict]]:
    """
    GraphRAG v2 retrieval: BFS expansion → scored subgraph → context assembly.

    Returns:
        context  — formatted evidence string for the LLM
        results  — list of dicts with 'id', 'score', 'kind' for metrics/logging
    """
    context, ranked_nodes = retrieve_and_build_context(
        question, retriever, graph, verbose=verbose
    )
    results = [
        {"kind": "node", "id": sn.node_id, "score": sn.score}
        for sn in ranked_nodes
    ]
    return context, results


def save_run_log(
    question: str,
    answer: str,
    results: list[dict],
    latency: float,
    path: str = QUESTION_LOG_PATH,
) -> None:
    record = {
        "question": question,
        "answer": answer,
        "retrieved_ids": [r.get("id") for r in results],
        "predicted_ids": extract_ids(answer),
        "retrieval_count": len(results),
        "model": GEMINI_MODEL,
        "latency_seconds": round(latency, 2),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    try:
        with open(path, "r", encoding="utf-8") as f:
            existing = f.read().strip()
            data = json.loads(existing) if existing else []
    except (FileNotFoundError, json.JSONDecodeError):
        data = []

    data.append(record)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def print_results_summary(results: list[dict]) -> None:
    print("\nRetrieved Results:")
    for i, result in enumerate(results, start=1):
        result_id = result.get("id", "N/A")
        result_kind = result.get("kind", "unknown")
        score = result.get("score", 0.0)
        print(f"{i}. kind={result_kind} | id={result_id} | score={score:.4f}")


def print_banner(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def main() -> None:
    try:
        configure_gemini()

        question = input("Ask a question: ").strip()
        if not question:
            print("Question cannot be empty.")
            return

        print_banner("LOADING GRAPH + RETRIEVER")
        retriever = Retriever(EMBEDDINGS_PATH, METADATA_PATH)
        graph = GraphStore(GRAPH_PATH)

        print_banner("RETRIEVAL (GraphRAG v2 — BFS)")
        context, results = retrieve_and_format(question, retriever, graph, verbose=True)
        print_results_summary(results)

        style = os.getenv("ANSWER_STYLE", "concise")
        print_banner(f"GENERATING ANSWER WITH GEMINI (style={style})")
        start_time = time.time()
        answer = ask_gemini(context, question, answer_style=style)
        elapsed = time.time() - start_time

        answer = answer.replace("**", "").strip()

        print_banner("FINAL ANSWER")
        print(answer)

        save_run_log(question, answer, results, elapsed)
        print(f"\nSaved run to: {QUESTION_LOG_PATH}")

        print_banner("RUN SUMMARY")
        print(f"Question: {question}")
        print(f"Retrieved results: {len(results)}")
        print(f"Predicted IDs: {extract_ids(answer)}")
        print(f"Model: {GEMINI_MODEL}")
        print(f"Latency: {elapsed:.2f} seconds")

    except Exception as e:
        print_banner("ERROR")
        print(str(e))


if __name__ == "__main__":
    main()
