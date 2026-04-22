import requests
from retriever import Retriever
from graph_utils import GraphStore
from bfs_retriever import retrieve_and_build_context, extract_ids

GRAPH_PATH = "data/graphrag/multi_layer/global_graph.json"
EMBEDDINGS_PATH = "data/graphrag/multi_layer/embeddings.npy"
METADATA_PATH = "data/graphrag/multi_layer/embeddings_metadata.json"

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "gemma3"


def ask_ollama(context: str, question: str) -> str:
    system_prompt = (
        "You are a cybersecurity graph reasoning assistant. "
        "Answer only from the provided graph evidence. "
        "If the evidence is insufficient, say so clearly. "
        "When relevant, explain the reasoning path through the graph."
    )

    user_prompt = f"""
Use the graph evidence below to answer the user's question.

{context}

Instructions:
- Be precise.
- Mention the most relevant nodes and edges.
- If a CVE, ATT&CK technique, or CVSS node appears, name it explicitly.
- Prefer graph-grounded reasoning over general knowledge.
- If there are multiple plausible paths, mention the strongest one first.

Question:
{question}
"""

    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()
    return data["message"]["content"]


def main():
    question = input("Ask a question: ").strip()
    if not question:
        print("Question cannot be empty.")
        return

    retriever = Retriever(EMBEDDINGS_PATH, METADATA_PATH)
    graph = GraphStore(GRAPH_PATH)

    print("\n[GraphRAG v2 — BFS retrieval]")
    context, ranked_nodes = retrieve_and_build_context(
        question, retriever, graph, verbose=True
    )

    print(f"\nRetrieved {len(ranked_nodes)} nodes")
    print("\nGenerating answer with local LLM...\n")
    answer = ask_ollama(context, question)

    print("=" * 90)
    print("FINAL ANSWER")
    print("=" * 90)
    print(answer)

    predicted = extract_ids(answer)
    if predicted:
        print(f"\nPredicted IDs: {predicted}")


if __name__ == "__main__":
    main()
