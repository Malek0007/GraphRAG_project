from retriever import Retriever
from graph_utils import GraphStore
from bfs_retriever import retrieve_and_build_context

GRAPH_PATH = "data/graphrag/multi_layer/global_graph.json"
EMBEDDINGS_PATH = "data/graphrag/multi_layer/embeddings.npy"
METADATA_PATH = "data/graphrag/multi_layer/embeddings_metadata.json"


def main():
    question = input("Ask a question: ").strip()
    if not question:
        print("Question cannot be empty.")
        return

    retriever = Retriever(EMBEDDINGS_PATH, METADATA_PATH)
    graph = GraphStore(GRAPH_PATH)

    print("\n[GraphRAG v2 — BFS retrieval]\n")
    context, ranked_nodes = retrieve_and_build_context(
        question, retriever, graph, verbose=True
    )

    print("\n" + "=" * 90)
    print("RETRIEVED CONTEXT")
    print("=" * 90)
    print(context)
    print(f"\nTotal nodes in subgraph: {len(ranked_nodes)}")


if __name__ == "__main__":
    main()
