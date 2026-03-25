import json
import numpy as np
from sentence_transformers import SentenceTransformer


class Retriever:
    def __init__(self, embeddings_path: str, metadata_path: str, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings = np.load(embeddings_path)

        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        if len(self.embeddings) != len(self.metadata):
            raise ValueError(
                f"Embeddings count ({len(self.embeddings)}) != metadata count ({len(self.metadata)})"
            )

        self.embeddings = self._normalize_matrix(self.embeddings)

    def _normalize_matrix(self, matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        return matrix / norms

    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        norm = max(np.linalg.norm(vector), 1e-12)
        return vector / norm

    def search(self, question: str, top_k: int = 10):
        query_embedding = self.model.encode(question, convert_to_numpy=True)
        query_embedding = self._normalize_vector(query_embedding)

        scores = self.embeddings @ query_embedding
        top_indices = np.argsort(scores)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            item = dict(self.metadata[idx])
            item["score"] = float(scores[idx])
            item["embedding_index"] = int(idx)
            results.append(item)

        return results