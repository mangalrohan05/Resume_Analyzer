import chromadb
import numpy as np
import pandas as pd
from src.config import CFG


class VectorStore:
    """Persistent ChromaDB-backed vector store for resume embeddings."""

    def __init__(self, collection_name: str = None, db_path: str = None):
        self.db_path = db_path or CFG.CHROMA_DB_PATH
        self.collection_name = collection_name or CFG.CHROMA_COLLECTION
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def is_populated(self) -> bool:
        """Return True if the collection already has documents."""
        return self.collection.count() > 0

    def add_resumes(self, df: pd.DataFrame, embeddings: np.ndarray) -> None:
        """Index all resumes. Skips already-existing IDs."""
        ids = [str(i) for i in df.index.tolist()]
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=df["Resume_str"].tolist(),
            metadatas=[{"category": str(c)} for c in df["Category"]],
        )

    def query(self, query_embedding: np.ndarray, top_k: int = None) -> dict:
        """Return top-K similar resumes for a given query embedding."""
        top_k = top_k or CFG.TOP_K
        return self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

    def results_to_df(self, query_result: dict) -> pd.DataFrame:
        """Convert a ChromaDB query result dict into a tidy DataFrame."""
        docs = query_result["documents"][0]
        metas = query_result["metadatas"][0]
        distances = query_result["distances"][0]
        rows = []
        for doc, meta, dist in zip(docs, metas, distances):
            rows.append(
                {
                    "Resume_str": doc,
                    "Category": meta.get("category", ""),
                    "similarity": round(1 - dist, 4),  # cosine distance → similarity
                }
            )
        return pd.DataFrame(rows)
