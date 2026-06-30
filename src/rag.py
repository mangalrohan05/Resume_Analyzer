import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
import pandas as pd
from src.config import CFG


class ResumeRAG:

    def __init__(self, df, embeddings, embed_model, ollama_model: str = None, vector_store=None):
        self.df = df.reset_index(drop=True)
        self.embeddings = embeddings
        self.embed_model = embed_model
        self.ollama_model = ollama_model or CFG.OLLAMA_MODEL
        self.ollama_url = CFG.OLLAMA_URL
        self.vector_store = vector_store  # Optional ChromaDB VectorStore

    def retrieve(self, query: str, preprocess_fn, top_k: int = None):

        top_k = top_k or CFG.TOP_K

        if self.vector_store is not None:
            
            cleaned_query = preprocess_fn(query)
            query_vec = np.array(self.embed_model.encode([cleaned_query]))[0]
            result = self.vector_store.query(query_vec, top_k=top_k)
            retrieved_df = self.vector_store.results_to_df(result)
            scores = retrieved_df["similarity"].tolist()
            return retrieved_df, scores

        # if chromadb not available then fall back to cosine_similarity
        
        cleaned_query = preprocess_fn(query)
        query_vec = self.embed_model.encode([cleaned_query])
        sims = cosine_similarity(query_vec, self.embeddings)[0]
        top_idx = np.argsort(sims)[::-1][:top_k]
        return self.df.iloc[top_idx][["Resume_str", "Category"]], sims[top_idx]

    def build_context(self, retrieved_df: pd.DataFrame, max_chars: int = None) -> str:
        max_chars = max_chars or CFG.MAX_CONTEXT_CHARS
        chunks = [
            f"[Resume {i+1} | Category: {row['Category']}]\n{row['Resume_str'][:600]}"
            for i, row in enumerate(retrieved_df.to_dict("records"))
        ]
        return "\n\n".join(chunks)[:max_chars]

    def _call_ollama(self, prompt: str) -> str:
        response = requests.post(
            self.ollama_url,
            json={"model": self.ollama_model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["response"]

    def answer(self, query: str, preprocess_fn, top_k: int = None) -> str:
        retrieved_df, scores = self.retrieve(query, preprocess_fn, top_k)
        context = self.build_context(retrieved_df)
        prompt = (
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer using only the context above. Cite resume numbers when relevant."
        )
        return self._call_ollama(prompt)