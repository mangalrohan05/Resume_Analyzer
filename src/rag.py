import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from src.config import CFG
import requests


class ResumeRAG:

    def __init__(self, df, embeddings, embed_model, model_name=None, vector_store=None):
        self.df = df.reset_index(drop=True)
        self.embeddings = embeddings
        self.embed_model = embed_model
        self.vector_store = vector_store  # Optional ChromaDB VectorStore
        self.model_name = model_name or CFG.OLLAMA_MODEL

    def retrieve(self, query: str, preprocess_fn, top_k: int = None):
        top_k = top_k or CFG.TOP_K

        if self.vector_store is not None:
            # Embed raw query — BERT performs better on natural language than preprocessed text
            query_vec = np.array(self.embed_model.encode([query]))[0]
            result = self.vector_store.query(query_vec, top_k=top_k)
            retrieved_df = self.vector_store.results_to_df(result)
            scores = retrieved_df["similarity"].tolist()
            return retrieved_df, scores

        # Fallback: in-memory cosine similarity if ChromaDB not available
        query_vec = self.embed_model.encode([query])
        sims = cosine_similarity(query_vec, self.embeddings)[0]
        top_idx = np.argsort(sims)[::-1][:top_k]
        return self.df.iloc[top_idx][["Resume_str", "Category"]], sims[top_idx]

    def build_context(self, retrieved_df: pd.DataFrame, max_chars: int = None) -> str:
        max_chars = max_chars or CFG.MAX_CONTEXT_CHARS
        skip = CFG.RAG_RESUME_SKIP
        take = CFG.RAG_RESUME_CHARS
        chunks = [
            f"[Resume {i+1} | Category: {row['Category']}]\n"
            f"{row['Resume_str'][skip:skip + take]}"
            for i, row in enumerate(retrieved_df.to_dict("records"))
        ]
        return "\n\n".join(chunks)[:max_chars]

    def answer(self, query: str, preprocess_fn, top_k: int = None) -> str:
        retrieved_df, scores = self.retrieve(query, preprocess_fn, top_k)
        context = self.build_context(retrieved_df)
        prompt = (
            "You are a recruiter assistant. You are given excerpts from candidate resumes "
            "and must answer questions based strictly on those excerpts.\n\n"
            f"--- RESUME EXCERPTS ---\n{context}\n--- END OF EXCERPTS ---\n\n"
            f"Question: {query}\n\n"
            "Instructions:\n"
            "- Base your answer ONLY on the resume excerpts above.\n"
            "- Cite resume numbers (e.g. Resume 1, Resume 3) when making specific claims.\n"
            "- If the excerpts do not contain enough information, say so clearly.\n"
            "- Be concise and structured.\n\n"
            "Answer:"
        )
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }
        
        try:
            response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=120)
            response.raise_for_status()
            return response.json().get("response", "")
        except requests.exceptions.RequestException as e:
            return f"Error communicating with Ollama: {str(e)}"