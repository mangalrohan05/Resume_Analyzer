import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests

class ResumeRAG:
    def __init__(self, df, embeddings, embed_model, ollama_model="llama3.1"):
        self.df = df.reset_index(drop=True)
        self.embeddings = embeddings
        self.embed_model = embed_model
        self.ollama_model = ollama_model
        self.ollama_url = "http://localhost:11434/api/generate"
        
    def retrieve(self, query, preprocess_fn, top_k=5):
        query_vec = self.embed_model.encode(preprocess_fn([query]))
        sims = cosine_similarity(query_vec, self.embeddings)[0]
        top_idx = np.argsort(sims)[::-1][:top_k]
        return self.df.iloc[top_idx][['Resume_str', 'Category']], sims[top_idx]

        
    def build_context(self, retrieved_df, max_chars=3000):
        chunks=[
            f"[Resume {i} | Category : {row['Category']}]\n{row['Resume_str'][:6000]}"
            for i, row in retrieved_df.iterrows()
        ]
        return "\n\n".join(chunks)[:max_chars]
    
    def _call_ollama(self, prompt):
        response = requests.post(self.ollama_url, json={
            "model" : self.ollama_model,
            "prompt" : prompt,
            "stream" : False
        })
        return response.json()["response"]
    
    def answer(self, query, preprocess_fn, top_k=5):
        retrieved_df, scores = self.retrieve(query, preprocess_fn, top_k)
        context = self.build_context(retrieved_df)
        prompt = f"""You are a recruiting assistant. Use the resume excerpts below to answer the question.
        
Context:
{context}

Question: {query}

Answer using only the context above. Cite resume numbers when relevant."""
        return self._call_ollama(prompt)
        
        

        