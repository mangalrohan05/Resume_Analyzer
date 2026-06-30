from sentence_transformers import SentenceTransformer as ST
import numpy as np

class EmbeddingModel:
    def __init__(self, model_name = 'all-MiniLM-L6-v2'):
        self.model = ST(model_name)
    
    def encode(self, texts, show_progress_bar=False):
        return self.model.encode(texts, show_progress_bar=show_progress_bar)
    
    def save(self, embeddings, path):
        np.save(path, embeddings)
        
    def load(self, path):
        return np.load(path)