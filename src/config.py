from dataclasses import dataclass

@dataclass
class Config:
    # Model settings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    OLLAMA_MODEL: str = "llama3.1"
    OLLAMA_URL: str = "http://localhost:11434/api/generate"

    # Retrieval settings
    TOP_K: int = 5
    MAX_CONTEXT_CHARS: int = 3000

    # Scoring weights
    SCORE_WEIGHT_SEMANTIC: float = 0.5
    SCORE_WEIGHT_SKILLS: float = 0.5

    # Classifier training
    CLASSIFIER_EPOCHS: int = 50
    CLASSIFIER_BATCH_SIZE: int = 32
    EARLY_STOPPING_PATIENCE: int = 5

    # Paths
    MODEL_SAVE_PATH: str = "models/classifier_model.h5"
    ENCODER_SAVE_PATH: str = "models/label_encoder.pkl"
    EMBEDDINGS_SAVE_PATH: str = "models/embeddings.npy"
    CHROMA_DB_PATH: str = "./chroma_db"
    DATA_PATH: str = "data/Resume.csv"
    CHROMA_COLLECTION: str = "resumes"

    # Evaluation
    MATCH_SCORE_THRESHOLD: float = 60.0

CFG = Config()
