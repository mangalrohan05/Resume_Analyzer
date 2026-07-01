from dataclasses import dataclass

@dataclass
class Config:
    # Embedding model
    EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"  # 768-dim

    # Gemini RAG model (google.genai SDK)
    GEMINI_MODEL: str = "gemini-2.0-flash"

    # Retrieval settings
    TOP_K: int = 5
    MAX_CONTEXT_CHARS: int = 3000
    RAG_RESUME_SKIP: int = 150   # chars to skip at start of each resume (contact info)
    RAG_RESUME_CHARS: int = 700  # chars to include per resume in RAG context

    # Scoring weights
    SCORE_WEIGHT_SEMANTIC: float = 0.65
    SCORE_WEIGHT_SKILLS: float = 0.35
    MIN_SKILL_DENOM: int = 3     # prevents inflated skill score when JD lists < 3 skills

    # Classifier training
    CLASSIFIER_EPOCHS: int = 30
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
