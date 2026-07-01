from dataclasses import dataclass

@dataclass
class Config:
    # Embedding model
    EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"  # 768-dim

    # Ollama RAG model
    OLLAMA_MODEL: str = "llama3.1"

    # Retrieval settings
    TOP_K: int = 5
    MAX_CONTEXT_CHARS: int = 3000
    RAG_RESUME_SKIP: int = 150   # chars to skip at start of each resume (contact info)
    RAG_RESUME_CHARS: int = 700  # chars to include per resume in RAG context

    # Scoring weights
    SCORE_WEIGHT_SEMANTIC: float = 0.65
    SCORE_WEIGHT_SKILLS: float = 0.35
    MIN_SKILL_DENOM: int = 15

    # Paths
    EMBEDDINGS_SAVE_PATH: str = "models/embeddings.npy"
    CHROMA_DB_PATH: str = "./chroma_db"
    DATA_PATH: str = "data/Resume.csv"
    CHROMA_COLLECTION: str = "resumes"

    # Evaluation
    MATCH_SCORE_THRESHOLD: float = 60.0

CFG = Config()
