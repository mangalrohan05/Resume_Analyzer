import os
import pandas as pd
from src.preprocessing import preprocessing
from src.embeddings import EmbeddingModel
from src.skill_extractor import SkillExtractor
from src.rag import ResumeRAG
from src.classifier import ResumeClassifier
from src.scoring import get_final_match_score
from src.vector_store import VectorStore
from src.config import CFG

# 1. Load & preprocess data
df = pd.read_csv(CFG.DATA_PATH, skip_blank_lines=True)
df['cleaned_text'] = df['Resume_str'].apply(preprocessing)

# 2. Embeddings (load cached if available)
embed_model = EmbeddingModel()
os.makedirs('models', exist_ok=True)
if os.path.exists(CFG.EMBEDDINGS_SAVE_PATH):
    print("Loading cached embeddings...")
    embeddings = embed_model.load(CFG.EMBEDDINGS_SAVE_PATH)
else:
    print("Generating embeddings...")
    embeddings = embed_model.encode(df['cleaned_text'].tolist(), show_progress_bar=True)
    embed_model.save(embeddings, CFG.EMBEDDINGS_SAVE_PATH)

# 3. Classifier (train only if model doesn't exist)
if not os.path.exists(CFG.MODEL_SAVE_PATH):
    print("Training classifier...")
    clf = ResumeClassifier(input_dim=embeddings.shape[1], num_classes=df['Category'].nunique())
    history, report = clf.train(embeddings, df['Category'])
    print(report)
    clf.save()
else:
    print("Classifier model found, skipping training.")

# 4. ChromaDB Vector Store
print("Setting up ChromaDB vector store...")
vector_store = VectorStore()
if not vector_store.is_populated():
    print("Populating ChromaDB with resume embeddings...")
    vector_store.add_resumes(df, embeddings)
    print(f"Indexed {df.shape[0]} resumes into ChromaDB.")
else:
    print("ChromaDB already populated.")

# 5. Skill Extractor
skill_extractor = SkillExtractor()

# 6. Scoring demo
score, (res_skills, jd_skills) = get_final_match_score(
    df['Resume_str'].iloc[0],
    "Looking for a Python developer with machine learning and NLP experience",
    embed_model, skill_extractor, preprocessing
)
print(f"\nMatch Score: {score}%")
print(f"Resume Skills: {res_skills}")
print(f"JD Skills: {jd_skills}")

# 7. RAG demo
rag = ResumeRAG(df, embeddings, embed_model, vector_store=vector_store)
answer = rag.answer(
    "Which candidates have strong Python and ML experience but lack leadership roles?",
    preprocessing
)
print(f"\nRAG Answer:\n{answer}")