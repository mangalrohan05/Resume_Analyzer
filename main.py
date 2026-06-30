import pandas as pd
from src.preprocessing import preprocessing
from src.embeddings import EmbeddingModel
from src.skill_extractor import SkillExtractor
from src.rag import ResumeRAG
from src.classifier import ResumeClassifier
from src.scoring import get_final_match_score

# 1. clean
df = pd.read_csv('data/Resume.csv', skip_blank_lines=True)
df['cleaned_text'] = df['Resume_str'].apply(preprocessing)

# 2. BERT
embed_model = EmbeddingModel()
embeddings = embed_model.encode(df['cleaned_text'].tolist(), show_progress_bar=True)

# 3. Classifier
clf = ResumeClassifier(input_dim=embeddings.shape[1], num_classes=df['Category'].nunique())
history, report = clf.train(embeddings, df['Category'])
print(report)

clf.save('models/classifier_model.h5', 'models/label_encoder.pkl')

# 4. Skill
skill_extractor = SkillExtractor()

# 5. Scoring
score, (res_skills, jd_skills) = get_final_match_score(
    df['Resume_str'].iloc[0],
    "Looking for a Python developer with machine learning and NLP experience",
    embed_model, skill_extractor, preprocessing
)

# 6. RAG
rag = ResumeRAG(df, embeddings, embed_model, ollama_model="llama3.1")
answer = rag.answer(
    "Which candidates have strong Python and ML experience but lack leadership roles?",
    preprocessing
)
print(answer)