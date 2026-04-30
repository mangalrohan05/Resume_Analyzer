import re
import warnings
from dataclasses import dataclass, field

import joblib
import nltk
import spacy

from nltk.corpus import stopwords as nltk_stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

for r in ("stopwords", "wordnet", "averaged_perceptron_tagger_eng"):
    nltk.download(r, quiet=True)

STOP_WORDS = set(nltk_stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def _wordnet_pos(tag: str) -> str:
    return {"J": wordnet.ADJ, "V": wordnet.VERB, "N": wordnet.NOUN, "R": wordnet.ADV}.get(
        tag[0], wordnet.NOUN
    )


def preprocess(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    words = text.split()
    pos_tags = nltk.pos_tag(words)
    cleaned = [
        LEMMATIZER.lemmatize(word, _wordnet_pos(tag))
        for word, tag in pos_tags
        if word not in STOP_WORDS and len(word) > 1
    ]
    return " ".join(cleaned)


def _build_nlp():
    nlp = spacy.load("en_core_web_sm")
    skill_patterns = [
        *[{"label": "SKILL", "pattern": p} for p in [
            [{"LOWER": "fmla"}], [{"LOWER": "osha"}],
            [{"LOWER": "strategic"}, {"LOWER": "planning"}],
            [{"LOWER": "workers"}, {"LOWER": "compensation"}],
            [{"LOWER": "leadership"}], [{"LOWER": "employee"}, {"LOWER": "relations"}],
            [{"LOWER": "labor"}, {"LOWER": "union"}],
            [{"LOWER": "project"}, {"LOWER": "management"}],
            [{"LOWER": "agile"}], [{"LOWER": "scrum"}], [{"LOWER": "kanban"}],
        ]],
        *[{"label": "SKILL", "pattern": p} for p in [
            [{"LOWER": "microsoft"}, {"LOWER": "excel"}], [{"LOWER": "excel"}],
            [{"LOWER": "microsoft"}, {"LOWER": "word"}],
            [{"LOWER": "powerpoint"}], [{"LOWER": "outlook"}],
            [{"LOWER": "access"}], [{"LOWER": "sharepoint"}],
        ]],
        *[{"label": "SKILL", "pattern": [{"LOWER": lang}]} for lang in [
            "python", "java", "javascript", "typescript", "c++", "c#",
            "r", "scala", "kotlin", "swift", "go", "rust", "ruby", "php",
        ]],
        *[{"label": "SKILL", "pattern": p} for p in [
            [{"LOWER": "machine"}, {"LOWER": "learning"}],
            [{"LOWER": "deep"}, {"LOWER": "learning"}],
            [{"LOWER": "nlp"}],
            [{"LOWER": "natural"}, {"LOWER": "language"}, {"LOWER": "processing"}],
            [{"LOWER": "computer"}, {"LOWER": "vision"}],
            [{"LOWER": "neural"}, {"LOWER": "network"}],
            [{"LOWER": "generative"}, {"LOWER": "ai"}],
        ]],
        *[{"label": "SKILL", "pattern": [{"LOWER": lib}]} for lib in [
            "tensorflow", "pytorch", "keras", "sklearn", "scikit-learn",
            "pandas", "numpy", "matplotlib", "seaborn", "plotly",
            "xgboost", "lightgbm", "spacy", "nltk", "huggingface", "langchain",
        ]],
        *[{"label": "SKILL", "pattern": [{"LOWER": db}]} for db in [
            "sql", "mysql", "postgresql", "mongodb", "redis",
            "cassandra", "oracle", "sqlite", "elasticsearch",
        ]],
        *[{"label": "SKILL", "pattern": p} for p in [
            [{"LOWER": "aws"}], [{"LOWER": "azure"}], [{"LOWER": "gcp"}],
            [{"LOWER": "docker"}], [{"LOWER": "kubernetes"}], [{"LOWER": "terraform"}],
            [{"LOWER": "ci/cd"}], [{"LOWER": "jenkins"}],
            [{"LOWER": "github"}, {"LOWER": "actions"}],
        ]],
        *[{"label": "SKILL", "pattern": [{"LOWER": fw}]} for fw in [
            "react", "angular", "vue", "django", "flask", "fastapi",
            "node", "express", "spring", "laravel",
        ]],
        *[{"label": "SKILL", "pattern": p} for p in [
            [{"LOWER": "communication"}], [{"LOWER": "teamwork"}],
            [{"LOWER": "problem"}, {"LOWER": "solving"}],
            [{"LOWER": "critical"}, {"LOWER": "thinking"}],
            [{"LOWER": "time"}, {"LOWER": "management"}],
        ]],
    ]
    ruler = nlp.add_pipe("entity_ruler", before="ner")
    ruler.add_patterns(skill_patterns)
    return nlp


NLP = _build_nlp()

_DEGREE_RE = re.compile(
    r"\b(bachelor|master|phd|ph\.d|mba|b\.sc|m\.sc|btech|mtech|b\.e|m\.e|"
    r"associate|diploma|certification|certified)\b",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\s*[-–]\s*(19|20)\d{2}|\b\d+\s+year", re.IGNORECASE)


def extract_info(text: str) -> dict:
    doc = NLP(text)
    info = {"SKILLS": set(), "ORGANIZATIONS": set(), "EDUCATION": set(), "EXPERIENCE_YEARS": []}
    for ent in doc.ents:
        t = ent.text.strip()
        if len(t) < 2:
            continue
        if ent.label_ == "SKILL":
            info["SKILLS"].add(t.title())
        elif ent.label_ in ("ORG", "NORP"):
            info["ORGANIZATIONS"].add(t)
        elif ent.label_ in ("DATE", "QUANTITY") and _YEAR_RE.search(t):
            info["EXPERIENCE_YEARS"].append(t)
    for sent in doc.sents:
        if _DEGREE_RE.search(sent.text):
            info["EDUCATION"].add(sent.text.strip()[:120])
    return info


@dataclass
class MatchReport:
    final_score: float
    cosine_score: float
    skill_score: float
    matched_skills: list = field(default_factory=list)
    missing_skills: list = field(default_factory=list)
    resume_skills: list = field(default_factory=list)
    jd_skills: list = field(default_factory=list)


def match_resume_to_jd(
    resume_text: str,
    jd_text: str,
    pipeline: Pipeline,
    weights: tuple = (0.5, 0.5),
) -> MatchReport:
    content_w, skill_w = weights
    tfidf: TfidfVectorizer = pipeline.named_steps["tfidf"]
    res_vec = tfidf.transform([preprocess(resume_text)])
    jd_vec = tfidf.transform([preprocess(jd_text)])
    cosine = float(cosine_similarity(res_vec, jd_vec)[0][0])

    res_info = extract_info(resume_text)
    jd_info = extract_info(jd_text)
    res_skills = res_info["SKILLS"]
    jd_skills = jd_info["SKILLS"]

    if jd_skills:
        matched = jd_skills & res_skills
        missing = jd_skills - res_skills
        skill_sc = len(matched) / len(jd_skills)
    else:
        matched = missing = set()
        skill_sc = 0.0

    final = round((content_w * cosine + skill_w * skill_sc) * 100, 2)
    return MatchReport(
        final_score=final,
        cosine_score=cosine,
        skill_score=skill_sc,
        matched_skills=sorted(matched),
        missing_skills=sorted(missing),
        resume_skills=sorted(res_skills),
        jd_skills=sorted(jd_skills),
    )


def predict_category(resume_text: str, pipeline: Pipeline, le: LabelEncoder) -> str:
    encoded = pipeline.predict([preprocess(resume_text)])[0]
    return le.inverse_transform([encoded])[0]


def load_models(
    model_path: str = "resume_model.joblib",
    encoder_path: str = "label_encoder.joblib",
):
    return joblib.load(model_path), joblib.load(encoder_path)
