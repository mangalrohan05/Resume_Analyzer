# Resume Analyzer 📄

> AI-powered resume screening — BERT semantic matching, ChromaDB vector store, Gemini RAG Q&A, spaCy skill extraction.

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-app-red?logo=streamlit)](https://streamlit.io)
[![Live Demo](https://img.shields.io/badge/Live-Demo-brightgreen)](https://resume-analyzer-chi-six.vercel.app)

🚀 **[Live Demo](https://resume-analyzer-chi-six.vercel.app)**

## Features

- **Semantic Matching** — BAAI/bge-base-en-v1.5 embeddings; cosine similarity between resume and job description
- **Skill Extraction** — spaCy EntityRuler with hundreds of patterns (ML, cloud, DevOps, data, soft skills)
- **Hybrid Score** — 65% semantic similarity + 35% skill overlap → single 0-100 match score
- **ChromaDB Vector Store** — persistent resume index; fast retrieval on repeat queries
- **Gemini RAG Q&A** — retrieves top-K resumes from ChromaDB, builds context, answers via gemini-2.0-flash
- **Evaluation Suite** — Precision, Recall, F1, Accuracy over 1300+ auto-generated JD pairs

## Architecture

```
One-time Setup (main.py)
  Resume.csv -> BERT Embeddings -> ChromaDB Index

Web App (streamlit run app.py)
  Scoring Tab   | RAG Q&A Tab  | Evaluation Tab
  PDF + JD      | Query text   | Auto pairs
  BERT embed    | ChromaDB K   | P / R / F1
  Hybrid %      | Gemini gen.  |
```

## Tech Stack

| Component | Technology |
|---|---|
| Embeddings | sentence-transformers (BAAI/bge-base-en-v1.5) |
| Skill Extraction | spaCy + custom EntityRuler |
| Vector Store | ChromaDB (persistent, cosine) |
| RAG Generation | Google Gemini (gemini-2.0-flash) |
| PDF Parsing | pdfplumber |
| Frontend | Streamlit |
| Preprocessing | NLTK (stopwords, lemmatization, POS) |

## Setup

```bash
git clone https://github.com/mangalrohan05/Resume_Analyzer
cd Resume_Analyzer
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Create .env:
```
GEMINI_API_KEY=your_gemini_api_key_here
```

```bash
python main.py        # one-time index build
streamlit run app.py  # launch app
```

## Project Structure

```
Resume_Analyzer/
├── data/Resume.csv          # Kaggle resume dataset
├── src/
│   ├── config.py
│   ├── embeddings.py        # SentenceTransformer wrapper
│   ├── skill_extractor.py   # spaCy EntityRuler
│   ├── scoring.py           # Hybrid similarity scorer
│   ├── vector_store.py      # ChromaDB client wrapper
│   ├── rag.py               # Gemini RAG generation
│   └── evaluation.py        # Evaluation metrics
├── main.py                  # CLI setup script
├── app.py                   # Streamlit app
└── requirements.txt
```