# Resume Analyzer

An AI-powered resume screening tool that matches resumes against job descriptions using semantic embeddings, rule-based skill extraction, a Keras neural network classifier, ChromaDB-backed retrieval, and a Retrieval-Augmented Generation (RAG) layer powered by Google's Gemini LLM for natural-language Q&A over a resume database.

## Features

- **PDF Resume Upload** — extracts text directly from uploaded PDF resumes using `pdfplumber` (text-based PDFs only; no OCR)
- **Semantic Matching** — uses `sentence-transformers` (`BAAI/bge-base-en-v1.5`) to compute resume–JD cosine similarity
- **Skill Extraction** — spaCy `EntityRuler` pipeline matches hundreds of hard-coded skill patterns across ML, AI, cloud, DevOps, data, testing, and soft skills
- **Hybrid Match Scoring** — combines semantic similarity (65%) and skill overlap (35%) into a single score out of 100
- **Neural Network Classifier** — three-layer Keras feedforward network classifies resumes into job categories using embeddings as input, showing top 3 category matches and confidence
- **ChromaDB Vector Store** — resumes are indexed into a persistent ChromaDB collection on first run; subsequent queries hit the DB
- **RAG-based Q&A** — retrieves top-K resumes from ChromaDB by embedding similarity, builds a context window, and sends a prompt to a local **Ollama** model for natural-language answers
- **Comprehensive Evaluation Suite** — automatically builds labeled evaluation pairs from a dataset of 1300+ lines of diverse job descriptions and reports Accuracy, Precision, Recall, and F1 Score for matching

## Project Flow

```text
                     One-time Setup  (main.py)
                     ─────────────────────────
  Resume.csv
       │
       ▼
  BERT Embeddings  ──────────────────────────► saved: models/embeddings.npy
  (bge-base-en-v1.5)
       │
       ▼
  Keras Classifier Training ────────────────► saved: models/classifier_model.h5
  (Dense 256→128→N)                                  models/label_encoder.pkl
       │
       ▼
  ChromaDB Indexing ────────────────────────► persisted: ./chroma_db/
  (cosine space, upsert all resumes)

                     Web App  (streamlit run app.py)
                     ──────────────────────────────
  Loads all cached artifacts (embeddings.npy, classifier_model.h5, ChromaDB)

  ┌──────────────────┬──────────────────┬──────────────────┬─────────────────┐
  │ Classification   │ Scoring          │ RAG Q&A          │ Evaluation      │
  ├──────────────────┼──────────────────┼──────────────────┼─────────────────┤
  │ Upload PDF       │ Upload PDF + JD  │ User query text  │ Auto-build      │
  │ pdfplumber       │ pdfplumber       │ Preprocess       │ eval pairs      │
  │ BERT embed       │ BERT embed both  │ ChromaDB top-K   │ Score each pair │
  │ Keras predict    │ Cosine sim       │ Build context    │ Report metrics  │
  │ → Top 3 Labels   │ + Skill overlap  │ Ollama local gen.│ (Acc, P, R, F1) │
  │ & Confidence     │ → Match score %  │ → Answer         │                 │
  └──────────────────┴──────────────────┴──────────────────┴─────────────────┘
```

## Project Structure

```text
Resume_Analyzer/
│
├── data/
│   └── Resume.csv                  # Kaggle resume dataset (Resume_str, Category)
│
├── src/
│   ├── config.py                   # Centralized config dataclass (CFG)
│   ├── preprocessing.py            # NLTK: lowercase, stopword removal, POS lemmatization
│   ├── embeddings.py               # SentenceTransformer wrapper (encode/save/load)
│   ├── skill_extractor.py          # spaCy EntityRuler with extensive skill patterns
│   ├── classifier.py               # Keras Sequential + LabelEncoder
│   ├── scoring.py                  # Hybrid score: cosine similarity + skill overlap
│   ├── vector_store.py             # ChromaDB PersistentClient wrapper
│   ├── rag.py                      # Retrieval + Gemini LLM generation
│   ├── evaluation.py               # evaluate_matcher, build_sample_eval_pairs
│   ├── jd_templates.py             # Extensive template repository for evaluation pairs
│   └── utils.py                    # PDF parsing, validation, safe_answer (rate limit handling)
│
├── models/                         # Auto-created on first run of main.py
│   ├── classifier_model.h5
│   ├── label_encoder.pkl
│   └── embeddings.npy
│
├── chroma_db/                      # Auto-created on first run of main.py (ChromaDB)
│
├── main.py                         # CLI: embed → train classifier → index ChromaDB → demo
├── app.py                          # Streamlit web app (4 tabs)
├── requirements.txt
└── README.md
```

## Tech Stack

| Component | Technology |
|---|---|
| Embeddings | `sentence-transformers` (`BAAI/bge-base-en-v1.5`) |
| Classifier | TensorFlow / `tf-keras` feedforward neural network |
| Skill Extraction | `spaCy` (`en_core_web_sm` + custom `EntityRuler` patterns) |
| Vector Store | `chromadb` (persistent, cosine space) |
| RAG Generation | **Google Gemini** (`google-genai` SDK, `gemini-2.0-flash`) |
| PDF Parsing | `pdfplumber` |
| Frontend | `streamlit` |
| Text Preprocessing | `nltk` (stopwords, WordNet lemmatization, POS tagging) |
| Similarity | `scikit-learn` (`cosine_similarity`) |

## Setup

### 1. Clone and install Python dependencies

```bash
git clone <repo-url>
cd Resume_Analyzer
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

> NLTK data (stopwords, WordNet, averaged_perceptron_tagger) downloads automatically on first run.

### 2. Configure Environment Variables

The RAG capabilities require a Gemini API key. Create a `.env` file in the root directory:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 3. Run the one-time setup script

Generates embeddings, trains the classifier, and populates ChromaDB. Cached files are reused on all subsequent runs (skipped automatically if they already exist).

```bash
python main.py
```

### 4. Launch the Streamlit app

```bash
streamlit run app.py
```

## Configuration

All parameters live in `src/config.py` as a single `Config` dataclass:

```python
EMBEDDING_MODEL         = "BAAI/bge-base-en-v1.5"
GEMINI_MODEL            = "gemini-2.0-flash"
TOP_K                   = 5
MAX_CONTEXT_CHARS       = 3000
SCORE_WEIGHT_SEMANTIC   = 0.65
SCORE_WEIGHT_SKILLS     = 0.35
MATCH_SCORE_THRESHOLD   = 70.0
CLASSIFIER_EPOCHS       = 50
CLASSIFIER_BATCH_SIZE   = 32
EARLY_STOPPING_PATIENCE = 5
DATA_PATH               = "data/Resume.csv"
CHROMA_DB_PATH          = "./chroma_db"
```

## When to Re-run `main.py`

| What changed | Required action |
|---|---|
| Skill patterns / `utils.py` / `scoring.py` | Just restart Streamlit — no re-run needed |
| `preprocessing.py` or `embeddings.py` | Delete `models/embeddings.npy` → re-run `python main.py` |
| `classifier.py` or training hyperparameters | Delete `models/classifier_model.h5` + `label_encoder.pkl` → re-run `python main.py` |
| Resume dataset (`Resume.csv`) | Delete everything in `models/` and `chroma_db/` → re-run `python main.py` |

## Known Limitations

- **PDF support is text-based only** — scanned/image-only PDFs (no embedded text layer) will fail extraction; OCR support is not implemented.
- **Skill taxonomy is rule-based** — the spaCy `EntityRuler` covers hundreds of fixed patterns; skills not in the list won't be detected without extending `skill_extractor.py`.
- **Classifier trained on a single dataset** — generalization to resume formats/industries outside the `Resume.csv` training distribution is not validated.

## Roadmap

- [ ] OCR fallback for scanned PDF resumes (e.g. `pytesseract`)
- [ ] Embedding-based skill matching to catch synonyms beyond exact EntityRuler patterns
- [ ] Unit test coverage for preprocessing, scoring, and extraction modules
