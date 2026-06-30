# Resume Analyzer

An AI-powered resume screening tool that matches resumes against job descriptions using BERT-based semantic embeddings, rule-based skill extraction, a Keras neural network classifier, ChromaDB-backed retrieval, and a Retrieval-Augmented Generation (RAG) layer powered by a local Ollama LLM for natural-language Q&A over a resume database.

## Features

- **PDF Resume Upload** — extracts text directly from uploaded PDF resumes using `pdfplumber` (text-based PDFs only; no OCR)
- **Semantic Matching** — uses `sentence-transformers` (`all-MiniLM-L6-v2`) to compute resume–JD cosine similarity
- **Skill Extraction** — spaCy `EntityRuler` pipeline matches 300+ hard-coded skill patterns across languages, frameworks, cloud, DevOps, data, testing, and soft skills
- **Hybrid Match Scoring** — combines semantic similarity (50%) and skill overlap (50%) into a single score out of 100
- **Neural Network Classifier** — three-layer Keras feedforward network (Dense 256→128→N, Adam + early stopping) classifies resumes into job categories using BERT embeddings as input
- **ChromaDB Vector Store** — resumes are indexed into a persistent ChromaDB collection (cosine space) on first run; subsequent queries hit the DB, not memory
- **RAG-based Q&A** — retrieves top-K resumes from ChromaDB by embedding similarity, builds a context window, and sends a prompt to a locally running **Ollama** LLM (default: `llama3.1`) for natural-language answers
- **Evaluation Suite** — automatically builds labeled positive/negative resume–JD pairs from the dataset and reports match accuracy, plus Precision@K and Recall@K for retrieval
- **Caching & Error Handling** — Streamlit `@st.cache_resource` / `@st.cache_data` caching, disk-cached embeddings (`.npy`), retry-with-backoff for Ollama errors, and input validation

## Project Flow

```
                     One-time Setup  (main.py)
                     ─────────────────────────
  Resume.csv
       │
       ▼
  Preprocessing (NLTK)
       │
       ▼
  BERT Embeddings  ──────────────────────────► saved: models/embeddings.npy
  (all-MiniLM-L6-v2)
       │
       ▼
  Keras Classifier Training ────────────────► saved: models/classifier_model.h5
  (Dense 256→128→N, Adam, EarlyStopping)             models/label_encoder.pkl
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
  │ pdfplumber       │ pdfplumber       │ Preprocess + BERT│ +/- eval pairs  │
  │ Preprocess       │ BERT embed both  │ ChromaDB top-K   │ Score each pair │
  │ BERT embed       │ Cosine sim       │ Build context    │ Report accuracy │
  │ Keras predict    │ + Skill overlap  │ Ollama LLM gen.  │ P@K / R@K       │
  │ → Category label │ → Match score %  │ → Answer         │                 │
  └──────────────────┴──────────────────┴──────────────────┴─────────────────┘
```

## Project Structure

```
Resume_Analyzer/
│
├── data/
│   └── Resume.csv                  # Kaggle resume dataset (Resume_str, Category)
│
├── src/
│   ├── config.py                   # Centralized config dataclass (CFG)
│   ├── preprocessing.py            # NLTK: lowercase, stopword removal, POS lemmatization
│   ├── embeddings.py               # SentenceTransformer wrapper (encode/save/load)
│   ├── skill_extractor.py          # spaCy EntityRuler with 300+ skill patterns
│   ├── classifier.py               # Keras Sequential (Dense 256→128→N) + LabelEncoder
│   ├── scoring.py                  # Hybrid score: cosine similarity + skill overlap
│   ├── vector_store.py             # ChromaDB PersistentClient wrapper
│   ├── rag.py                      # Retrieval (ChromaDB/cosine fallback) + Ollama call
│   ├── evaluation.py               # evaluate_matcher, build_sample_eval_pairs, P@K/R@K
│   └── utils.py                    # extract_text_from_pdf, validate_text, safe_answer
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
└── Readme.md
```

## Tech Stack

| Component | Technology |
|---|---|
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Classifier | TensorFlow / `tf-keras` feedforward neural network |
| Skill Extraction | `spaCy` (`en_core_web_sm` + custom `EntityRuler` patterns) |
| Vector Store | `chromadb` (persistent, cosine space) |
| RAG Generation | **Ollama** (local LLM, default: `llama3.1`) |
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

### 2. Install and start Ollama (required for RAG Q&A tab)

Ollama runs the LLM locally — no API key or internet connection needed after the initial model download.

```bash
# Install from https://ollama.com
ollama pull llama3.1        # ~4 GB one-time download
ollama serve                 # start local server at localhost:11434
```

> If Ollama is not running, the RAG tab shows a connection-refused warning and returns a graceful error. All other tabs continue to work normally.

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
EMBEDDING_MODEL         = "all-MiniLM-L6-v2"
OLLAMA_MODEL            = "llama3.1"
OLLAMA_URL              = "http://localhost:11434/api/generate"
TOP_K                   = 5
MAX_CONTEXT_CHARS       = 3000
SCORE_WEIGHT_SEMANTIC   = 0.5
SCORE_WEIGHT_SKILLS     = 0.5
MATCH_SCORE_THRESHOLD   = 70.0   # % threshold for positive match classification
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
- **Ollama must be running locally** — the RAG tab requires `ollama serve` and a pulled model; there is no cloud-hosted LLM fallback.
- **Skill taxonomy is rule-based** — the spaCy `EntityRuler` covers ~300 fixed patterns; skills not in the list won't be detected without extending `skill_extractor.py`.
- **Classifier trained on a single dataset** — generalization to resume formats/industries outside the `Resume.csv` training distribution is not validated.
- **Evaluation pairs are synthetic** — `build_sample_eval_pairs` generates JD text from category names (e.g. *"We are hiring a Data Science professional"*) rather than real job postings.

## Roadmap

- [ ] OCR fallback for scanned PDF resumes (e.g. `pytesseract`)
- [ ] Embedding-based skill matching to catch synonyms beyond exact EntityRuler patterns
- [ ] Real labeled resume–JD evaluation pairs for rigorous benchmark accuracy
- [ ] Unit test coverage for preprocessing, scoring, and extraction modules
- [ ] Support for additional Ollama models (e.g., Mistral, Llama 3.2)
