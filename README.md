# ResumeAI — Resume Match Analyzer

A machine learning powered resume analyzer that compares a resume against a job description and gives a match score with detailed skill breakdown. No API keys, no paid services — runs entirely on a locally trained model.

---

## What it does

- Upload a resume as a PDF
- Paste a job description
- Get a match score out of 100 with two sub-scores — content similarity and skill coverage
- See exactly which skills match, which are missing, and all skills detected in the resume
- Predicts the resume's job category (Data Science, HR, Engineering, etc.)
- Extracts education and experience periods from the resume

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML Model | Random Forest Classifier |
| Text Processing | TF-IDF, NLTK, spaCy |
| API | FastAPI |
| Frontend | HTML, CSS, Vanilla JS |
| PDF Parsing | pypdf |
| Model Storage | joblib |
| Training | Google Colab |
| Backend Deploy | Render |
| Frontend Deploy | Vercel |

---

## Project Structure

```
├── app.py                  FastAPI backend
├── resume_analyzer.py      ML logic — preprocessing, NER, scoring
├── static/
│   └── index.html          Frontend UI
├── train_models.ipynb      Google Colab training notebook
├── requirements.txt
├── resume_model.joblib     trained model (generate via Colab)
└── label_encoder.joblib    label encoder (generate via Colab)
```

---

## Setup & Running Locally

### 1. Train the model on Google Colab

- Open `train_models.ipynb` in [Google Colab](https://colab.research.google.com)
- Upload `Resume.csv` using the 📁 sidebar ([get the dataset here](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset))
- Run all cells
- Two files will auto-download: `resume_model.joblib` and `label_encoder.joblib`
- Place both in the project root next to `app.py`

### 2. Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Run

```bash
uvicorn app:app --reload
```

Open [http://localhost:8000](http://localhost:8000)

---

## Deployment

### Backend — Render

1. Push the repo to GitHub (include the `.joblib` files)
2. Create a new Web Service on [render.com](https://render.com)
3. Set the following:
   - **Build command:** `pip install -r requirements.txt && python -m spacy download en_core_web_sm`
   - **Start command:** `uvicorn app:app --host 0.0.0.0 --port $PORT`
4. Deploy

### Frontend — Vercel

1. In `static/index.html`, update the API base URL:
   ```js
   const API = 'https://your-app-name.onrender.com';
   ```
2. In `app.py`, update the CORS allowed origins:
   ```python
   allow_origins=["https://your-app-name.vercel.app"]
   ```
3. Deploy the `static/` folder to [vercel.com](https://vercel.com)

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/upload` | Upload a PDF → returns extracted text |
| POST | `/analyze` | resume text + JD text → full match report |
| GET | `/health` | Check if the model is loaded and server is running |

---

## How the Scoring Works

The final match score is a weighted combination of two signals:

**Content Similarity (50%)** — TF-IDF vectors of the resume and JD are compared using cosine similarity. This catches general language and theme overlap even when exact skill keywords aren't matched.

**Skill Coverage (50%)** — spaCy NER with a custom EntityRuler detects skills in both documents and calculates what percentage of the JD's required skills appear in the resume.

```
Final Score = 0.5 × cosine_similarity + 0.5 × (matched_skills / jd_skills)
```

---

## AI Tools Used

**[Claude.ai](https://claude.ai)** — Used for building the complete frontend UI (`static/index.html`), the FastAPI integration layer (`app.py`), and wiring everything together end to end.

**[Google Gemini](https://gemini.google.com)** — Used as a guide throughout the project to understand concepts, debug ideas, and plan the ML pipeline before implementation.

---

## Dataset

[Resume Dataset — Kaggle](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset) by Sneha Anbhawal. Contains 2400+ resumes across 24 job categories.
