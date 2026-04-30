import io
import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pypdf import PdfReader

from resume_analyzer import extract_info, load_models, match_resume_to_jd, predict_category


app = FastAPI(title="ResumeAI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://resume-analyzer-chi-six.vercel.app",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    allow_credentials=False
)


MODEL_PATH   = os.getenv("MODEL_PATH",   "resume_model.joblib")
ENCODER_PATH = os.getenv("ENCODER_PATH", "label_encoder.joblib")

pipeline, le = None, None


@app.on_event("startup")
def load():
    global pipeline, le
    if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
        raise RuntimeError(
            "Model files not found. Run the Colab notebook first and place "
            "resume_model.joblib + label_encoder.joblib in the project folder."
        )
    pipeline, le = load_models(MODEL_PATH, ENCODER_PATH)


class AnalyzeRequest(BaseModel):
    resume_text: str
    jd_text: str


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")
    contents = await file.read()
    reader   = PdfReader(io.BytesIO(contents))
    text     = "\n".join(p.extract_text() or "" for p in reader.pages).strip()
    if not text:
        raise HTTPException(status_code=422, detail="Could not extract text from this PDF.")
    return {"text": text}


@app.post("/analyze")
def analyze(body: AnalyzeRequest):
    if not body.resume_text.strip():
        raise HTTPException(status_code=400, detail="Resume text is empty.")
    if not body.jd_text.strip():
        raise HTTPException(status_code=400, detail="Job description is empty.")

    report   = match_resume_to_jd(body.resume_text, body.jd_text, pipeline)
    info     = extract_info(body.resume_text)
    category = predict_category(body.resume_text, pipeline, le)

    return {
        "match_score":        report.final_score,
        "content_similarity": round(report.cosine_score * 100),
        "skill_coverage":     round(report.skill_score  * 100),
        "predicted_category": category,
        "matched_skills":     report.matched_skills,
        "missing_skills":     report.missing_skills,
        "resume_skills":      report.resume_skills,
        "jd_skills":          report.jd_skills,
        "education":          list(info["EDUCATION"]),
        "experience_years":   info["EXPERIENCE_YEARS"],
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": pipeline is not None}

app.mount("/", StaticFiles(directory="static", html=True), name="static")
