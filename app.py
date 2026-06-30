import streamlit as st
import pandas as pd
import numpy as np
import os

from src.preprocessing import preprocessing
from src.embeddings import EmbeddingModel
from src.skill_extractor import SkillExtractor
from src.rag import ResumeRAG
from src.classifier import ResumeClassifier
from src.scoring import get_final_match_score
from src.vector_store import VectorStore
from src.utils import safe_answer, validate_text, extract_text_from_pdf
from src.evaluation import evaluate_matcher, build_sample_eval_pairs, precision_recall_at_k
from src.config import CFG

st.set_page_config(page_title="Resume Analyzer", layout="wide", page_icon="📄")

@st.cache_resource
def load_embedding_model():
    return EmbeddingModel()

@st.cache_resource
def load_skill_extractor():
    return SkillExtractor()

@st.cache_resource
def load_vector_store():
    return VectorStore()

@st.cache_data
def load_data():
    if os.path.exists(CFG.DATA_PATH):
        df = pd.read_csv(CFG.DATA_PATH, skip_blank_lines=True)
        return df
    return None

@st.cache_data
def get_cleaned_texts(_df):
    return _df['Resume_str'].apply(preprocessing).tolist()

@st.cache_data
def get_resume_embeddings(_embed_model, texts):
    """Encode and cache all resume embeddings. Leading _ prevents hashing the model."""
    return _embed_model.encode(texts, show_progress_bar=False)


def main():
    st.title("📄 Resume Analyzer")
    st.markdown(
        "Analyze resumes, match them against job descriptions, "
        "query candidates via RAG, and run evaluation metrics."
    )

    # Load all resources
    with st.spinner("Loading models…"):
        embed_model = load_embedding_model()
        skill_extractor = load_skill_extractor()
        vector_store = load_vector_store()
        df = load_data()

    if df is None:
        st.error(f"Dataset not found at `{CFG.DATA_PATH}`.")
        st.stop()

    # Embeddings (cached on disk too)
    if os.path.exists(CFG.EMBEDDINGS_SAVE_PATH):
        embeddings = embed_model.load(CFG.EMBEDDINGS_SAVE_PATH)
    else:
        with st.spinner("Generating embeddings (one-time setup, may take a minute)…"):
            os.makedirs("models", exist_ok=True)
            texts = get_cleaned_texts(df)
            embeddings = get_resume_embeddings(embed_model, texts)
            embed_model.save(embeddings, CFG.EMBEDDINGS_SAVE_PATH)
            st.success("Embeddings generated and saved!")

    # Populate ChromaDB if needed
    if not vector_store.is_populated():
        with st.spinner("Indexing resumes into ChromaDB (one-time setup)…"):
            vector_store.add_resumes(df, embeddings)
            st.success(f"Indexed {df.shape[0]} resumes into ChromaDB.")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Classification", "🎯 Scoring", "💬 RAG Q&A", "📈 Evaluation"]
    )

    # ── Tab 1: Classification ─────────────────────────────────────────────────
    with tab1:
        st.header("Resume Classification")
        st.write("Predict the job category of a given resume text using the trained neural classifier.")

        col1, col2 = st.columns([2, 1])
        with col1:
            uploaded_clf = st.file_uploader(
                "Upload Resume PDF", type=["pdf"], key="clf_pdf"
            )
            resume_text = ""
            if uploaded_clf is not None:
                try:
                    resume_text = extract_text_from_pdf(uploaded_clf)
                    with st.expander("📄 Extracted Text Preview"):
                        st.text(resume_text[:1500] + ("…" if len(resume_text) > 1500 else ""))
                except ValueError as e:
                    st.error(str(e))

        with col2:
            st.markdown("**Model Status**")
            is_trained = (
                os.path.exists(CFG.MODEL_SAVE_PATH)
                and os.path.exists(CFG.ENCODER_SAVE_PATH)
            )
            if is_trained:
                st.success("✅ Trained model found")
            else:
                st.warning("⚠️ No trained model. Run `python main.py` first.")

        if st.button("Predict Category", type="primary", key="btn_clf"):
            if not resume_text or not resume_text.strip():
                st.warning("Please enter some resume text.")
            elif not is_trained:
                st.error("Cannot predict — no trained model found.")
            else:
                with st.spinner("Predicting…"):
                    try:
                        clf = ResumeClassifier(
                            input_dim=embeddings.shape[1],
                            num_classes=df["Category"].nunique()
                        )
                        clf.load()
                        cleaned_res = preprocessing(resume_text)
                        res_vector = embed_model.encode([cleaned_res])
                        pred_category = clf.predict(res_vector)[0]
                        st.success(f"🏷️ **Predicted Category:** {pred_category}")
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")

    # ── Tab 2: Scoring ────────────────────────────────────────────────────────
    with tab2:
        st.header("Resume ↔ Job Description Matching")
        st.write(
            "Get a match score combining semantic similarity and skill overlap. "
            f"Weights: Semantic {CFG.SCORE_WEIGHT_SEMANTIC*100:.0f}% / "
            f"Skills {CFG.SCORE_WEIGHT_SKILLS*100:.0f}%"
        )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Resume (PDF Upload)**")
            uploaded_score = st.file_uploader(
                "Upload Resume PDF", type=["pdf"], key="score_pdf"
            )
            res_input = ""
            if uploaded_score is not None:
                try:
                    res_input = extract_text_from_pdf(uploaded_score)
                    with st.expander("📄 Extracted Resume Text Preview"):
                        st.text(res_input[:1500] + ("…" if len(res_input) > 1500 else ""))
                except ValueError as e:
                    st.error(str(e))

        with col2:
            jd_input = st.text_area("Job Description", height=300, key="score_jd")

        if st.button("Calculate Match Score", type="primary", key="btn_score"):
            errors = []
            try:
                validate_text(res_input, "Resume")
            except ValueError as e:
                errors.append(str(e))
            try:
                validate_text(jd_input, "Job Description")
            except ValueError as e:
                errors.append(str(e))

            if errors:
                for err in errors:
                    st.warning(err)
            else:
                with st.spinner("Scoring…"):
                    score, (res_skills, jd_skills) = get_final_match_score(
                        res_input, jd_input, embed_model, skill_extractor, preprocessing
                    )
                    threshold = CFG.MATCH_SCORE_THRESHOLD
                    color = "green" if score >= threshold else "orange"
                    st.metric("Final Match Score", f"{score}%")
                    if score >= threshold:
                        st.success("✅ Good match — score is above threshold.")
                    else:
                        st.warning("⚠️ Below match threshold — candidate may not be ideal.")

                    st.divider()
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**Resume Skills Extracted**")
                        st.write(", ".join(res_skills) if res_skills else "_None detected_")
                    with c2:
                        st.markdown("**JD Skills Extracted**")
                        st.write(", ".join(jd_skills) if jd_skills else "_None detected_")

    # ── Tab 3: RAG Q&A ────────────────────────────────────────────────────────
    with tab3:
        st.header("Q&A with Resume Dataset (RAG)")
        st.write(
            "Ask natural language questions about the candidates. "
            "Uses ChromaDB for retrieval and Ollama (llama3.1) for generation."
        )

        query = st.text_input("Enter your question:", key="rag_query")
        top_k = st.slider("Number of resumes to retrieve", min_value=1, max_value=10,
                           value=CFG.TOP_K, key="rag_top_k")

        if st.button("Ask Ollama", type="primary", key="btn_rag"):
            if not query or not query.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner("Retrieving and generating answer…"):
                    rag = ResumeRAG(
                        df, embeddings, embed_model,
                        vector_store=vector_store
                    )
                    # Phase 2: safe_answer with retry logic
                    answer = safe_answer(rag, query, preprocessing, top_k=top_k)
                    st.markdown("### 💬 Answer")
                    st.write(answer)

                    # Show retrieved context
                    with st.expander("🔍 Retrieved Resume Excerpts"):
                        retrieved_df, scores = rag.retrieve(query, preprocessing, top_k=top_k)
                        for i, (_, row) in enumerate(retrieved_df.iterrows()):
                            sim = scores[i] if not isinstance(scores[i], list) else scores[i]
                            st.markdown(
                                f"**Resume {i+1}** | Category: `{row['Category']}` "
                                f"| Similarity: `{sim:.4f}`"
                            )
                            st.caption(str(row["Resume_str"])[:400] + "…")
                            st.divider()

    # ── Tab 4: Evaluation ─────────────────────────────────────────────────────
    with tab4:
        st.header("📈 Evaluation Metrics")
        st.write(
            "Automatically evaluate the matching quality using labeled pairs "
            "generated from the dataset."
        )

        n_pos = st.slider("Positive pairs", 2, 10, 5, key="eval_pos")
        n_neg = st.slider("Negative pairs", 2, 10, 5, key="eval_neg")

        if st.button("Run Evaluation", type="primary", key="btn_eval"):
            with st.spinner("Building eval set and scoring…"):
                eval_pairs = build_sample_eval_pairs(df, n_positive=n_pos, n_negative=n_neg)

                def scoring_fn(resume, jd):
                    return get_final_match_score(resume, jd, embed_model, skill_extractor, preprocessing)

                accuracy, df_eval = evaluate_matcher(eval_pairs, scoring_fn)

            st.metric("Matching Accuracy", f"{accuracy * 100:.1f}%")
            st.caption(
                f"Threshold: a score ≥ {CFG.MATCH_SCORE_THRESHOLD}% is considered a match."
            )
            st.divider()

            # Colour correct/incorrect
            def highlight_row(row):
                color = "background-color: #d4edda" if row["correct"] else "background-color: #f8d7da"
                return [color] * len(row)

            styled = df_eval.style.apply(highlight_row, axis=1)
            st.dataframe(styled, use_container_width=True)

            correct = df_eval["correct"].sum()
            st.caption(
                f"✅ {correct}/{len(df_eval)} pairs correctly classified "
                f"({'positive' if correct > len(df_eval)//2 else 'negative'} leaning)"
            )


if __name__ == "__main__":
    main()
