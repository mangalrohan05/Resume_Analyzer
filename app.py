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

st.set_page_config(page_title="Resume Analyzer", layout="wide", page_icon="📄")

# Cache models to load them only once
@st.cache_resource
def load_models():
    embed_model = EmbeddingModel()
    skill_extractor = SkillExtractor()
    return embed_model, skill_extractor

@st.cache_data
def load_data():
    if os.path.exists('data/Resume.csv'):
        df = pd.read_csv('data/Resume.csv', skip_blank_lines=True)
        if 'cleaned_text' not in df.columns:
            df['cleaned_text'] = df['Resume_str'].apply(preprocessing)
        return df
    return None

def main():
    st.title("📄 Resume Analyzer App")
    st.markdown("Analyze resumes, match them against job descriptions, and query candidates using RAG.")
    
    with st.spinner("Loading models and data..."):
        embed_model, skill_extractor = load_models()
        df = load_data()
        
    if df is None:
        st.error("Dataset not found at `data/Resume.csv`.")
        st.stop()
        
    # Generate or Load Embeddings
    embed_path = 'models/embeddings.npy'
    if os.path.exists(embed_path):
        embeddings = embed_model.load(embed_path)
    else:
        with st.spinner("Generating embeddings for the dataset. This might take a while..."):
            os.makedirs('models', exist_ok=True)
            embeddings = embed_model.encode(df['cleaned_text'].tolist(), show_progress_bar=True)
            embed_model.save(embeddings, embed_path)
            st.success("Embeddings generated and saved successfully!")

    tab1, tab2, tab3 = st.tabs(["📊 Classification", "🎯 Scoring", "💬 RAG Q&A"])
    
    with tab1:
        st.header("Resume Classification")
        st.write("Predict the job category of a given resume.")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            resume_text = st.text_area("Enter Resume Text", height=250, key="clf_text")
        
        with col2:
            st.write("Model Status:")
            model_path = 'models/classifier_model.h5'
            encoder_path = 'models/label_encoder.pkl'
            
            if os.path.exists(model_path) and os.path.exists(encoder_path):
                st.success("Trained model found.")
                is_trained = True
            else:
                st.warning("Model not found. Please train the model using `main.py` first.")
                is_trained = False
                
        if st.button("Predict Category", type="primary"):
            if not resume_text:
                st.warning("Please enter some text.")
            elif not is_trained:
                st.error("Cannot predict without a trained model.")
            else:
                with st.spinner("Predicting..."):
                    clf = ResumeClassifier(input_dim=embeddings.shape[1], num_classes=df['Category'].nunique())
                    clf.load(model_path, encoder_path)
                    
                    cleaned_res = preprocessing(resume_text)
                    res_vector = embed_model.encode([cleaned_res])
                    pred_category = clf.predict(res_vector)[0]
                    
                    st.success(f"**Predicted Category:** {pred_category}")

    with tab2:
        st.header("Resume Scoring vs Job Description")
        st.write("Calculate a match score between a resume and a job description.")
        
        col1, col2 = st.columns(2)
        with col1:
            res_input = st.text_area("Resume Content", height=300)
        with col2:
            jd_input = st.text_area("Job Description", height=300)
            
        if st.button("Calculate Match Score", type="primary"):
            if res_input and jd_input:
                with st.spinner("Scoring..."):
                    score, (res_skills, jd_skills) = get_final_match_score(
                        res_input, jd_input, embed_model, skill_extractor, preprocessing
                    )
                    
                    st.metric("Final Match Score", f"{score}%")
                    st.write("---")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Extracted Resume Skills:**")
                        if res_skills:
                            st.write(", ".join(res_skills))
                        else:
                            st.write("No skills found.")
                            
                    with col2:
                        st.write("**Extracted JD Skills:**")
                        if jd_skills:
                            st.write(", ".join(jd_skills))
                        else:
                            st.write("No skills found.")
            else:
                st.warning("Please provide both Resume and Job Description.")
                
    with tab3:
        st.header("Q&A with Resumes (RAG)")
        st.write("Ask questions about the candidates. The system uses Ollama (llama3.1) to answer based on the resume dataset.")
        
        query = st.text_input("Enter your question:")
        
        if st.button("Ask Ollama", type="primary"):
            if query:
                with st.spinner("Searching and generating answer..."):
                    try:
                        rag = ResumeRAG(df, embeddings, embed_model, ollama_model="llama3.1")
                        answer = rag.answer(query, preprocessing)
                        st.write("### Answer:")
                        st.write(answer)
                    except Exception as e:
                        st.error(f"Error querying Ollama. Please ensure Ollama is running and `llama3.1` model is pulled. Details: {e}")
            else:
                st.warning("Please enter a question.")

if __name__ == "__main__":
    main()
