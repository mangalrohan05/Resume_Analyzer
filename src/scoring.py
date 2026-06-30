from sklearn.metrics.pairwise import cosine_similarity

def get_final_match_score(resume_text, jd_text, embed_model, skill_extractor, preprocess_fn):
    res_clean = preprocess_fn(resume_text)
    jd_clean = preprocess_fn(jd_text)
    
    vectors = embed_model.encode([res_clean, jd_clean])
    
    cosine_sim = cosine_similarity([vectors[0]], [vectors[1]])[0][0]
    
    res_skills = skill_extractor.extract(resume_text)["SKILLS"]
    jd_skills = skill_extractor.extract(jd_text)["SKILLS"]
    
    skill_score = 0 if not jd_skills else len(jd_skills & res_skills) / len(jd_skills)
    final_score = (0.5*cosine_sim) + (0.5*skill_score)
    
    return round(final_score * 100, 2), (list(res_skills), list(jd_skills))