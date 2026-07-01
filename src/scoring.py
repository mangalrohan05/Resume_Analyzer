from sklearn.metrics.pairwise import cosine_similarity
from src.config import CFG
from src.utils import validate_text


def get_final_match_score(resume_text, jd_text, embed_model, skill_extractor, preprocess_fn):

    try:
        resume_text = validate_text(resume_text, name="Resume")
        jd_text = validate_text(jd_text, name="Job Description")
    except ValueError as e:
        return 0.0, ([], [])

    # Embed raw text — BERT performs better on natural language than preprocessed text
    vectors = embed_model.encode([resume_text, jd_text])
    cosine_sim = cosine_similarity([vectors[0]], [vectors[1]])[0][0]

    res_skills = skill_extractor.extract(resume_text)["SKILLS"]
    jd_skills = skill_extractor.extract(jd_text)["SKILLS"]

    if not jd_skills:
        skill_score = 0.0
    else:
        # Use MIN_SKILL_DENOM to prevent inflated scores when JD lists very few skills
        denom = max(len(jd_skills), CFG.MIN_SKILL_DENOM)
        skill_score = len(jd_skills & res_skills) / denom

    final_score = (CFG.SCORE_WEIGHT_SEMANTIC * cosine_sim) + (CFG.SCORE_WEIGHT_SKILLS * skill_score)

    return round(final_score * 100, 2), (list(res_skills), list(jd_skills))