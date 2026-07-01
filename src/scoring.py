from sklearn.metrics.pairwise import cosine_similarity
from src.config import CFG
from src.utils import validate_text


def get_final_match_score(resume_text, jd_text, embed_model, skill_extractor, preprocess_fn):

    try:
        resume_text = validate_text(resume_text, name="Resume")
        jd_text     = validate_text(jd_text, name="Job Description")
    except ValueError:
        return 0.0, ([], [])

    # Skill extraction
    res_skills = skill_extractor.extract(resume_text)["SKILLS"]
    jd_skills  = skill_extractor.extract(jd_text)["SKILLS"]

    vectors    = embed_model.encode([resume_text, jd_text])
    cosine_sim = cosine_similarity([vectors[0]], [vectors[1]])[0][0]

    # Skill overlap score
    if not jd_skills:
        skill_score = 0.0
    else:
        matched         = jd_skills & res_skills
        effective_total = min(len(jd_skills), CFG.MIN_SKILL_DENOM)
        skill_score     = min(len(matched) / effective_total, 1.0)

    # Weighted final score
    final_score = (CFG.SCORE_WEIGHT_SEMANTIC * cosine_sim) + \
                  (CFG.SCORE_WEIGHT_SKILLS   * skill_score)

    return round(final_score * 100, 2), (list(res_skills), list(jd_skills))