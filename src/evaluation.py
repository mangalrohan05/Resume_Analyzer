import pandas as pd
from src.config import CFG


def evaluate_matcher(eval_pairs: list, scoring_fn) -> tuple[float, pd.DataFrame]:
    """Evaluate the resume-JD matching function against labeled examples.

    Args:
        eval_pairs: List of (resume_text, jd_text, expected_match: bool).
        scoring_fn: A callable matching the signature of get_final_match_score,
                    returning (score, (res_skills, jd_skills)).

    Returns:
        Tuple of (accuracy, results_DataFrame).
    """
    results = []
    for resume, jd, expected in eval_pairs:
        try:
            score, (res_skills, jd_skills) = scoring_fn(resume, jd)
        except Exception as e:
            score = 0.0
            res_skills, jd_skills = [], []
        predicted = score >= CFG.MATCH_SCORE_THRESHOLD
        results.append(
            {
                "score": score,
                "expected_match": expected,
                "predicted_match": predicted,
                "correct": predicted == expected,
                "resume_skills": ", ".join(res_skills) if res_skills else "—",
                "jd_skills": ", ".join(jd_skills) if jd_skills else "—",
            }
        )

    df_eval = pd.DataFrame(results)
    accuracy = df_eval["correct"].mean() if len(df_eval) > 0 else 0.0
    return accuracy, df_eval


def precision_recall_at_k(retrieved_ids: list, relevant_ids: set, k: int) -> dict:
    """Compute Precision@K and Recall@K for a single retrieval result.

    Args:
        retrieved_ids: Ordered list of retrieved document IDs (top-K).
        relevant_ids: Set of ground-truth relevant document IDs.
        k: Cut-off rank.

    Returns:
        Dict with keys: 'precision_at_k', 'recall_at_k', 'k'.
    """
    top_k = retrieved_ids[:k]
    hits = len([rid for rid in top_k if rid in relevant_ids])
    precision = hits / k if k > 0 else 0.0
    recall = hits / len(relevant_ids) if relevant_ids else 0.0
    return {"precision_at_k": precision, "recall_at_k": recall, "k": k}


def build_sample_eval_pairs(df: pd.DataFrame, n_positive: int = 5, n_negative: int = 5) -> list:
    """Build a small labeled eval set from the loaded dataset.

    Positive pairs: same-category resume vs. a JD that mentions the category name.
    Negative pairs: cross-category resume vs. a JD from a different category.

    Args:
        df: The full resume DataFrame with 'Resume_str' and 'Category' columns.
        n_positive: Number of positive (matching) pairs to generate.
        n_negative: Number of negative (non-matching) pairs to generate.

    Returns:
        List of (resume_text, jd_text, expected_match: bool).
    """
    categories = df["Category"].unique().tolist()
    pairs = []

    # Positive pairs
    for cat in categories[:n_positive]:
        sample = df[df["Category"] == cat].iloc[0]
        jd_text = f"We are hiring a {cat} professional with relevant skills and experience."
        pairs.append((sample["Resume_str"], jd_text, True))

    # Negative pairs
    for i, cat in enumerate(categories[:n_negative]):
        other_cat = categories[(i + 1) % len(categories)]
        sample = df[df["Category"] == cat].iloc[0]
        jd_text = f"We are hiring a {other_cat} professional with relevant skills and experience."
        pairs.append((sample["Resume_str"], jd_text, False))

    return pairs
