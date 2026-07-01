# src/evaluation.py
import logging
import pandas as pd
from src.config import CFG
from src.jd_templates import JD_TEMPLATES, build_eval_pairs

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------

def evaluate_matcher(eval_pairs: list, scoring_fn) -> tuple[float, pd.DataFrame]:
    """
    Run scoring_fn over every (resume, jd, expected_match) triple and
    return accuracy + a detailed results DataFrame.

    Args:
        eval_pairs: List of (resume_text: str, jd_text: str, is_match: bool)
        scoring_fn: Callable(resume_text, jd_text) -> (score: float, (res_skills, jd_skills))

    Returns:
        (accuracy: float, df_eval: pd.DataFrame)
    """
    results = []

    for i, (resume, jd, expected) in enumerate(eval_pairs):
        try:
            score, (res_skills, jd_skills) = scoring_fn(resume, jd)
        except Exception as e:
            logger.warning(f"Pair {i} scoring failed: {e}")
            score = 0.0
            res_skills, jd_skills = [], []

        predicted = score >= CFG.MATCH_SCORE_THRESHOLD

        results.append({
            "pair_index":       i,
            "score":            round(score, 2),
            "expected_match":   expected,
            "predicted_match":  predicted,
            "correct":          predicted == expected,
            "resume_skills":    ", ".join(res_skills) if res_skills else "—",
            "jd_skills":        ", ".join(jd_skills)  if jd_skills  else "—",
        })

    df_eval = pd.DataFrame(results)
    accuracy = df_eval["correct"].mean() if len(df_eval) > 0 else 0.0
    return accuracy, df_eval


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

def precision_recall_at_k(retrieved_ids: list, relevant_ids: set, k: int) -> dict:
    """
    Precision@K and Recall@K for a single retrieval result.

    Args:
        retrieved_ids: Ordered list of retrieved document IDs (top-K).
        relevant_ids:  Set of ground-truth relevant document IDs.
        k:             Cut-off rank.

    Returns:
        Dict with keys: 'precision_at_k', 'recall_at_k', 'k'.
    """
    if k <= 0:
        return {"precision_at_k": 0.0, "recall_at_k": 0.0, "k": k}

    top_k = retrieved_ids[:k]
    hits  = sum(1 for rid in top_k if rid in relevant_ids)

    precision = hits / k
    recall    = hits / len(relevant_ids) if relevant_ids else 0.0

    return {"precision_at_k": precision, "recall_at_k": recall, "k": k}


# ---------------------------------------------------------------------------
# Pair builder — separated from evaluation so pairs are reusable
# ---------------------------------------------------------------------------

def build_sample_eval_pairs(df: pd.DataFrame, n_negative: int = 2) -> list:
    """
    Build evaluation pairs from the resume DataFrame using realistic JD templates.

    Args:
        df:         Resume DataFrame with 'Resume_str' and 'Category' columns.
        n_negative: Number of negative (non-matching) JDs to pair per resume.

    Returns:
        List of (resume_text: str, jd_text: str, is_match: bool)
    """
    missing_cols = {"Resume_str", "Category"} - set(df.columns)
    if missing_cols:
        raise ValueError(f"DataFrame is missing required columns: {missing_cols}")

    pairs = build_eval_pairs(df, JD_TEMPLATES, n_negative=n_negative)
    logger.info(f"Built {len(pairs)} eval pairs "
                f"({sum(m for _, _, m in pairs)} positive, "
                f"{sum(not m for _, _, m in pairs)} negative)")
    return pairs


# ---------------------------------------------------------------------------
# Convenience: build pairs AND evaluate in one call
# ---------------------------------------------------------------------------

def run_full_evaluation(
    df: pd.DataFrame,
    scoring_fn,
    n_negative: int = 2
) -> tuple[float, pd.DataFrame]:
    """
    Build eval pairs from df and immediately evaluate them with scoring_fn.

    Args:
        df:         Resume DataFrame.
        scoring_fn: Callable(resume_text, jd_text) -> (score, (res_skills, jd_skills))
        n_negative: Negative pairs per resume sample.

    Returns:
        (accuracy: float, df_eval: pd.DataFrame)
    """
    pairs = build_sample_eval_pairs(df, n_negative=n_negative)
    accuracy, df_eval = evaluate_matcher(pairs, scoring_fn)

    # Summary log
    total     = len(df_eval)
    correct   = df_eval["correct"].sum()
    tp        = ((df_eval["predicted_match"]) & (df_eval["expected_match"])).sum()
    fp        = ((df_eval["predicted_match"]) & (~df_eval["expected_match"])).sum()
    fn        = ((~df_eval["predicted_match"]) & (df_eval["expected_match"])).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    logger.info(
        f"\n{'='*40}\n"
        f"Evaluation Results ({total} pairs)\n"
        f"  Accuracy  : {accuracy*100:.1f}%\n"
        f"  Precision : {precision*100:.1f}%\n"
        f"  Recall    : {recall*100:.1f}%\n"
        f"  F1 Score  : {f1*100:.1f}%\n"
        f"  Correct   : {correct}/{total}\n"
        f"{'='*40}"
    )

    df_eval["precision"] = precision
    df_eval["recall"]    = recall
    df_eval["f1"]        = f1

    return accuracy, df_eval