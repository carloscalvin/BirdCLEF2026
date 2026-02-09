import numpy as np
import sklearn.metrics

def macro_auc(solution_probs, submission_probs, threshold=0.30):
    solution_binary = (solution_probs > threshold).astype(int)
    solution_sums = solution_binary.sum(axis=0)
    scored_cols_indices = np.where(solution_sums > 0)[0]

    if len(scored_cols_indices) == 0:
        return 0.5

    solution_filtered = solution_binary[:, scored_cols_indices]
    submission_filtered = submission_probs[:, scored_cols_indices]
    score = sklearn.metrics.roc_auc_score(
        solution_filtered, 
        submission_filtered, 
        average='macro'
    )

    return score