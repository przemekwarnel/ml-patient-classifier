from __future__ import annotations

import numpy as np


def find_threshold_for_min_recall(y_true, y_proba, min_recall: float) -> float | None:
    """
    Returns the highest threshold that still achieves recall >= min_recall.
    If impossible, returns None.
    """
    thresholds = np.unique(y_proba)
    thresholds.sort()

    best = None
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)

        tp = ((y_true == 1) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if recall >= min_recall:
            best = float(t)  # keep increasing; final best is the highest threshold meeting constraint

    return best