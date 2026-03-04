from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def build_model(name: str):
    """
    Returns an sklearn estimator based on a string key from config.
    Keep this small now; we'll expand with tuning grids in the next step.
    """
    name = name.lower().strip()

    if name == "logistic_regression":
        return LogisticRegression(max_iter=2000)

    if name == "svm":
        # probability=True required for ROC/AUC via predict_proba
        return SVC(probability=True)

    if name == "random_forest":
        return RandomForestClassifier()

    raise ValueError(
        f"Unknown model '{name}'. Valid: logistic_regression | svm | random_forest"
    )