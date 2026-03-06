from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from ml_patient_classifier.config import load_config
from ml_patient_classifier.data import load_dataframe, make_split, split_xy
from ml_patient_classifier.thresholds import find_threshold_for_min_recall


def evaluate_at_threshold(y_true, y_proba, threshold: float, setting: str) -> dict:
    pred = (y_proba >= threshold).astype(int)

    tp = ((y_true == 1) & (pred == 1)).sum()
    tn = ((y_true == 0) & (pred == 0)).sum()
    fp = ((y_true == 0) & (pred == 1)).sum()
    fn = ((y_true == 1) & (pred == 0)).sum()

    return {
        "setting": setting,
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_proba)),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def run_threshold_analysis(config_path: str) -> pd.DataFrame:
    cfg = load_config(config_path)

    df = load_dataframe(cfg.data.path)
    X, y = split_xy(df, cfg.data.target_col)
    _, X_test, _, y_test = make_split(
        X, y, test_size=cfg.data.test_size, random_state=cfg.data.random_state
    )

    pipeline = joblib.load(cfg.output.model_path)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_true = y_test.to_numpy()

    rows = []

    # Default threshold
    rows.append(evaluate_at_threshold(y_true, y_proba, threshold=0.5, setting="default"))

    # Screening @ recall >= 0.90
    t90 = find_threshold_for_min_recall(y_true, y_proba, min_recall=0.90)
    if t90 is not None:
        rows.append(evaluate_at_threshold(y_true, y_proba, threshold=t90, setting="screening_90"))

    # Screening @ recall >= 0.95
    t95 = find_threshold_for_min_recall(y_true, y_proba, min_recall=0.95)
    if t95 is not None:
        rows.append(evaluate_at_threshold(y_true, y_proba, threshold=t95, setting="screening_95"))

    result = pd.DataFrame(rows)

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    result.to_csv(reports_dir / "threshold_comparison.csv", index=False)
    result.to_markdown(reports_dir / "threshold_comparison.md", index=False)

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()

    df = run_threshold_analysis(args.config)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()