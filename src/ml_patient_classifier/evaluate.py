from __future__ import annotations

import argparse
import json

import joblib # type: ignore
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ml_patient_classifier.config import load_config
from ml_patient_classifier.data import load_dataframe, make_split, split_xy
from ml_patient_classifier.thresholds import find_threshold_for_min_recall


def evaluate(config_path: str, threshold: float | None, min_recall: float | None) -> dict:
    cfg = load_config(config_path)

    # Load data (same split logic as training for consistent holdout)
    df = load_dataframe(cfg.data.path)
    X, y = split_xy(df, cfg.data.target_col)
    _, X_test, _, y_test = make_split(
        X, y, test_size=cfg.data.test_size, random_state=cfg.data.random_state
    )

    # Load trained pipeline
    pipeline = joblib.load(cfg.output.model_path)

    proba = pipeline.predict_proba(X_test)[:, 1]

    selected_threshold = threshold
    if selected_threshold is None and min_recall is not None:
        selected_threshold = find_threshold_for_min_recall(y_test.to_numpy(), proba, min_recall)

    if selected_threshold is None:
        raise ValueError("Could not determine threshold. Provide --threshold or --min-recall.")

    pred = (proba >= selected_threshold).astype(int)
    tp = ((y_test == 1) & (pred == 1)).sum()
    tn = ((y_test == 0) & (pred == 0)).sum()
    fp = ((y_test == 0) & (pred == 1)).sum()
    fn = ((y_test == 1) & (pred == 0)).sum()    

    if min_recall is not None:
        # compute recall for selected threshold
        achieved_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if achieved_recall < min_recall:
            raise ValueError(
            f"min_recall constraint not met: achieved {achieved_recall:.4f} < {min_recall:.4f} "
            f"at threshold={selected_threshold:.4f}"
            )

    metrics = {
        "threshold": float(selected_threshold),
        "min_recall": (float(min_recall) if min_recall is not None else None),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),        
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "auc": float(roc_auc_score(y_test, proba)),
    }

    # Ensure reports dir exists
    cfg.output.metrics_path.parent.mkdir(parents=True, exist_ok=True)

    # Save ROC curve
    RocCurveDisplay.from_predictions(y_test, proba)
    plt.title("ROC Curve (Test)")
    plt.savefig("reports/roc_curve.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Save Confusion Matrix
    ConfusionMatrixDisplay.from_predictions(y_test, pred)
    plt.title(f"Confusion Matrix (Test) @ threshold={selected_threshold:.2f}")
    plt.savefig("reports/confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Save eval metrics separately (do not overwrite training metrics yet)
    with open("reports/eval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--min-recall", type=float, default=None)
    args = parser.parse_args()

    metrics = evaluate(args.config, args.threshold, args.min_recall)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()