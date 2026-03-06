from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from ml_patient_classifier.config import load_config
from ml_patient_classifier.train import train


MODELS_TO_COMPARE = [
    "logistic_regression",
    "svm",
    "random_forest",
]


def compare_models(config_path: str) -> pd.DataFrame:
    base_cfg = load_config(config_path)
    rows = []

    for model_name in MODELS_TO_COMPARE:
        cfg_dict = {
            "data": {
                "path": str(base_cfg.data.path),
                "target_col": base_cfg.data.target_col,
                "test_size": base_cfg.data.test_size,
                "random_state": base_cfg.data.random_state,
            },
            "training": {
                "model": model_name,
                "cv_folds": base_cfg.training.cv_folds,
                "scoring": base_cfg.training.scoring,
            },
            "output": {
                "model_path": str(base_cfg.output.model_path),
                "metrics_path": str(base_cfg.output.metrics_path),
            },
        }

        temp_config_path = Path("reports") / f"temp_{model_name}.yaml"
        temp_config_path.parent.mkdir(parents=True, exist_ok=True)
        temp_config_path.write_text(yaml.safe_dump(cfg_dict, sort_keys=False), encoding="utf-8")

        metrics = train(temp_config_path)

        rows.append(
            {
                "model": metrics["model"],
                "best_cv_score": metrics["best_cv_score"],
                "auc": metrics["auc"],
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "best_params": json.dumps(metrics["best_params"]),
            }
        )

    result = pd.DataFrame(rows).sort_values(by="best_cv_score", ascending=False)

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    result.to_csv(reports_dir / "model_comparison.csv", index=False)
    result.to_markdown(reports_dir / "model_comparison.md", index=False)

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()

    result = compare_models(args.config)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()