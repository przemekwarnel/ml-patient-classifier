from __future__ import annotations

import argparse
import json

import joblib # type: ignore
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from ml_patient_classifier.config import load_config
from ml_patient_classifier.data import load_dataframe, make_split, split_xy
from ml_patient_classifier.modeling import build_model
from ml_patient_classifier.preprocessing import build_preprocessor
from ml_patient_classifier.tuning import get_param_grid


def train(config_path: str) -> dict:
    cfg = load_config(config_path)

    df = load_dataframe(cfg.data.path)
    X, y = split_xy(df, cfg.data.target_col)
    X_train, X_test, y_train, y_test = make_split(
        X, y, test_size=cfg.data.test_size, random_state=cfg.data.random_state
    )

    preprocessor = build_preprocessor(X_train)
    model = build_model(cfg.training.model)

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    #pipeline.fit(X_train, y_train)

    param_grid = get_param_grid(cfg.training.model)

    cv = StratifiedKFold(
        n_splits=cfg.training.cv_folds,
        shuffle=True, 
        random_state=cfg.data.random_state
    )

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=cfg.training.scoring,
        cv=cv,
        n_jobs=-1,
    )

    search.fit(X_train, y_train)

    best_pipeline = search.best_estimator_

    #proba = pipeline.predict_proba(X_test)[:, 1]
    proba = best_pipeline.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    metrics = {
        "model": cfg.training.model,
        "cv_folds": int(cfg.training.cv_folds),
        "scoring": str(cfg.training.scoring),
        "best_cv_score": float(search.best_score_),
        "best_params": search.best_params_,
        "n_rows": int(df.shape[0]),
        "n_features": int(X.shape[1]),
        "test_size": float(cfg.data.test_size),
        "random_state": int(cfg.data.random_state),
        "threshold": 0.5,
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "auc": float(roc_auc_score(y_test, proba)),
    }

    cfg.output.model_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.output.metrics_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_pipeline, cfg.output.model_path)
    cfg.output.metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()

    metrics = train(args.config)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()