import argparse
import json
from pathlib import Path

import joblib # type: ignore
import pandas as pd

def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_input(record: dict, expected_columns: list[str]) -> None:
    missing = [col for col in expected_columns if col not in record]
    extra = [col for col in record if col not in expected_columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if extra:
        raise ValueError(f"Unexpected columns: {extra}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local inference for a single patient.")
    parser.add_argument("--input", required=True, help="Path to input JSON file.")
    parser.add_argument(
        "--model",
        default="models/pipeline.joblib",
        help="Path to serialized sklearn pipeline.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for positive class.",
    )
    args = parser.parse_args()

    pipeline = joblib.load(args.model)
    record = load_json(args.input)

    expected_columns = pipeline.feature_names_in_.tolist()
    validate_input(record, expected_columns)

    X = pd.DataFrame([record])
    X = X[expected_columns]

    # predict_proba returns shape (n_samples, n_classes)
    # [0, 1] -> first sample, probability of positive class 
    positive_proba = float(pipeline.predict_proba(X)[0, 1])
    prediction = int(positive_proba >= args.threshold)

    result = {
        "prediction": prediction,
        "label": "disease" if prediction == 1 else "healthy",
        "probability_positive_class": positive_proba,
        "threshold": args.threshold,
    }

    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()