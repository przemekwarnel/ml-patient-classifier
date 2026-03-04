from __future__ import annotations

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def load_dataframe(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("Loaded dataframe is empty.")
    return df


def split_xy(df: pd.DataFrame, target_col: str):
    if target_col not in df.columns:
        raise KeyError(f"target_col '{target_col}' not found. Available: {list(df.columns)}")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def make_split(X, y, test_size: float, random_state: int):
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )