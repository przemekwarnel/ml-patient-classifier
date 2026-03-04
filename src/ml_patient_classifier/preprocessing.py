from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Preprocessor inferred from X dtypes:
      - numeric: median impute + standard scale
      - categorical: most_frequent impute + one-hot
    This is leakage-safe when used inside a Pipeline and fit on train only.
    """
    num_selector = make_column_selector(dtype_include=["number"])
    cat_selector = make_column_selector(dtype_exclude=["number"])

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, num_selector),
            ("cat", categorical_pipeline, cat_selector),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )