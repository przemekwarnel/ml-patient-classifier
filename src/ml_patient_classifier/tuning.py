from __future__ import annotations


def get_param_grid(model_name: str) -> dict:
    """
    Param grids for GridSearchCV.
    Keys use the Pipeline step name: 'model__...'
    """
    name = model_name.lower().strip()

    if name == "logistic_regression":
        return {
            "model__C": [0.1, 1.0, 10.0],
            "model__penalty": ["l2"],
            "model__solver": ["lbfgs", "liblinear"],
            "model__class_weight": [None, "balanced"],
        }

    if name == "svm":
        return {
            "model__C": [0.1, 1.0, 10.0],
            "model__kernel": ["linear", "rbf"],
            "model__class_weight": [None, "balanced"],
        }

    if name == "random_forest":
        return {
            "model__n_estimators": [200, 500],
            "model__max_depth": [None, 5, 10],
            "model__min_samples_split": [2, 5, 10],
            "model__class_weight": [None, "balanced"],
        }

    raise ValueError(
        f"Unknown model '{model_name}'. Valid: logistic_regression | svm | random_forest"
    )