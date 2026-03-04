from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass(frozen=True)
class DataConfig:
    path: Path
    target_col: str
    test_size: float
    random_state: int


@dataclass(frozen=True)
class TrainingConfig:
    model: str
    cv_folds: int
    scoring: str


@dataclass(frozen=True)
class OutputConfig:
    model_path: Path
    metrics_path: Path


@dataclass(frozen=True)
class AppConfig:
    data: DataConfig
    training: TrainingConfig
    output: OutputConfig


def load_config(path: str | Path) -> AppConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    return AppConfig(
        data=DataConfig(
            path=Path(raw["data"]["path"]),
            target_col=str(raw["data"]["target_col"]),
            test_size=float(raw["data"]["test_size"]),
            random_state=int(raw["data"]["random_state"]),
        ),
        training=TrainingConfig(
            model=str(raw["training"]["model"]),
            cv_folds=int(raw["training"]["cv_folds"]),
            scoring=str(raw["training"]["scoring"]),
        ),
        output=OutputConfig(
            model_path=Path(raw["output"]["model_path"]),
            metrics_path=Path(raw["output"]["metrics_path"]),
        ),
    )