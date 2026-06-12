from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


DEFAULT_EXCLUDED_COLUMNS = {
    "ground_truth_no2",
    "target_no2",
    "observed_no2",
    "no2_observed",
    "no2_target",
}


@dataclass(frozen=True)
class ModelReport:
    """Evaluation summary for one train/test split."""

    rmse: float
    mae: float
    r2: float
    train_rows: int
    test_rows: int
    feature_names: tuple[str, ...]


def normalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with consistent snake_case column names."""

    data = frame.copy()
    data.columns = [
        column.strip().lower().replace(" ", "_").replace("-", "_")
        for column in data.columns
    ]
    return data


def infer_target_column(data: pd.DataFrame, target: str | None) -> str:
    if target:
        target_name = target.strip().lower().replace(" ", "_").replace("-", "_")
        if target_name not in data.columns:
            raise ValueError(f"Target column '{target}' was not found in the dataset.")
        return target_name

    for candidate in DEFAULT_EXCLUDED_COLUMNS:
        if candidate in data.columns:
            return candidate

    numeric_columns = list(data.select_dtypes(include=[np.number]).columns)
    if not numeric_columns:
        raise ValueError("No numeric columns are available for modelling.")
    return numeric_columns[-1]


def build_feature_matrix(
    data: pd.DataFrame,
    target_column: str,
    feature_columns: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build validated numeric features and target values."""

    if feature_columns:
        features = [name.strip().lower().replace(" ", "_").replace("-", "_") for name in feature_columns]
    else:
        features = [
            column
            for column in data.select_dtypes(include=[np.number]).columns
            if column != target_column
        ]

    if not features:
        raise ValueError("At least one numeric feature column is required.")

    missing = [column for column in features if column not in data.columns]
    if missing:
        raise ValueError(f"Feature columns were not found: {', '.join(missing)}")

    modelling_data = data[[*features, target_column]].replace([np.inf, -np.inf], np.nan).dropna()
    if modelling_data.empty:
        raise ValueError("No rows remain after removing missing or infinite values.")

    return modelling_data[features], modelling_data[target_column]


class NO2Downscaler:
    """Train and evaluate a Random Forest model for NO2 concentration downscaling."""

    def __init__(
        self,
        *,
        n_estimators: int = 300,
        random_state: int = 42,
        test_size: float = 0.2,
    ) -> None:
        self.random_state = random_state
        self.test_size = test_size
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=random_state,
        )
        self.report_: ModelReport | None = None
        self.feature_importances_: pd.Series | None = None
        self.predictions_: pd.DataFrame | None = None

    def fit_evaluate(self, features: pd.DataFrame, target: pd.Series) -> ModelReport:
        x_train, x_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=self.test_size,
            random_state=self.random_state,
        )
        self.model.fit(x_train, y_train)
        predictions = self.model.predict(x_test)

        rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
        mae = float(mean_absolute_error(y_test, predictions))
        r2 = float(r2_score(y_test, predictions))
        self.report_ = ModelReport(
            rmse=rmse,
            mae=mae,
            r2=r2,
            train_rows=len(x_train),
            test_rows=len(x_test),
            feature_names=tuple(features.columns),
        )
        self.feature_importances_ = pd.Series(
            self.model.feature_importances_,
            index=features.columns,
            name="importance",
        ).sort_values(ascending=False)
        self.predictions_ = pd.DataFrame(
            {"actual": y_test.to_numpy(), "predicted": predictions},
            index=y_test.index,
        )
        return self.report_

    def save_diagnostics(self, output_dir: str | Path) -> None:
        if self.predictions_ is None or self.feature_importances_ is None:
            raise RuntimeError("Call fit_evaluate before saving diagnostics.")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        self.predictions_.to_csv(output_path / "predictions.csv", index=False)
        self.feature_importances_.to_csv(output_path / "feature_importance.csv")

        sns.set_theme(style="whitegrid")
        self._plot_actual_vs_predicted(output_path / "actual_vs_predicted.png")
        self._plot_residuals(output_path / "residuals.png")
        self._plot_feature_importance(output_path / "feature_importance.png")

    def _plot_actual_vs_predicted(self, path: Path) -> None:
        assert self.predictions_ is not None
        plt.figure(figsize=(7, 6))
        sns.scatterplot(data=self.predictions_, x="actual", y="predicted", alpha=0.75)
        limits = [
            min(self.predictions_["actual"].min(), self.predictions_["predicted"].min()),
            max(self.predictions_["actual"].max(), self.predictions_["predicted"].max()),
        ]
        plt.plot(limits, limits, color="black", linestyle="--", linewidth=1)
        plt.title("NO2 Prediction Quality")
        plt.tight_layout()
        plt.savefig(path, dpi=180)
        plt.close()

    def _plot_residuals(self, path: Path) -> None:
        assert self.predictions_ is not None
        residuals = self.predictions_["actual"] - self.predictions_["predicted"]
        plt.figure(figsize=(7, 5))
        sns.scatterplot(x=self.predictions_["predicted"], y=residuals, alpha=0.75)
        plt.axhline(0, color="black", linestyle="--", linewidth=1)
        plt.xlabel("Predicted NO2")
        plt.ylabel("Residual")
        plt.title("Residuals by Predicted NO2")
        plt.tight_layout()
        plt.savefig(path, dpi=180)
        plt.close()

    def _plot_feature_importance(self, path: Path) -> None:
        assert self.feature_importances_ is not None
        top_features = self.feature_importances_.head(12).sort_values()
        plt.figure(figsize=(8, 5))
        top_features.plot(kind="barh", color="#2878b5")
        plt.xlabel("Random Forest importance")
        plt.title("Top NO2 Downscaling Features")
        plt.tight_layout()
        plt.savefig(path, dpi=180)
        plt.close()


def load_csv_dataset(path: str | Path, target: str | None = None) -> tuple[pd.DataFrame, str]:
    data = normalise_columns(pd.read_csv(path))
    return data, infer_target_column(data, target)
