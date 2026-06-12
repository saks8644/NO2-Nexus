from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
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


@dataclass(frozen=True)
class EvaluationResult:
    """Evaluation summary for one model and one validation strategy."""

    model_name: str
    split_strategy: str
    rmse: float
    mae: float
    r2: float
    train_rows: int
    test_rows: int


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


def compute_metrics(actual: pd.Series | np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mae": float(mean_absolute_error(actual, predicted)),
        "r2": float(r2_score(actual, predicted)),
    }


def split_random(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
    )


def split_spatial_holdout(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    latitude_column: str = "latitude",
    longitude_column: str = "longitude",
    test_size: float = 0.2,
    random_state: int = 42,
    grid_bins: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Hold out complete lat/lon grid cells to reduce spatial leakage."""

    if latitude_column not in features.columns or longitude_column not in features.columns:
        raise ValueError(
            "Spatial holdout requires latitude and longitude feature columns. "
            "Pass --split random if coordinates are unavailable."
        )

    lat_bins = pd.qcut(features[latitude_column], q=grid_bins, labels=False, duplicates="drop")
    lon_bins = pd.qcut(features[longitude_column], q=grid_bins, labels=False, duplicates="drop")
    cells = (lat_bins.astype(str) + "_" + lon_bins.astype(str)).rename("cell")
    unique_cells = np.array(sorted(cells.unique()))
    if len(unique_cells) < 2:
        raise ValueError("Spatial holdout requires at least two spatial grid cells.")

    rng = np.random.default_rng(random_state)
    shuffled_cells = rng.permutation(unique_cells)
    test_cell_count = max(1, int(np.ceil(len(unique_cells) * test_size)))
    test_cells = set(shuffled_cells[:test_cell_count])
    test_mask = cells.isin(test_cells)

    if test_mask.all() or not test_mask.any():
        raise ValueError("Spatial holdout produced an empty train or test split.")

    return (
        features.loc[~test_mask],
        features.loc[test_mask],
        target.loc[~test_mask],
        target.loc[test_mask],
    )


def make_baseline_models(random_state: int = 42) -> dict[str, object]:
    return {
        "linear_regression": make_pipeline(StandardScaler(), LinearRegression()),
        "spatial_knn": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5)),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=random_state,
        ),
        "gradient_boosting": GradientBoostingRegressor(random_state=random_state),
    }


def evaluate_models(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    split_strategy: str = "random",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare baseline models on a random or spatial holdout split."""

    if split_strategy == "random":
        x_train, x_test, y_train, y_test = split_random(
            features,
            target,
            test_size=test_size,
            random_state=random_state,
        )
    elif split_strategy == "spatial":
        x_train, x_test, y_train, y_test = split_spatial_holdout(
            features,
            target,
            test_size=test_size,
            random_state=random_state,
        )
    else:
        raise ValueError("split_strategy must be either 'random' or 'spatial'.")

    results: list[EvaluationResult] = []
    predictions = pd.DataFrame({"actual": y_test.to_numpy()}, index=y_test.index)
    for model_name, estimator in make_baseline_models(random_state).items():
        fitted_model = clone(estimator)
        fitted_model.fit(x_train, y_train)
        predicted = fitted_model.predict(x_test)
        metrics = compute_metrics(y_test, predicted)
        predictions[model_name] = predicted
        results.append(
            EvaluationResult(
                model_name=model_name,
                split_strategy=split_strategy,
                train_rows=len(x_train),
                test_rows=len(x_test),
                **metrics,
            )
        )

    comparison = pd.DataFrame([result.__dict__ for result in results]).sort_values("rmse")
    return comparison, predictions


def save_model_comparison(
    comparison: pd.DataFrame,
    predictions: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_path / "model_comparison.csv", index=False)
    predictions.to_csv(output_path / "baseline_predictions.csv", index=False)

    plt.figure(figsize=(8, 5))
    sns.barplot(data=comparison, x="rmse", y="model_name", color="#2878b5")
    plt.xlabel("RMSE")
    plt.ylabel("Model")
    plt.title("Baseline Model Comparison")
    plt.tight_layout()
    plt.savefig(output_path / "model_comparison.png", dpi=180)
    plt.close()


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

        metrics = compute_metrics(y_test, predictions)
        self.report_ = ModelReport(
            rmse=metrics["rmse"],
            mae=metrics["mae"],
            r2=metrics["r2"],
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
