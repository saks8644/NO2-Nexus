from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import (
    NO2Downscaler,
    build_feature_matrix,
    evaluate_models,
    load_csv_dataset,
    save_model_comparison,
    spatial_cross_validate_models,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a Random Forest NO2 downscaling model."
    )
    parser.add_argument("--data", required=True, help="Path to a CSV dataset.")
    parser.add_argument(
        "--target",
        default=None,
        help="Target column name. If omitted, a common NO2 target name or the final numeric column is used.",
    )
    parser.add_argument(
        "--features",
        nargs="*",
        default=None,
        help="Optional feature column names. Defaults to all numeric non-target columns.",
    )
    parser.add_argument("--output-dir", default="outputs", help="Directory for plots and CSV outputs.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Evaluation split fraction.")
    parser.add_argument("--n-estimators", type=int, default=300, help="Number of Random Forest trees.")
    parser.add_argument(
        "--skip-primary",
        action="store_true",
        help="Skip the primary Random Forest run and only execute comparison/CV tasks.",
    )
    parser.add_argument(
        "--no-engineered-features",
        action="store_true",
        help="Disable cyclic, spatial-distance, and interaction feature engineering.",
    )
    parser.add_argument(
        "--target-quantile",
        type=float,
        default=None,
        help="Optional central target quantile to keep, e.g. 0.99 removes the outer 1%% tails.",
    )
    parser.add_argument(
        "--log-target",
        action="store_true",
        help="Compare log-transformed target models with inverse-transformed predictions.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare selected baseline models.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["linear_regression", "hist_gradient_boosting", "random_forest"],
        choices=[
            "linear_regression",
            "spatial_knn",
            "hist_gradient_boosting",
            "extra_trees",
            "random_forest",
        ],
        help="Models to compare. KNN and ExtraTrees can be slow on million-row datasets.",
    )
    parser.add_argument(
        "--split",
        choices=["random", "spatial", "temporal"],
        default="random",
        help="Validation split strategy used for baseline comparison.",
    )
    parser.add_argument(
        "--spatial-cv-folds",
        type=int,
        default=0,
        help="Run multi-fold spatial cross-validation when greater than 1.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data, target_column = load_csv_dataset(args.data, args.target)
    features, target = build_feature_matrix(
        data,
        target_column,
        args.features,
        engineer_features=not args.no_engineered_features,
        target_quantile=args.target_quantile,
    )

    if not args.skip_primary:
        downscaler = NO2Downscaler(
            n_estimators=args.n_estimators,
            test_size=args.test_size,
        )
        report = downscaler.fit_evaluate(features, target)
        downscaler.save_diagnostics(Path(args.output_dir))

        print("NO2 Nexus evaluation")
        print(f"Rows: train={report.train_rows}, test={report.test_rows}")
        print(f"Features: {', '.join(report.feature_names)}")
        print(f"RMSE: {report.rmse:.4f}")
        print(f"MAE: {report.mae:.4f}")
        print(f"R2: {report.r2:.4f}")
        print(f"Diagnostics saved to: {Path(args.output_dir).resolve()}")
    else:
        print("NO2 Nexus evaluation")
        print(f"Rows after preprocessing: {len(features)}")
        print(f"Features: {', '.join(features.columns)}")
        print("Primary Random Forest skipped.")

    if args.compare:
        comparison, predictions = evaluate_models(
            features,
            target,
            split_strategy=args.split,
            test_size=args.test_size,
            log_target=args.log_target,
            model_names=args.models,
        )
        save_model_comparison(comparison, predictions, Path(args.output_dir), features)
        print("")
        print(f"Baseline comparison ({args.split} split)")
        print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    if args.spatial_cv_folds > 1:
        spatial_cv = spatial_cross_validate_models(
            features,
            target,
            n_splits=args.spatial_cv_folds,
            log_target=args.log_target,
            model_names=args.models,
        )
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        spatial_cv.to_csv(output_path / "spatial_cv_summary.csv", index=False)
        print("")
        print(f"Spatial CV summary ({args.spatial_cv_folds} folds)")
        print(spatial_cv.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
