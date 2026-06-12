from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import (
    NO2Downscaler,
    build_feature_matrix,
    evaluate_models,
    load_csv_dataset,
    save_model_comparison,
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
        "--compare",
        action="store_true",
        help="Compare Linear Regression, spatial KNN, Random Forest, and Gradient Boosting baselines.",
    )
    parser.add_argument(
        "--split",
        choices=["random", "spatial", "temporal"],
        default="random",
        help="Validation split strategy used for baseline comparison.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data, target_column = load_csv_dataset(args.data, args.target)
    features, target = build_feature_matrix(data, target_column, args.features)

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

    if args.compare:
        comparison, predictions = evaluate_models(
            features,
            target,
            split_strategy=args.split,
            test_size=args.test_size,
        )
        save_model_comparison(comparison, predictions, Path(args.output_dir), features)
        print("")
        print(f"Baseline comparison ({args.split} split)")
        print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
