# NO2 Nexus

NO2 Nexus is a reproducible machine learning pipeline for estimating fine-grained
Nitrogen Dioxide (NO2) concentration patterns from coarse satellite-style
measurements and auxiliary spatial features.

The project uses a Random Forest regressor as a strong, interpretable baseline:
it handles nonlinear interactions, works well on tabular geospatial features, and
provides feature-importance diagnostics that are easy to explain in an ML review.

![no2](https://github.com/user-attachments/assets/15acf3fb-37a9-48df-b160-13f51da3db3f)

## Why This Project Matters

High-resolution air-quality maps help researchers, planners, and public-health
teams understand pollution exposure at a more actionable spatial scale. Satellite
products such as Sentinel-5P/TROPOMI provide valuable NO2 observations, but the
raw spatial resolution is often too coarse for neighborhood-level decisions.

NO2 Nexus demonstrates the ML workflow needed for a downscaling experiment:

- clean and validate tabular geospatial features
- train a reproducible Random Forest model
- compare simple baselines against tree-based models
- evaluate with random, spatial, and temporal holdout splits
- run multi-fold spatial cross-validation
- engineer cyclic time, spatial-distance, and interaction features
- evaluate with RMSE, MAE, and R2
- export prediction and feature-importance artifacts
- create a lightweight prediction map
- generate diagnostic plots for model inspection

## Repository Structure

```text
.
|-- data/sample_no2.csv      # Synthetic sample dataset for quick runs
|-- docs/                    # Real-data export workflow
|-- model.ipynb              # Polished notebook walkthrough
|-- pyproject.toml           # Installable package metadata
|-- requirements.txt         # Runtime and test dependencies
|-- results.md               # Metrics interpretation and limitations
|-- scripts/                 # Public data preparation utilities
|-- src/no2_nexus/
|   |-- cli.py               # Command-line training entry point
|   `-- pipeline.py          # Data validation, training, metrics, plots
`-- tests/
    `-- test_pipeline.py     # Unit tests with synthetic NO2-like data
```

## Installation

```bash
git clone https://github.com/saks8644/NO2-Nexus.git
cd NO2-Nexus
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

On macOS/Linux, activate the environment with:

```bash
source .venv/bin/activate
```

## Input Data

Use a CSV file with numeric feature columns and one numeric target column. Common
target names such as `target_NO2`, `ground_truth_NO2`, `observed_NO2`, and
`NO2_target` are detected automatically. If your target has a different name,
pass it with `--target`.

Example columns:

```text
latitude, longitude, coarse_NO2, traffic_index, population_density, target_NO2
```

The pipeline normalizes column names, removes rows with missing or infinite
modeling values, and uses all numeric non-target columns as features unless
specific features are provided.

## Usage

Train and evaluate the model:

```bash
no2-nexus --data data/no2_training.csv --target target_NO2 --output-dir outputs
```

Try the included synthetic sample dataset:

```bash
no2-nexus --data data/sample_no2.csv --target target_NO2 --output-dir outputs/sample_run
```

Compare baselines with a spatial holdout:

```bash
no2-nexus --data data/sample_no2.csv --target target_NO2 --output-dir outputs/sample_run --compare --split spatial
```

Compare baselines with a temporal holdout:

```bash
no2-nexus --data data/sample_no2.csv --target target_NO2 --output-dir outputs/temporal_run --compare --split temporal
```

Run the improved public real-data experiment after preparing HSG-AIML data:

```bash
no2-nexus --data data/processed/hsg_no2_training.csv --target surfaceconcentration --output-dir outputs/hsg_improved_spatial --skip-primary --compare --split spatial --target-quantile 0.99 --models linear_regression hist_gradient_boosting random_forest
```

Run multi-fold spatial cross-validation:

```bash
no2-nexus --data data/processed/hsg_no2_training.csv --target surfaceconcentration --output-dir outputs/hsg_improved_spatial_cv --skip-primary --target-quantile 0.99 --models linear_regression hist_gradient_boosting --spatial-cv-folds 4
```

Or run it as a module:

```bash
python -m no2_nexus.cli --data data/no2_training.csv --target target_NO2
```

Optional feature selection:

```bash
no2-nexus ^
  --data data/no2_training.csv ^
  --target target_NO2 ^
  --features latitude longitude coarse_NO2 traffic_index population_density
```

The command prints evaluation metrics and writes:

- `outputs/predictions.csv`
- `outputs/feature_importance.csv`
- `outputs/model_comparison.csv` when `--compare` is used
- `outputs/model_comparison.png` when `--compare` is used
- `outputs/prediction_map.png` when coordinates are available
- `outputs/actual_vs_predicted.png`
- `outputs/residuals.png`
- `outputs/feature_importance.png`

## Model Evaluation

The CLI reports:

- **RMSE:** penalizes large NO2 prediction errors
- **MAE:** average absolute prediction error
- **R2:** explained variance on the held-out test split

The exported plots help diagnose whether the model is biased, whether errors
grow for higher NO2 values, and which features drive predictions.

Use `--split spatial` for a held-out-region estimate and `--split temporal` for
a held-out-future estimate. A random split can overstate performance when nearby
or same-period observations share similar pollution patterns.

Temporal validation requires a complete `date` column. The included synthetic
sample has one; the prepared public HSG-AIML file exposes month, weekday, and
hour but not a complete timestamp, so spatial validation is the main real-data
generalization check.

## Real Data Workflow

The included sample CSV is synthetic. Use it to verify the software, not to claim
real-world model performance. For real experiments, follow
`docs/google_earth_engine_export.md` to export Sentinel-5P NO2 features from
Google Earth Engine and join them with ground-truth measurements.

You can also prepare the public HSG-AIML aligned Sentinel-5P/ground-station NO2
dataset locally:

```bash
python scripts/prepare_hsg_no2_dataset.py
no2-nexus --data data/processed/hsg_no2_training.csv --target surfaceconcentration --output-dir outputs/hsg_improved_spatial --skip-primary --compare --split spatial --target-quantile 0.99 --models linear_regression hist_gradient_boosting random_forest
```

The raw and processed large-data directories are ignored by Git, which keeps this
repository lightweight while making the real-data path reproducible.

See `results.md` for metric interpretation, current limitations, and future work.

## Development Checks

Run tests before sharing changes:

```bash
pytest
```

The tests cover column normalization, target/feature validation, missing-value
handling, and end-to-end model training on deterministic synthetic data.

## Next Improvements

- Add raster export support for generating GeoTIFF prediction maps.
- Track experiments with MLflow or Weights & Biases for stronger reproducibility.

## Contact

For support, questions, or contributions, contact Saksham Balsane at
sakshambalsane19@gmail.com.
