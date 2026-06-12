# Results

These results use the included synthetic sample dataset in `data/sample_no2.csv`.
They verify that the pipeline runs end to end, but they should not be presented as
real-world Sentinel-5P performance.

## Sample Run

Command:

```bash
python -m no2_nexus.cli --data data/sample_no2.csv --target target_NO2 --output-dir outputs/sample_run --compare --split spatial
```

Outputs:

- `outputs/sample_run/model_comparison.csv`
- `outputs/sample_run/model_comparison.png`
- `outputs/sample_run/prediction_map.png`
- `outputs/sample_run/predictions.csv`
- `outputs/sample_run/baseline_predictions.csv`
- `outputs/sample_run/actual_vs_predicted.png`
- `outputs/sample_run/residuals.png`
- `outputs/sample_run/feature_importance.png`

Spatial holdout baseline comparison on the synthetic sample:

| Model | RMSE | MAE | R2 |
| --- | ---: | ---: | ---: |
| Linear Regression | 0.0876 | 0.0786 | 0.9988 |
| Gradient Boosting | 2.3627 | 2.0285 | 0.1460 |
| Random Forest | 3.2674 | 2.9635 | -0.6332 |
| Spatial KNN | 3.8262 | 3.4703 | -1.2395 |

Linear Regression performs best on this synthetic sample because the target was
generated from a mostly linear relationship. On real Sentinel-5P and ground
monitoring data, the ranking may change, which is why the project compares
multiple baselines instead of assuming the most complex model is best.

Temporal holdout comparison on the synthetic sample:

| Model | RMSE | MAE | R2 |
| --- | ---: | ---: | ---: |
| Linear Regression | 0.0767 | 0.0636 | 0.9998 |
| Gradient Boosting | 0.9506 | 0.6161 | 0.9629 |
| Random Forest | 1.6665 | 0.9178 | 0.8859 |
| Spatial KNN | 1.9021 | 1.6205 | 0.8514 |

The temporal split asks a harder question than random validation: can the model
generalize to later observations?

## Real-Data Path

The repository includes `scripts/prepare_hsg_no2_dataset.py`, which downloads and
prepares the public HSG-AIML NO2 dataset when run locally. That dataset is
described by its authors as temporally and spatially aligned NO2 measurements
from European air-quality stations, Sentinel-5P, and supplementary sources.

The archive is about 100 MB, so raw and processed files are intentionally ignored
by Git. After preparing it, run:

```bash
no2-nexus --data data/processed/hsg_no2_training.csv --target target_no2 --output-dir outputs/hsg_run --compare --split spatial
no2-nexus --data data/processed/hsg_no2_training.csv --target target_no2 --output-dir outputs/hsg_temporal --compare --split temporal
```

## How To Interpret Metrics

- **RMSE** highlights larger errors and is useful when high-pollution misses are
  especially costly.
- **MAE** is easier to explain as the average absolute error.
- **R2** explains how much held-out variance the model captures.

For real geospatial experiments, prioritize the spatial split results over the
random split results. Random splits often look better because nearby observations
can share very similar pollution patterns.

## Limitations

- The included sample data is synthetic and only intended for reproducibility
  checks.
- Sentinel-5P NO2 column density is not the same physical quantity as ground-level
  NO2 concentration.
- Auxiliary features such as traffic, land use, meteorology, population density,
  and elevation strongly affect real downscaling quality.
- Spatial autocorrelation can inflate scores when nearby points appear in both
  train and test sets.
- Real deployment would need uncertainty estimates, temporal validation, data
  drift monitoring, and careful QA on satellite retrieval quality.

## Failure Cases To Watch

- Rural or low-monitor-density regions may have weak ground-truth coverage.
- Satellite retrievals can be missing or noisy because of clouds, snow, aerosols,
  or low-quality retrieval flags.
- A model trained in one city or season may fail under different meteorology,
  emission patterns, or sensor networks.
- Feature importance can be misleading when features are correlated, so it should
  be treated as a diagnostic rather than a causal explanation.

## Future Work

- Compare Random Forest with XGBoost or LightGBM when those dependencies are
  acceptable.
- Export predicted maps as GeoTIFF files for GIS tools.
