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
- `outputs/sample_run/predictions.csv`
- `outputs/sample_run/baseline_predictions.csv`
- `outputs/sample_run/actual_vs_predicted.png`
- `outputs/sample_run/residuals.png`
- `outputs/sample_run/feature_importance.png`

Spatial holdout baseline comparison on the synthetic sample:

| Model | RMSE | MAE | R2 |
| --- | ---: | ---: | ---: |
| Linear Regression | 0.0598 | 0.0524 | 0.9995 |
| Gradient Boosting | 2.3678 | 2.0744 | 0.1423 |
| Random Forest | 3.2510 | 2.9326 | -0.6168 |
| Spatial KNN | 3.9697 | 3.6053 | -1.4107 |

Linear Regression performs best on this synthetic sample because the target was
generated from a mostly linear relationship. On real Sentinel-5P and ground
monitoring data, the ranking may change, which is why the project compares
multiple baselines instead of assuming the most complex model is best.

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

## Future Work

- Add a public real-world benchmark dataset.
- Compare Random Forest with XGBoost or LightGBM when those dependencies are
  acceptable.
- Add temporal validation, for example training on one month and testing on a
  later month.
- Export predicted maps as GeoTIFF files for GIS tools.
