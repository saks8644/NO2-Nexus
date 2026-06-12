# Public Real Dataset

NO2 Nexus supports a public real-data path through the HSG-AIML NO2 dataset:

https://github.com/HSG-AIML/NO2-dataset

The dataset authors describe it as temporally and spatially aligned NO2
measurements from European air-quality stations, Sentinel-5P satellite data, and
supplementary sources. This makes it a much better benchmark than the small
synthetic `data/sample_no2.csv` file.

## Why The Dataset Is Not Committed

The compressed archive is about 100 MB. Committing it would make this portfolio
repository heavy and harder to clone, so raw and processed large-data paths are
ignored:

```text
data/raw/
data/processed/
```

## Prepare Locally

```bash
python scripts/prepare_hsg_no2_dataset.py
```

Then run spatial and temporal evaluations:

```bash
no2-nexus --data data/processed/hsg_no2_training.csv --target target_no2 --output-dir outputs/hsg_spatial --compare --split spatial
no2-nexus --data data/processed/hsg_no2_training.csv --target target_no2 --output-dir outputs/hsg_temporal --compare --split temporal
```

If the upstream dataset schema changes, inspect the extracted files under
`data/raw/hsg_no2/extracted` and create a cleaned CSV with:

```text
latitude, longitude, coarse_no2, target_no2
```

Additional numeric features are used automatically.
