# Google Earth Engine Export Workflow

This project includes a synthetic sample CSV so the code can be reviewed quickly.
For real experiments, export Sentinel-5P NO2 features from Google Earth Engine and
join them with ground observations or a trusted high-resolution reference source.

Official Earth Engine dataset:
`COPERNICUS/S5P/OFFL/L3_NO2`

Relevant band:
`tropospheric_NO2_column_number_density`

Dataset catalog:
https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S5P_OFFL_L3_NO2

For a ready-made public benchmark, see `docs/public_real_dataset.md`.

## Export Template

Paste this into the Earth Engine Code Editor and update the region, dates, and
asset/table destination.

```javascript
var region = ee.Geometry.Rectangle([77.35, 12.75, 77.85, 13.25]); // Bengaluru example
var startDate = '2019-06-01';
var endDate = '2019-06-30';

var no2 = ee.ImageCollection('COPERNICUS/S5P/OFFL/L3_NO2')
  .select('tropospheric_NO2_column_number_density')
  .filterDate(startDate, endDate)
  .filterBounds(region)
  .mean()
  .rename('coarse_NO2');

var lonLat = ee.Image.pixelLonLat();
var featureImage = no2.addBands(lonLat);

var samples = featureImage.sample({
  region: region,
  scale: 1000,
  geometries: true
});

Export.table.toDrive({
  collection: samples,
  description: 'no2_nexus_sentinel5p_export',
  fileFormat: 'CSV'
});
```

## Joining Ground Truth

The exported file gives satellite-derived NO2 features, not ground truth. To train
a supervised model, join the export with target measurements such as:

- government air-quality monitoring station data
- calibrated local sensor observations
- validated higher-resolution model output

Expected final CSV schema:

```text
latitude, longitude, coarse_NO2, traffic_index, population_density, industrial_index, target_NO2
```

## Important Caveats

- Sentinel-5P measures atmospheric columns, while many ground monitors report
  surface concentration. The two are related but not identical.
- Random train/test splits can leak spatial information. Use `--split spatial`
  for a more realistic held-out-region estimate.
- Report the date range, region, target source, spatial resolution, and filtering
  choices for every experiment.
