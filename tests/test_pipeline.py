import numpy as np
import pandas as pd
import pytest

from no2_nexus.pipeline import NO2Downscaler, build_feature_matrix, infer_target_column, normalise_columns


def make_synthetic_no2_data(rows: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    latitude = rng.uniform(12.8, 13.2, rows)
    longitude = rng.uniform(77.4, 77.8, rows)
    coarse_no2 = rng.normal(28, 4, rows)
    traffic_index = rng.normal(0.6, 0.15, rows)
    target_no2 = 0.7 * coarse_no2 + 8 * traffic_index + 0.1 * latitude - 0.05 * longitude
    return pd.DataFrame(
        {
            "Latitude": latitude,
            "Longitude": longitude,
            "coarse_NO2": coarse_no2,
            "traffic_index": traffic_index,
            "target_NO2": target_no2,
        }
    )


def test_normalise_columns_creates_snake_case_names():
    data = normalise_columns(pd.DataFrame({"Target NO2": [1.0], "coarse-NO2": [0.8]}))

    assert list(data.columns) == ["target_no2", "coarse_no2"]


def test_build_feature_matrix_drops_missing_values():
    data = normalise_columns(make_synthetic_no2_data(10))
    data.loc[0, "coarse_no2"] = np.nan
    target_column = infer_target_column(data, None)

    features, target = build_feature_matrix(data, target_column)

    assert len(features) == 9
    assert len(target) == 9
    assert "target_no2" not in features.columns


def test_downscaler_trains_and_reports_metrics():
    data = normalise_columns(make_synthetic_no2_data())
    target_column = infer_target_column(data, None)
    features, target = build_feature_matrix(data, target_column)

    report = NO2Downscaler(n_estimators=40).fit_evaluate(features, target)

    assert report.train_rows == 64
    assert report.test_rows == 16
    assert report.rmse >= 0
    assert report.r2 > 0.75


def test_build_feature_matrix_rejects_missing_feature():
    data = normalise_columns(make_synthetic_no2_data())

    with pytest.raises(ValueError, match="Feature columns were not found"):
        build_feature_matrix(data, "target_no2", ["missing_feature"])
