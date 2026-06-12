"""NO2 Nexus: reproducible NO2 downscaling experiments."""

from .pipeline import (
    EvaluationResult,
    ModelReport,
    NO2Downscaler,
    add_engineered_features,
    build_feature_matrix,
    evaluate_models,
    spatial_cross_validate_models,
)

__all__ = [
    "EvaluationResult",
    "ModelReport",
    "NO2Downscaler",
    "add_engineered_features",
    "build_feature_matrix",
    "evaluate_models",
    "spatial_cross_validate_models",
]
