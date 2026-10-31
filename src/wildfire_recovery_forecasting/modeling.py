"""Small constructors for the approved development pipelines."""
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL_FEATURES = ("dominant_evt", "ignition_month")


def build_ridge_pipeline(predictors, *, alpha=1.0):
    """Build an unfitted pipeline from the approved predictor allowlist.

    Excludes the exact NBR difference, drops all other input columns, and fits
    scaling/encoding only when fit is called (inside each training CV fold).
    The caller supplies roles['predictors'], never the full dataset columns.
    """
    features = list(predictors)
    if len(features) != len(set(features)):
        raise ValueError("Duplicate predictor names")
    if not set(CATEGORICAL_FEATURES).issubset(features):
        raise ValueError("Both approved categorical predictors are required")
    if {"nbr_5_mean", "fire_id", "patch_id", "split"}.intersection(features):
        raise ValueError("Target and identifiers must not be predictors")
    numeric = [f for f in features
               if f not in (*CATEGORICAL_FEATURES, "nbr_drop_patch")]
    return Pipeline([
        ("preprocess", ColumnTransformer([
            ("numeric", StandardScaler(), numeric),
            ("categorical", OneHotEncoder(handle_unknown="ignore"),
             list(CATEGORICAL_FEATURES)),
        ], remainder="drop")),
        ("model", Ridge(alpha=alpha)),
    ])


def build_rf_pipeline(predictors):
    """Unfitted initial RF checkpoint; use the approved predictor allowlist.

    Retains NBR drop, passes numerical inputs unchanged, and learns categorical
    encoding inside each fit. Parameters reproduce the saved Day 3 experiment.
    """
    features = list(predictors)
    if len(features) != len(set(features)):
        raise ValueError("Duplicate predictor names")
    if not set(CATEGORICAL_FEATURES).issubset(features):
        raise ValueError("Both approved categorical predictors are required")
    if {"nbr_5_mean", "fire_id", "patch_id", "split"}.intersection(features):
        raise ValueError("Target and identifiers must not be predictors")
    numeric = [f for f in features if f not in CATEGORICAL_FEATURES]
    return Pipeline([
        ("preprocess", ColumnTransformer([
            ("numeric", "passthrough", numeric),
            ("categorical", OneHotEncoder(handle_unknown="ignore"),
             list(CATEGORICAL_FEATURES)),
        ], remainder="drop")),
        ("model", RandomForestRegressor(
            n_estimators=100, max_depth=16, min_samples_leaf=10,
            max_features=0.8, bootstrap=True, random_state=42, n_jobs=-1,
        )),
    ])
