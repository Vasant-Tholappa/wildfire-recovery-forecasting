"""Patch regression metrics with an equal-fire error summary."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model(y_true, y_pred, fire_ids):
    """Evaluate positionally aligned finite 1-D targets and predictions.

    Equal-fire MAE averages per-fire patch MAEs. Pandas indices are not keys.
    R-squared follows sklearn defaults for constant targets and single rows.
    """
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    fires = np.asarray(fire_ids)
    if any(values.ndim != 1 for values in (actual, predicted, fires)):
        raise ValueError("Inputs must be one-dimensional")
    if not len(actual) or not (len(actual) == len(predicted) == len(fires)):
        raise ValueError("Inputs must be nonempty and have matching lengths")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Targets and predictions must contain only finite numbers")
    if pd.isna(fires).any():
        raise ValueError("Fire IDs must not be missing")
    fire_errors = pd.DataFrame({
        "fire_id": fires,
        "absolute_error": np.abs(actual - predicted),
    })
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
        "equal_fire_mae": float(
            fire_errors.groupby("fire_id")["absolute_error"].mean().mean()
        ),
    }
