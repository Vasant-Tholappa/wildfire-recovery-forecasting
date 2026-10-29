"""Offline evaluation tests using synthetic targets only."""
import numpy as np
import pandas as pd
import pytest

from wildfire_recovery_forecasting.evaluation import evaluate_model


def test_metrics_and_unequal_fire_sizes():
    # Three patches in A, one in B: patch MAE=1.5, equal-fire MAE=2.
    scores = evaluate_model([0, 1, 2, 3], [1, 2, 3, 6], ["A", "A", "A", "B"])
    assert scores == pytest.approx({
        "mae": 1.5, "rmse": np.sqrt(3), "r2": -1.4, "equal_fire_mae": 2,
    })


def test_perfect_predictions_and_positional_series():
    scores = evaluate_model(pd.Series([1, 2], index=[3, 4]),
                            pd.Series([1, 2], index=[9, 8]), ["A", "B"])
    assert scores == {"mae": 0, "rmse": 0, "r2": 1, "equal_fire_mae": 0}


@pytest.mark.parametrize("actual,predicted,fires", [
    ([1, 2], [1], ["A", "B"]),
    ([1, 2], [1, 2], ["A"]),
    ([], [], []),
    ([[1], [2]], [1, 2], ["A", "B"]),
    ([1, 2], [[1], [2]], ["A", "B"]),
    ([1, 2], [1, 2], [["A"], ["B"]]),
    ([1, np.nan], [1, 2], ["A", "B"]),
    ([1, 2], [1, np.nan], ["A", "B"]),
    ([1, np.inf], [1, 2], ["A", "B"]),
    ([1, 2], [1, -np.inf], ["A", "B"]),
    ([1, "invalid"], [1, 2], ["A", "B"]),
    ([1, 2], [1, "invalid"], ["A", "B"]),
    ([1, 2], [1, 2], ["A", None]),
])
def test_invalid_inputs(actual, predicted, fires):
    with pytest.raises(ValueError):
        evaluate_model(actual, predicted, fires)
