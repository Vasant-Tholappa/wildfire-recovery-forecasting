"""Synthetic-only tests for Ridge feature selection and fold isolation."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from wildfire_recovery_forecasting.modeling import build_ridge_pipeline
from wildfire_recovery_forecasting.pipeline import dataset_roles


def test_approved_roles_and_pipeline_configuration():
    roles = dataset_roles(["nbr_drop_patch", "forest_fraction"])
    model = build_ridge_pipeline(roles["predictors"], alpha=3000)
    prep = model.named_steps["preprocess"]
    numeric = prep.transformers[0][2]
    categorical = prep.transformers[1][2]
    assert len(numeric) == 31
    assert categorical == ["dominant_evt", "ignition_month"]
    assert set(numeric + categorical) == set(roles["predictors"]) - {"nbr_drop_patch"}
    assert not {roles["target"], "forest_fraction", "fire_id", "split"}.intersection(numeric)
    assert prep.remainder == "drop"
    assert prep.transformers[1][1].handle_unknown == "ignore"
    assert model.named_steps["model"].alpha == 3000


def test_scaling_unseen_categories_and_extra_columns():
    x = pd.DataFrame({"x": [0., 2., 4., 6.], "dominant_evt": [1, 1, 2, 2],
                      "ignition_month": [6, 6, 7, 7], "nbr_drop_patch": [9]*4,
                      "nbr_5_mean": [999]*4})
    model = build_ridge_pipeline(["x", "dominant_evt", "ignition_month", "nbr_drop_patch"])
    model.fit(x, [0., 1., 2., 3.])
    prep = model.named_steps["preprocess"]
    np.testing.assert_allclose(prep.named_transformers_["numeric"].mean_, [3.])
    transformed = prep.transform(x)
    np.testing.assert_allclose(transformed[:, 0].mean(), 0., atol=1e-12)
    np.testing.assert_allclose(transformed[:, 0].std(), 1.)
    unseen = x.iloc[:1].assign(dominant_evt=99, ignition_month=12, nbr_5_mean=-999)
    encoded = prep.transform(unseen)
    np.testing.assert_array_equal(encoded[:, 1:], 0.)
    assert np.isfinite(model.predict(unseen)).all()
    np.testing.assert_allclose(prep.named_transformers_["numeric"].mean_, [3.])
    np.testing.assert_allclose(model.predict(x), model.predict(x.assign(nbr_drop_patch=-10, nbr_5_mean=-999)))


def test_grouped_cv_fits_preprocessing_only_on_training_fires():
    groups = np.repeat(["A", "B", "C", "D", "E", "F"], 2)
    x = pd.DataFrame({"x": np.arange(12.)**2, "dominant_evt": groups,
                      "ignition_month": [6]*12})
    y = np.arange(12.)
    template = build_ridge_pipeline(list(x.columns))
    held_out = []
    for train, holdout in GroupKFold(n_splits=3).split(x, y, groups):
        assert set(groups[train]).isdisjoint(groups[holdout])
        model = clone(template).fit(x.iloc[train], y[train])
        prep = model.named_steps["preprocess"]
        np.testing.assert_allclose(prep.named_transformers_["numeric"].mean_, [x.iloc[train].x.mean()])
        assert set(prep.named_transformers_["categorical"].categories_[0]) == set(groups[train])
        assert np.isfinite(model.predict(x.iloc[holdout])).all()
        held_out.extend(holdout)
    assert sorted(held_out) == list(range(12))
    assert not hasattr(template.named_steps["preprocess"], "transformers_")


@pytest.mark.parametrize("features", [
    ["dominant_evt"],
    ["dominant_evt", "ignition_month", "dominant_evt"],
    ["dominant_evt", "ignition_month", "nbr_5_mean"],
    ["dominant_evt", "ignition_month", "fire_id"],
])
def test_invalid_feature_lists(features):
    with pytest.raises(ValueError):
        build_ridge_pipeline(features)


def test_saved_notebook_grouping_and_target_guard():
    notebook = json.loads((Path(__file__).resolve().parents[1] / "notebooks/phase5_ml.ipynb").read_text(encoding="utf-8"))
    source = "\n".join("".join(c.get("source", [])) for c in notebook["cells"])
    assert 'GroupKFold(n_splits=3)' in source
    assert 'groups = train["fire_id"]' in source
    assert 'training_fires.isdisjoint(holdout_fires)' in source
    assert 'filters=[("split", "in", ["train", "validation"])]' in source
    assert 'build_ridge_pipeline(features, alpha=1.0)' in source
