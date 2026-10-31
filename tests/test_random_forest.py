"""Tiny synthetic tests; never fit the production dataset."""
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from wildfire_recovery_forecasting.modeling import build_rf_pipeline
from wildfire_recovery_forecasting.pipeline import dataset_roles


def test_rf_roles_and_checkpoint_parameters():
    roles = dataset_roles(["nbr_drop_patch", "forest_fraction"])
    pipe = build_rf_pipeline(roles["predictors"])
    prep = pipe.named_steps["preprocess"]
    assert prep.transformers[0][1] == "passthrough"
    assert len(prep.transformers[0][2]) == 32
    assert "nbr_drop_patch" in prep.transformers[0][2]
    assert prep.transformers[1][2] == ["dominant_evt", "ignition_month"]
    assert prep.remainder == "drop"
    assert set(prep.transformers[0][2] + prep.transformers[1][2]) == set(roles["predictors"])
    settings = pipe.named_steps["model"].get_params()
    expected = dict(n_estimators=100, max_depth=16, min_samples_leaf=10,
                    max_features=0.8, bootstrap=True, random_state=42,
                    n_jobs=-1, min_samples_split=2, criterion="squared_error")
    assert {key: settings[key] for key in expected} == expected


def test_rf_fold_encoding_unseen_categories_and_no_scaling():
    groups = np.repeat(["a", "b", "c", "d", "e", "f"], 2)
    x = pd.DataFrame(dict(value=np.arange(12.)*100, dominant_evt=groups,
                          ignition_month=[7]*12, nbr_drop_patch=np.arange(12.)/10))
    template = build_rf_pipeline(list(x))
    # Only tiny synthetic fits, not the 100-tree production experiment.
    template.set_params(model__n_estimators=3, model__min_samples_leaf=1, model__n_jobs=1)
    for train, holdout in GroupKFold(3).split(x, groups=groups):
        assert set(groups[train]).isdisjoint(groups[holdout])
        pipe = clone(template).fit(x.iloc[train], np.arange(12.)[train])
        prep = pipe.named_steps["preprocess"]
        encoded = prep.transform(x.iloc[holdout])
        if hasattr(encoded, "toarray"): encoded = encoded.toarray()
        np.testing.assert_allclose(encoded[:, :2], x.iloc[holdout][["value", "nbr_drop_patch"]])
        encoder = prep.named_transformers_["categorical"]
        assert set(encoder.categories_[0]) == set(groups[train])
        np.testing.assert_array_equal(encoded[:, 2:2+len(encoder.categories_[0])], 0)
        predictions = pipe.predict(x.iloc[holdout])
        assert np.isfinite(predictions).all()
        np.testing.assert_allclose(predictions, pipe.predict(x.iloc[holdout].assign(nbr_5_mean=999)))
        repeat = clone(template).fit(x.iloc[train], np.arange(12.)[train])
        np.testing.assert_allclose(predictions, repeat.predict(x.iloc[holdout]))
    assert not hasattr(template.named_steps["preprocess"], "transformers_")


@pytest.mark.parametrize("features", [
    ["dominant_evt"],
    ["dominant_evt", "ignition_month", "dominant_evt"],
    ["dominant_evt", "ignition_month", "nbr_5_mean"],
    ["dominant_evt", "ignition_month", "fire_id"],
])
def test_rf_rejects_invalid_allowlists(features):
    with pytest.raises(ValueError): build_rf_pipeline(features)
