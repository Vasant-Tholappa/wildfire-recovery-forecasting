"""Small synthetic XGBoost checks: no production fits or Optuna execution."""
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from wildfire_recovery_forecasting.modeling import build_xgb_pipeline
from wildfire_recovery_forecasting.pipeline import dataset_roles


def test_roles_and_initial_parameters():
    roles = dataset_roles(["nbr_drop_patch", "forest_fraction"])
    pipe = build_xgb_pipeline(roles["predictors"])
    prep = pipe.named_steps["preprocess"]
    assert prep.transformers[0][1] == "passthrough"
    assert len(prep.transformers[0][2]) == 32
    assert "nbr_drop_patch" in prep.transformers[0][2]
    assert prep.transformers[1][2] == ["dominant_evt", "ignition_month"]
    assert set(prep.transformers[0][2] + prep.transformers[1][2]) == set(roles["predictors"])
    assert prep.remainder == "drop"
    params = pipe.named_steps["model"].get_params()
    expected = dict(n_estimators=300, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                    reg_lambda=5.0, objective="reg:squarederror",
                    tree_method="hist", random_state=42, n_jobs=4)
    assert {k: params[k] for k in expected} == expected
    tuned = clone(pipe).set_params(model__max_depth=9, model__min_child_weight=13)
    assert tuned.named_steps["model"].max_depth == 9
    assert pipe.named_steps["model"].max_depth == 4


def test_grouped_fits_unknown_categories_no_scaling():
    groups = np.repeat(["a", "b", "c", "d", "e", "f"], 2)
    x = pd.DataFrame(dict(value=np.arange(12.)*10, dominant_evt=groups,
                          ignition_month=[7]*12, nbr_drop_patch=np.arange(12.)/10))
    template = build_xgb_pipeline(list(x)).set_params(
        model__n_estimators=2, model__max_depth=2, model__min_child_weight=1, model__n_jobs=1)
    held_out = []
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
        preds = pipe.predict(x.iloc[holdout])
        assert np.isfinite(preds).all()
        np.testing.assert_allclose(preds, pipe.predict(x.iloc[holdout].assign(nbr_5_mean=999)))
        held_out.extend(holdout)
    assert sorted(held_out) == list(range(12))
    assert not hasattr(template.named_steps["preprocess"], "transformers_")


@pytest.mark.parametrize("features", [
    ["dominant_evt"],
    ["dominant_evt", "ignition_month", "dominant_evt"],
    ["dominant_evt", "ignition_month", "nbr_5_mean"],
    ["dominant_evt", "ignition_month", "fire_id"],
])
def test_invalid_allowlists(features):
    with pytest.raises(ValueError): build_xgb_pipeline(features)


def test_saved_objectives_are_training_only_and_clone_preprocessing():
    path = Path(__file__).resolve().parents[1] / "notebooks/phase5_ml.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    functions = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code": continue
        tree = ast.parse("".join(cell["source"]))
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in {
                "xgb_objective", "xgb_objective_wide", "xgb_final_objective"}:
                functions[node.name] = ast.unparse(node)
    assert len(functions) == 3
    for source in functions.values():
        assert "cv_splits" in source
        assert "clone(xgb_pipeline)" in source
        assert "X_train.iloc" in source and "y_train.iloc" in source
        assert "mean_absolute_error" in source
        assert "X_val" not in source and "y_val" not in source
        assert "test" not in source
