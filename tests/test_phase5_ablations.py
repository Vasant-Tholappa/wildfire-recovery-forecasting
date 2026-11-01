"""Small synthetic checks for the frozen Phase 5 ablation feature contracts."""
import numpy as np
import pandas as pd
import pytest

from wildfire_recovery_forecasting.modeling import build_rf_pipeline, build_xgb_pipeline
from wildfire_recovery_forecasting.pipeline import dataset_roles


@pytest.mark.parametrize('builder', [build_rf_pipeline, build_xgb_pipeline])
def test_no_climate_preprocessing_preserves_18_inputs_and_ignores_removed_columns(builder):
    full = dataset_roles(['nbr_drop_patch'])['predictors']
    climate = [f for f in full if f.startswith(('hist_', 'pre_s1_', 's1_'))]
    retained = [f for f in full if f not in climate]
    assert len(full) == 34 and len(climate) == 16 and len(retained) == 18
    assert {'nbr_pre_mean', 'nbr_1_mean', 'nbr_drop_patch'} <= set(retained)
    assert not {'nbr_5_mean', 'fire_id', 'split'} & set(retained)
    frame = pd.DataFrame({name: [1., 2., 3.] for name in full})
    frame['dominant_evt'] = [3014, 3129, 3014]
    frame['ignition_month'] = [6, 7, 8]
    prep = builder(retained).named_steps['preprocess']
    expected = prep.fit_transform(frame)
    changed = frame.copy()
    changed[climate] = -99999.
    actual = prep.transform(changed)
    if hasattr(expected, 'toarray'):
        expected, actual = expected.toarray(), actual.toarray()
    np.testing.assert_array_equal(actual, expected)
    assert len(prep.transformers_[0][2]) == 16
    assert prep.named_transformers_['categorical'].handle_unknown == 'ignore'
