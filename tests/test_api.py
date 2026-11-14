import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from wildfire_recovery_forecasting import api


def example():
    return dict(nbr_pre_mean=.5, dominant_evt=3008, dominant_evt_fraction=.9,
                tree_cover_mean=50., tree_height_mean=10., elevation_mean=1000.,
                slope_mean=20., northness_mean=0., eastness_mean=0., fire_size_acres=10000.,
                ignition_month=8, mtbs_unburned_low_fraction=.1, mtbs_low_fraction=.2,
                mtbs_moderate_fraction=.3, mtbs_high_fraction=.4,
                mtbs_increased_greenness_fraction=0., nbr_1_mean=.2, nbr_drop_patch=.3)


class SyntheticPipeline:
    def predict(self, frame):
        assert list(frame.columns) == list(api.FEATURES)
        assert frame.iloc[0].nbr_pre_mean == .5
        return np.array([1.2])  # No prediction clipping.


def test_health_prediction_order_and_docs():
    with TestClient(api.create_app(lambda _: SyntheticPipeline())) as client:
        assert client.get('/health').json()['model_id'] == api.MODEL_ID
        request = dict(reversed(list(example().items())))
        assert client.post('/predict', json=request).json()['nbr_5_mean'] == 1.2
        assert client.get('/docs').status_code == 200


@pytest.mark.parametrize('key,value', [('ignition_month',13),('dominant_evt',3008.0),
    ('nbr_pre_mean','0.5'),('nbr_pre_mean',True),('slope_mean',-1),('tree_cover_mean',101),
    ('mtbs_high_fraction',2),('fire_size_acres',0)])
def test_invalid(key,value):
    data=example();data[key]=value
    with TestClient(api.create_app(lambda _: SyntheticPipeline())) as c:
        assert c.post('/predict',json=data).status_code==422


@pytest.mark.parametrize('change',['missing','extra','batch','null'])
def test_schema(change):
    data=example()
    if change=='missing':del data['nbr_pre_mean']
    if change=='extra':data['nbr_5_mean']=.4
    if change=='batch':data=[data]
    if change=='null':data['nbr_pre_mean']=None
    with TestClient(api.create_app(lambda _: SyntheticPipeline())) as c:
        assert c.post('/predict',json=data).status_code==422


@pytest.mark.parametrize('value',[float('nan'),float('inf'),-float('inf')])
def test_nonfinite(value):
    data=example();data['nbr_pre_mean']=value
    with pytest.raises(ValueError):api.PredictRequest(**data)
    with TestClient(api.create_app(lambda _: SyntheticPipeline())) as c:
        response=c.post('/predict',content=json.dumps(data),headers={'Content-Type':'application/json'})
        assert response.status_code==422


def test_unavailable():
    def fail(_):raise FileNotFoundError('secret/private/path')
    with TestClient(api.create_app(fail)) as c:
        assert c.get('/health').status_code==503
        response=c.post('/predict',json=example())
        assert response.status_code==503 and 'secret' not in response.text


def test_hash_before_deserialization(tmp_path):
    p=tmp_path/'bad.pkl';p.write_bytes(b'untrusted')
    with patch.object(api.cloudpickle,'loads',side_effect=AssertionError('must not load')):
        with pytest.raises(ValueError,match='hash'):api.load_pipeline(p)


def test_nonfinite_prediction():
    class Bad:
        def predict(self,_):return [np.nan]
    with TestClient(api.create_app(lambda _:Bad())) as c:
        assert c.post('/predict',json=example()).status_code==503


@pytest.mark.skipif(not os.environ.get('WILDFIRE_REAL_SMOKE'),reason='Explicit trusted-model smoke only')
def test_real_pipeline_matches_api():
    path=Path(os.environ['WILDFIRE_MODEL_PATH'])
    before=hashlib.sha256(path.read_bytes()).hexdigest()
    model=api.load_pipeline(path)
    import pandas as pd
    request=example()  # Synthetic inputs, not held-out targets or rows.
    direct=float(model.predict(pd.DataFrame([request],columns=api.FEATURES))[0])
    with TestClient(api.create_app()) as c:
        assert c.get('/health').status_code==200
        assert c.post('/predict',json=request).json()['nbr_5_mean']==direct
    assert hashlib.sha256(path.read_bytes()).hexdigest()==before==api.MODEL_SHA256
