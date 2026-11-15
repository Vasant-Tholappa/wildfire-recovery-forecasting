"""Presentation-only checks; no model loads or new evaluation."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from explorer.views import DATA, LABELS, load_public, fire_view, selected_fire


def test_bundle():
    fires,patches,per_fire,summary=load_public()
    assert fires.fire_id.is_unique and fires.patches.sum()==30606
    assert patches.lon.between(-125,-113).all() and patches.lat.between(32,43).all()
    assert len(per_fire)==37*8 and summary['studied_fires']==151
    manifest=json.loads((DATA/'provenance.json').read_text())
    for name,digest in manifest['files'].items():
        assert hashlib.sha256((DATA/name).read_bytes()).hexdigest()==digest
    assert sum(p.stat().st_size for p in DATA.iterdir())<5_000_000


def test_fire_model_layers_and_selection():
    fires,patches,_,_=load_public()
    for fire in fires.fire_id:
        for model in LABELS:
            frame=fire_view(patches,fire,model,'Predicted NBR5')
            assert len(frame)==int(fires.set_index('fire_id').loc[fire,'patches'])
            np.testing.assert_array_equal(frame.value,frame[model])
        obs=fire_view(patches,fire,'option_a_xgb','Observed NBR5')
        np.testing.assert_array_equal(obs.value,obs.nbr_5_mean)
    event={'selection':{'objects':{'fires':[{'fire_id':fires.iloc[1].fire_id}]}}}
    assert selected_fire(event,fires.iloc[0].fire_id)==fires.iloc[1].fire_id
    assert selected_fire({},fires.iloc[0].fire_id)==fires.iloc[0].fire_id


def test_geographic_roundtrip():
    pyproj=pytest.importorskip('pyproj')  # Builder QA only; not an app dependency.
    patches=load_public()[1]
    rc=patches.patch_id.str.extract(r'_r(-?\d+)_c(-?\d+)$').astype(int)
    transform=pyproj.Transformer.from_crs(4326,5070,always_xy=True)
    x,y=transform.transform(patches.lon,patches.lat)
    np.testing.assert_allclose(x,(rc[1]+.5)*90,rtol=0,atol=1e-5)
    np.testing.assert_allclose(y,(rc[0]+.5)*90,rtol=0,atol=1e-5)
    for i,(dx,dy) in enumerate(((0,0),(1,0),(1,1),(0,1))):
        x,y=transform.transform(patches[f'x{i}'],patches[f'y{i}'])
        np.testing.assert_allclose(x,(rc[1]+dx)*90,rtol=0,atol=1e-5)
        np.testing.assert_allclose(y,(rc[0]+dy)*90,rtol=0,atol=1e-5)


def test_saved_metrics_when_private_sources_available():
    root=Path(__file__).resolve().parents[1]
    path=root/'data/processed/phase7_test_v1/comparison.csv'
    if not path.exists():return  # Public checkout only needs the presentation bundle.
    summary=load_public()[3]
    original=pd.read_csv(path).set_index('model')
    public=pd.DataFrame(summary['models']).set_index('model')
    for col in ('mae','rmse','r2','equal_fire_mae','historical_validation_mae'):
        np.testing.assert_allclose(public[col],original.loc[public.index,col],rtol=0,atol=1e-14,equal_nan=True)


def test_app_and_widgets():
    app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'explorer/streamlit_app.py'),default_timeout=30).run()
    assert not app.exception
    fires=load_public()[0]
    app.selectbox(key='fire_pick').set_value(fires.iloc[-1].fire_id).run()
    assert not app.exception
    app.selectbox(key='model_pick').set_value('option_a_fusion').run()
    app.radio(key='layer_pick').set_value('Absolute error').run()
    assert not app.exception
    assert any(fires.iloc[-1].fire_name.title() in m.value for m in app.markdown)
