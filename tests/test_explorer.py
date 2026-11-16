"""Presentation-only checks; no model loads or new evaluation."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from explorer.views import DATA, LABELS, load_public, fire_view, selected_fire, MODES, COLORS


def test_scatter_svg_preserves_all_points_for_largest_fire():
    import base64

    def values(encoded):
        if isinstance(encoded, dict):
            return np.frombuffer(base64.b64decode(encoded['bdata']), dtype=encoded['dtype'])
        return np.asarray(encoded)

    fires, patches, _, _ = load_public()
    fire = fires.loc[fires.patches.idxmax(), 'fire_id']
    expected = patches[patches.fire_id.eq(fire)]
    app = AppTest.from_file(str(Path('explorer/streamlit_app.py').resolve()), default_timeout=30).run()
    app.selectbox(key='fire_pick').set_value(fire).run()
    for model in LABELS:
        app.selectbox(key='model_pick').set_value(model).run()
        assert not app.exception
        charts = [json.loads(c.proto.spec) for c in app.get('plotly_chart')]
        assert all(t['type'] in {'scatter', 'bar'} for c in charts for t in c['data'])
        points, reference = charts[0]['data']
        assert points['type'] == 'scatter' and points['mode'] == 'markers'
        np.testing.assert_array_equal(values(points['x']), expected.nbr_5_mean)
        np.testing.assert_array_equal(values(points['y']), expected[model])
        assert points['marker'] == dict(size=5, color=COLORS[model], opacity=.55)
        assert points['hovertemplate'] == 'Observed %{x:.4f}<br>Predicted %{y:.4f}<extra></extra>'
        assert reference['mode'] == 'lines' and reference['x'] == reference['y']
        assert charts[0]['layout']['yaxis']['scaleanchor'] == 'x'


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
            frame=fire_view(patches,fire,model,MODES[0])
            assert len(frame)==int(fires.set_index('fire_id').loc[fire,'patches'])
            np.testing.assert_array_equal(frame.value,frame[model])
        obs=fire_view(patches,fire,'option_a_xgb',MODES[1])
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
    app.radio(key='layer_pick').set_value(MODES[2]).run()
    assert not app.exception
    assert any(fires.iloc[-1].fire_name.title() in m.value for m in app.markdown)


def test_year1_exact_alignment_and_unchanged_predictions():
    patches=pd.read_csv(DATA/'patches.csv.gz',float_precision='round_trip')
    source=Path('data/processed/phase6_cnn_dataset_v1/test/tabular.parquet')
    if not source.exists():pytest.skip('Private source available only to maintainer')
    early=pd.read_parquet(source,columns=['fire_id','patch_id','nbr_1_mean'])
    assert patches[['fire_id','patch_id']].equals(early[['fire_id','patch_id']])
    np.testing.assert_array_equal(patches.nbr_1_mean,early.nbr_1_mean)
    for mode in MODES:
        view=fire_view(patches,patches.iloc[0].fire_id,'option_a_xgb',mode)
        assert np.isfinite(view.value).all()


def test_selection_ignores_empty_ambiguous_unknown_and_overlay_events():
    current='first';allowed={'first','second'}
    for objects in ([],[{}],[{'fire_id':'unknown'}],[{'fire_id':'first'},{'fire_id':'second'}]):
        assert selected_fire({'selection':{'objects':{'fires':objects}}},current,allowed)==current
    assert selected_fire({'selection':{'objects':{'selected-highlight':[{'fire_id':'second'}]}}},current,allowed)==current
    assert selected_fire({'selection':{'objects':{'fires':[{'fire_id':'second'}]}}},current,allowed)=='second'


def test_same_vegetation_scale_and_individual_voice():
    patches=load_public()[1].iloc[:2].copy()
    patches['nbr_1_mean']=.25;patches['nbr_5_mean']=.25;patches['option_a_xgb']=.25
    colors=[fire_view(patches,patches.iloc[0].fire_id,'option_a_xgb',m).color.tolist() for m in MODES[:2]]
    assert colors[0]==colors[1]
    import re
    source=Path('explorer/streamlit_app.py').read_text(encoding='utf-8')
    source=source.replace('What do the results tell us?','').replace('What if we could predict what comes next?','')  # Approved rhetorical exception.
    source=re.sub(r'https?://[^\s\"\']+', '', source)  # URLs are not project voice.
    assert not re.search(r'\b(we|our|us)\b',source.replace('US Geological Survey','United States Geological Survey'),re.I)
    assert "scaleanchor='x',scaleratio=1" in source
    assert len(set(COLORS.values()))==4


def test_hero_asset_and_timeline_placement():
    from PIL import Image
    root=Path(__file__).resolve().parents[1]
    asset=root/'explorer/assets/bobcat_landsat.webp'
    with Image.open(asset) as image:
        assert image.size==(900,1000)
        image.verify()
    assert asset.stat().st_size<400_000
    source=(root/'explorer/streamlit_app.py').read_text(encoding='utf-8')
    assert source.count('class="timeline"')==1
    assert source.count("with st.expander('Why test on wildfires from 2020?')")==1
    assert source.index("key='fire_map'")<source.index('class="timeline"')<source.index("cols[0].metric('Landscape patches analyzed'")
    assert 'California wildfires have burned millions of acres in recent years.' in source and 'comparable year-to-date intervals' in source
    assert 'alt="False-color Landsat' in source and 'not a project prediction' in source


def test_best_fire_saved_scores_and_missing_ties():
    from explorer.views import best_fire_models
    fires,_,scores,_=load_public()
    non_xgb=[]
    for fire in fires.fire_id:
        winners,value=best_fire_models(scores,fire)
        expected=scores[scores.fire_id.eq(fire)&scores.model.isin(LABELS)].set_index('model').mae
        assert value==expected.min()
        assert winners==[m for m in LABELS if expected[m]==value]
        if 'option_a_xgb' not in winners:non_xgb.append(fire)
    assert non_xgb
    fire=fires.iloc[0].fire_id
    subset=scores[scores.fire_id.eq(fire)&scores.model.isin(LABELS)].copy()
    subset['mae']=.1
    assert best_fire_models(subset,fire)==(list(LABELS),.1)
    for bad in (subset.iloc[:-1],pd.concat([subset,subset.iloc[:1]]),subset.assign(mae=np.nan),subset.assign(mae=-1)):
        with pytest.raises(ValueError):best_fire_models(bad,fire)


def test_map_change_notification_state():
    from explorer.views import apply_map_selection
    state={'fire_pick':'a'}
    event={'selection':{'objects':{'fires':[{'fire_id':'b'}]}}}
    assert not apply_map_selection(state,{}, {'a','b'})
    assert 'map_selection_notice' not in state
    assert apply_map_selection(state,event,{'a','b'})
    assert state['fire_pick']=='b' and state.pop('map_selection_notice')=='b'
    assert not apply_map_selection(state,event,{'a','b'})
    assert 'map_selection_notice' not in state


def test_part1_summary_independent_of_display_model():
    from explorer.views import best_fire_models
    fires,_,scores,_=load_public()
    fire=next(f for f in fires.fire_id if 'option_a_xgb' not in best_fire_models(scores,f)[0])
    app=AppTest.from_file(str(Path('explorer/streamlit_app.py').resolve()),default_timeout=30).run()
    app.selectbox(key='fire_pick').set_value(fire).run()
    winners,value=best_fire_models(scores,fire)
    def summary():return next(m.value for m in app.metric if m.label=='Lowest prediction error on this fire')
    assert summary()==f'{value:.4f} NBR'
    assert next(m.value for m in app.metric if m.label=='Landscape patches analyzed')==f'{int(fires.set_index("fire_id").loc[fire,"patches"]):,}'
    for model in ('option_a_cnn_only','option_a_fusion'):
        app.selectbox(key='model_pick').set_value(model).run()
        assert not app.exception and summary()==f'{value:.4f} NBR'
    source=Path('explorer/streamlit_app.py').read_text(encoding='utf-8')
    assert "Explore California's wildfires, then examine groups of landscape patches within each fire to see how well the models predicted vegetation conditions." in source
    assert 'Click an orange marker or search for a wildfire by name. Zoom in if a marker is difficult to select.' in source
    assert 'How the 2020 wildfire predictions were tested' in source
    assert '{int(fire.patches):,}</strong> landscape patches analyzed' in source


def test_three_modes_color_mapping_and_legacy_state():
    from explorer.views import VEGETATION_BOUNDS
    assert len(MODES)==3 and MODES[0]=='Predicted vegetation — Year 5'
    assert VEGETATION_BOUNDS==(-.2,.8)
    patches=load_public()[1].iloc[:5].copy()
    for model in LABELS:
        observed=fire_view(patches,patches.iloc[0].fire_id,model,MODES[1])
        np.testing.assert_array_equal(observed.value,patches.nbr_5_mean)
        pred=fire_view(patches,patches.iloc[0].fire_id,model,MODES[0])
        np.testing.assert_array_equal(pred.value,patches[model])
        err=fire_view(patches,patches.iloc[0].fire_id,model,MODES[2])
        expected=np.abs(patches[model]-patches.nbr_5_mean)
        np.testing.assert_array_equal(err.value,expected)
        rgb=np.array([33,48,45])+(np.array([255,166,95])-np.array([33,48,45]))*np.clip(expected.to_numpy()/.5,0,1)[:,None]
        assert err.color.tolist()==[list(map(int,c))+[220] for c in rgb]
    patches['nbr_5_mean']=[-.3,-.2,.3,.8,.9]
    patches['option_a_rf']=patches.nbr_5_mean
    obs=fire_view(patches,patches.iloc[0].fire_id,'option_a_xgb',MODES[1])
    pred=fire_view(patches,patches.iloc[0].fire_id,'option_a_rf',MODES[0])
    assert obs.color.tolist()==pred.color.tolist()
    assert obs.color.iloc[0]==obs.color.iloc[1] and obs.color.iloc[3]==obs.color.iloc[4]
    app=AppTest.from_file(str(Path('explorer/streamlit_app.py').resolve()),default_timeout=30)
    app.session_state['layer_pick']='Early vegetation — Year 1'
    app.run()
    assert not app.exception and app.radio(key='layer_pick').value==MODES[0]
    assert app.radio(key='layer_pick').options==MODES
    for mode in MODES:
        app.radio(key='layer_pick').set_value(mode).run()
        assert not app.exception


def test_showdown_four_models_baselines_and_preserved_reversal():
    import re
    app=AppTest.from_file(str(Path('explorer/streamlit_app.py').resolve()),default_timeout=30).run()
    assert not app.exception
    summary=load_public()[3]
    saved=pd.DataFrame(summary['models']).set_index('model')
    ordered=sorted(LABELS,key=lambda m:saved.loc[m,'mae'])
    table=app.dataframe[0].value
    assert len(table)==4 and table['Model'].tolist()==[LABELS[m] for m in ordered]
    for public,internal in [('Patch MAE','mae'),('RMSE','rmse'),('R²','r2'),('Equal-fire MAE','equal_fire_mae')]:
        np.testing.assert_array_equal(table[public],saved.loc[ordered,internal])
    expanders={e.label:e for e in app.expander}
    assert expanders['What are the two baselines?'].proto.expanded is False
    assert expanders['Explore detailed model performance'].proto.expanded is False
    baseline_text=' '.join(m.value for m in expanders['What are the two baselines?'].markdown)
    assert 'Predicts the same vegetation-index value for every patch, based on the average observed in the original training data.' in baseline_text
    assert "Assumes each patch's vegetation condition stays the same between the first and fifth growing seasons." in baseline_text
    visible=' '.join(m.value for m in app.markdown)+' '+' '.join(e.label for e in app.expander)+' '+table.to_string()
    assert not re.search(r'Option [AB]|Strategy [AB]|eight models|retraining',visible,re.I)
    charts=[json.loads(c.proto.spec) for c in app.get('plotly_chart')]
    main=next(c for c in charts if len(c['data'])==1 and len(c['data'][0].get('y',[]))==6)
    trace=main['data'][0]
    assert trace['y']==[LABELS[m] for m in ordered]+['Baseline 1 · Historical average','Baseline 2 · No change']
    baselines=pd.DataFrame(summary['baselines']).set_index('baseline')
    assert trace['x']==[saved.loc[m,'mae'] for m in ordered]+[baselines.loc[b,'mae'] for b in ['training_mean','nbr1_persistence']]
    assert trace['marker']['color']==[COLORS[m] for m in ordered]+['#69736e','#69736e']
    reversal=charts[-1]
    for trace,model in zip(reversal['data'],['option_a_xgb','option_a_fusion']):
        assert trace['y']==[saved.loc[model,'historical_validation_mae'],saved.loc[model,'mae']]
        assert trace['line']['color']==COLORS[model]
    assert any(m.value=='51.45%' for m in app.metric)
    assert any(m.value=='63.39%' for m in app.metric)


def test_closing_cards_and_readable_details():
    app=AppTest.from_file(str(Path('explorer/streamlit_app.py').resolve()),default_timeout=30).run()
    assert not app.exception
    details=next(e for e in app.expander if e.label=='About the data and how to interpret these results')
    assert details.proto.expanded is False
    import re,base64,xml.etree.ElementTree as ET
    detail_html=' '.join(e.proto.body for e in details.get('html'))
    questions=['**'+q+'**' for q in re.findall(r'<p class="decision-question">(.*?)</p>',detail_html)]
    assert detail_html.count('class="decision-answer"')==5
    assert detail_html.count('href=')==2
    assert 'Project technical documentation' not in detail_html
    assert questions==['**What exactly was predicted?**','**How much of each wildfire was analyzed?**','**What do the results tell us?**','**Does this measure complete forest recovery?**','**Are these live predictions?**']
    html=' '.join(e.proto.body for e in app.get('html'))
    assert html.count('<article class="decision-card">')==4
    icons=re.findall(r'data:image/svg\+xml;base64,([^"]+)',html)
    assert len(icons)==4
    for icon in icons:
        svg=ET.fromstring(base64.b64decode(icon))
        assert svg.attrib['stroke']=='#F29B62'
        assert svg.attrib['width']==svg.attrib['height']=='34'
        assert len(svg)>0
    assert html.count('class="decision-icon"')==4
    for title in ['Identify potential recovery hotspots','Prioritize field monitoring','Inform restoration planning','Support watershed protection']:
        assert title in html
    assert 'grid-auto-rows:1fr' in html and '@media(max-width:520px)' in html
    assert 'https://www.usgs.gov/faqs/are-there-any-restrictions-use-or-redistribution-landsat-data' in html
    assert 'USFS_GTAC_MTBS_burned_area_boundaries_v1' in html
    assert 'target="_blank" rel="noopener noreferrer"' in html
