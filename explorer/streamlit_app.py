"""After the Fire: public, precomputed explorer. No model inference at runtime."""
import math
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st
from views import load_public, fire_view, selected_fire, LABELS

st.set_page_config(page_title='After the Fire | Wildfire Recovery Explorer',page_icon='🌲',layout='wide')
st.html('''<style>
.block-container {max-width:1280px;padding-top:2rem;padding-bottom:4rem;}
h1,h2,h3 {letter-spacing:-.035em;}
.hero {padding:clamp(24px,5vw,64px);border:1px solid #42564b;border-radius:24px;
background:radial-gradient(ellipse at 100% 0%,#6a3726 0%,transparent 55%),linear-gradient(145deg,#233c31,#111918 72%);margin-bottom:20px;}
.eyebrow {color:#F29B62;letter-spacing:.22em;font-size:.75rem;text-transform:uppercase;font-weight:700;}
.hero h1 {font-size:clamp(3.1rem,8vw,6.8rem);line-height:1;margin:.4em 0 .18em;color:#F4EEE3;}
.hero h2 {font-size:clamp(1.1rem,3vw,1.8rem);font-weight:400;color:#c2d5c7;margin:0;}
.hero p {max-width:650px;font-size:1.1rem;color:#e1ded5;line-height:1.6;}
.hero a {display:inline-block;color:#F29B62;text-decoration:none;border-bottom:1px solid #F29B62;padding:8px 0;}
.legend {height:10px;border-radius:8px;margin:10px 0 4px;}
.step {border-top:2px solid #cb8158;padding:18px 0;min-height:130px;}
.step h3 {color:#c2d5c7;font-size:1.25rem;}
</style>''')

@st.cache_data(show_spinner=False)
def data():
    return load_public()

try:
    fires,patches,per_fire,summary=data()
except (OSError,ValueError,KeyError) as exc:
    st.error('The presentation dataset could not be loaded. Restore the reviewed presentation bundle and restart.')
    st.stop()

st.html('''<div class="hero"><div class="eyebrow">A geospatial machine-learning field study</div>
<h1>After the Fire</h1><h2>Predicting California’s Wildfire Recovery</h2>
<p>Can we forecast vegetation conditions five growing seasons after a wildfire using only information available after the first?</p>
<a href="https://github.com/Vasant-Tholappa/wildfire-recovery-forecasting" target="_blank" rel="noopener noreferrer">Explore the project on GitHub ↗</a>
<p style="font-size:.85rem;color:#a6b7ac">37 unseen-year fires. A surprising model comparison. Start with the map below.</p></div>''')
hero=st.columns(3)
hero[0].metric('Wildfires studied',f"{summary['studied_fires']:,}")
reduction=next(r for r in summary['reductions'] if r['model']=='option_a_xgb')
hero[1].metric('Lower test MAE vs historical mean',f"{reduction['reduction_vs_training_mean_percent']:.2f}%")
hero[2].metric('Held-out patches · 37 fires',f"{summary['test_patches']:,}")
st.caption('Relative reduction in average absolute NBR error—not prediction accuracy or percentage ecological recovery. All results below are precomputed, from the independent 2020 cohort.')

def chart_style(fig,height=360):
    fig.update_layout(height=height,paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#F4EEE3',family='Arial'),margin=dict(l=15,r=25,t=25,b=35),
        legend=dict(orientation='h',y=-.2),hoverlabel=dict(bgcolor='#22352d'))
    fig.update_xaxes(gridcolor='#2c3932',zeroline=False)
    fig.update_yaxes(gridcolor='#2c3932',zeroline=False)
    return fig

st.divider()
st.markdown('## 01 / Explore the Fires')
st.write('From a statewide view to a 90-meter patch. Choose a fire to see where a forecast matched—and where it missed.')
if 'fire_pick' not in st.session_state:
    st.session_state.fire_pick=fires.iloc[0].fire_id

def map_choice():
    chosen=selected_fire(st.session_state.get('fire_map',{}),st.session_state.fire_pick)
    if chosen in set(fires.fire_id):st.session_state.fire_pick=chosen

markers=fires.copy()
markers['color']=[[242,155,98,240] if i==st.session_state.fire_pick else [110,179,146,200] for i in markers.fire_id]
markers['radius']=(markers.patches**.5)*90+1800
deck=pdk.Deck(map_style='https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json',
    initial_view_state=pdk.ViewState(latitude=37.35,longitude=-119.6,zoom=5.25,pitch=0),
    layers=[pdk.Layer('ScatterplotLayer',markers,id='fires',get_position='[lon, lat]',
        get_radius='radius',radius_min_pixels=6,radius_max_pixels=20,get_fill_color='color',
        pickable=True,auto_highlight=True,stroked=True,get_line_color=[242,238,227,140],line_width_min_pixels=1)],
    tooltip={'text':'{fire_name}\n{patches} evaluated patches\n{fire_id}'})
st.pydeck_chart(deck,height=450,key='fire_map',on_select=map_choice,selection_mode='single-object')
st.caption('Click a marker or use the selector. Markers are interior representative points of MTBS fire perimeters—not ignition sites. Size reflects evaluated patch count. Basemap: © CARTO, © OpenStreetMap contributors.')
names={r.fire_id:f'{r.fire_name.title()} · {r.patches:,} patches' for r in fires.itertuples()}
st.selectbox('Choose a 2020 fire',fires.fire_id.tolist(),format_func=lambda x:names[x],key='fire_pick')
fire_id=st.session_state.fire_pick
fire=fires.set_index('fire_id').loc[fire_id]
metrics=per_fire[per_fire.fire_id.eq(fire_id)].set_index('model')
st.markdown(f"### {fire.fire_name.title()}")
st.caption(f"MTBS {fire_id} · representative location {fire.lat:.3f}° N, {abs(fire.lon):.3f}° W · one selected tile, not the whole burn area")
cols=st.columns(3)
cols[0].metric('Evaluated 90 × 90 m patches',f'{int(fire.patches):,}')
cols[1].metric('A XGBoost fire MAE',f"{metrics.loc['option_a_xgb','mae']:.4f}")
cols[2].metric('MTBS fire area',f'{int(fire.fire_size_acres):,} acres')
if fire.patches<25:st.warning('Small sample: fewer than 25 evaluated patches. Interpret this fire’s summary cautiously.')
controls=st.columns(2)
model=controls[0].selectbox('Display model (frozen Option A)',list(LABELS),format_func=lambda x:LABELS[x],key='model_pick')
mode=controls[1].radio('Patch layer',['Predicted NBR5','Observed NBR5','Absolute error'],horizontal=True,key='layer_pick')
frame=fire_view(patches,fire_id,model,mode)
spread=max(frame.lon.max()-frame.lon.min(),frame.lat.max()-frame.lat.min(),.005)
zoom=max(8,min(13,math.log2(250/spread)))
patchdeck=pdk.Deck(map_style='https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json',
    initial_view_state=pdk.ViewState(latitude=float(frame.lat.mean()),longitude=float(frame.lon.mean()),zoom=zoom),
    layers=[pdk.Layer('PolygonLayer',frame[['polygon','color','display_value','patch_id']],id='patches',
        get_polygon='polygon',get_fill_color='color',get_line_color=[240,240,230,45],
        line_width_min_pixels=.25,stroked=True,pickable=True)],
    tooltip={'text':mode+' (unitless): {display_value}\n{patch_id}'})
st.pydeck_chart(patchdeck,height=440,key=f'patch_map_{fire_id}')
if mode=='Absolute error':
    st.html('<div class="legend" style="background:linear-gradient(90deg,#21302d,#ffa65f)"></div>')
    st.caption('Absolute error: 0 (dark) → 0.5+ (ember). Color saturates at 0.5; tooltip values are unchanged.')
else:
    st.html('<div class="legend" style="background:linear-gradient(90deg,#b75535,#928a66,#6dc097)"></div>')
    st.caption('Common NBR5 color scale: −1 (rust) → 0 → +1 (green). Colors saturate beyond endpoints; underlying values are unchanged.')
st.caption(f'All {len(frame):,} evaluated patches for this fire, no display sampling. Footprints follow the original 90 m EPSG:5070 grid, transformed to longitude/latitude. Higher NBR is spectral condition, not a verified recovery percentage.')
left,right=st.columns(2)
with left:
    st.markdown('#### Where did the predictions land?')
    lo=min(frame.nbr_5_mean.min(),frame[model].min());hi=max(frame.nbr_5_mean.max(),frame[model].max())
    scatter=go.Figure(go.Scattergl(x=frame.nbr_5_mean,y=frame[model],mode='markers',
        marker=dict(size=5,color='#f29b62',opacity=.45),name=LABELS[model],
        hovertemplate='Observed %{x:.4f}<br>Predicted %{y:.4f}<extra></extra>'))
    scatter.add_trace(go.Scatter(x=[lo,hi],y=[lo,hi],mode='lines',line=dict(color='#96b9a5',dash='dot'),name='Perfect agreement'))
    scatter.update_layout(xaxis_title='Observed NBR5',yaxis_title='Predicted NBR5')
    st.plotly_chart(chart_style(scatter),width='stretch')
with right:
    st.markdown('#### Same fire. Four approaches.')
    bar=go.Figure(go.Bar(x=[metrics.loc[m,'mae'] for m in LABELS],y=list(LABELS.values()),orientation='h',
        marker_color=['#f29b62','#78a98e','#728a96','#b6b99a'],hovertemplate='%{y}<br>MAE %{x:.4f}<extra></extra>'))
    bar.update_layout(xaxis_title='Saved fire MAE · lower is better',yaxis=dict(autorange='reversed'))
    st.plotly_chart(chart_style(bar),width='stretch')

st.divider()
st.markdown('## 02 / The Machine Learning Showdown')
st.markdown('### Deep learning won validation. XGBoost won the independent test.')
st.write('The more complex fusion model led during development, but the tree model generalized better on primary patch-weighted MAE in the unseen-year cohort. This was not a uniform win across fires.')
comparison=pd.DataFrame(summary['models']).set_index('model')
base=pd.DataFrame(summary['baselines']).set_index('baseline')
labels=list(LABELS.values())+['Historical training mean','Season 1 persistence']
scores=[comparison.loc[m,'mae'] for m in LABELS]+[base.loc[b,'mae'] for b in ('training_mean','nbr1_persistence')]
fig=go.Figure(go.Bar(x=scores,y=labels,orientation='h',text=[f'{x:.4f}' for x in scores],textposition='outside',
    marker_color=['#f29b62','#78a98e','#728a96','#b6b99a','#515c58','#515c58']))
fig.update_layout(xaxis_title='Independent 2020 MAE · unitless NBR · lower is better',yaxis=dict(autorange='reversed'),xaxis_range=[0,max(scores)*1.2])
st.plotly_chart(chart_style(fig,370),width='stretch')
st.caption(f"XGBoost reduced MAE by {reduction['reduction_vs_training_mean_percent']:.2f}% vs the original 2015–2017 target mean and {reduction['reduction_vs_persistence_percent']:.2f}% vs Season 1 persistence. Baselines were fixed, not fitted to 2020.")
reversal=go.Figure()
for m,c in [('option_a_xgb','#f29b62'),('option_a_fusion','#8db69d')]:
    reversal.add_trace(go.Scatter(x=['Development validation · 2018–19','Independent test · 2020'],
        y=[comparison.loc[m,'historical_validation_mae'],comparison.loc[m,'mae']],
        name=LABELS[m],mode='lines+markers+text',line=dict(color=c,width=3),marker=dict(size=12),
        text=[f"{comparison.loc[m,'historical_validation_mae']:.4f}",f"{comparison.loc[m,'mae']:.4f}"],textposition='top center'))
reversal.update_layout(yaxis_title='MAE (different cohorts)',yaxis_range=[.06,.09])
st.plotly_chart(chart_style(reversal,300),width='stretch')
with st.expander('Under the hood · all eight frozen models and four metrics'):
    table=comparison.reset_index()[['model','mae','rmse','r2','equal_fire_mae']]
    table['Strategy']=table.model.map(lambda m:'A · original fitted model' if m.startswith('option_a') else 'B · fresh 2015–19 fit')
    table['model']=table.model.map(lambda m:('A ' if m.startswith('option_a') else 'B ')+LABELS[m.replace('option_b','option_a')])
    st.dataframe(table.rename(columns={'model':'Model','mae':'Patch MAE','rmse':'RMSE','r2':'R²','equal_fire_mae':'Equal-fire MAE'}),hide_index=True,width='stretch')
    st.write('Option B XGBoost has the lowest equal-fire MAE. Option A XGBoost wins primary patch-weighted MAE. Equal-fire averaging gives a small fire the same weight as a large sampled fire; neither weighting makes nearby patches statistically independent.')
    st.caption('A: original training on 2015–2017, validation-selected configurations. B: predeclared fresh fit on 2015–2019 with frozen settings. B has no independent 2018–2019 validation score. Selecting a serving candidate after testing is a post-test choice, not another independent test.')

st.divider()
st.markdown('## 03 / Why It Matters')
for col,title,body in zip(st.columns(3),['01 · After Season 1','02 · Look toward Season 5','03 · Look more closely'],[
    'Use early spectral condition, vegetation, terrain and fire characteristics.',
    'Forecast fifth-season NBR: a spectral vegetation-condition indicator.',
    'Potentially prioritize geographically targeted monitoring and investigation.']):
    with col:st.html(f'<div class="step"><h3>{title}</h3><p>{body}</p></div>')
st.write('Predictive skill is demonstrated here. Operational benefits are not. These forecasts alone do not justify restoration spending, land-use decisions or claims about ecological recovery.')
with st.expander('Scientific limitations & source transparency'):
    st.markdown('''- One selected tile per fire, retrospectively screened eligible patches—not every burned hectare.
- Fire-disjoint chronological cohorts contain clustered patches. Limited training/validation context overlap was documented.
- Repeated development-validation selection can favor optimistic configurations; 2020 is one independent fire-year cohort.
- NBR5 is not a percentage of ecological recovery. Input shifts and error patterns are associations, not established causes.
- Fusion versus trees does not isolate the causal or predictive contribution of spatial arrangement.
- All maps/charts are precomputed results. No models, private databases, Earth Engine or live FastAPI service are called.
- MTBS names/locations: USDA Forest Service / US Geological Survey, Monitoring Trends in Burn Severity. Locations are not legal boundaries.
- NBR observations derive from USGS Landsat; predictions are project-generated frozen outputs.''')
    st.markdown('[MTBS provenance and terms](https://developers.google.com/earth-engine/datasets/catalog/USFS_GTAC_MTBS_burned_area_boundaries_v1) · [USGS Landsat reuse policy](https://www.usgs.gov/faqs/are-there-any-restrictions-use-or-redistribution-landsat-data)')
st.caption('After the Fire · A reproducible geospatial ML study · Precomputed research demonstration, not an operational forecasting service.')
