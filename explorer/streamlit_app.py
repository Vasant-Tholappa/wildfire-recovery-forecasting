"""After the Fire: public, precomputed explorer. No model inference at runtime."""
import math
import base64
from pathlib import Path
from html import escape
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st
from views import load_public, fire_view, selected_fire, LABELS, COLORS, MODES, best_fire_models, apply_map_selection

st.set_page_config(page_title='After the Fire | Wildfire Recovery Explorer',page_icon='🌲',layout='wide')
st.html('''<style>
.block-container {max-width:1280px;padding-top:2rem;padding-bottom:4rem;}
h1,h2,h3 {letter-spacing:-.035em;}
.hero {padding:clamp(20px,3vw,36px);border:1px solid #42564b;border-radius:24px;
background:radial-gradient(ellipse at 100% 0%,#6a3726 0%,transparent 55%),linear-gradient(145deg,#233c31,#111918 72%);margin-bottom:20px;}
.eyebrow {color:#F29B62;letter-spacing:.22em;font-size:.75rem;text-transform:uppercase;font-weight:700;}
.hero h1 {font-size:clamp(1.8rem,2.8vw,2.65rem);line-height:1.08;margin:.2em 0 .18em;color:#F4EEE3;}
.hero h2 {font-size:clamp(1.1rem,3vw,1.8rem);font-weight:400;color:#c2d5c7;margin:0;}
.hero p {max-width:650px;font-size:1rem;color:#e1ded5;line-height:1.5;margin:.8rem 0;}
.hero a {color:#F29B62;text-decoration:underline;text-underline-offset:3px;}
.hero-top {display:flex;justify-content:space-between;align-items:center;gap:16px;margin-bottom:18px;}
.hero-links {display:flex;align-items:center;justify-content:flex-end;gap:16px;flex-wrap:wrap;}
.hero-wordmark {color:#F4EEE3;font-size:.75rem;font-weight:700;letter-spacing:.13em;white-space:nowrap;}
.hero-top a {white-space:nowrap;font-size:.82rem;}
.hero .source {font-size:.72rem;color:#b6c7bc;}
.hero .question {font-size:1.2rem;color:#F4EEE3;}
.legend {height:10px;border-radius:8px;margin:10px 0 4px;}
.step {border-top:2px solid #cb8158;padding:18px 0;min-height:130px;}
.step h3 {color:#c2d5c7;font-size:1.25rem;}
.hero-grid {display:grid;grid-template-columns:1.35fr 1fr;gap:28px;align-items:center;min-width:0;}
.hero-visual {margin:0;min-width:0;}
.hero-visual img {display:block;width:100%;height:390px;object-fit:cover;object-position:50% 55%;border-radius:16px;border:1px solid #52675c;}
.hero-visual figcaption {font-size:.72rem;color:#b6c7bc;line-height:1.4;margin-top:8px;}
.hero-stats {display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin-bottom:16px;}
.hero-stat {padding:15px 18px;border:1px solid #42564b;border-radius:14px;background:#192722;}
.hero-stat h2 {margin:0 0 4px;font-size:2rem;color:#f4eee3;}
.hero-stat strong {font-size:.94rem;}
.hero-stat p {margin:6px 0 0;font-size:.8rem;line-height:1.4;color:#b6c7bc;}
.card {padding:20px;border:1px solid #42564b;border-radius:14px;background:#192722;height:100%;}
.timeline {display:flex;justify-content:space-between;gap:14px;padding:8px 0;color:#c2d5c7;}
.timeline strong {color:#f29b62;display:block;}
@media(max-width:760px){.hero-title-break{display:none;}.hero-grid{grid-template-columns:1fr;gap:16px;}.hero-visual img{height:240px;}.hero-top{align-items:flex-start;flex-wrap:wrap;}.hero-links{gap:12px;}.hero-top .eyebrow{font-size:.62rem;}.hero{padding:20px;}.hero-stats{grid-template-columns:1fr;gap:8px;}.hero-stat{padding:12px 16px;}.hero-stat h2{font-size:1.7rem;}.timeline{flex-direction:column;}}
</style>''')

@st.cache_data(show_spinner=False)
def data():
    return load_public()

try:
    fires,patches,per_fire,summary=data()
except (OSError,ValueError,KeyError) as exc:
    st.error('The presentation dataset could not be loaded. Restore the reviewed presentation bundle and restart.')
    st.stop()

@st.cache_data(show_spinner=False)
def hero_image():
    return base64.b64encode((Path(__file__).parent/'assets/bobcat_landsat.webp').read_bytes()).decode('ascii')

st.html(f'''<div class="hero"><div class="hero-top"><div class="eyebrow">CALIFORNIA · SATELLITE DATA · WILDFIRE RECOVERY</div>
<div class="hero-links"><span class="hero-wordmark">AFTER THE FIRE</span><a href="https://github.com/Vasant-Tholappa/wildfire-recovery-forecasting" target="_blank" rel="noopener noreferrer">View project on GitHub ↗</a></div></div>
<div class="hero-grid"><div><h1>Predicting California's Wildfire<br class="hero-title-break"> Recovery with Machine Learning</h1>
<p>California wildfires have burned millions of acres in recent years.
<a class="source" href="https://www.fire.ca.gov/our-impact/statistics" title="CAL FIRE: five-year average of 1,103,821 acres for comparable year-to-date intervals through November 11, reported November 11, 2026; preliminary, not a full-year average.">CAL FIRE source</a></p>
<p>Fire can transform a landscape in days. Understanding how vegetation recovers can take years.</p>
<p class="question"><strong>What if we could predict what comes next?</strong></p>
<p>I built machine-learning models using satellite imagery and environmental data to predict vegetation conditions five years after a wildfire—using information available just one year after it.</p>
</div><figure class="hero-visual"><img src="data:image/webp;base64,{hero_image()}" width="900" height="1000" alt="False-color Landsat view of California's Bobcat Fire: a red-brown burn scar surrounded by green vegetation in the San Gabriel Mountains, September 21, 2020.">
<figcaption>California wildfire burn scar · illustrative, not a project prediction.<br>
<a href="https://science.nasa.gov/earth/earth-observatory/bobcat-fire-scorches-southern-california-147324/">NASA Earth Observatory / Lauren Dauphin</a> · Landsat data: USGS</figcaption></figure></div></div>''')
reduction=next(r for r in summary['reductions'] if r['model']=='option_a_xgb')
st.html(f'''<div class="hero-stats">
<div class="hero-stat"><h2>151</h2><strong>California wildfires studied</strong><p>2015–2020 fires · vegetation observations through 2025</p></div>
<div class="hero-stat"><h2>{reduction['reduction_vs_training_mean_percent']:.2f}%</h2><strong>Less prediction error</strong><p>Versus a simple historical-average prediction</p><p>Relative MAE reduction—not accuracy or recovery percentage.</p></div>
<div class="hero-stat"><h2>30,606</h2><strong>Landscape patches evaluated</strong><p>90 × 90 m areas across 37 test wildfires from 2020</p></div>
</div>''')

def chart_style(fig,height=360):
    fig.update_layout(height=height,paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#F4EEE3',family='Arial'),margin=dict(l=15,r=25,t=25,b=35),
        legend=dict(orientation='h',y=-.25,bgcolor='#26332e',font=dict(size=12),itemsizing='constant'),hoverlabel=dict(bgcolor='#22352d'))
    fig.update_xaxes(gridcolor='#2c3932',zeroline=False)
    fig.update_yaxes(gridcolor='#2c3932',zeroline=False)
    return fig

st.divider()
st.markdown('## 01 / Explore Wildfires & ML Predictions')
st.write("Explore California's wildfires, then examine groups of landscape patches within each fire to see how well the models predicted vegetation conditions.")
if 'fire_pick' not in st.session_state:
    st.session_state.fire_pick=fires.iloc[0].fire_id

def fire_name(name):
    name=name.title()
    return name if name.endswith('Fire') or 'Complex' in name else name+' Fire'

def map_choice():
    apply_map_selection(st.session_state,st.session_state.get('fire_map',{}),set(fires.fire_id))

st.write('Click an orange marker or search for a wildfire by name. Zoom in if a marker is difficult to select.')
# Keep pickable data identical across reruns. Highlight is a separate non-pickable
# layer: selection state no longer mutates the objects being selected.
markers=fires.copy()
markers['radius']=(markers.patches**.5)*90+1800
chosen=markers[markers.fire_id.eq(st.session_state.fire_pick)].copy()
chosen['radius']=chosen.radius*1.15
map_col,card_col=st.columns([.65,.35])
with map_col:
    deck=pdk.Deck(map_style='https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json',
        initial_view_state=pdk.ViewState(latitude=37.35,longitude=-119.6,zoom=5.25,pitch=0),
        layers=[pdk.Layer('ScatterplotLayer',markers,id='fires',get_position='[lon, lat]',
            get_radius='radius',radius_min_pixels=7,radius_max_pixels=18,get_fill_color=[242,155,98,230],
            pickable=True,auto_highlight=True,stroked=True,get_line_color=[242,238,227,140],line_width_min_pixels=1),
            pdk.Layer('ScatterplotLayer',chosen,id='selected-highlight',get_position='[lon, lat]',get_radius='radius',
            radius_min_pixels=8,radius_max_pixels=21,get_fill_color=[110,215,174,240],
            stroked=True,get_line_color=[226,255,239],line_width_min_pixels=2,pickable=False)],
        tooltip={'text':'{fire_name}\n{patches} analyzed patches'})
    st.pydeck_chart(deck,height=420,key='fire_map',on_select=map_choice,selection_mode='single-object')
with card_col:
    names={r.fire_id:f'{fire_name(r.fire_name)} · {r.patches:,} patches' for r in fires.itertuples()}
    st.selectbox('Search for a 2020 fire',fires.fire_id.tolist(),format_func=lambda x:names[x],key='fire_pick')
    fire_id=st.session_state.fire_pick
    fire=fires.set_index('fire_id').loc[fire_id]
    title=fire_name(fire.fire_name)
    metrics=per_fire[per_fire.fire_id.eq(fire_id)].set_index('model')
    st.html(f'<div class="card"><div class="eyebrow">Selected wildfire · 2020</div><h3>{escape(title)}</h3><p style="font-size:clamp(.85rem,1.25vw,1rem);overflow-wrap:break-word"><strong>{int(fire.patches):,}</strong> landscape patches analyzed</p><strong style="color:#F29B62">Explore detailed predictions ↓</strong></div>')
notice=st.session_state.pop('map_selection_notice',None)
if notice==fire_id:
    st.toast(f'{title} selected — explore its predictions below ↓')
st.html('<strong>How the 2020 wildfire predictions were tested</strong><div class="timeline"><div><strong>2020 · Fire</strong>Wildfire occurs</div><div><strong>2021 · Early signals</strong>First growing-season measurements</div><div><strong>2025 · Verification</strong>Fifth growing-season predictions compared with observations</div></div>')
with st.expander('Why test on wildfires from 2020?'):
    st.write('Forecast verification requires fifth-season outcomes. The 2020 cohort has 2025 observations prepared and verified in this study. Newer fires could potentially receive forecasts, but their future fifth-season outcomes would not yet support equivalent verification. These are historical forecasts, not statewide future projections.')
    st.caption('Seasons mean the first and fifth complete growing seasons, not exact calendar anniversaries. MAE is the average absolute difference between each patch prediction and its observed vegetation index.')

st.markdown(f'### {title} — California, 2020')
cols=st.columns(3)
cols[0].metric('Landscape patches analyzed',f'{int(fire.patches):,}')
try:
    winners,best_mae=best_fire_models(per_fire,fire_id)
except ValueError:
    cols[1].metric('Lowest prediction error on this fire','Unavailable')
    cols[1].caption('A complete, unambiguous set of four saved model scores is required.')
else:
    cols[1].metric('Lowest prediction error on this fire',f'{best_mae:.4f} NBR')
    cols[1].caption(', '.join(LABELS[m] for m in winners)+(' · Best of four tested models' if len(winners)==1 else ' · Tied best of four tested models'))
cols[2].metric('Acres within mapped fire boundary',f'{int(fire.fire_size_acres):,}')
st.caption('Each patch measures 90 × 90 meters. These patches come from one selected study area within this wildfire; they do not cover the entire burned area. Acreage describes the whole mapped fire.')
if fire.patches<25:st.warning('Small sample: fewer than 25 analyzed patches. Interpret this fire’s summary cautiously.')
controls=st.columns(2)
with controls[0].container(key='patch_model_control'):
    st.html('<style>.st-key-patch_model_control [data-testid="stWidgetLabel"] p {color:#F29B62;font-size:1.08rem;font-weight:600;}</style>')
    model=st.selectbox('Choose a prediction model',list(LABELS),format_func=lambda x:LABELS[x],key='model_pick')
    st.caption('Changing models updates predictions and errors—not observed vegetation.')
if st.session_state.get('layer_pick') not in MODES:
    st.session_state.layer_pick=MODES[0]
mode=controls[1].radio('What would you like to see?',MODES,horizontal=True,key='layer_pick')
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
if mode==MODES[2]:
    st.html('<div class="legend" style="background:linear-gradient(90deg,#21302d,#ffa65f)"></div>')
    st.caption('Smaller error: 0 | 0.25 | 0.5+ : Larger error (NBR units)')
else:
    st.html('<div class="legend" style="background:linear-gradient(90deg,#cf6949,#cfbb7d,#41ae77)"></div>')
    st.caption('Lower vegetation index: −0.2 | 0.3 | 0.8 : Higher vegetation index (NBR). Colors saturate outside this fixed range; tooltips retain actual values.')
st.caption('NBR is a satellite-based vegetation-condition indicator, not a direct percentage of forest recovery.')
with st.expander('How to read these maps'):
    st.markdown('**What am I looking at?** Each square represents a 90 × 90 meter landscape patch. The map shows one selected study area within the wildfire, not the entire burned area.')
    st.markdown('**What do the colors mean?** Normalized Burn Ratio (NBR) is a satellite-based indicator of vegetation condition. Predicted and observed vegetation use the same fixed color scale for every fire and model so they can be compared fairly.')
    st.markdown("**How is prediction error calculated?** Prediction error is the absolute difference between the model's prediction and the observed value for each patch. Smaller errors mean closer predictions. Error colors saturate above 0.5; tooltip values are unchanged.")
left,right=st.columns(2)
with left:
    st.markdown(f'#### How well did {LABELS[model]} predict the {title}?')
    lo=min(frame.nbr_5_mean.min(),frame[model].min());hi=max(frame.nbr_5_mean.max(),frame[model].max())
    # SVG avoids competing with the PyDeck maps for browser WebGL contexts.
    scatter=go.Figure(go.Scatter(x=frame.nbr_5_mean,y=frame[model],mode='markers',
        marker=dict(size=5,color=COLORS[model],opacity=.55),name=LABELS[model],
        hovertemplate='Observed %{x:.4f}<br>Predicted %{y:.4f}<extra></extra>'))
    scatter.add_trace(go.Scatter(x=[lo,hi],y=[lo,hi],mode='lines',line=dict(color='#96b9a5',dash='dot'),name='Perfect prediction (reference line)'))
    scatter.update_layout(xaxis_title='Observed NBR5',yaxis_title='Predicted NBR5')
    pad=max((hi-lo)*.05,.01)
    scatter.update_xaxes(range=[lo-pad,hi+pad],constrain='domain')
    scatter.update_yaxes(range=[lo-pad,hi+pad],scaleanchor='x',scaleratio=1,constrain='domain')
    st.plotly_chart(chart_style(scatter,420),width='stretch')
    st.caption('Each dot represents one analyzed landscape patch. The diagonal shows where predictions would exactly match observations. Dots farther from the line represent larger errors.')
with right:
    st.markdown(f'#### Which model performed best on the {title}?')
    bar=go.Figure(go.Bar(x=[metrics.loc[m,'mae'] for m in LABELS],y=list(LABELS.values()),orientation='h',
        text=[f"{metrics.loc[m,'mae']:.4f}" for m in LABELS],textposition='outside',
        textfont=dict(color='#F4EEE3',size=13),cliponaxis=False,
        marker_color=['#f29b62','#78a98e','#728a96','#b6b99a'],hovertemplate='%{y}<br>MAE %{x:.4f}<extra></extra>'))
    bar.update_layout(xaxis_title='Average prediction error (MAE) · lower is better',
        xaxis_range=[0,max(float(metrics.loc[m,'mae']) for m in LABELS)*1.25],yaxis=dict(autorange='reversed'))
    st.plotly_chart(chart_style(bar),width='stretch')

st.divider()
st.markdown('## 02 / The Machine Learning Showdown')
st.write('Four approaches, one question: how well do forecasts work on previously unseen wildfires?')
for col,label,body in zip(st.columns(3),['Training · 2015–2017 · 78 fires','Validation · 2018–2019 · 36 fires','Independent test · 2020 · 37 fires'],['Fit the original models.','Compare and select configurations.','Evaluate on previously unseen fires.']):
    with col:st.html(f'<div class="step"><h3>{label}</h3><p>{body}</p></div>')
st.markdown("### Which models best predicted vegetation conditions in 2025 following California's 2020 wildfires?")
st.write('Comparing average prediction error across 30,606 landscape patches from 37 wildfires. Shorter bars mean more accurate predictions on this cohort.')
comparison=pd.DataFrame(summary['models']).set_index('model')
base=pd.DataFrame(summary['baselines']).set_index('baseline')
ordered=sorted(LABELS,key=lambda m:comparison.loc[m,'mae'])
labels=[LABELS[m] for m in ordered]+['Baseline 1 · Historical average','Baseline 2 · No change']
scores=[comparison.loc[m,'mae'] for m in ordered]+[base.loc[b,'mae'] for b in ('training_mean','nbr1_persistence')]
fig=go.Figure(go.Bar(x=scores,y=labels,orientation='h',text=[f'{x:.4f}' for x in scores],textposition='outside',
    marker_color=[COLORS[m] for m in ordered]+['#69736e','#69736e']))
fig.update_layout(xaxis_title='Independent 2020 MAE · unitless NBR · lower is better',yaxis=dict(autorange='reversed'),xaxis_range=[0,max(scores)*1.2])
st.write('Two simple baseline predictions provide a benchmark for judging whether machine learning actually helps.')
st.plotly_chart(chart_style(fig,370),width='stretch')
with st.expander('What are the two baselines?',expanded=False):
    st.markdown('**Baseline 1 — Historical average:** Predicts the same vegetation-index value for every patch, based on the average observed in the original training data.')
    st.markdown("**Baseline 2 — No change:** Assumes each patch's vegetation condition stays the same between the first and fifth growing seasons.")
a,b=st.columns(2)
a.metric('Less error vs historical-average prediction',f"{reduction['reduction_vs_training_mean_percent']:.2f}%")
b.metric('Less error vs no-change prediction',f"{reduction['reduction_vs_persistence_percent']:.2f}%")
st.caption('Relative reductions in MAE, not accuracy percentages. Baselines were fixed without fitting to 2020 outcomes.')
st.markdown('### The unexpected result: deep learning won during development—but XGBoost won the final test.')
st.write('The CNN + tabular fusion model performed best when I evaluated models using 2018–2019 wildfires. But when the original models were tested on 37 different fires from 2020, XGBoost achieved the lowest overall average prediction error.')
reversal=go.Figure()
for m,c in [('option_a_xgb','#f29b62'),('option_a_fusion','#b6b99a')]:
    reversal.add_trace(go.Scatter(x=['Development validation · 2018–19','Independent test · 2020'],
        y=[comparison.loc[m,'historical_validation_mae'],comparison.loc[m,'mae']],
        name=LABELS[m],mode='lines+markers+text',line=dict(color=c,width=3),marker=dict(size=12),
        text=[f"{comparison.loc[m,'historical_validation_mae']:.4f}",f"{comparison.loc[m,'mae']:.4f}"],textposition='top center'))
reversal.update_layout(yaxis_title='MAE (different cohorts)',yaxis_range=[.06,.09])
st.plotly_chart(chart_style(reversal,300),width='stretch')
st.caption('Two different fire groups, not a continuous year-by-year trend. XGBoost won on average; other models performed better on individual fires.')
with st.expander('Explore detailed model performance',expanded=False):
    st.write('These are the same four machine-learning models featured in the visualizations above. All four were evaluated on the same 37 previously unseen wildfires from 2020.')
    st.markdown('**Patch MAE:** Average absolute prediction error across all evaluated landscape patches. This is the primary metric; lower is better.')
    st.markdown('**RMSE:** Similar to MAE, but penalizes larger mistakes more heavily. Lower is better.')
    st.markdown('**R²:** Measures predictive fit relative to a constant test-set-mean reference. Higher is better.')
    st.markdown("**Equal-fire MAE:** Calculates each wildfire's average error and then averages across fires, giving every fire equal weight. Lower is better.")
    table=comparison.loc[ordered,['mae','rmse','r2','equal_fire_mae']].reset_index()
    table['model']=table.model.map(LABELS)
    st.dataframe(table.rename(columns={'model':'Model','mae':'Patch MAE','rmse':'RMSE','r2':'R²','equal_fire_mae':'Equal-fire MAE'}),hide_index=True,width='stretch',
        column_config={name:st.column_config.NumberColumn(format='%.4f') for name in ['Patch MAE','RMSE','R²','Equal-fire MAE']})
    st.write('XGBoost achieved the lowest overall average prediction error across the evaluated landscape patches. However, the best-performing model varied between individual wildfires.')
    st.caption('Choosing a serving model after inspecting test results is a post-test decision, not another independent validation.')

st.divider()
st.markdown('## 03 / From Forecasts to Decisions')
st.markdown('### What could land managers do if they knew where vegetation might struggle to recover—years before those conditions were observed?')
# Image-embedded SVG survives st.html's HTML-only sanitizer; no dependencies.
application_icons=[
    '<path d="M20 10c0 6-8 12-8 12S4 16 4 10a8 8 0 1 1 16 0Z"/><circle cx="12" cy="10" r="2.5"/>',
    '<circle cx="10" cy="10" r="7"/><path d="m15 15 6 6"/>',
    '<path d="M12 22V12M12 15C4 15 3 9 3 5c6 0 9 3 9 10ZM12 12c0-7 4-10 9-10 0 6-3 10-9 10Z"/>',
    '<path d="M12 2s-7 8-7 13a7 7 0 0 0 14 0c0-5-7-13-7-13Z"/><path d="M8 16c0 2 2 3 4 3"/>',
]
application_titles=['Identify potential recovery hotspots','Prioritize field monitoring','Inform restoration planning','Support watershed protection']
application_bodies=[
    'Spot areas where vegetation may remain weaker years after a wildfire.',
    'Help direct surveys and satellite monitoring toward areas needing closer attention.',
    'Provide an early signal when evaluating where replanting or reseeding may be worth considering.',
    'Help flag areas for further erosion and runoff assessment when combined with terrain and watershed data.',
]
application_icon_images=[base64.b64encode(
    f'<svg xmlns="http://www.w3.org/2000/svg" width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="#F29B62" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">{icon}</svg>'.encode()).decode('ascii')
    for icon in application_icons]
application_cards=''.join(
    f'<article class="decision-card"><img class="decision-icon" src="data:image/svg+xml;base64,{icon}" width="34" height="34" alt=""><h3>{escape(title)}</h3><p>{escape(body)}</p></article>'
    for icon,title,body in zip(application_icon_images,application_titles,application_bodies))
st.html('''<style>
.decision-grid {display:grid;grid-template-columns:repeat(4,minmax(0,1fr));grid-auto-rows:1fr;gap:16px;align-items:stretch;}
.decision-card {padding:20px;border:1px solid #42564b;border-radius:14px;background:#192722;display:flex;flex-direction:column;gap:14px;min-width:0;}
.decision-card .decision-icon {display:block;width:34px;height:34px;flex-shrink:0;}
.decision-card h3 {margin:0;font-size:1.25rem;line-height:1.3;min-height:3.9em;}
.decision-card p {margin:0;line-height:1.5;}
@media(max-width:900px){.decision-grid{grid-template-columns:repeat(2,minmax(0,1fr));}}
@media(max-width:520px){.decision-grid{grid-template-columns:1fr;}.decision-card h3{min-height:0;}}
</style><div class="decision-grid">'''+application_cards+'</div>')
st.caption('These are potential applications of predictive modeling. Real-world use would require additional validation alongside ecological expertise and field observations.')
with st.expander('About the data and how to interpret these results',expanded=False):
    faq_pairs=[('What exactly was predicted?', 'I used satellite and environmental data collected around one year after a wildfire to predict vegetation conditions approximately five years after the fire.'), ('How much of each wildfire was analyzed?', 'Each square on the maps represents a 90 × 90 meter landscape patch. The visualization covers one selected study area per wildfire, not the entire burned area.'), ('What do the results tell us?', 'The models were evaluated on 37 previously unseen wildfires from 2020, using vegetation observations through 2025. Their results demonstrate predictive performance on these historical fires, but future wildfires may behave differently.'), ('Does this measure complete forest recovery?', 'No. The predictions estimate a satellite-based vegetation index. They do not measure the percentage of ecological recovery or predict exactly when a forest will fully recover.'), ('Are these live predictions?', 'No. The interactive maps display saved model results. A separate prediction API is available in the GitHub project.')]
    faq_html=''.join(f'<div class="decision-qa"><p class="decision-question">{escape(question)}</p><p class="decision-answer">{escape(answer)}</p></div>' for question,answer in faq_pairs)
    st.html('''<style>
.decision-faq {max-width:78ch;}
.decision-qa {margin:0 0 20px;}
.decision-qa:last-child {margin-bottom:0;}
.decision-qa .decision-question {font-weight:700;color:#F4EEE3;margin:0 0 7px;line-height:1.4;}
.decision-qa .decision-answer {font-weight:400;margin:0;line-height:1.5;}
</style><div class="decision-faq">'''+faq_html+'</div>')
    st.markdown('**Data sources and technical documentation**')
    st.caption('MTBS: USDA Forest Service / US Geological Survey, Monitoring Trends in Burn Severity. Landsat observations: USGS. Map locations are not legal boundaries.')
    st.html('''<p style="font-size:.85rem;line-height:1.6">
<a href="https://developers.google.com/earth-engine/datasets/catalog/USFS_GTAC_MTBS_burned_area_boundaries_v1" target="_blank" rel="noopener noreferrer">MTBS provenance and terms</a> ·
<a href="https://www.usgs.gov/faqs/are-there-any-restrictions-use-or-redistribution-landsat-data" target="_blank" rel="noopener noreferrer">USGS Landsat reuse guidance</a>
</p>''')
st.markdown('**After the Fire** | [View project on GitHub](https://github.com/Vasant-Tholappa/wildfire-recovery-forecasting)')
st.caption('Interactive visualizations use saved model predictions. The website does not generate new forecasts in real time.')
