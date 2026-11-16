"""Small presentation helpers; no model, private-data or cloud imports."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

COLORS = {'option_a_xgb':'#f29b62','option_a_rf':'#78a98e','option_a_cnn_only':'#728a96','option_a_fusion':'#b6b99a'}
MODES = ['Predicted vegetation — Year 5','Observed vegetation — Year 5','Prediction error']
# Outward-rounded envelope of each series' 0.5th–99.5th percentiles in the public bundle.
# Fixed presentation-only bounds; never recomputed per fire or model.
VEGETATION_BOUNDS = (-0.2, 0.8)

DATA = Path(__file__).parent/'presentation'
LABELS = {'option_a_xgb':'XGBoost','option_a_rf':'Random Forest',
          'option_a_cnn_only':'CNN','option_a_fusion':'CNN + tabular fusion'}


def load_public(root=DATA):
    fires=pd.read_csv(root/'fires.csv')
    patches=pd.read_csv(root/'patches.csv.gz')
    per_fire=pd.read_csv(root/'per_fire.csv')
    summary=json.loads((root/'summary.json').read_text())
    if len(fires)!=37 or len(patches)!=30606 or patches.duplicated(['fire_id','patch_id']).any():
        raise ValueError('Presentation cohort mismatch')
    if set(patches.fire_id)!=set(fires.fire_id):raise ValueError('Fire identity mismatch')
    return fires,patches,per_fire,summary


def fire_view(patches,fire_id,model,mode):
    if model not in LABELS:raise ValueError('Unknown display model')
    frame=patches[patches.fire_id.eq(fire_id)].copy()
    if frame.empty:raise ValueError('No evaluated patches for this fire')
    values={MODES[0]:frame[model],MODES[1]:frame.nbr_5_mean,
            MODES[2]:abs(frame[model]-frame.nbr_5_mean)}
    if mode not in values:raise ValueError('Unknown display layer')
    frame['value']=values[mode]
    frame['display_value']=frame.value.map(lambda v:f'{v:.4f}')
    frame['polygon']=[[[r[f'x{i}'],r[f'y{i}']] for i in range(4)] for r in frame.to_dict('records')]
    # Color saturation only: original values and tooltips remain unchanged.
    if mode==MODES[2]:
        t=np.clip(frame.value.to_numpy()/.5,0,1)
        colors=np.array([33,48,45])+(np.array([255,166,95])-np.array([33,48,45]))*t[:,None]
    else:
        t=np.clip((frame.value.to_numpy()-VEGETATION_BOUNDS[0])/(VEGETATION_BOUNDS[1]-VEGETATION_BOUNDS[0]),0,1)
        stops=np.array([[207,105,73],[207,187,125],[65,174,119]])
        colors=np.column_stack([np.interp(t,[0,.5,1],stops[:,i]) for i in range(3)])
    frame['color']=[list(map(int,c))+[220] for c in colors]
    return frame


def selected_fire(event,current,allowed=None):
    """Only accept an unambiguous pick from the stable, pickable fire layer.

    Empty deselections retain the dropdown choice; overlays cannot select a fire.
    """
    objects=event.get('selection',{}).get('objects',{}).get('fires',[])
    if len(objects)!=1 or not isinstance(objects[0],dict):return current
    chosen=objects[0].get('fire_id')
    return chosen if chosen and (allowed is None or chosen in allowed) else current


def best_fire_models(per_fire,fire_id):
    """Retrospective Option A minimum; exact ties retain the defined model order."""
    rows=per_fire[per_fire.fire_id.eq(fire_id) & per_fire.model.isin(LABELS)]
    if len(rows)!=4 or not rows.model.is_unique or set(rows.model)!=set(LABELS):
        raise ValueError('Four unique Option A scores required')
    scores=rows.set_index('model').loc[list(LABELS),'mae']
    if not np.isfinite(scores.to_numpy(dtype=float)).all() or (scores<0).any():
        raise ValueError('Invalid saved MAE')
    minimum=float(scores.min())
    return [m for m in LABELS if scores[m]==minimum],minimum


def apply_map_selection(state,event,allowed):
    """Queue one notification only for a genuine map-driven selection change."""
    current=state['fire_pick']
    chosen=selected_fire(event,current,allowed)
    if chosen==current:return False
    state['fire_pick']=chosen
    state['map_selection_notice']=chosen
    return True
