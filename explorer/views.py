"""Small presentation helpers; no model, private-data or cloud imports."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

DATA = Path(__file__).parent/'presentation'
LABELS = {'option_a_xgb':'XGBoost','option_a_rf':'Random Forest',
          'option_a_cnn_only':'CNN-only','option_a_fusion':'CNN + tabular fusion'}


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
    values={'Predicted NBR5':frame[model],'Observed NBR5':frame.nbr_5_mean,
            'Absolute error':abs(frame[model]-frame.nbr_5_mean)}
    if mode not in values:raise ValueError('Unknown display layer')
    frame['value']=values[mode]
    frame['display_value']=frame.value.map(lambda v:f'{v:.4f}')
    frame['polygon']=[[[r[f'x{i}'],r[f'y{i}']] for i in range(4)] for r in frame.to_dict('records')]
    # Color saturation only: original values and tooltips remain unchanged.
    t=np.clip(frame.value.to_numpy()/.5 if mode=='Absolute error' else (frame.value.to_numpy()+1)/2,0,1)
    low=np.array([33,48,45]) if mode=='Absolute error' else np.array([183,85,53])
    high=np.array([255,166,95]) if mode=='Absolute error' else np.array([109,192,151])
    frame['color']=[list(map(int,c))+[220] for c in low+(high-low)*t[:,None]]
    return frame


def selected_fire(event,current):
    objects=event.get('selection',{}).get('objects',{}).get('fires',[])
    return objects[0]['fire_id'] if objects else current
