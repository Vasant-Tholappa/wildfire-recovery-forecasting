import numpy as np
import pandas as pd
import pytest
from manual_phase6_verify_pixels import choose_pixels, compare
from wildfire_recovery_forecasting import imagery


def test_point_selection_and_zero():
    rows=pd.DataFrame([dict(fire_id='f',patch_id='f_r0_c0',batch_id='r0_c0',grid_row=0,grid_col=0)])
    region=imagery.prepare_region(rows,fire_id='f',batch_id='r0_c0')
    array=np.ones((2,15,15));array[0,3,4]=0
    samples=choose_pixels(rows,region,{'nbrpre':array,'season1':array})
    assert samples==choose_pixels(rows,region,{'nbrpre':array,'season1':array})
    assert len(samples)<=12
    assert {'boundary','eligible_center','surrounding_context','observed_zero_nbrpre'}<=set(k for p in samples for k in p['kinds'])
    for p in samples: assert region['transform']*(p['col']+.5,p['row']+.5)==(p['x'],p['y'])


def test_comparison_fails_value_mask_or_missing_id():
    samples=[dict(pixel_id='a',row=0,col=0)]
    array=np.array([[[0.]],[[1.]]])
    features=[{'properties':dict(pixel_id='a',source_nbr=0.,source_mask=1)}]
    assert compare(samples,features,array)['value_pass']==1
    features[0]['properties']['source_nbr']=2e-6
    assert compare(samples,features,array)['value_pass']==0
    features[0]['properties']['source_mask']=0
    assert compare(samples,features,array)['mask_pass']==0
    with pytest.raises(ValueError): compare(samples,[],array)
