"""Synthetic dataset preprocessing/alignment safeguards; no cloud/model fits."""
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
import manual_phase6_build_dataset as b


def test_clip_fill_and_valid_zero():
    x=np.array([[[[0.,-9999.,2.,-3.]],[[.5,-9999.,.2,.3]]]],dtype='float32')
    masks=x!=-9999
    y=b.fill_images(x,masks,[.25,.75])
    np.testing.assert_array_equal(y[0,0,0],[0,.25,1,-1])
    assert y[0,1,0,1]==.75 and x[0,0,0,1]==-9999
    assert not masks[0,0,0,1]


def test_training_only_medians_exclude_nodata():
    train=np.array([[[[-9999.,0.,2.]],[[.2,.4,.6]]]],dtype='float32')
    medians=b.training_medians(train,train!=-9999)
    assert medians==pytest.approx([.5,.4])
    validation=np.full((1,2,1,3),-9999,dtype='float32')
    y=b.fill_images(validation,validation!=-9999,medians)
    assert y[0,0,0,0]==.5
    assert b.training_medians(train,train!=-9999)==medians


def test_alignment_reorders_and_rejects_missing_duplicates():
    left=pd.DataFrame({'fire_id':['f','f'],'patch_id':['a','b'],'split':['train','train']})
    right=left.iloc[::-1].copy();right['value']=[2,1]
    assert b.align(left,right,['value']).value.tolist()==[1,2]
    with pytest.raises(ValueError):b.align(left,right.iloc[:1],['value'])
    with pytest.raises(ValueError):b.align(left,pd.concat([right,right]),['value'])


def test_test_target_guard_and_dev_filter():
    with patch.object(b.pd,'read_parquet') as read:
        with pytest.raises(ValueError,match='forbidden'):b.load_targets('never_read','test')
        read.assert_not_called()
        b.load_targets('source','validation')
        assert read.call_args.kwargs['filters']==[('split','==','validation')]
        assert read.call_args.kwargs['columns']==b.KEYS+['nbr_5_mean']


def test_context_overlap_distinguishes_target_and_touching():
    rows=pd.DataFrame([dict(fire_id='a',patch_id='a',split='train',grid_row=0,grid_col=0),
        dict(fire_id='b',patch_id='b',split='validation',grid_row=0,grid_col=4),
        dict(fire_id='c',patch_id='c',split='test',grid_row=5,grid_col=0)])
    found,ids=b.context_overlap(rows)
    assert len(found)==1 and found[0]['pairs']==1
    assert found[0]['central_overlap_pairs']==0 and ids==['b']


def test_invalid_valid_pixels_rejected():
    x=np.full((1,2,1,1),np.nan,dtype='float32')
    with pytest.raises(ValueError):b.fill_images(x,np.ones_like(x,dtype=bool),[0,0])
