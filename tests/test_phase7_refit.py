"""Synthetic Phase 7 contract checks; no real data or cloud."""
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
import torch
import manual_phase7_refit as r

def test_masked_pooled_medians():
    a=np.array([[[[.1,.3]],[[.2,.4]]]],dtype='float32')
    b=np.array([[[[999,.7]],[[999,.8]]]],dtype='float32')
    va=np.ones_like(a,dtype=bool);vb=va.copy();vb[:,:,:,0]=False
    np.testing.assert_allclose(r.pooled_medians([(a,va),(b,vb)]),[.3,.4])
    x=r.fill_images(b,vb,[.3,.4]);assert b[0,0,0,0]==999
    np.testing.assert_allclose(x[0,:,0,0],[.3,.4])

def test_fixed_categories_70_columns():
    numeric=[f'n{i}' for i in range(16)]
    s=dict(numeric=numeric,categorical=['evt','month'],categories=[list(range(44)),list(range(10))])
    df=pd.DataFrame({k:[1.,3.,5.] for k in numeric}|{'evt':[0,1,99],'month':[0,1,99]})
    p=r.fusion_preprocessor(s);x=p.fit_transform(df)
    assert x.shape==(3,70) and x[2,16:].sum()==0
    np.testing.assert_allclose(p.named_transformers_['numeric'].mean_,[3.]*16)

def test_key_alignment():
    left=pd.DataFrame(dict(fire_id=['a','a'],patch_id=['1','2'],split=['train','train']))
    right=left.iloc[::-1].copy();right['x']=[20,10]
    assert r.align(left,right,['x']).x.tolist()==[10,20]
    with pytest.raises(ValueError):r.key_hash(pd.concat([left,left]))
    with pytest.raises(ValueError):r.align(left,right.iloc[:1],['x'])

@pytest.mark.parametrize('split',['test','2020',None])
def test_target_guard_before_io(split):
    with patch.object(r.pd,'read_parquet',side_effect=AssertionError('No IO')):
        with pytest.raises(ValueError,match='forbidden'):r.load_targets('unused',split)

def test_target_projection_filtered():
    with patch.object(r.pd,'read_parquet') as read:
        r.load_targets('unused','validation')
        assert read.call_args.kwargs['filters']==[('split','==','validation')]
        assert read.call_args.kwargs['columns']==r.KEYS+['nbr_5_mean']

@pytest.mark.parametrize('epochs',[5,14])
def test_fixed_epochs(epochs):
    model=torch.nn.Linear(1,1);history=[]
    with patch.object(r,'train_epoch',return_value={'loss':1.,'mae':.5}) as train:
        result=r.fixed_fit(model,[],epochs,.001,.0001,history.append)
    assert train.call_count==epochs
    assert [x['epoch'] for x in result]==list(range(1,epochs+1))
    assert not any('val' in k for x in history for k in x)

@pytest.mark.parametrize('family',['cnn_only','fusion'])
def test_fresh_initialization_and_update(family):
    r.seed_everything(42,1);a=r.CNNRegressor(family)
    r.seed_everything(42,1);b=r.CNNRegressor(family)
    assert all(torch.equal(x,y) for x,y in zip(a.parameters(),b.parameters()))
    ds=r.PreparedDataset(np.zeros((2,2,15,15),dtype='float32'),np.zeros((2,70),dtype='float32'),np.array([.2,.4]))
    r.fixed_fit(a,r.DataLoader(ds,batch_size=2),1,.001,.0001,lambda x:None)
    assert any(not torch.equal(x,y) for x,y in zip(a.parameters(),b.parameters()))

def test_no_overwrite(tmp_path):
    path=tmp_path/'record.json';r.write(path,{'seed':42})
    with pytest.raises(ValueError,match='overwrite'):r.write(path,{'seed':123})

def test_shape_alignment():
    with pytest.raises(ValueError,match='alignment'):
        r.PreparedDataset(np.zeros((2,2,15,15)),np.zeros((3,70)),np.zeros(2))

def test_command_boundaries():
    assert r.requested_models('prepare',None)==()
    assert r.requested_models('train','fusion')==('fusion',)
    assert len(r.requested_models('verify',None))==4
    with pytest.raises(ValueError):r.requested_models('train',None)
    with pytest.raises(ValueError):r.requested_models('verify','rf')

def test_completed_recovery_record_preferred(tmp_path):
    recovered=tmp_path/'rf_verified_recovery';recovered.mkdir()
    (recovered/'record.json').write_text('{}')
    assert r.existing_folder(tmp_path,'rf')==recovered
    assert r.existing_folder(tmp_path,'fusion')==tmp_path/'fusion'
