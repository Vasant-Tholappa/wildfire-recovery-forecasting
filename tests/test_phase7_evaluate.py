from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
import manual_phase7_evaluate as e

def keys():
 return pd.DataFrame({'fire_id':['a','a','b'],'patch_id':['1','2','3']})

def test_guard_before_io():
 with patch.object(e.pd,'read_parquet',side_effect=AssertionError('IO forbidden')):
  with pytest.raises(ValueError,match='authorization'):e.read_targets(keys())
 with patch.object(e,'verify_freeze',side_effect=AssertionError('No preflight')):
  with pytest.raises(ValueError,match='authorize'):e.run('evaluate')

def test_filtered_target_alignment():
 frame=keys().iloc[::-1].copy();frame['nbr_5_mean']=[3.,2.,1.]
 with patch.object(e.pd,'read_parquet',return_value=frame) as read:
  np.testing.assert_array_equal(e.read_targets(keys(),True),[1.,2.,3.])
  assert read.call_args.kwargs['filters']==[('split','==','test')]
  assert read.call_args.kwargs['columns']==e.KEYS+['nbr_5_mean']

@pytest.mark.parametrize('kind',['duplicate','missing','wrong','nonfinite'])
def test_bad_targets(kind):
 f=keys();f['nbr_5_mean']=[1.,2.,3.]
 if kind=='duplicate':f=pd.concat([f,f.iloc[:1]])
 if kind=='missing':f=f.iloc[:2]
 if kind=='wrong':f.loc[0,'patch_id']='other'
 if kind=='nonfinite':f.loc[0,'nbr_5_mean']=np.nan
 with patch.object(e.pd,'read_parquet',return_value=f):
  with pytest.raises(ValueError):e.read_targets(keys(),True)

def test_artifact_integrity(tmp_path):
 f=tmp_path/'model';f.write_bytes(b'original')
 with patch.object(e,'ROOT',tmp_path):
  assert e.checked_path(f,e.sha(f))==f
  with pytest.raises(ValueError,match='hash'):e.checked_path(f,'bad')

def test_output_is_exclusive(tmp_path):
 out=tmp_path/'results';e.create_output(out);(out/'evidence').write_text('keep')
 with pytest.raises(FileExistsError):e.create_output(out)
 assert (out/'evidence').read_text()=='keep'

def test_metrics_and_paired_sign():
 ix=keys();y=np.array([0.,0.,1.]);pred={}
 for f in e.FAMILIES:
  pred['option_a_'+f]=np.array([1.,1.,1.]);pred['option_b_'+f]=y.copy()
 table,fire,paired=e.summaries(ix,y,pred)
 row=table.set_index('model').loc['option_a_rf']
 assert row.mae==pytest.approx(2/3) and row.equal_fire_mae==pytest.approx(.5)
 assert row.rmse==pytest.approx(np.sqrt(2/3))
 assert paired.query("family=='rf' and scope=='all_patches'").mae.iloc[0]==pytest.approx(-2/3)
 assert set(fire.patch_count)=={1,2}

def test_frozen_encoding_without_fit():
 schema={'input_order':e.refit.FEATURES,'numeric':[f'n{i}' for i in range(16)],'mean':[1]*16,'scale':[2]*16,'categorical':['evt','month'],'categories':[list(range(44)),list(range(10))]}
 f=pd.DataFrame({n:[3.] for n in schema['numeric']}|{'evt':[99],'month':[2]})
 x=e.encode(f,schema)
 assert x.shape==(1,70) and (x[0,:16]==1).all() and x[0,16:60].sum()==0 and x[0,62]==1

def test_model_choice_neural_uses_saved_architecture(tmp_path):
 import torch
 from wildfire_recovery_forecasting.cnn import CNNRegressor
 model=CNNRegressor('fusion',embedding_dim=128,head_dim=64)
 path=tmp_path/'weights.pt';torch.save(model.state_dict(),path)
 rec={'artifact':str(path),'sha256':e.sha(path),'architecture':model.configuration}
 with patch.object(e,'ROOT',tmp_path):
  loaded=e.load_model('fusion',rec,{})
 assert loaded.configuration==model.configuration and not loaded.training
