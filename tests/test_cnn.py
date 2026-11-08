"""Tiny synthetic tests only. Never open the frozen dataset or test split."""
import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

from manual_phase6_train import SyntheticDataset
from wildfire_recovery_forecasting.cnn import CNNRegressor
from wildfire_recovery_forecasting.cnn_training import (
    PatchDataset, seed_everything, make_loaders, evaluate, EarlyStopping, fit,
)


@pytest.fixture(autouse=True)
def seed():
    seed_everything(42, threads=1)


def tiny_files(tmp_path):
    (tmp_path/'manifest.json').write_text(json.dumps({'status':'VALIDATED',
        'image_shape_per_row':[2,15,15], 'counts':{'train':4,'validation':4}}))
    for split in ('train','validation'):
        folder=tmp_path/split;folder.mkdir()
        images=np.stack([np.full((2,15,15),i/10,dtype='float32') for i in range(4)])
        np.save(folder/'images.npy',images)
        np.save(folder/'tabular_fusion.npy',np.repeat(np.arange(4,dtype='float32')[:,None],70,axis=1))
        np.save(folder/'targets.npy',np.arange(4,dtype='float64'))
        pd.DataFrame(dict(fire_id=[split]*4,patch_id=[split+str(i) for i in range(4)],
                          split=[split]*4,array_index=range(4))).to_parquet(folder/'index.parquet',index=False)
    return tmp_path


def test_memory_mapping_alignment_and_shuffle(tmp_path):
    root=tiny_files(tmp_path)
    train,val=PatchDataset(root,'train'),PatchDataset(root,'validation')
    assert isinstance(train.images,np.memmap)
    a,b=make_loaders(train,val,batch_size=2,seed=42)
    seen=[]
    for batch in a:
        seen.extend(batch['index'].tolist())
        torch.testing.assert_close(batch['image'][:,0,0,0]*10,batch['target'])
        torch.testing.assert_close(batch['tabular'][:,0],batch['target'])
    assert sorted(seen)==list(range(4))
    assert torch.cat([x['index'] for x in b]).tolist()==list(range(4))
    again,_=make_loaders(train,val,2,42)
    assert torch.cat([x['index'] for x in again]).tolist()==seen


def test_test_split_rejected_before_io():
    with patch('pandas.read_parquet') as read,patch('numpy.load') as load,patch('pathlib.Path.read_text') as text:
        with pytest.raises(ValueError,match='forbidden'):PatchDataset('not_read','test')
    read.assert_not_called();load.assert_not_called();text.assert_not_called()


def test_bad_index_and_fire_overlap_rejected(tmp_path):
    root=tiny_files(tmp_path)
    train,val=PatchDataset(root,'train'),PatchDataset(root,'validation')
    val.index['fire_id']='train'
    with pytest.raises(ValueError,match='overlap'):make_loaders(train,val)
    path=root/'train/index.parquet'
    frame=pd.read_parquet(path);frame['array_index']=[1,0,2,3];frame.to_parquet(path,index=False)
    with pytest.raises(ValueError,match='alignment'):PatchDataset(root,'train')


@pytest.mark.parametrize('kind,count',[('cnn_only',25553),('fusion',28849)])
def test_forward_backward_and_evaluation_isolation(kind,count):
    model=CNNRegressor(kind)
    assert sum(p.numel() for p in model.parameters())==count
    train,val=make_loaders(SyntheticDataset('train',1),SyntheticDataset('validation',2),4,42)
    batch=next(iter(train))
    assert batch['image'].shape==(4,2,15,15) and batch['tabular'].shape==(4,70)
    old={k:v.clone() for k,v in model.state_dict().items()}
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    pred=model(batch['image'],batch['tabular'])
    assert pred.shape==(4,) and torch.isfinite(pred).all()
    torch.nn.functional.mse_loss(pred,batch['target']).backward();optimizer.step()
    assert any(not torch.equal(old[k],v) for k,v in model.state_dict().items())
    before={k:v.clone() for k,v in model.state_dict().items()}
    gradients=[p.grad.clone() for p in model.parameters()]
    metrics=evaluate(model,val,'cpu')
    assert all(np.isfinite(v) for v in metrics.values()) and model.training
    assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())
    assert all(torch.equal(g,p.grad) for g,p in zip(gradients,model.parameters()))


def test_lowest_checkpoint_restored_with_patience(tmp_path):
    model=torch.nn.Linear(1,1)
    stop=EarlyStopping(tmp_path/'best.pt',patience=2,min_delta=.01)
    for epoch,score in enumerate([.3,.2,.195,.21],1):
        with torch.no_grad():model.weight.fill_(epoch)
        should_stop=stop.update(model,epoch,{'mae':score})
    assert should_stop and stop.best_epoch==3  # Save even sub-min_delta improvements.
    stop.restore(model,'cpu')
    assert model.weight.item()==3


@pytest.mark.parametrize('kind',['cnn_only','fusion'])
def test_tiny_fit_restores_best_not_last(tmp_path,kind):
    model=CNNRegressor(kind)
    train,val=make_loaders(SyntheticDataset('train',1),SyntheticDataset('validation',2),8,42)
    result=fit(model,train,val,'cpu',tmp_path/kind,epochs=2,patience=1)
    saved=torch.load(tmp_path/kind/'best_state.pt',weights_only=True)
    assert all(torch.equal(v,saved[k]) for k,v in model.state_dict().items())
    history=pd.read_csv(tmp_path/kind/'history.csv')
    assert result['best_epoch']==int(history.loc[history.val_mae.idxmin(),'epoch'])
    assert evaluate(model,val,'cpu')['mae']==pytest.approx(result['best_validation']['mae'])


def test_bad_model_shapes():
    with pytest.raises(ValueError):CNNRegressor('fusion')(torch.zeros(2,2,15,15),torch.zeros(2,18))
    with pytest.raises(ValueError):CNNRegressor()(torch.zeros(2,2,16,16))

def test_cli_smoke_is_synthetic_and_logs_locally(tmp_path,monkeypatch):
    import sys
    from unittest.mock import MagicMock
    import manual_phase6_train as cli
    tracker=MagicMock()
    tracker.active_run.return_value = None
    tracker.start_run.return_value.info.run_id = 'synthetic-id'
    monkeypatch.setitem(sys.modules,'mlflow',tracker)
    monkeypatch.setattr(cli,'ROOT',tmp_path)
    monkeypatch.setattr(sys,'argv',['manual_phase6_train.py','--model','fusion','--smoke-test','--threads','1'])
    with patch.object(cli,'PatchDataset') as dataset:
        cli.main()
    dataset.assert_not_called()
    tracker.set_tracking_uri.assert_called_once_with('sqlite:///'+(tmp_path/'mlflow.db').as_posix())
    assert tracker.log_metrics.call_count==2
    configs=list((tmp_path/'data/processed/phase6_training').glob('*/configuration.json'))
    config=json.loads(configs[0].read_text())
    assert config['synthetic_only'] and config['train_rows']==16 and config['max_epochs']==1
    tracker.log_artifacts.assert_called_once()
