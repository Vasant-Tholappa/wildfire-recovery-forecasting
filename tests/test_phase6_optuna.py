"""Small local studies and synthetic training only; no frozen test data."""
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import optuna
from optuna.trial import TrialState, create_trial
import pytest
import torch

import manual_phase6_optuna as driver
from wildfire_recovery_forecasting.cnn import CNNRegressor
from wildfire_recovery_forecasting.cnn_training import seed_everything


def studies(tmp_path, suffix=''):
    storage='sqlite:///'+(tmp_path/f'{suffix}studies.sqlite').as_posix()
    return {m:driver.open_study(storage,m,{'model':m,'identity':'frozen'}) for m in ('cnn_only','fusion')}


@pytest.mark.parametrize('family',['cnn_only','fusion'])
@pytest.mark.parametrize('widths',['16_32','32_64'])
def test_architectures_and_serialization(family,widths):
    seed_everything(42,1)
    params=dict(driver.BASELINE,conv_widths=widths,embedding_dim=32,head_dim=16,tabular_dim=64,dropout=.3,tabular_dropout=.2)
    model=CNNRegressor(**driver.architecture(params,family))
    pred=model(torch.zeros(2,2,15,15),torch.zeros(2,70))
    assert pred.shape==(2,) and torch.isfinite(pred).all()
    clone=CNNRegressor(**json.loads(json.dumps(model.configuration)))
    clone.load_state_dict(model.state_dict())
    assert clone.configuration==model.configuration


def test_default_checkpoint_keys_compatible():
    model=CNNRegressor('fusion')
    assert sum(p.numel() for p in model.parameters())==28849
    assert set(model.state_dict())=={'encoder.layers.0.weight','encoder.layers.0.bias','encoder.layers.3.weight','encoder.layers.3.bias',
                                    'encoder.layers.7.weight','encoder.layers.7.bias','tabular.0.weight','tabular.0.bias',
                                    'head.0.weight','head.0.bias','head.2.weight','head.2.bias'}


def test_sampling_all_parameters_apply():
    for family in ('cnn_only','fusion'):
        trial=optuna.trial.FixedTrial(driver.BASELINE)
        params=driver.sample(trial,family)
        assert ('tabular_dim' in params)==(family=='fusion')
        model=CNNRegressor(**driver.architecture(params,family))
        assert model.configuration['dropout']==params['dropout']
        assert set(params)==set(driver.SPACE)-(set() if family=='fusion' else {'tabular_dim','tabular_dropout'})


def test_cumulative_resume_and_fairness(tmp_path):
    ss=studies(tmp_path)
    sequence=[]
    def objective(trial,model):
        sequence.append(model)
        return .1
    first=driver.schedule(ss,objective,target=3,minutes=1,sampler_seed=5,max_new=1)
    assert first['reason']=='session completion cap'
    reopened=studies(tmp_path)
    result=driver.schedule(reopened,objective,target=3,minutes=1,sampler_seed=5)
    assert result['reason']=='target achieved'
    assert sequence==['cnn_only','fusion']*3
    assert all(driver.counts(s)['COMPLETE']==3 for s in reopened.values())
    assert all([t.number for t in s.trials]==[0,1,2] for s in reopened.values())


def test_pruned_failed_not_complete_and_no_starvation(tmp_path):
    ss=studies(tmp_path)
    order=[]
    def objective(trial,model):
        order.append(model)
        if model=='cnn_only' and trial.number<3:
            raise optuna.TrialPruned('fixture')
        if model=='fusion' and trial.number==0:
            raise ValueError('one bad configuration')
        return .1
    driver.schedule(ss,objective,target=2,minutes=1,sampler_seed=5)
    assert all(driver.counts(s)['COMPLETE']==2 for s in ss.values())
    assert driver.counts(ss['cnn_only'])['PRUNED']==3
    assert driver.counts(ss['fusion'])['FAIL']==1
    assert order[:4]==['cnn_only','fusion','cnn_only','fusion']


def test_global_deadline_does_not_start_another_trial(tmp_path):
    ss=studies(tmp_path);now=[0.0];calls=[]
    def objective(t,m):
        calls.append(m);now[0]+=61;return .2
    result=driver.schedule(ss,objective,target=75,minutes=1,sampler_seed=1,clock=lambda:now[0])
    assert result['reason']=='deadline' and len(calls)==1
    assert sum(driver.counts(s)['COMPLETE'] for s in ss.values())==1


def test_repeated_failures_stop(tmp_path):
    ss=studies(tmp_path)
    def bad(t,m):raise RuntimeError('systemic')
    result=driver.schedule(ss,bad,target=75,minutes=1,sampler_seed=1)
    assert result['reason'].startswith('error')
    assert sum(driver.counts(s)['FAIL'] for s in ss.values())==3
    # A restart cannot silently reset the systemic-failure guard.
    again=driver.schedule(ss,bad,target=75,minutes=1,sampler_seed=1)
    assert again['reason'].startswith('error')
    assert sum(driver.counts(s)['FAIL'] for s in ss.values())==3


def test_milestone_continues(tmp_path):
    s=optuna.create_study(storage='sqlite:///'+(tmp_path/'milestones.sqlite').as_posix(),study_name='milestone')
    for i in range(24):s.add_trial(create_trial(value=.2))
    result=driver.schedule({'cnn_only':s},lambda t,m:.1,target=26,minutes=1,sampler_seed=1)
    assert result['reason']=='target achieved'
    assert driver.counts(s)['COMPLETE']==26
    assert s.user_attrs['milestone_25']['counts']['COMPLETE']==25


def test_contract_mismatch_rejected(tmp_path):
    ss=studies(tmp_path)
    with pytest.raises(ValueError,match='identity mismatch'):
        driver.open_study('sqlite:///'+(tmp_path/'studies.sqlite').as_posix(),'cnn_only',{'model':'cnn_only','identity':'changed'})


def test_interruption_and_recovery(tmp_path):
    ss=studies(tmp_path)
    result=driver.schedule(ss,lambda t,m:(_ for _ in ()).throw(KeyboardInterrupt()),target=2,minutes=1,sampler_seed=1)
    assert result['reason']=='interruption'
    s=ss['fusion'];t=s.ask();t.set_user_attr('mlflow_run_id','saved-id')
    client=MagicMock()
    client.get_run.return_value=SimpleNamespace(info=SimpleNamespace(status='FINISHED'),data=SimpleNamespace(
        tags={'trial_identifier':f'fusion:{t.number}','outcome':'COMPLETE'}, metrics={'best_val_mae':.12,'objective_value':.12}))
    driver.recover(s,client)
    assert driver.counts(s)['COMPLETE']==1
    client.set_terminated.assert_not_called()
    abandoned=s.ask()
    driver.recover(s,client)
    assert driver.counts(s)['FAIL']==1
    driver.schedule(ss,lambda t,m:.1,target=2,minutes=1,sampler_seed=1)
    assert all(driver.counts(s)['COMPLETE']==2 for s in ss.values())


def test_exclusive_lock(tmp_path):
    with driver.exclusive_lock(tmp_path/'lock'):
        with pytest.raises(OSError):
            with driver.exclusive_lock(tmp_path/'lock'):
                pass


def test_source_protects_test_access():
    import inspect
    source=inspect.getsource(driver.main)
    assert "PatchDataset(args.data_root,'train')" in source
    assert "PatchDataset(args.data_root,'validation')" in source
    assert "PatchDataset(args.data_root,'test')" not in source


def test_real_tracking_synthetic_interruption_resume(tmp_path):
    from manual_phase6_train import SyntheticDataset
    from mlflow.tracking import MlflowClient
    uri='sqlite:///'+(tmp_path/'tracking.sqlite').as_posix()
    ss=studies(tmp_path)
    contract=dict(epochs=1,dataset_hash='synthetic-hash',dataset_version='synthetic',source_hashes={'fixture':'v1'})
    contracts={m:contract for m in ss}
    args=SimpleNamespace(seed=42,threads=1,synthetic_smoke=True,smoke_test=False,data_root=tmp_path)
    objective=driver.make_objective(args,(SyntheticDataset('train',42),SyntheticDataset('validation',43)),contracts,uri,tmp_path/'weights')
    # Simulate death after MLflow completion but before Optuna tell, then reload from SQLite.
    t=ss['cnn_only'].ask()
    value=objective(t,'cnn_only')
    reopened=studies(tmp_path)
    client=MlflowClient(tracking_uri=uri)
    driver.recover(reopened['cnn_only'],client)
    assert reopened['cnn_only'].best_value==value
    result=driver.schedule(reopened,objective,target=1,minutes=1,sampler_seed=42)
    assert result['reason']=='target achieved'
    ids=[]
    for s in reopened.values():
        t=s.best_trial
        rid=t.user_attrs['mlflow_run_id'];ids.append(rid)
        run=client.get_run(rid)
        assert run.info.status=='FINISHED' and run.data.tags['outcome']=='COMPLETE'
        assert t.value==run.data.metrics['best_val_mae']==run.data.metrics['objective_value']
        assert run.data.tags['trial_identifier']==f'{s.study_name}:{t.number}'
        assert run.data.metrics['epochs_completed']==1
        assert len(run.data.tags['best_checkpoint_sha256'])==64
        for metric in ('train_loss','train_mae','val_mae','val_rmse','val_r2','val_loss','val_equal_fire_mae','seconds'):
            history=client.get_metric_history(rid,metric)
            assert len(history)==1 and history[0].step==1
    assert len(set(ids))==2


@pytest.mark.parametrize('outcome',['PRUNED','FAILED'])
def test_partial_trial_artifacts_and_status(tmp_path,monkeypatch,outcome):
    from manual_phase6_train import SyntheticDataset
    from mlflow.tracking import MlflowClient
    from wildfire_recovery_forecasting import cnn_training
    uri='sqlite:///'+(tmp_path/'tracking.sqlite').as_posix()
    ss=studies(tmp_path)
    contract=dict(epochs=30,dataset_hash='synthetic',dataset_version='synthetic',source_hashes={})
    args=SimpleNamespace(seed=42,threads=1,synthetic_smoke=True,smoke_test=False,data_root=tmp_path)
    def interrupted_fit(model,train,val,device,output,**kwargs):
        output.mkdir(parents=True)
        (output/'best_state.pt').write_bytes(b'fixture')
        for epoch in range(1,10):
            kwargs['log_epoch'](dict(epoch=epoch,seconds=.01,train_loss=.1,train_mae=.2,
                                    val_loss=.1,val_mae=.2,val_rmse=.3,val_r2=.4,val_equal_fire_mae=.2))
            if outcome=='FAILED':raise RuntimeError('fixture failure')
    monkeypatch.setattr(cnn_training,'fit',interrupted_fit)
    monkeypatch.setattr(optuna.Trial,'should_prune',lambda self:True)
    objective=driver.make_objective(args,(SyntheticDataset('train',42),SyntheticDataset('validation',43)),
                                    {m:contract for m in ss},uri,tmp_path/'weights')
    trial=ss['cnn_only'].ask()
    with pytest.raises(optuna.TrialPruned if outcome=='PRUNED' else RuntimeError):objective(trial,'cnn_only')
    run=MlflowClient(tracking_uri=uri).get_run(trial.user_attrs['mlflow_run_id'])
    assert run.data.tags['outcome']==outcome
    assert run.data.metrics['epochs_completed']==(9 if outcome=='PRUNED' else 1)
    assert run.data.metrics['best_val_mae']==.2
    assert run.data.tags['best_checkpoint']=='training/best_state.pt'
    assert run.info.status==('FINISHED' if outcome=='PRUNED' else 'FAILED')


def test_sampler_history_reproducible_across_resume(tmp_path):
    a=studies(tmp_path,'a');b=studies(tmp_path,'b')
    def objective(t,m):return driver.sample(t,m)['learning_rate']
    driver.schedule(a,objective,target=3,minutes=1,sampler_seed=42)
    driver.schedule(b,objective,target=3,minutes=1,sampler_seed=42,max_new=1)
    b=studies(tmp_path,'b')
    driver.schedule(b,objective,target=3,minutes=1,sampler_seed=42)
    assert {m:[t.params for t in s.trials] for m,s in a.items()}=={m:[t.params for t in s.trials] for m,s in b.items()}


def test_smoke_never_repeats_failed_epoch(tmp_path):
    ss=studies(tmp_path)
    calls=[]
    def objective(t,m):
        calls.append(m)
        if m=='cnn_only':raise ValueError('failed smoke')
        return .1
    result=driver.schedule(ss,objective,target=1,minutes=1,sampler_seed=42,max_attempts=1)
    assert calls==['cnn_only','fusion']
    assert result['reason'].startswith('error: smoke')
