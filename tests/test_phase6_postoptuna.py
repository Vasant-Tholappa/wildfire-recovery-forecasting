"""Focused post-Optuna checks; synthetic/mocked inputs only."""
import json
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import optuna

import manual_phase6_postoptuna as post


def trial(number,value,params,state=optuna.trial.TrialState.COMPLETE):
    return SimpleNamespace(number=number,value=value,params=params,state=state)


def test_top_distinct_complete_only():
    rows=[trial(0,.1,{'a':1}),trial(1,.09,{'a':1}),trial(2,.11,{'a':2}),
          trial(3,.01,{'a':3},optuna.trial.TrialState.PRUNED)]
    assert [t.number for t in post.top_distinct(rows)]==[1,2]
    with pytest.raises(ValueError):post.top_distinct(rows[:2])


def test_exact_key_alignment_and_rejection():
    index=pd.DataFrame({'fire_id':['a','b'],'patch_id':['p','q']})
    frame=pd.DataFrame({'fire_id':['b','a'],'patch_id':['q','p'],'x':[20,10]})
    assert post.aligned_frame(index,frame).x.tolist()==[10,20]
    with pytest.raises(ValueError):post.aligned_frame(index,pd.concat([frame.iloc[:1]]*2))
    with pytest.raises(ValueError):post.aligned_frame(index,frame.iloc[:1])


def test_test_split_rejected_before_io():
    with pytest.raises(ValueError,match='forbidden'):post.development('test')


def test_common_population_paired_and_overlap_sensitivity():
    index=pd.DataFrame({'fire_id':['a']*60+['b']*40,'patch_id':[f'p{i}' for i in range(100)]})
    y=np.linspace(0,1,100)
    metrics,fire,pairs=post.comparisons(index,y,{'a':y+.1,'b':y+.2},index.patch_id[:53].tolist())
    assert metrics['a']['full']['mae']==pytest.approx(.1)
    assert metrics['a']['without_shared_context']['mae']==pytest.approx(.1)
    assert pairs[0]['fires_a_better']==2
    assert pairs[0]['equal_fire_difference_a_minus_b']==pytest.approx(-.1)
    with pytest.raises(ValueError):post.comparisons(index,y,{'a':y},['bad']*53)


def test_stability_sd_and_incomplete_reporting():
    rows=[dict(model='cnn_only',trial=17,seed=s,mae=v) for s,v in [(42,.1),(123,.2),(456,.3)]]
    summary=post.seed_summary(rows)[0]
    assert summary['complete'] and summary['mean_mae']==pytest.approx(.2)
    assert summary['std_mae']==pytest.approx(.1)
    assert not post.seed_summary(rows[:1])[0]['complete']


def test_local_artifact_boundary(tmp_path,monkeypatch):
    monkeypatch.setattr(post,'ROOT',tmp_path)
    assert post.local_path((tmp_path/'artifact').as_uri())==tmp_path/'artifact'
    with pytest.raises(ValueError):post.local_path('https://example.com/model.pkl')
    with pytest.raises(ValueError):post.local_path(tmp_path.parent.as_uri())


def test_seed_resume_after_finished_run_before_ledger_completion(tmp_path,monkeypatch):
    from wildfire_recovery_forecasting import cnn_tracking,cnn_training
    import mlflow.tracking
    from manual_phase6_optuna import BASELINE
    monkeypatch.setattr(post,'ROOT',tmp_path);monkeypatch.setattr(post,'DATA',tmp_path/'data')
    monkeypatch.setattr(post,'OUT',tmp_path/'output');post.OUT.mkdir()
    monkeypatch.setattr(post,'sha',lambda path:'hash')
    monkeypatch.setattr(post,'development',lambda split:(split,None,None))
    monkeypatch.setattr(cnn_training,'seed_everything',lambda *a:None)
    monkeypatch.setattr(cnn_training,'make_loaders',lambda *a:(None,None))
    fake=SimpleNamespace(runs={})
    fake.get_run=lambda rid:fake.runs[rid]
    fake.log_metric=lambda rid,k,v:fake.runs[rid].data.metrics.update({k:v})
    monkeypatch.setattr(mlflow.tracking,'MlflowClient',lambda **kw:fake)
    calls=[];interrupt=[True]
    def fit(model,train,val,device,output,**kw):
        rid=str(len(calls));calls.append(rid)
        output.mkdir(parents=True)
        (output/'best_state.pt').write_bytes(b'fixture')
        params={k:str(v) for k,v in kw['metadata'].items()}
        fake.runs[rid]=SimpleNamespace(info=SimpleNamespace(status='FINISHED',artifact_uri=output.parent.as_uri()),data=SimpleNamespace(
            tags=dict(kw['tags'],outcome='COMPLETE',best_checkpoint_sha256='hash'),params=params,
            metrics={'best_val_mae':.1,'best_val_rmse':.2,'best_val_r2':.5,'best_val_equal_fire_mae':.11,'fit_seconds':1.,'best_epoch':2}))
        # Map the mocked artifact root directly; production paths are checked separately.
        kw['run_hook'](rid)
        if len(calls)==3 and interrupt[0]:
            interrupt[0]=False;raise KeyboardInterrupt()
        return {}
    monkeypatch.setattr(cnn_tracking,'tracked_fit',fit)
    # Only hash checking uses this path in this test; no checkpoint deserialization occurs.
    monkeypatch.setattr(post,'local_path',lambda uri:tmp_path)
    cfg=tmp_path/'artifact';cfg.mkdir();(cfg/'configuration.json').write_text(json.dumps(dict(BASELINE,dataset_manifest_sha256='hash',seed=42)))
    audited={'dataset_hash':'hash','models':{}}
    for family in post.MODELS:
        params={k:v for k,v in BASELINE.items() if family=='fusion' or not k.startswith('tabular_')}
        contract={'source_hashes':{k:'hash' for k in ['src/wildfire_recovery_forecasting/cnn.py','src/wildfire_recovery_forecasting/cnn_training.py']},'training_seed':42,'epochs':30,'threads':4,'torch_version':str(post.torch.__version__),'numpy_version':np.__version__}
        top=[dict(trial=t,params=params,artifact=str(cfg),metrics={'mae':.1,'rmse':.2,'r2':.5,'equal_fire_mae':.11},run_id=f'{family}{t}',duration=2,best_epoch=1) for t in (1,2)]
        audited['models'][family]={'contract':contract,'top':top}
    # Add model/params metadata through the actual stability path.
    with pytest.raises(KeyboardInterrupt):post.stability(audited)
    rows=post.stability(audited)
    assert len(rows)==12 and len(calls)==8
    assert sum(r['reused'] for r in rows)==4
    assert len(post.stability(audited))==12 and len(calls)==8
    fake.runs['0'].info.status='RUNNING'
    with pytest.raises(ValueError,match='requires review'):post.stability(audited)
    assert len(calls)==8


def test_changed_checkpoint_is_rejected_before_loading(tmp_path):
    (tmp_path/'best_state.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed since audit'):
        post.load_checkpoint({'artifact':str(tmp_path),'checkpoint_sha256':'incorrect'})


@pytest.mark.parametrize('tamper',['history','checkpoint','identity'])
def test_audit_detects_missing_history_hash_and_identity(tmp_path,monkeypatch,tamper):
    import sqlite3
    from manual_phase6_optuna import BASELINE,architecture
    from wildfire_recovery_forecasting.cnn import CNNRegressor
    monkeypatch.setattr(post,'ROOT',tmp_path);monkeypatch.setattr(post,'DATA',tmp_path/'data')
    post.DATA.mkdir();(post.DATA/'manifest.json').write_text('{}');(post.DATA/'tabular_preprocessing.json').write_text('{}')
    root=tmp_path/'artifacts';path=root/'training';path.mkdir(parents=True)
    meta=root/'dataset_metadata';meta.mkdir()
    for name in ('manifest.json','tabular_preprocessing.json'):(meta/name).write_text('{}')
    params={k:v for k,v in BASELINE.items() if not k.startswith('tabular_')}
    cfg=dict(params,seed=42,dataset_manifest_sha256=post.sha(post.DATA/'manifest.json'),train_rows=61073,validation_rows=19561,max_epochs=30,patience=5,min_delta=.0001,threads=4,training_loss='MSE')
    (path/'configuration.json').write_text(json.dumps(cfg))
    (path/'model_configuration.json').write_text(json.dumps(CNNRegressor(**architecture(params,'cnn_only')).configuration))
    (path/'architecture.txt').write_text('fixture');(path/'best_state.pt').write_bytes(b'fixture')
    hist=pd.DataFrame([dict(epoch=e,train_loss=.1,train_mae=.2,val_loss=.1,val_mae=v,val_rmse=.3,val_r2=.5,val_equal_fire_mae=.2,seconds=1.) for e,v in [(1,.2),(2,.1)]])
    hist.to_csv(path/'history.csv',index=False)
    selection={'best_epoch':2,'epochs_completed':2,'best_validation':{k[4:]:float(v) for k,v in hist.iloc[1].items() if k.startswith('val_')}}
    (path/'selection.json').write_text(json.dumps(selection))
    db=sqlite3.connect(':memory:')
    db.executescript('create table runs(run_uuid text,status text,artifact_uri text);create table params(run_uuid text,key text,value text);create table tags(run_uuid text,key text,value text);create table metrics(run_uuid text,key text,step integer,value real);')
    db.execute('insert into runs values(?,?,?)',('r','FINISHED',root.as_uri()))
    db.executemany('insert into params values(?,?,?)',[('r',k,str(v)) for k,v in cfg.items()])
    tags={'outcome':'COMPLETE','trial_identifier':'s:0','best_checkpoint_sha256':post.sha(path/'best_state.pt'),'best_checkpoint':'training/best_state.pt'}
    db.executemany('insert into tags values(?,?,?)',[('r',k,v) for k,v in tags.items()])
    for row in hist.to_dict('records'):
        db.executemany('insert into metrics values(?,?,?,?)',[('r',k,int(row['epoch']),v) for k,v in row.items() if k!='epoch'])
    best={'best_epoch':2,'objective_value':.1,**{'best_val_'+k:v for k,v in selection['best_validation'].items()}}
    db.executemany('insert into metrics values(?,?,?,?)',[('r',k,0,v) for k,v in best.items()])
    t=trial(0,.1,params);t.user_attrs={'mlflow_run_id':'r'};t.intermediate_values={1:.2,2:.1}
    study=SimpleNamespace(study_name='s',user_attrs={'contract':{'model':'cnn_only'}})
    assert post.audit_trial(db,t,study,cfg['dataset_manifest_sha256'])['run_id']=='r'
    if tamper=='history':db.execute("delete from metrics where key='val_mae' and step=2")
    elif tamper=='checkpoint':(path/'best_state.pt').write_bytes(b'tampered')
    else:db.execute("update tags set value='wrong' where key='trial_identifier'")
    with pytest.raises(ValueError):post.audit_trial(db,t,study,cfg['dataset_manifest_sha256'])
