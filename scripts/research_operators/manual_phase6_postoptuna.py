"""Local post-Optuna audit and frozen validation evaluation. Never read test arrays."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
from urllib.parse import urlparse, unquote
from urllib.request import url2pathname

import numpy as np
import pandas as pd
import optuna
import torch

from wildfire_recovery_forecasting.cnn import CNNRegressor

ROOT = Path(__file__).resolve().parent
DATA = ROOT/'data/processed/phase6_cnn_dataset_v1'
OUT = ROOT/'data/processed/phase6_postoptuna_v1'
MODELS = ('cnn_only','fusion')
CLASSICAL = {'rf':'b175c6f10eeb4ebfa8c0930192468d39','xgb':'2cb3267c3dac4d4c80fe118e274637be'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def local_path(uri):
    parsed=urlparse(uri)
    require(parsed.scheme=='file' and not parsed.netloc, 'Only local file artifacts permitted')
    path=Path(url2pathname(unquote(parsed.path))).resolve()
    require(path.is_relative_to(ROOT.resolve()), 'Artifact outside trusted repository')
    return path


def db_read():
    return sqlite3.connect((ROOT/'mlflow.db').as_uri()+'?mode=ro', uri=True)


def run_record(db, rid):
    row=db.execute('select status,artifact_uri from runs where run_uuid=?',(rid,)).fetchone()
    require(row is not None, f'Missing MLflow run {rid}')
    params=dict(db.execute('select key,value from params where run_uuid=?',(rid,)))
    tags=dict(db.execute('select key,value from tags where run_uuid=?',(rid,)))
    return row[0],local_path(row[1]),params,tags


def top_distinct(trials, count=2):
    found=[];seen=set()
    for trial in sorted((t for t in trials if t.state==optuna.trial.TrialState.COMPLETE),key=lambda t:(t.value,t.number)):
        key=json.dumps(trial.params,sort_keys=True)
        if key not in seen:
            found.append(trial);seen.add(key)
        if len(found)==count:break
    require(len(found)==count, 'Insufficient distinct completed configurations')
    return found


def audit_trial(db, trial, study, dataset_hash):
    rid=trial.user_attrs.get('mlflow_run_id')
    require(bool(rid), f'Trial {trial.number} has no MLflow link')
    status,path,params,tags=run_record(db,rid)
    expected='COMPLETE' if trial.state==optuna.trial.TrialState.COMPLETE else 'PRUNED'
    require(status=='FINISHED' and tags.get('outcome')==expected, f'{rid}: outcome mismatch')
    require(tags.get('trial_identifier')==f'{study.study_name}:{trial.number}', f'{rid}: identity mismatch')
    require(params.get('dataset_manifest_sha256')==dataset_hash and params.get('seed')=='42', f'{rid}: dataset/seed mismatch')
    for key,value in trial.params.items():
        require(params.get(key)==str(value), f'{rid}: parameter mismatch {key}')
    path=path/'training'
    cfg=json.loads((path/'configuration.json').read_text())
    require(all(params.get(k)==str(v) for k,v in cfg.items()), f'{rid}: saved configuration mismatch')
    hist=pd.read_csv(path/'history.csv')
    require(list(hist.epoch)==list(range(1,len(hist)+1)) and 1<=len(hist)<=30, f'{rid}: epoch sequence')
    metric_rows=db.execute('select key,step,value from metrics where run_uuid=?',(rid,)).fetchall()
    for key in ('train_loss','train_mae','val_loss','val_mae','val_rmse','val_r2','val_equal_fire_mae','seconds'):
        rows=sorted((step,value) for k,step,value in metric_rows if k==key)
        require(len(rows)==len(hist) and [s for s,v in rows]==list(hist.epoch),f'{rid}: missing/duplicate {key} steps')
        require(np.allclose([v for s,v in rows],hist[key],atol=1e-10,rtol=0),f'{rid}: history mismatch {key}')
    require(np.isfinite(hist.select_dtypes('number')).all().all(),f'{rid}: nonfinite history')
    selection=json.loads((path/'selection.json').read_text())
    best=hist.loc[hist.val_mae.idxmin()]
    require(selection['best_epoch']==int(best.epoch) and selection['epochs_completed']==len(hist),f'{rid}: best epoch mismatch')
    metrics={k:v for k,s,v in metric_rows if k.startswith('best_') or k in ('objective_value','trial_seconds','fit_seconds')}
    for key,value in selection['best_validation'].items():
        require(abs(value-best['val_'+key])<=1e-10 and abs(metrics['best_val_'+key]-value)<=1e-10,f'{rid}: best metric {key}')
    require(metrics['best_epoch']==selection['best_epoch'],f'{rid}: best epoch metric')
    require(sha(path/'best_state.pt')==tags.get('best_checkpoint_sha256') and tags.get('best_checkpoint')=='training/best_state.pt',f'{rid}: checkpoint hash/path')
    for name in ('architecture.txt','model_configuration.json'):
        require((path/name).is_file(),f'{rid}: missing {name}')
    from manual_phase6_optuna import architecture
    model_cfg=json.loads((path/'model_configuration.json').read_text())
    expected_cfg=dict(tabular_dim=32,tabular_dropout=0.0)
    expected_cfg.update(architecture(trial.params,study.user_attrs['contract']['model']))
    require(model_cfg==expected_cfg,f'{rid}: architecture/configuration mismatch')
    require(params.get('train_rows')=='61073' and params.get('validation_rows')=='19561'
            and params.get('max_epochs')=='30' and params.get('patience')=='5'
            and params.get('min_delta')=='0.0001' and params.get('threads')=='4'
            and params.get('training_loss')=='MSE',f'{rid}: training protocol mismatch')
    require(sha(path.parent/'dataset_metadata/manifest.json')==dataset_hash,f'{rid}: logged manifest mismatch')
    require((path.parent/'dataset_metadata/tabular_preprocessing.json').read_bytes()==(DATA/'tabular_preprocessing.json').read_bytes(),f'{rid}: preprocessing mismatch')
    if expected=='COMPLETE':
        require(abs(trial.value-best.val_mae)<=1e-10 and abs(metrics['objective_value']-trial.value)<=1e-10,f'{rid}: objective mismatch')
    else:
        require(int(tags['pruned_epoch'])==len(hist) and bool(trial.user_attrs.get('pruning_reason')),f'{rid}: pruning metadata')
    require(set(trial.intermediate_values)==set(hist.epoch),f'{rid}: Optuna epoch history')
    require(all(abs(trial.intermediate_values[int(r.epoch)]-r.val_mae)<1e-10 for r in hist.itertuples()),f'{rid}: Optuna values')
    return dict(trial=trial.number,run_id=rid,state=expected,params=trial.params,
                best_epoch=selection['best_epoch'],metrics=selection['best_validation'],epochs=len(hist),
                duration=metrics.get('trial_seconds',metrics.get('fit_seconds')),artifact=str(path),checkpoint_sha256=sha(path/'best_state.pt'))


def load_checkpoint(record):
    path=Path(record['artifact'])
    require(sha(path/'best_state.pt')==record['checkpoint_sha256'],'Checkpoint changed since audit')
    cfg=json.loads((path/'model_configuration.json').read_text())
    model=CNNRegressor(**cfg)
    model.load_state_dict(torch.load(path/'best_state.pt',map_location='cpu',weights_only=True),strict=True)
    model.eval()
    with torch.no_grad():
        pred=model(torch.zeros(2,2,15,15),torch.zeros(2,70))
    require(pred.shape==(2,) and torch.isfinite(pred).all(),'Checkpoint reconstruction failed')
    return model


def audit():
    result={'dataset_hash':sha(DATA/'manifest.json'),'models':{}}
    seen=set()
    with db_read() as db:
        for model,expected_pruned in [('cnn_only',12),('fusion',3)]:
            study=optuna.load_study(study_name='phase6_tpe_v1_'+model,storage='sqlite:///'+(ROOT/'data/processed/phase6_optuna/studies.sqlite').as_posix())
            counts={s.name:sum(t.state==s for t in study.trials) for s in optuna.trial.TrialState}
            require(counts==dict(RUNNING=0,COMPLETE=75,PRUNED=expected_pruned,FAIL=0,WAITING=0),f'{model}: unexpected states {counts}')
            records=[]
            for trial in study.trials:
                record=audit_trial(db,trial,study,result['dataset_hash'])
                require(record['run_id'] not in seen,'Duplicate MLflow run ID')
                seen.add(record['run_id']);records.append(record)
            by_number={r['trial']:r for r in records}
            top=[by_number[t.number] for t in top_distinct(study.trials)]
            for record in top:load_checkpoint(record)
            result['models'][model]=dict(counts=counts,top=top,trials=records,contract=study.user_attrs['contract'])
            print(model,counts,'top',[(r['trial'],r['metrics']['mae']) for r in top],flush=True)
    result['status']='PASS'
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'audit.json').write_text(json.dumps(result,indent=2))
    return result


def aligned_frame(index, frame):
    keys=['fire_id','patch_id']
    require(not index.duplicated(keys).any() and not frame.duplicated(keys).any(),'Duplicate patch keys')
    require(len(frame)==len(index) and set(map(tuple,frame[keys].to_numpy()))==set(map(tuple,index[keys].to_numpy())),'Patch-key set mismatch')
    return frame.set_index(keys).loc[pd.MultiIndex.from_frame(index[keys])].reset_index()


def development(split):
    require(split in ('train','validation'),'Test data access forbidden')
    from wildfire_recovery_forecasting.cnn_training import PatchDataset
    ds=PatchDataset(DATA,split)
    frame=aligned_frame(ds.index,pd.read_parquet(DATA/split/'tabular.parquet'))
    require(frame['split'].eq(split).all(),'Split mismatch')
    return ds,frame,np.asarray(ds.targets,dtype=float)


def load_classical(name, features):
    import yaml
    rid=CLASSICAL[name]
    with db_read() as db:
        status,runpath,params,tags=run_record(db,rid)
        require(status=='FINISHED','Classical run not finished')
        feature_set=json.loads((runpath/'inputs/feature_set.json').read_text())
        require(feature_set['features']==features and feature_set['trained_on']=='2015-2017','Frozen feature/training mismatch')
        rows=db.execute('select artifact_location from logged_models where source_run_id=?',(rid,)).fetchall()
        require(len(rows)==1,'Ambiguous classical artifact')
        artifact=local_path(rows[0][0])
        meta=yaml.safe_load((artifact/'MLmodel').read_text())
        require(meta['run_id']==rid,'Classical model provenance mismatch')
        flavor=meta['flavors']['sklearn'];file=artifact/flavor['pickled_model']
        require(file.resolve().is_relative_to(artifact) and file.stat().st_size==meta['model_size_bytes'],'Classical artifact size/path mismatch')
        fingerprint=sha(file)
        if name=='rf':
            import skops.io as sio
            require(flavor['serialization_format']=='skops','Unexpected RF serialization')
            unknown=sio.get_untrusted_types(file=file)
            require(set(unknown)<= {'sklearn.tree._tree.Tree'},f'Unexpected skops types: {unknown}')
            pipeline=sio.load(file,trusted=['sklearn.tree._tree.Tree'])
        else:
            require(flavor['serialization_format']=='cloudpickle','Unexpected XGBoost serialization')
            # Only this allowlisted, locally produced run; never accept arbitrary pickle paths.
            import cloudpickle
            with file.open('rb') as handle:pipeline=cloudpickle.load(handle)
        require(list(pipeline.feature_names_in_)==features,'Fitted model predictor order mismatch')
        prep=pipeline.named_steps['preprocess']
        require(prep.remainder=='drop','Unexpected preprocessing remainder')
        numeric=[f for f in features if f not in ('dominant_evt','ignition_month')]
        transforms={k:(transform,list(cols)) for k,transform,cols in prep.transformers_ if k!='remainder'}
        require(transforms['numeric'][1]==numeric and list(transforms['categorical'][1])==['dominant_evt','ignition_month'],'Preprocessing roles mismatch')
        # sklearn replaces fitted passthrough with an identity FunctionTransformer.
        from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
        transformer=transforms['numeric'][0]
        require(isinstance(transformer,FunctionTransformer) and transformer.func is None,'Unexpected numerical transformation')
        encoder=transforms['categorical'][0]
        require(isinstance(encoder,OneHotEncoder) and encoder.handle_unknown=='ignore','Unexpected categorical encoding')
        actual=pipeline.named_steps['model'].get_params()
        for key,value in params.items():
            if key in actual:require(str(actual[key])==value,f'Classical hyperparameter mismatch: {key}')
        metrics=dict(db.execute('select key,value from metrics where run_uuid=?',(rid,)))
    return pipeline,dict(run_id=rid,artifact=str(file),sha256=fingerprint,recorded_metrics=metrics,
                         encoded_features=prep.get_feature_names_out().tolist(),
                         integrity_limit='No historical cryptographic checksum; verified local run provenance, recorded size, schema/settings and reproduced validation metrics')


@torch.no_grad()
def predict_cnn(record, ds):
    from torch.utils.data import DataLoader
    model=load_checkpoint(record)
    pred=[];positions=[]
    for batch in DataLoader(ds,batch_size=256,shuffle=False,num_workers=0):
        pred.append(model(batch['image'],batch['tabular']).numpy());positions.extend(batch['index'].tolist())
    require(positions==list(range(len(ds))),'Prediction ordering changed')
    result=np.concatenate(pred)
    require(np.isfinite(result).all(),'Nonfinite predictions')
    return result


def comparisons(index, y, predictions, overlap):
    from wildfire_recovery_forecasting.evaluation import evaluate_model
    require(len(index)==len(y),'Index/target mismatch')
    require(len(set(overlap))==53 and set(overlap)<=set(index.patch_id),'Expected 53 unique validation overlap IDs')
    keep=~index.patch_id.isin(overlap).to_numpy()
    metrics={};per_fire={}
    for name,pred in predictions.items():
        metrics[name]=dict(full=evaluate_model(y,pred,index.fire_id),without_shared_context=evaluate_model(y[keep],pred[keep],index.fire_id[keep]))
        per_fire[name]=pd.DataFrame({'fire_id':index.fire_id,'absolute_error':np.abs(y-pred)}).groupby('fire_id').absolute_error.mean()
    fire=pd.DataFrame(per_fire)
    fire['patch_count']=index.groupby('fire_id').size()
    pairs=[]
    from itertools import combinations
    for a,b in combinations(predictions,2):
        delta=fire[a]-fire[b]
        pairs.append(dict(model_a=a,model_b=b,patch_mae_difference_a_minus_b=metrics[a]['full']['mae']-metrics[b]['full']['mae'],
                          equal_fire_difference_a_minus_b=float(delta.mean()),fires_a_better=int((delta<0).sum()),
                          fires_b_better=int((delta>0).sum()),ties=int((delta==0).sum())))
        fire[a+'_minus_'+b]=delta
    return metrics,fire,pairs


def evaluate_frozen(audited):
    from wildfire_recovery_forecasting.evaluation import evaluate_model
    from wildfire_recovery_forecasting.cnn_training import seed_everything
    seed_everything(42,4)
    features=json.loads((DATA/'tabular_preprocessing.json').read_text())['input_order']
    train,xt,yt=development('train');val,xv,yv=development('validation')
    require(len(val)==19561 and val.index.fire_id.nunique()==36,'Validation population changed')
    # Input-only crosscheck against the frozen Phase 5 source; no target column loaded.
    source=pd.read_parquet(ROOT/'data/processed/patch_model_eligible_terrain_v2.parquet',
                           columns=['fire_id','patch_id']+features,filters=[('split','==','validation')])
    source=aligned_frame(val.index,source)
    require(np.array_equal(source[features].to_numpy(),xv[features].to_numpy()),'Phase 5/6 predictor values differ')
    predictions={};train_metrics={};provenance={}
    for name in ('rf','xgb'):
        model,record=load_classical(name,features)
        predictions[name]=model.predict(xv[features])
        train_metrics[name]=evaluate_model(yt,model.predict(xt[features]),train.index.fire_id)
        score=evaluate_model(yv,predictions[name],val.index.fire_id)
        require(all(abs(score[k]-record['recorded_metrics']['val_'+k])<1e-8 for k in score),f'{name}: frozen validation metrics not reproduced')
        provenance[name]=record
        print(name,score,flush=True)
    for name in MODELS:
        record=audited['models'][name]['top'][0]
        predictions[name]=predict_cnn(record,val)
        train_metrics[name]=evaluate_model(yt,predict_cnn(record,train),train.index.fire_id)
        metrics=evaluate_model(yv,predictions[name],val.index.fire_id)
        require(abs(metrics['mae']-record['metrics']['mae'])<1e-8,f'{name}: best-checkpoint score not reproduced')
        provenance[name]=dict(run_id=record['run_id'],trial=record['trial'],checkpoint_sha256=record['checkpoint_sha256'])
        print(name,metrics,flush=True)
    overlap=pd.read_csv(DATA/'shared_context_validation_ids.csv').patch_id.tolist()
    metrics,fire,pairs=comparisons(val.index,yv,predictions,overlap)
    output=val.index[['fire_id','patch_id']].copy();output['target']=yv
    for name,pred in predictions.items():output[name]=pred
    output.to_parquet(OUT/'validation_predictions.parquet',index=False)
    fire.to_csv(OUT/'per_fire.csv');pd.DataFrame(pairs).to_csv(OUT/'paired_differences.csv',index=False)
    result=dict(population=19561,fires=36,sensitivity_population=19508,metrics=metrics,training_metrics=train_metrics,
                primary='Single-best Optuna configurations, seed 42; not selected across stability seeds',
                train_population=61073,provenance=provenance,paired_differences=pairs,
                target_precision='Common metrics use frozen float64 targets; original CNN epoch metrics used float32 targets. Agreement tolerance 1e-8.',
                limitations='Validation reused for 75-trial selection; these are development results, not unbiased generalization estimates. No test evaluation.')
    (OUT/'comparison.json').write_text(json.dumps(result,indent=2))
    return result


def save_json(path, value):
    import os,time,uuid
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temp.open('w') as handle:
        json.dump(value,handle,indent=2);handle.flush();os.fsync(handle.fileno())
    for attempt in range(5):
        try:os.replace(temp,path);return
        except PermissionError:
            if attempt==4:raise
            time.sleep(.1*(attempt+1))


def seed_summary(rows):
    frame=pd.DataFrame(rows)
    summaries=[]
    for (model,trial),part in frame.groupby(['model','trial']):
        summaries.append(dict(model=model,trial=int(trial),seeds=sorted(part.seed.tolist()),
                              complete=len(part)==3 and set(part.seed)=={42,123,456},
                              mean_mae=float(part.mae.mean()),std_mae=float(part.mae.std(ddof=1)) if len(part)>1 else None))
    return summaries


def stability(audited):
    """Eight additional fits; resume verified completed jobs, never retry ambiguous jobs."""
    from wildfire_recovery_forecasting.cnn_training import seed_everything,make_loaders
    from wildfire_recovery_forecasting.cnn_tracking import tracked_fit
    from mlflow.tracking import MlflowClient
    from manual_phase6_optuna import architecture
    import time,uuid
    ledger_path=OUT/'seed_jobs.json'
    ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
    train,_,_=development('train');val,_,_=development('validation')
    uri='sqlite:///'+(ROOT/'mlflow.db').as_posix();client=MlflowClient(tracking_uri=uri)
    rows=[]
    for family in MODELS:
        contract=audited['models'][family]['contract']
        require(contract['torch_version']==str(torch.__version__) and contract['numpy_version']==np.__version__,
                'Training library versions changed; seed-equivalence review required')
        for file in ('src/wildfire_recovery_forecasting/cnn.py','src/wildfire_recovery_forecasting/cnn_training.py'):
            require(sha(ROOT/file)==contract['source_hashes'][file],'Model/training implementation changed; cannot claim seed equivalence')
        for record in audited['models'][family]['top']:
            require(contract['training_seed']==42 and contract['epochs']==30 and contract['threads']==4,'Original training protocol mismatch')
            for seed in (42,123,456):
                key=f'{family}_trial{record["trial"]}_seed{seed}'
                identity=dict(model=family,trial=record['trial'],seed=seed,params=record['params'],dataset_hash=audited['dataset_hash'])
                if seed==42:
                    metrics=record['metrics'];rid=record['run_id'];duration=record['duration'];epoch=record['best_epoch']
                else:
                    if key not in ledger:
                        ledger[key]=dict(identity=identity,state='INTENT')
                        save_json(ledger_path,ledger)
                        seed_everything(seed,4)
                        model=CNNRegressor(**architecture(record['params'],family))
                        loaders=make_loaders(train,val,record['params']['batch_size'],seed)
                        metadata=json.loads((Path(record['artifact'])/'configuration.json').read_text())
                        for field in ('study_name','trial_number','contract_hash'):metadata.pop(field,None)
                        metadata.update(seed=seed,experiment_role='post_optuna_seed_stability',parent_run_id=record['run_id'],
                                        source_optuna_trial=record['trial'],model=family,torch_version=str(torch.__version__),
                                        numpy_version=np.__version__,optuna_version=optuna.__version__,
                                        evaluation_workflow_sha256=sha(Path(__file__)))
                        output=OUT/'training'/(key+'_'+uuid.uuid4().hex[:8])
                        def record_run(rid):
                            ledger[key].update(run_id=rid,state='RUNNING');save_json(ledger_path,ledger)
                        started=time.monotonic()
                        tracked_fit(model,*loaders,'cpu',output,metadata=metadata,tracking_uri=uri,dataset_root=DATA,
                                    tags={'postoptuna_job':key,'experiment_role':'seed_stability','parent_run_id':record['run_id']},
                                    run_hook=record_run,epochs=30,lr=record['params']['learning_rate'],weight_decay=record['params']['weight_decay'],patience=5,min_delta=.0001)
                        client.log_metric(ledger[key]['run_id'],'trial_seconds',time.monotonic()-started)
                    job=ledger[key]
                    require(job['identity']==identity and 'run_id' in job,'Ambiguous seed job; inspect rather than retrain')
                    run=client.get_run(job['run_id'])
                    require(run.info.status=='FINISHED' and run.data.tags.get('outcome')=='COMPLETE' and run.data.tags.get('postoptuna_job')==key,'Incomplete/failed seed job requires review; not automatically retried')
                    require(run.data.params['seed']==str(seed) and run.data.params['model']==family
                            and run.data.params['dataset_manifest_sha256']==audited['dataset_hash']
                            and all(run.data.params[k]==str(v) for k,v in record['params'].items()),
                            'Seed run configuration mismatch')
                    path=local_path(run.info.artifact_uri)/'training'
                    require(sha(path/'best_state.pt')==run.data.tags['best_checkpoint_sha256'],'Seed checkpoint hash mismatch')
                    metrics={k:run.data.metrics['best_val_'+k] for k in ('mae','rmse','r2','equal_fire_mae')}
                    rid=job['run_id'];duration=run.data.metrics.get('trial_seconds',run.data.metrics['fit_seconds']);epoch=int(run.data.metrics['best_epoch'])
                    ledger[key]['state']='COMPLETE';save_json(ledger_path,ledger)
                rows.append(dict(model=family,trial=record['trial'],seed=seed,run_id=rid,best_epoch=epoch,duration_seconds=duration,
                                 reused=seed==42,**{k:metrics[k] for k in ('mae','rmse','r2','equal_fire_mae')}))
                pd.DataFrame(rows).to_csv(OUT/'seed_results.csv',index=False)
                save_json(OUT/'seed_summary.json',seed_summary(rows))
                print('SEED RESULT',rows[-1],flush=True)
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluate',action='store_true',help='Audit then compare fixed seed-42 winners and frozen classical models')
    parser.add_argument('--train-seeds',action='store_true',help='Audit then run/resume eight additional 30-epoch-capped stability fits')
    args=parser.parse_args()
    from manual_phase6_optuna import exclusive_lock
    with exclusive_lock(OUT/'evaluation.lock'):
        audited=audit()
        if args.train_seeds:stability(audited)
        if args.evaluate:evaluate_frozen(audited)


if __name__=='__main__':main()
