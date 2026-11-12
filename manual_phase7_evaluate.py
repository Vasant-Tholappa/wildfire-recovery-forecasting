"""Frozen eight-model inference; outcomes require explicit CLI authorization."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

import manual_phase7_refit as refit
from wildfire_recovery_forecasting.cnn import CNNRegressor
from wildfire_recovery_forecasting.cnn_training import seed_everything
from wildfire_recovery_forecasting.evaluation import evaluate_model

ROOT=refit.ROOT
DATA=refit.DATA
FREEZE=refit.OUTPUT/'eight_model_freeze.json'
OUT=ROOT/'data/processed/phase7_test_v1'
KEYS=['fire_id','patch_id']
FAMILIES=('rf','xgb','cnn_only','fusion')
require=refit.require
sha=refit.sha


def exact_align(index, frame):
    require(not index[KEYS].isna().any().any() and not frame[KEYS].isna().any().any(), 'Null keys')
    require(not index.duplicated(KEYS).any() and not frame.duplicated(KEYS).any(), 'Duplicate keys')
    require(len(index)==len(frame) and set(map(tuple,index[KEYS].to_numpy()))==set(map(tuple,frame[KEYS].to_numpy())), 'Exact key set mismatch')
    return frame.set_index(KEYS).loc[pd.MultiIndex.from_frame(index[KEYS])].reset_index()


def read_targets(index, authorized=False, source=refit.SOURCE):
    require(authorized is True, 'Explicit 2020 target authorization required before IO')
    frame=pd.read_parquet(source,columns=KEYS+['nbr_5_mean'],filters=[('split','==','test')])
    aligned=exact_align(index,frame)
    y=aligned.nbr_5_mean.to_numpy(dtype='float64')
    require(np.isfinite(y).all(), 'Missing/nonfinite targets')
    return y


def checked_path(path, digest):
    path=(ROOT/path).resolve()
    require(path.is_relative_to(ROOT) and path.is_file(), 'Untrusted/missing artifact path')
    require(sha(path)==digest, f'Artifact hash mismatch: {path}')
    return path


def verify_freeze():
    freeze=json.loads(FREEZE.read_text())
    require(freeze['status']=='FROZEN', 'Not frozen')
    require(freeze['evaluation']['rows']==30606 and freeze['evaluation']['fires']==37
            and freeze['evaluation']['ordered_keys_sha256']==refit.TEST_KEYS
            and freeze['evaluation']['primary']=='MAE'
            and freeze['evaluation']['secondary']==['RMSE','R2','equal_fire_MAE']
            and freeze['evaluation']['prediction_clipping'] is False, 'Evaluation contract changed')
    checked_path(DATA/'manifest.json',refit.DATA_HASH)
    checked_path(DATA/'tabular_preprocessing.json',refit.PREP_HASH)
    for path,digest in freeze['artifact_files'].items():
        checked_path(refit.OUTPUT/path,digest)
    checked_path(freeze['protocol_path'],freeze['protocol_sha256'])
    schema=json.loads((DATA/'tabular_preprocessing.json').read_text())
    for option in ('option_a','option_b'):
        require(set(freeze[option])==set(FAMILIES), 'Candidate set mismatch')
        for name,rec in freeze[option].items():
            checked_path(rec['artifact'],rec['sha256'])
            if option=='option_a':
                checked_path(rec['preprocessing']['path'],rec['preprocessing']['sha256'])
                require(rec['input_schema']['features']==refit.FEATURES
                        and rec['input_schema']['encoded_order']==schema['output_order']
                        and rec['input_schema']['channel_order']==['nbrpre','season1'], 'Option A schema')
            else:
                checked_path(refit.OUTPUT/'preprocessing.json',rec['preprocessing_sha256'])
                require(rec['feature_order']==refit.FEATURES and rec['encoded_order']==schema['output_order']
                        and rec['verification']=='PASS', 'Option B schema/verification')
                require(rec['parent']['run_id']==freeze['option_a'][name]['run_id'], 'Parent mismatch')
                if name in ('cnn_only','fusion'):
                    require(rec['epochs']==(14 if name=='cnn_only' else 5)
                            and rec['selection']=='FINAL_FIXED_EPOCH','Epoch policy mismatch')
            with refit.evidence.db_read() as db:
                status,_,_,_=refit.evidence.run_record(db,rec['run_id'])
                require(status=='FINISHED','Candidate run not finished')
    return freeze,schema


def encode(frame, schema):
    require(schema['input_order']==refit.FEATURES, 'Raw feature order')
    numeric=(frame[schema['numeric']].to_numpy(dtype=float)-np.array(schema['mean']))/np.array(schema['scale'])
    blocks=[numeric]
    for name,categories in zip(schema['categorical'],schema['categories']):
        blocks.append((frame[name].to_numpy()[:,None]==np.array(categories)[None,:]).astype(float))
    result=np.concatenate(blocks,axis=1).astype('float32')
    require(result.shape==(len(frame),70) and np.isfinite(result).all(),'Encoded input mismatch')
    return result


def load_model(name, rec, schema):
    path=checked_path(rec['artifact'],rec['sha256'])
    if name in ('rf','xgb'):
        if name=='rf':
            import skops.io as sio
            require(set(sio.get_untrusted_types(file=path)) <= {'sklearn.tree._tree.Tree'},'Untrusted RF type')
            model=sio.load(path,trusted=['sklearn.tree._tree.Tree'])
        else:
            # Only the hash-verified local frozen model, never a user-supplied pickle.
            import cloudpickle
            with path.open('rb') as f:model=cloudpickle.load(f)
        require(list(model.feature_names_in_)==refit.FEATURES,'Fitted feature order')
        prep=model.named_steps['preprocess']
        require(prep.get_feature_names_out().tolist()==schema['output_order'],'Encoded schema')
        encoder=prep.named_transformers_['categorical']
        require([c.tolist() for c in encoder.categories_]==schema['categories']
                and encoder.handle_unknown=='ignore','Category vocabulary')
        params=rec.get('parameters',rec.get('parent',{}).get('parameters'))
        require(refit.clean(model.named_steps['model'].get_params())==params,'Estimator parameters changed')
    else:
        cfg=rec.get('architecture',rec.get('parent',{}).get('architecture'))
        model=CNNRegressor(**cfg)
        model.load_state_dict(torch.load(path,weights_only=True,map_location='cpu'),strict=True)
        require(all(torch.isfinite(v).all() for v in model.state_dict().values()),'Nonfinite model state')
        model.eval()
    return model


def infer(freeze,schema):
    manifest=json.loads((DATA/'manifest.json').read_text())
    for name in ('index.parquet','images.npy','validity.npy','tabular.parquet','tabular_fusion.npy'):
        checked_path(DATA/'test'/name,manifest['files'][str(Path('test')/name)]['sha256'])
    index=pd.read_parquet(DATA/'test/index.parquet',columns=KEYS+['array_index','split'])
    require(len(index)==30606 and index.fire_id.nunique()==37 and index.split.eq('test').all()
            and np.array_equal(index.array_index,np.arange(30606))
            and refit.key_hash(index)==refit.TEST_KEYS,'Test identity mismatch')
    checked_path(DATA/'test/index.parquet',freeze['evaluation']['index_sha256'])
    frame=exact_align(index,pd.read_parquet(DATA/'test/tabular.parquet',columns=KEYS+refit.FEATURES))
    original=pd.read_parquet(refit.SOURCE,columns=KEYS+refit.FEATURES+['fire_year'],filters=[('split','==','test')])
    original=exact_align(index,original)
    require(original.fire_year.eq(2020).all() and np.array_equal(frame[refit.FEATURES].to_numpy(),original[refit.FEATURES].to_numpy()),'Original input mismatch')
    images=np.load(DATA/'test/images.npy',mmap_mode='r',allow_pickle=False)
    masks=np.load(DATA/'test/validity.npy',mmap_mode='r',allow_pickle=False)
    require(images.shape==masks.shape==(30606,2,15,15) and images.dtype==np.float32 and masks.dtype==bool,'Image dimensions/dtypes')
    tab_a=encode(frame,schema);tab_b=encode(frame,freeze['preprocessing'])
    require(np.array_equal(tab_a,np.load(DATA/'test/tabular_fusion.npy',mmap_mode='r')),'Option A preprocessing reconstruction')
    seed_everything(42,4)
    predictions={}
    for option in ('option_a','option_b'):
        for name in FAMILIES:
            model=load_model(name,freeze[option][name],schema)
            if name in ('rf','xgb'):
                pred=model.predict(frame[refit.FEATURES])
            else:
                chunks=[]
                for start in range(0,len(index),512):
                    block=np.array(images[start:start+512],copy=True)
                    require(np.isfinite(block).all() and (abs(block)<=1).all(),'Invalid image input')
                    if option=='option_b':
                        block=refit.fill_images(block,masks[start:start+512],freeze['preprocessing']['fill_medians'])
                    tab=tab_a if option=='option_a' else tab_b
                    with torch.no_grad():chunks.append(model(torch.from_numpy(block),torch.from_numpy(tab[start:start+512])).numpy())
                pred=np.concatenate(chunks)
            require(pred.shape==(30606,) and np.isfinite(pred).all(),'Prediction coverage/finite check')
            key=option+'_'+name;predictions[key]=pred.astype(float)
            print('TARGET-FREE INFERENCE PASS',key,len(pred),flush=True)
            del model
    return index,frame,predictions,masks


def summaries(index,y,predictions):
    metrics=[];per_fire=[];paired=[]
    for name,pred in predictions.items():
        metrics.append(dict(model=name,**evaluate_model(y,pred,index.fire_id)))
        for fire,positions in index.groupby('fire_id',sort=True).indices.items():
            error=y[positions]-pred[positions]
            per_fire.append(dict(model=name,fire_id=fire,patch_count=len(positions),
                                 mae=float(np.abs(error).mean()),rmse=float(np.sqrt(np.mean(error**2)))))
    table=pd.DataFrame(metrics).sort_values(['mae','model']).reset_index(drop=True)
    table['mae_rank']=np.arange(1,len(table)+1)
    fire_table=pd.DataFrame(per_fire)
    for family in FAMILIES:
        a='option_a_'+family;b='option_b_'+family
        aa=table.set_index('model').loc[a];bb=table.set_index('model').loc[b]
        paired.append(dict(family=family,scope='all_patches',**{k:float(bb[k]-aa[k]) for k in ('mae','rmse','r2','equal_fire_mae')}))
        for fire in sorted(index.fire_id.unique()):
            rows=fire_table[fire_table.fire_id.eq(fire)].set_index('model')
            paired.append(dict(family=family,scope=fire,mae=float(rows.loc[b,'mae']-rows.loc[a,'mae']),
                               rmse=float(rows.loc[b,'rmse']-rows.loc[a,'rmse'])))
    return table,fire_table,pd.DataFrame(paired)


def shift_diagnostics(frame,masks,schema):
    parts=[pd.read_parquet(DATA/s/'tabular.parquet',columns=refit.FEATURES) for s in ('train','validation')]
    rows=[]
    for cohort,df in [('A_neural_train',parts[0]),('B_neural_development',pd.concat(parts)),('test_inputs',frame)]:
        for feature in schema['numeric']:
            v=df[feature]
            rows.append(dict(cohort=cohort,feature=feature,rows=len(df),missing=int(v.isna().sum()),
                             mean=float(v.mean()),std=float(v.std()),q05=float(v.quantile(.05)),
                             q50=float(v.quantile(.5)),q95=float(v.quantile(.95))))
    categories={cohort:{name:int((~df[name].isin(cats)).sum()) for name,cats in zip(schema['categorical'],schema['categories'])}
                for cohort,df in [('A_neural_train',parts[0]),('B_neural_development',pd.concat(parts)),('test_inputs',frame)]}
    coverage={'full_mean_by_channel':masks.mean(axis=(0,2,3)).tolist(),
              'central_mean_by_channel':masks[:,:,6:9,6:9].mean(axis=(0,2,3)).tolist(),
              'note':'Development baselines use CNN-retained rows; 17 additional classical training rows are not included.'}
    return pd.DataFrame(rows),dict(unknown_categories=categories,test_coverage=coverage)


def create_output(path):
    Path(path).mkdir(parents=True,exist_ok=False)


def run(mode,authorized=False):
    require(mode in ('preflight','evaluate'),'Unknown mode')
    if mode=='evaluate':require(authorized is True,'Use --authorize-2020-targets; no outcomes read')
    start=time.monotonic();freeze,schema=verify_freeze()
    if mode=='evaluate':
        create_output(OUT)  # Exclusive reservation before any target access.
        refit.write(OUT/'started.json',dict(status='STARTED',freeze_sha256=sha(FREEZE),code_sha256=sha(__file__)))
        refit.write(OUT/'frozen_protocol.json',freeze)
    try:
        index,frame,predictions,masks=infer(freeze,schema)
        if mode=='preflight':
            print('PREFLIGHT PASS: 8 models, 30606 rows, 37 fires; NO TARGET ACCESS',flush=True)
            return
        y=read_targets(index,authorized=authorized)
        table,per_fire,paired=summaries(index,y,predictions)
        historical=json.loads((refit.evidence.OUT/'comparison.json').read_text())
        for metric in ('mae','rmse','r2','equal_fire_mae'):
            table['historical_validation_'+metric]=[historical['metrics'][name.removeprefix('option_a_')]['full'][metric]
                if name.startswith('option_a_') else np.nan for name in table.model]
        table.to_csv(OUT/'comparison.csv',index=False);per_fire.to_csv(OUT/'per_fire_metrics.csv',index=False)
        paired.to_csv(OUT/'paired_comparisons.csv',index=False)
        pd.DataFrame(dict(fire_id=index.fire_id,patch_id=index.patch_id,nbr_5_mean=y,**predictions)).to_parquet(OUT/'predictions.parquet',index=False)
        shifts,shiftmeta=shift_diagnostics(frame,masks,schema)
        shifts.to_csv(OUT/'input_shift.csv',index=False);refit.write(OUT/'input_shift.json',shiftmeta)
        refit.write(OUT/'comparison.json',dict(metrics=table.astype(object).where(pd.notna(table),None).to_dict('records'),
                    interpretation='A validation: 2018-2019, 19561 rows; test: 2020, 30606 rows. B has no independent validation. No causal spatial attribution. Clustered patches are not independent. No further tuning.'))
        audit=dict(status='PASS',freeze_sha256=sha(FREEZE),code_sha256=sha(__file__),
                   target_source=str(refit.SOURCE.relative_to(ROOT)),target_filter="split == 'test' (already eligible v2 source)",
                   target_source_metadata_sha256=sha(refit.SOURCE.with_suffix('.metadata.json')),
                   roles_sha256=sha(refit.SOURCE.with_suffix('.roles.json')),
                   historical_validation_sha256=sha(refit.evidence.OUT/'comparison.json'),
                   ordered_keys_sha256=refit.key_hash(index),aligned_target_sha256=__import__('hashlib').sha256(y.astype('<f8').tobytes()).hexdigest(),
                   rows=len(index),fires=index.fire_id.nunique(),seconds=time.monotonic()-start,
                   model_provenance={k:freeze[k] for k in ('option_a','option_b')},
                   outputs={p.name:sha(p) for p in OUT.iterdir() if p.is_file()})
        lines=['# Independent 2020 evaluation','','```text',table.to_string(index=False),'```','',
               'All eight models use identical rows. Option A is primary; B is secondary.',
               'No independent 2018-2019 validation exists for B. Ranking is not an unbiased post-selection guarantee.',
               'Patch dependence and cohort shift limit interpretation. No causal spatial-information claim or further tuning.']
        (OUT/'summary.md').write_text('\n'.join(lines),encoding='utf-8')
        audit['outputs']={p.name:sha(p) for p in OUT.iterdir() if p.is_file()}
        refit.write(OUT/'evaluation_audit.json',audit)  # Completion marker written last.
        print('EVALUATION PASS',OUT,flush=True)
    except BaseException as exc:
        if mode=='evaluate':refit.write(OUT/'failure.json',dict(status='FAILED_REVIEW_REQUIRED',error=repr(exc)))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['preflight','evaluate'])
    p.add_argument('--authorize-2020-targets',action='store_true')
    args=p.parse_args();run(args.mode,args.authorize_2020_targets)
