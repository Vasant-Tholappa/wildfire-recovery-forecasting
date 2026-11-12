"""Fixed Option B refits. Development targets only; no test-target gate exists here."""
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sqlite3
import time

import numpy as np
import pandas as pd
import torch
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset

import manual_phase6_postoptuna as evidence
from manual_phase6_build_dataset import FEATURES, KEYS, align, fill_images, load_targets
from wildfire_recovery_forecasting.cnn import CNNRegressor
from wildfire_recovery_forecasting.cnn_training import seed_everything, train_epoch

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data/processed/phase6_cnn_dataset_v1'
SOURCE = ROOT / 'data/processed/patch_model_eligible_terrain_v2.parquet'
OUTPUT = ROOT / 'data/processed/phase7_option_b_v1'
DATA_HASH = '1d7019a466e459a625b0e6ee00a73d92219dda846c1cbddce142285221d1b063'
PREP_HASH = '3cef2cbb2f178251fe47563bd6664b385d1a28f2830d192063f761877d218757'
TEST_KEYS = 'dddc96134b7d1ac581632e673ccdeff70dd990428373f890cb14658e4948ec42'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def clean(value):
    """Strict JSON, with unset NaN estimator defaults recorded explicitly."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def write(path, value):
    path = Path(path)
    require(not path.exists(), f'Refusing overwrite: {path}')
    with path.open('x', encoding='utf-8') as f:
        json.dump(clean(value), f, indent=2, allow_nan=False)


def key_hash(rows):
    require(not rows.duplicated(['fire_id', 'patch_id']).any(), 'Duplicate keys')
    return hashlib.sha256(''.join(f'{r.fire_id}\t{r.patch_id}\n'
                                 for r in rows.itertuples()).encode()).hexdigest()


def pooled_medians(pairs):
    medians = []
    for ch in range(2):
        valid = np.concatenate([np.asarray(x[:, ch])[np.asarray(v[:, ch])]
                                for x, v in pairs])
        require(len(valid) > 0 and np.isfinite(valid).all(), 'Invalid valid pixels')
        require((abs(valid) <= 1).all(), 'Expected already clipped valid pixels')
        medians.append(float(np.median(valid, overwrite_input=True)))
    return medians


def fusion_preprocessor(schema):
    return ColumnTransformer([
        ('numeric', StandardScaler(), schema['numeric']),
        ('categorical', OneHotEncoder(categories=schema['categories'],
                                     handle_unknown='ignore', sparse_output=False),
         schema['categorical'])], remainder='drop')


class PreparedDataset(Dataset):
    def __init__(self, images, tabular, targets):
        require(images.shape == (len(targets), 2, 15, 15)
                and tabular.shape == (len(targets), 70), 'Array alignment mismatch')
        self.images, self.tabular, self.targets = images, tabular, targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        return dict(image=torch.from_numpy(np.array(self.images[i], copy=True)),
                    tabular=torch.from_numpy(np.array(self.tabular[i], copy=True)),
                    target=torch.tensor(self.targets[i], dtype=torch.float32))


def fixed_fit(model, loader, epochs, lr, weight_decay, log_epoch):
    require(epochs > 0, 'Positive fixed duration required')
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = []
    for epoch in range(1, epochs + 1):
        start = time.monotonic()
        metrics = train_epoch(model, loader, optimizer, torch.device('cpu'))
        record = dict(epoch=epoch, seconds=time.monotonic()-start,
                      **{'train_'+k: v for k, v in metrics.items()})
        require(all(np.isfinite(v) for v in record.values()), 'Nonfinite training history')
        history.append(record)
        log_epoch(record)
    return history


def historical_runs():
    """Snapshot historical run rows without invoking a writing MLflow client."""
    with evidence.db_read() as db:
        tables = {}
        for table in ('runs', 'params', 'tags', 'metrics', 'latest_metrics'):
            rows = db.execute(f'SELECT * FROM {table}').fetchall()
            names = [r[1] for r in db.execute(f'PRAGMA table_info({table})')]
            position = names.index('run_uuid')
            grouped = {}
            for row in rows:
                grouped.setdefault(row[position], []).append(row)
            tables[table] = {rid: hashlib.sha256(repr(sorted(rs, key=repr)).encode()).hexdigest()
                             for rid, rs in grouped.items()}
        return tables


def prepare(output, schema, manifest, reuse=False):
    indexes, frames, targets, pairs, classical = [], [], [], [], []
    source_hashes = {}
    for split, count in [('train', 61073), ('validation', 19561)]:
        folder = DATA / split
        for name in ('index.parquet', 'tabular.parquet', 'images.npy', 'validity.npy', 'targets.npy'):
            path = folder / name
            expected = manifest['files'][str(Path(split)/name)]['sha256']
            require(sha(path) == expected, f'Source changed: {path}')
            source_hashes[str(path.relative_to(ROOT))] = expected
        ix = pd.read_parquet(folder/'index.parquet', columns=KEYS+['array_index'])
        require(len(ix) == count and ix.split.eq(split).all()
                and np.array_equal(ix.array_index, np.arange(count)), 'Frozen index mismatch')
        key_hash(ix)
        frame = pd.read_parquet(folder/'tabular.parquet', columns=KEYS+FEATURES)
        ordered = align(ix, frame, FEATURES)
        source_inputs = pd.read_parquet(SOURCE, columns=KEYS+FEATURES+['fire_year'],
                                        filters=[('split', '==', split)])
        allowed = [2015, 2016, 2017] if split == 'train' else [2018, 2019]
        require(source_inputs.fire_year.isin(allowed).all(), 'Wrong development year')
        require(len(source_inputs) == (61090 if split == 'train' else 19561), 'Classical count')
        require(np.array_equal(ordered.to_numpy(), align(ix, source_inputs, FEATURES).to_numpy()),
                'Original predictor values/order mismatch')
        labels = load_targets(SOURCE, split)  # Explicit filtered development-only projection.
        y = align(ix, labels, ['nbr_5_mean']).to_numpy().ravel()
        require(np.array_equal(y, np.load(folder/'targets.npy', allow_pickle=False)), 'Target/key mismatch')
        source_inputs['nbr_5_mean'] = align(source_inputs, labels, ['nbr_5_mean']).to_numpy().ravel()
        classical.append(source_inputs)
        indexes.append(ix); frames.append(ordered); targets.append(y)
        pairs.append((np.load(folder/'images.npy', mmap_mode='r', allow_pickle=False),
                      np.load(folder/'validity.npy', mmap_mode='r', allow_pickle=False)))
    require(not set(indexes[0].fire_id) & set(indexes[1].fire_id), 'Development fire overlap')
    ix = pd.concat(indexes, ignore_index=True); raw = pd.concat(frames, ignore_index=True)
    y = np.concatenate(targets); classic = pd.concat(classical, ignore_index=True)
    require(len(ix) == 80634 and len(classic) == 80651 and ix.fire_id.nunique() == 114,
            'Pooled population mismatch')
    key_hash(ix); key_hash(classic)
    require(np.isfinite(classic[FEATURES+['nbr_5_mean']].to_numpy()).all(), 'Nonfinite development')
    medians = pooled_medians(pairs)
    images = (np.load(output/'images.npy', mmap_mode='r') if reuse else
              np.lib.format.open_memmap(output/'images.npy', mode='w+', dtype='float32',
                                       shape=(len(ix), 2, 15, 15)))
    offset = 0
    for x, v in pairs:
        for start in range(0, len(x), 2048):
            block = fill_images(x[start:start+2048], v[start:start+2048], medians)
            if reuse:
                require(np.array_equal(images[offset+start:offset+start+len(block)], block), 'Prepared image changed')
            else:
                images[offset+start:offset+start+len(block)] = block
        offset += len(x)
    if not reuse: images.flush()
    transformer = fusion_preprocessor(schema)
    tab = transformer.fit_transform(raw).astype('float32')
    require(tab.shape == (80634, 70) and np.isfinite(tab).all(), 'Fusion schema/finite mismatch')
    require(transformer.get_feature_names_out().tolist() == schema['output_order'], 'Encoded order')
    scaler = transformer.named_transformers_['numeric']
    prep = dict(schema, mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(),
                variance=scaler.var_.tolist(), fit_split='train+validation', fit_rows=len(ix),
                fill_medians=medians, channel_order=['nbrpre', 'season1'],
                median_policy='Originally valid clipped pixel occurrences; masks exclude old fills',
                ordered_keys_sha256=key_hash(ix), source_hashes=source_hashes)
    if reuse:
        require(json.loads((output/'preprocessing.json').read_text()) == prep, 'Preprocessing changed')
        require(np.array_equal(np.load(output/'tabular_fusion.npy'), tab)
                and np.array_equal(np.load(output/'targets.npy'), y), 'Prepared data changed')
        require(pd.read_parquet(output/'development_index.parquet').equals(ix), 'Prepared index changed')
    else:
        write(output/'preprocessing.json', prep)
        np.save(output/'tabular_fusion.npy', tab); np.save(output/'targets.npy', y)
        ix.to_parquet(output/'development_index.parquet', index=False)
        classic[KEYS].to_parquet(output/'classical_index.parquet', index=False)
    print('PREPARED', len(classic), len(ix), 'medians', medians, flush=True)
    return PreparedDataset(images, tab, y), classic, ix, prep


def existing_folder(output, name):
    recovered = output/'rf_verified_recovery'
    return recovered if name == 'rf' and (recovered/'record.json').exists() else output/name


def requested_models(action, selected):
    require(action in ('prepare','train','verify'), 'Unknown action')
    require((selected in ('rf','xgb','cnn_only','fusion')) if action=='train' else selected is None,
            'Train requires one explicit model; other actions take no model')
    return () if action=='prepare' else ((selected,) if action=='train' else ('rf','xgb','cnn_only','fusion'))


def verify_record(folder, parent, schema, dataset, classical, ix, output):
    import skops.io as sio
    import cloudpickle
    rec = json.loads((folder/'record.json').read_text())
    name = rec['model']; rows = classical if name in ('rf','xgb') else ix
    artifact = (ROOT/rec['artifact']).resolve()
    require(artifact.is_relative_to(output) and sha(artifact)==rec['sha256'], 'Saved artifact hash/path mismatch')
    require(rec['parent']==parent and rec['rows']==len(rows)
            and rec['ordered_keys_sha256']==key_hash(rows)
            and rec['preprocessing_sha256']==sha(output/'preprocessing.json')
            and rec['feature_order']==FEATURES and rec['encoded_order']==schema['output_order'], 'Saved fit identity mismatch')
    with evidence.db_read() as db:
        status,_,_,tags=evidence.run_record(db,rec['run_id'])
        require(status=='FINISHED' and tags.get('artifact_sha256')==rec['sha256'], 'Unfinished or conflicting MLflow run')
    if name in ('rf','xgb'):
        if name=='rf':
            require(set(sio.get_untrusted_types(file=artifact)) <= {'sklearn.tree._tree.Tree'}, 'RF types')
            model=sio.load(artifact,trusted=['sklearn.tree._tree.Tree'])
            require(len(model.named_steps['model'].estimators_)==100,'RF tree count')
        else:
            with artifact.open('rb') as f:model=cloudpickle.load(f)
            require(model.named_steps['model'].get_booster().num_boosted_rounds()==300,'XGB rounds')
        require(clean(model.named_steps['model'].get_params())==parent['parameters'],'Estimator configuration changed')
        require(list(model.feature_names_in_)==FEATURES and model.named_steps['preprocess'].get_feature_names_out().tolist()==schema['output_order'],'Fitted schema changed')
        predicted=model.predict(classical[FEATURES].iloc[:4])
    else:
        epochs=14 if name=='cnn_only' else 5
        history=pd.read_csv(folder/'history.csv')
        require(rec['epochs']==epochs and list(history.epoch)==list(range(1,epochs+1))
                and rec['selection']=='FINAL_FIXED_EPOCH','Incomplete fixed duration')
        model=CNNRegressor(**parent['architecture'])
        model.load_state_dict(torch.load(artifact,map_location='cpu',weights_only=True),strict=True)
        require(all(torch.isfinite(v).all() for v in model.state_dict().values()),'Nonfinite weights')
        model.eval()
        with torch.no_grad():
            predicted=model(torch.from_numpy(np.array(dataset.images[:4],copy=True)),
                            torch.from_numpy(dataset.tabular[:4])).numpy()
    require(predicted.shape==(4,) and np.isfinite(predicted).all(),'Reload smoke failed')
    print('ALREADY VERIFIED; NO RETRAINING',name,rec['run_id'],flush=True)
    return rec


def run(output, action='prepare', selected=None, restart_incomplete=False):
    import mlflow
    import skops.io as sio
    import cloudpickle
    output = Path(output).resolve()
    require(output == OUTPUT.resolve(), 'Only isolated versioned output allowed')
    requested=requested_models(action,selected)
    resume = output.exists()
    require(action=='prepare' or resume, 'Run prepare first')
    require(sha(DATA/'manifest.json') == DATA_HASH and sha(DATA/'tabular_preprocessing.json') == PREP_HASH,
            'Frozen dataset identity changed')
    schema = json.loads((DATA/'tabular_preprocessing.json').read_text())
    require(schema['input_order'] == FEATURES and len(schema['numeric']) == 16
            and list(map(len, schema['categories'])) == [44, 10], 'Frozen features changed')
    manifest = json.loads((DATA/'manifest.json').read_text())
    audit = json.loads((evidence.OUT/'audit.json').read_text())
    comparison = json.loads((evidence.OUT/'comparison.json').read_text())
    parents, models, protected = {}, {}, {}
    for name in ('rf', 'xgb'):
        rec = comparison['provenance'][name]
        require(sha(rec['artifact']) == rec['sha256'], 'Option A artifact mismatch')
        model, verified = evidence.load_classical(name, FEATURES)
        require(verified['encoded_features'] == schema['output_order'], 'Classical schema mismatch')
        models[name] = model
        parents[name] = dict(run_id=verified['run_id'], artifact=verified['artifact'],
                             sha256=verified['sha256'], parameters=clean(model.named_steps['model'].get_params()))
        protected[verified['artifact']] = verified['sha256']
    for name, trial, epochs in [('cnn_only', 17, 14), ('fusion', 61, 5)]:
        rec = audit['models'][name]['top'][0]
        require(rec['trial'] == trial and rec['best_epoch'] == epochs, 'Parent identity mismatch')
        folder = Path(rec['artifact']); artifact = folder/'best_state.pt'
        require(sha(artifact) == rec['checkpoint_sha256'], 'Parent checkpoint changed')
        cfg = json.loads((folder/'model_configuration.json').read_text())
        parents[name] = dict(run_id=rec['run_id'], artifact=str(artifact), sha256=sha(artifact),
                             parameters=rec['params'], architecture=cfg, selected_epoch=epochs)
        for path in folder.iterdir():
            if path.is_file(): protected[str(path)] = sha(path)
    for path in list((ROOT/'src/wildfire_recovery_forecasting').glob('cnn*.py')) + list(ROOT.glob('manual_phase6*.py')) + [ROOT/'docs/phase6_technical_notes.md']:
        protected[str(path)] = sha(path)
    before = historical_runs()
    if not resume:
        output.mkdir()
        (output/'protocol_before_fit.md').write_bytes((ROOT/'docs/phase7_execution_plan.md').read_bytes())
        write(output/'preservation_before.json', dict(files=protected, historical_runs=before))
    else:
        original=json.loads((output/'preservation_before.json').read_text())
        require(all(sha(p)==h for p,h in original['files'].items()), 'Original protected file changed')
        require(all(before[t].get(r)==h for t,rs in original['historical_runs'].items() for r,h in rs.items()), 'Historical record changed')
    dataset, classical, ix, prep = prepare(output, schema, manifest, reuse=resume)
    env = {p: importlib.metadata.version(p) for p in
           ('torch', 'numpy', 'pandas', 'scikit-learn', 'xgboost', 'mlflow', 'skops')}
    code = {str(Path(__file__).relative_to(ROOT)): sha(__file__),
            'tests/test_phase7_refit.py': sha(ROOT/'tests/test_phase7_refit.py')}
    if not resume: write(output/'environment.json', env)
    else: require(json.loads((output/'environment.json').read_text())==env, 'Environment changed')
    if action=='prepare':
        print('PREPARATION VERIFIED; NO MODEL TRAINING',flush=True)
        return
    mlflow.set_tracking_uri('sqlite:///'+(ROOT/'mlflow.db').as_posix())
    require(mlflow.active_run() is None, 'Existing active MLflow run')
    mlflow.set_experiment('P1 - Wildfire Recovery - Phase 7 Option B')
    records = {}
    for name in requested:
        completed=existing_folder(output,name)
        if (completed/'record.json').exists():
            records[name]=verify_record(completed,parents[name],schema,dataset,classical,ix,output)
            continue
        require(action=='train',f'Missing verified {name}; freeze blocked, no training from verify')
        folder = output/name
        interrupted_run = None
        if folder.exists() and name in ('cnn_only','fusion'):
            require(restart_incomplete, 'Incomplete neural attempt: explicit --restart-incomplete required')
            interrupted_run=json.loads((folder/'intent.json').read_text())['run_id']
            # Preserve evidence; the new model starts at seed 42, not partial weights.
            archive=output/(name+'_interrupted_'+str(time.time_ns()))
            require(folder.resolve().is_relative_to(output) and archive.resolve().is_relative_to(output),'Unsafe archive path')
            folder.rename(archive)
            print('PRESERVED INCOMPLETE ATTEMPT',archive,flush=True)
        recovering = folder.exists()
        recovery_artifact = folder/'model.skops'
        if recovering:
            require(resume and name=='rf' and recovery_artifact.is_file()
                    and not (folder/'record.json').exists(), 'Ambiguous existing model; manual review required')
            old_intent=json.loads((folder/'intent.json').read_text())
            folder=output/'rf_verified_recovery'
        folder.mkdir()
        parent = parents[name]; start = time.monotonic()
        rows = classical if name in models else ix
        metadata = dict(model=name, option='B', parent=parent, seed=42, rows=len(rows),
                        ordered_keys_sha256=key_hash(rows), dataset_manifest_sha256=DATA_HASH,
                        preprocessing_sha256=sha(output/'preprocessing.json'), environment=env,
                        code_sha256=code, target_policy='2015-2019 only; 2020 sealed',
                        feature_order=FEATURES, encoded_order=schema['output_order'])
        metadata['preprocessing_policy'] = ('Numeric passthrough; frozen categorical vocabulary; fitted pipeline contains preprocessing'
                                             if name in models else 'Pooled image medians; pooled numeric scaling; frozen categorical vocabulary')
        if interrupted_run:
            metadata['preserved_interrupted_run_id']=interrupted_run
            metadata['restart_policy']='Fresh seed-42 fit for full fixed duration; no partial-weight reuse'
        if recovering:
            require(old_intent['ordered_keys_sha256']==metadata['ordered_keys_sha256']
                    and old_intent['parent']==parent, 'Recovery identity mismatch')
            metadata['recovered_fit_run_id']=old_intent['run_id']
            metadata['recovery']='Saved completed RF fit; verification-only new run; no retraining'
        with mlflow.start_run(run_name='Option B '+name) as active:
            metadata['run_id'] = active.info.run_id
            write(folder/'intent.json', metadata)
            mlflow.set_tags({'option':'B', 'parent_run_id':parent['run_id'], 'target_gate':'SEALED'})
            mlflow.log_params(dict(model=name, seed=42, fit_rows=len(rows),
                                  dataset_manifest_sha256=DATA_HASH, ordered_keys_sha256=key_hash(rows)))
            try:
                if name in models:
                    model = clone(models[name])
                    model.set_params(preprocess__categorical__categories=schema['categories'])
                    require(clean(model.named_steps['model'].get_params()) == parent['parameters'], 'Estimator changed')
                    if recovering:
                        require(set(sio.get_untrusted_types(file=recovery_artifact)) <= {'sklearn.tree._tree.Tree'}, 'RF recovery types')
                        model=sio.load(recovery_artifact, trusted=['sklearn.tree._tree.Tree'])
                        require(clean(model.named_steps['model'].get_params())==parent['parameters'], 'Recovery settings')
                    else:
                        model.fit(classical[FEATURES], classical.nbr_5_mean.to_numpy())
                    require(model.named_steps['preprocess'].get_feature_names_out().tolist() == schema['output_order'], 'Fitted schema')
                    artifact = folder/('model.skops' if name == 'rf' else 'model.pkl')
                    if name == 'rf':
                        if recovering:
                            artifact.write_bytes(recovery_artifact.read_bytes())
                        else:
                            sio.dump(model, artifact)
                        unknown = sio.get_untrusted_types(file=artifact)
                        require(set(unknown) <= {'sklearn.tree._tree.Tree'}, 'Unexpected saved RF types')
                        loaded = sio.load(artifact, trusted=['sklearn.tree._tree.Tree'])
                        require(len(loaded.named_steps['model'].estimators_) == 100, 'RF trees')
                    else:
                        with artifact.open('xb') as f: cloudpickle.dump(model, f)
                        with artifact.open('rb') as f: loaded = cloudpickle.load(f)
                        require(loaded.named_steps['model'].get_booster().num_boosted_rounds() == 300, 'XGB rounds')
                    require(list(loaded.feature_names_in_) == FEATURES, 'Reload schema')
                    predicted = loaded.predict(classical[FEATURES].iloc[:4])
                    np.testing.assert_allclose(predicted, model.predict(classical[FEATURES].iloc[:4]), rtol=0, atol=1e-12)
                else:
                    seed_everything(42, 4)
                    model = CNNRegressor(**parent['architecture'])  # Fresh weights, never load A.
                    loader = DataLoader(dataset, batch_size=parent['parameters']['batch_size'], shuffle=True,
                                        generator=torch.Generator().manual_seed(42), num_workers=0, drop_last=False)
                    epochs = 14 if name == 'cnn_only' else 5
                    metadata.update(epochs=epochs, selection='FINAL_FIXED_EPOCH', threads=4,
                                    optimizer='AdamW', training_loss='MSE',
                                    steps_per_epoch=len(loader), architecture=parent['architecture'])
                    mlflow.log_params(dict(epochs=epochs, **parent['parameters']))
                    def log_epoch(record):
                        mlflow.log_metrics({k:v for k,v in record.items() if k != 'epoch'}, step=record['epoch'])
                        print(name, json.dumps(record), flush=True)
                    history = fixed_fit(model, loader, epochs, parent['parameters']['learning_rate'],
                                        parent['parameters']['weight_decay'], log_epoch)
                    pd.DataFrame(history).to_csv(folder/'history.csv', index=False)
                    require(len(history) == epochs, 'Incomplete fit')
                    artifact = folder/'final_state.pt'
                    torch.save(model.state_dict(), artifact)
                    loaded = CNNRegressor(**parent['architecture'])
                    loaded.load_state_dict(torch.load(artifact, map_location='cpu', weights_only=True), strict=True)
                    loaded.eval(); model.eval()
                    with torch.no_grad():
                        x=torch.from_numpy(np.array(dataset.images[:4], copy=True)); t=torch.from_numpy(dataset.tabular[:4])
                        predicted=loaded(x,t).numpy()
                        np.testing.assert_array_equal(predicted, model(x,t).numpy())
                require(predicted.shape == (4,) and np.isfinite(predicted).all(), 'Smoke inference failed')
                metadata.update(artifact=str(artifact.relative_to(ROOT)), sha256=sha(artifact),
                                seconds=time.monotonic()-start, verification='PASS',
                                smoke='4 development inputs; reload agreement atol=1e-12, rtol=0')
                write(folder/'record.json', metadata)
                mlflow.log_metrics({'fit_and_verification_seconds':metadata['seconds']})
                mlflow.set_tags({'verification':'PASS', 'artifact_sha256':metadata['sha256']})
                mlflow.log_artifact(str(output/'preprocessing.json'), artifact_path='preprocessing')
                mlflow.log_artifacts(str(folder), artifact_path='model')
                records[name] = metadata
                print('VERIFIED', name, metadata['run_id'], metadata['sha256'], flush=True)
            except BaseException as exc:
                mlflow.set_tag('error', str(exc)[:3000])
                raise
    require(all(sha(path) == h for path,h in protected.items()), 'Protected Option A/Phase 6 file changed')
    after = historical_runs()
    require(all(after[t].get(rid) == h for t, runs in before.items() for rid,h in runs.items()), 'Historical run modified')
    require(all(sha(ROOT/path)==h for path,h in prep['source_hashes'].items()), 'Development source changed')
    if action=='train':
        print('REQUESTED MODEL COMPLETE; RUN verify AFTER ALL FOUR',flush=True)
        return
    test = pd.read_parquet(DATA/'test/index.parquet', columns=['fire_id','patch_id','array_index'])
    require(len(test)==30606 and test.fire_id.nunique()==37 and key_hash(test)==TEST_KEYS
            and np.array_equal(test.array_index,np.arange(30606)), 'Test key identity changed')
    for parent in parents.values():
        parent.update(input_schema=dict(features=FEATURES, encoded_order=schema['output_order'],
                                       image_shape=[2,15,15], channel_order=['nbrpre','season1']),
                      preprocessing=dict(path=str((DATA/'tabular_preprocessing.json').relative_to(ROOT)),
                                         sha256=PREP_HASH, dataset_manifest_sha256=DATA_HASH,
                                         image_medians=manifest['fill_medians']))
    freeze = dict(status='FROZEN', target_gate='SEALED_REQUIRES_SEPARATE_AUTHORIZATION',
                  option_a=parents, option_b=records, preprocessing=prep,
                  evaluation=dict(rows=30606, fires=37, ordered_keys_sha256=TEST_KEYS,
                                  index_sha256=sha(DATA/'test/index.parquet'), primary='MAE',
                                  secondary=['RMSE','R2','equal_fire_MAE'],
                                  per_fire=['MAE','RMSE','patch_count'], paired='Option B minus A per family',
                                  prediction_clipping=False, option_a_primary=True,
                                  interpretation='B has no independent 2018-2019 validation; report all eight'),
                  protocol_path=str((output/'protocol_before_fit.md').relative_to(ROOT)),
                  protocol_sha256=sha(output/'protocol_before_fit.md'),
                  artifact_files={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file()},
                  preservation='All protected files and historical MLflow rows unchanged',
                  test_inference='Deferred; no test arrays or targets read')
    freeze_path=output/'eight_model_freeze.json'
    if freeze_path.exists():
        saved=json.loads(freeze_path.read_text())
        require(saved['option_a']==clean(parents) and saved['option_b']==clean(records)
                and saved['evaluation']==freeze['evaluation'],'Existing freeze identity mismatch')
        require(all(sha(output/p)==h for p,h in saved['artifact_files'].items()),'Frozen file changed')
        print('EXISTING FREEZE VERIFIED; NOT OVERWRITTEN',flush=True)
    else:
        write(freeze_path, freeze)
    print('EIGHT MODEL FREEZE READY', output/'eight_model_freeze.json', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','train','verify'])
    parser.add_argument('--model',choices=['rf','xgb','cnn_only','fusion'])
    parser.add_argument('--restart-incomplete',action='store_true',help='Archive an incomplete neural attempt and restart same seed/configuration; completed fits are always skipped')
    args = parser.parse_args()
    run(OUTPUT,action=args.action,selected=args.model,restart_incomplete=args.restart_incomplete)
