"""Bounded, sequential, resumable Phase 6 Optuna studies; CPU development data only."""
import argparse
from contextlib import contextmanager
import gc
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import statistics
import time
import traceback
import uuid

import optuna
from optuna.trial import TrialState

ROOT = Path(__file__).resolve().parent
VERSION = 'phase6_tpe_v1'
SPACE = dict(learning_rate=[1e-4, 3e-3, 'log'], batch_size=[64,128,256],
             weight_decay=[1e-6,1e-2,'log'], dropout=[0.0,0.4],
             conv_widths=['16_32','32_64'], embedding_dim=[32,64,128], head_dim=[16,32,64],
             tabular_dim=[16,32,64], tabular_dropout=[0.0,0.4])
BASELINE = dict(learning_rate=.001, batch_size=128, weight_decay=.0001, dropout=0.0,
                conv_widths='16_32', embedding_dim=64, head_dim=32, tabular_dim=32, tabular_dropout=0.0)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def sample(trial, model):
    params = {}
    for name, spec in SPACE.items():
        if name.startswith('tabular_') and model != 'fusion':
            continue
        if name in ('learning_rate','weight_decay'):
            params[name] = trial.suggest_float(name, spec[0], spec[1], log=True)
        elif name in ('dropout','tabular_dropout'):
            params[name] = trial.suggest_float(name, *spec)
        else:
            params[name] = trial.suggest_categorical(name, spec)
    return params


def architecture(params, model):
    return dict(model_type=model, conv_widths=[int(x) for x in params['conv_widths'].split('_')],
                **{k:params[k] for k in ('embedding_dim','head_dim','dropout','tabular_dim','tabular_dropout') if k in params})


@contextmanager
def exclusive_lock(path):
    """OS-held lock releases on process death; never steal another live invocation's lock."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as handle:
        if handle.tell() == 0:
            handle.write(b'0'); handle.flush()
        handle.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == 'nt':
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def counts(study):
    trials = study.get_trials(deepcopy=False)
    return {state.name:sum(t.state == state for t in trials) for state in TrialState}


def open_study(storage, name, contract):
    study = optuna.create_study(storage=storage, study_name=name, direction='minimize', load_if_exists=True,
                               pruner=optuna.pruners.MedianPruner(n_startup_trials=12, n_warmup_steps=8, interval_steps=1, n_min_trials=5))
    existing = study.user_attrs.get('contract')
    if existing is None and not study.trials:
        study.set_user_attr('contract', contract)
    elif existing != contract:
        raise ValueError(f'{name}: dataset/search/training identity mismatch; refusing resume')
    if not study.trials:
        study.enqueue_trial({k:v for k,v in BASELINE.items() if contract['model']=='fusion' or not k.startswith('tabular_')})
    return study


def recover(study, client):
    """Only called under the exclusive lock. Never repeat an interrupted trial."""
    for trial in study.get_trials(states=(TrialState.RUNNING,)):
        rid = trial.user_attrs.get('mlflow_run_id')
        if rid:
            run = client.get_run(rid)  # Unreadable identity is a blocker, not an invented failure.
            if run.data.tags.get('trial_identifier') != f'{study.study_name}:{trial.number}':
                raise ValueError('Interrupted trial MLflow identity mismatch')
            if run.info.status == 'FINISHED' and run.data.tags.get('outcome') == 'COMPLETE':
                value = run.data.metrics['best_val_mae']
                if not math.isfinite(value) or abs(value-run.data.metrics['objective_value']) > 1e-12:
                    raise ValueError('Interrupted trial objective mismatch')
                study.tell(trial.number, value)
                continue
            if run.info.status == 'FINISHED' and run.data.tags.get('outcome') == 'PRUNED':
                study.tell(trial.number, state=TrialState.PRUNED)
                continue
            client.set_tag(rid, 'interruption_recovery', 'Interrupted trial retained; not retried')
            client.set_tag(rid, 'outcome', 'FAILED')
            if run.info.status == 'RUNNING':
                client.set_terminated(rid, status='FAILED')
        study.tell(trial.number, state=TrialState.FAIL)


def ordered_finished(studies):
    return sorted([(t.datetime_complete, model, t) for model,s in studies.items() for t in s.trials
                   if t.state.is_finished()], key=lambda x:(x[0], x[1], x[2].number))


def choose(studies, limits):
    candidates = [m for m,s in studies.items() if counts(s)['COMPLETE'] < limits[m]]
    if not candidates:
        return None
    recent = ordered_finished(studies)
    if len(candidates)>1 and len(recent)>=2 and recent[-1][1] == recent[-2][1]:
        candidates = [m for m in candidates if m != recent[-1][1]]
    return min(candidates, key=lambda m:(counts(studies[m])['COMPLETE'],
                sum(t.state.is_finished() for t in studies[m].trials), m))


def summary(study):
    done = study.get_trials(states=(TrialState.COMPLETE,))
    last = [t for t in study.trials if t.state.is_finished()][-12:]
    durations = [(t.datetime_complete-t.datetime_start).total_seconds() for t in last]
    best = study.best_trial if done else None
    return dict(study=study.study_name, counts=counts(study), attempted=sum(t.state != TrialState.WAITING for t in study.trials),
                best_trial=best.number if best else None, best_mae=best.value if best else None,
                best_params=best.params if best else None,
                best_mlflow_run_id=best.user_attrs.get('mlflow_run_id') if best else None,
                total_trial_seconds=sum((t.datetime_complete-t.datetime_start).total_seconds() for t in study.trials if t.state.is_finished()),
                recent=[dict(number=t.number, state=t.state.name, value=t.value) for t in last],
                median_recent_seconds=statistics.median(durations) if durations else None)


def milestones(study):
    n = counts(study)['COMPLETE']
    if n in (25,50,75) and f'milestone_{n}' not in study.user_attrs:
        report = summary(study)
        previous = study.user_attrs.get(f'milestone_{n-25}')
        report['improvement_since_previous'] = (previous['best_mae']-report['best_mae']) if previous else None
        report['improved_since_previous'] = report['improvement_since_previous'] > 0 if previous else None
        study.set_user_attr(f'milestone_{n}', report)  # Durable SQLite summary; never a stopping condition.
        print('MILESTONE', json.dumps(report), flush=True)


def schedule(studies, objective, *, target, minutes, sampler_seed, max_new=None, max_attempts=None, clock=time.monotonic):
    start = clock()
    limits = {m:min(target, counts(s)['COMPLETE']+max_new) if max_new else target for m,s in studies.items()}
    initial_attempts = {m:sum(t.state != TrialState.WAITING for t in s.trials) for m,s in studies.items()}
    reason = 'target achieved'
    while True:
        recent = ordered_finished(studies)
        if len(recent)>=3 and all(t.state == TrialState.FAIL for _,_,t in recent[-3:]):
            reason = 'error: three consecutive failed trials; intervention required'; break
        eligible = {m:s for m,s in studies.items() if max_attempts is None or
                    sum(t.state != TrialState.WAITING for t in s.trials) < initial_attempts[m]+max_attempts}
        model = choose(eligible, limits)
        if model is None:
            reason = 'target achieved' if all(counts(s)['COMPLETE']>=target for s in studies.values()) else ('error: smoke attempt failed; inspect artifacts' if max_attempts is not None else 'session completion cap'); break
        if clock()-start >= minutes*60:
            reason = 'deadline'; break
        study = studies[model]
        # Seed each ask deterministically from identity/attempt index. TPE reads ALL stored history.
        seed = int(digest([sampler_seed, study.study_name, len(study.trials)])[:8],16)
        study.sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=12)
        trial = study.ask()
        try:
            value = objective(trial, model)
            if not math.isfinite(value):
                raise ValueError('Nonfinite objective')
            study.tell(trial, value)
        except optuna.TrialPruned:
            study.tell(trial, state=TrialState.PRUNED)
        except KeyboardInterrupt:
            study.tell(trial, state=TrialState.FAIL)
            reason = 'interruption'; break
        except Exception:
            traceback.print_exc()
            study.tell(trial, state=TrialState.FAIL)
        finally:
            gc.collect()
        milestones(study)
        print('PROGRESS', model, json.dumps(summary(study)), flush=True)
    return dict(reason=reason, elapsed_seconds=clock()-start,
                models={m:summary(s) for m,s in studies.items()})


def make_objective(args, datasets, contracts, tracking_uri, output_root):
    from wildfire_recovery_forecasting.cnn import CNNRegressor
    from wildfire_recovery_forecasting.cnn_training import seed_everything, make_loaders
    from wildfire_recovery_forecasting.cnn_tracking import tracked_fit
    from mlflow.tracking import MlflowClient
    import torch
    import numpy as np
    client = MlflowClient(tracking_uri=tracking_uri)

    def train_trial(trial, model_type):
        params = sample(trial, model_type)
        seed_everything(args.seed, args.threads)
        model = CNNRegressor(**architecture(params, model_type))
        loaders = make_loaders(*datasets, params['batch_size'], args.seed)
        contract = contracts[model_type]
        identity = f'{trial.study.study_name}:{trial.number}'
        metadata = dict(model=model_type, **params, seed=args.seed, threads=args.threads, device='cpu',
                        max_epochs=contract['epochs'], patience=5, min_delta=.0001, optimizer='AdamW',
                        training_loss='MSE', selection_metric='val_mae', num_workers=0, deterministic_algorithms=True,
                        parameters=sum(p.numel() for p in model.parameters()), train_rows=len(datasets[0]), validation_rows=len(datasets[1]),
                        torch_version=str(torch.__version__), numpy_version=np.__version__, optuna_version=optuna.__version__,
                        dataset_manifest_sha256=contract['dataset_hash'], dataset_version=contract['dataset_version'],
                        study_name=trial.study.study_name, trial_number=trial.number, search_space_version=VERSION,
                        contract_hash=digest(contract), source_hashes=json.dumps(contract['source_hashes'],sort_keys=True))
        output = output_root/trial.study.study_name/f'trial_{trial.number:04d}_{uuid.uuid4().hex[:8]}'

        def epoch_hook(record):
            trial.report(record['val_mae'], step=record['epoch'])
            if record['epoch'] >= 9 and trial.should_prune():
                trial.set_user_attr('pruning_reason', f'Median pruner at epoch {record["epoch"]}')
                raise optuna.TrialPruned(trial.user_attrs['pruning_reason'])

        result = tracked_fit(model, *loaders, 'cpu', output, metadata=metadata, tracking_uri=tracking_uri,
                             dataset_root=None if args.synthetic_smoke else args.data_root,
                             tags={'trial_identifier':identity,'optuna.study_name':trial.study.study_name,
                                   'optuna.trial_number':trial.number,'smoke_test':args.smoke_test or args.synthetic_smoke},
                             run_hook=lambda rid:trial.set_user_attr('mlflow_run_id',rid), epoch_hook=epoch_hook,
                             epochs=contract['epochs'], lr=params['learning_rate'], weight_decay=params['weight_decay'],
                             patience=5, min_delta=.0001)
        value = result['best_validation']['mae']
        run = client.get_run(result['run_id'])
        if run.info.status != 'FINISHED' or run.data.tags.get('outcome') != 'COMPLETE' or abs(run.data.metrics['best_val_mae']-value)>1e-12:
            raise RuntimeError('MLflow objective consistency failed')
        trial.set_user_attr('output',str(output))
        trial.set_user_attr('epochs_completed', result['epochs_completed'])
        return value
    def objective(trial, model_type):
        started = time.monotonic()
        try:
            return train_trial(trial, model_type)
        except BaseException as exc:
            if not trial.user_attrs.get('mlflow_run_id'):
                # Configuration/setup failures still receive their own traceable run.
                import mlflow
                mlflow.set_tracking_uri(tracking_uri)
                mlflow.set_experiment('P1 - Wildfire Recovery - Phase 6')
                with mlflow.start_run(run_name=model_type) as run:
                    trial.set_user_attr('mlflow_run_id',run.info.run_id)
                    mlflow.log_params(dict(model=model_type, **trial.params, seed=args.seed, threads=args.threads,
                                           dataset_version=contracts[model_type]['dataset_version'],
                                           dataset_manifest_sha256=contracts[model_type]['dataset_hash'],
                                           search_space_version=VERSION, optimizer='AdamW', training_loss='MSE',
                                           contract_hash=digest(contracts[model_type])))
                    mlflow.log_dict(contracts[model_type], 'training/study_contract.json')
                    mlflow.set_tags({'trial_identifier':f'{trial.study.study_name}:{trial.number}',
                                     'optuna.study_name':trial.study.study_name,'optuna.trial_number':trial.number,
                                     'outcome':'FAILED','error':f'{type(exc).__name__}: {exc}'[:4000]})
                    mlflow.end_run(status='FAILED')
            elif not isinstance(exc, optuna.TrialPruned):
                rid = trial.user_attrs['mlflow_run_id']
                client.set_tag(rid, 'outcome', 'FAILED')
                client.set_tag(rid, 'error', f'{type(exc).__name__}: {exc}'[:4000])
                client.set_terminated(rid, status='FAILED')
            raise
        finally:
            duration = time.monotonic()-started
            trial.set_user_attr('trial_seconds',duration)
            rid = trial.user_attrs.get('mlflow_run_id')
            if rid:
                client.log_metric(rid,'trial_seconds',duration)
    return objective


def main():
    invocation_started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',choices=['both','cnn_only','fusion'],default='both')
    parser.add_argument('--target-completed-per-model',type=int,default=75)
    parser.add_argument('--time-budget-minutes',type=float,default=540)
    parser.add_argument('--max-new-completed-per-model',type=int)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--sampler-seed',type=int,default=2026)
    parser.add_argument('--study-storage',type=Path,default=ROOT/'data/processed/phase6_optuna/studies.sqlite')
    parser.add_argument('--data-root',type=Path,default=ROOT/'data/processed/phase6_cnn_dataset_v1')
    parser.add_argument('--dry-run',action='store_true')
    smoke = parser.add_mutually_exclusive_group()
    smoke.add_argument('--smoke-test',action='store_true',help='Isolated real-data studies: exactly one epoch/trial per model')
    smoke.add_argument('--synthetic-smoke',action='store_true',help='Tiny synthetic data and isolated local tracking')
    args = parser.parse_args()
    if (not 1<=args.target_completed_per_model<=75 or not math.isfinite(args.time_budget_minutes) or args.time_budget_minutes<=0
            or args.threads<1 or not 0<=args.seed<2**32 or not 0<=args.sampler_seed<2**32
            or (args.max_new_completed_per_model is not None and args.max_new_completed_per_model<1)):
        parser.error('Invalid bounds; cumulative target must be 1..75')
    if os.environ.get('MLFLOW_RUN_ID'):
        parser.error('Unset MLFLOW_RUN_ID; each trial requires a fresh run')
    is_smoke = args.smoke_test or args.synthetic_smoke
    prefix = VERSION
    if is_smoke:
        prefix += '_smoke_'+uuid.uuid4().hex[:10]
        args.study_storage = args.study_storage.parent/prefix/'studies.sqlite'
        args.target_completed_per_model = 1
    if args.synthetic_smoke:
        dataset_hash, dataset_version = digest('synthetic16_8_v1'), 'synthetic16_8_v1'
    else:
        raw = (args.data_root/'manifest.json').read_bytes()  # Metadata only, no test arrays.
        manifest = json.loads(raw)
        if manifest['status']!='VALIDATED' or manifest['counts'] != {'train':61073,'validation':19561,'test':30606}:
            raise ValueError('Frozen dataset metadata mismatch')
        dataset_hash, dataset_version = hashlib.sha256(raw).hexdigest(), manifest['version']
    sources = ['manual_phase6_optuna.py','src/wildfire_recovery_forecasting/cnn.py',
               'src/wildfire_recovery_forecasting/cnn_training.py','src/wildfire_recovery_forecasting/cnn_tracking.py']
    source_hashes = {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources}
    models = ['cnn_only','fusion'] if args.models=='both' else [args.models]
    contracts = {m:dict(version=VERSION, model=m, space=SPACE, dataset_hash=dataset_hash,
                       dataset_version=dataset_version, objective='minimum 2018-2019 validation patch MAE',
                       epochs=1 if is_smoke else 30, patience=5,min_delta=.0001,training_seed=args.seed,
                       sampler_seed=args.sampler_seed,threads=args.threads,source_hashes=source_hashes,
                       optuna_version=optuna.__version__,torch_version=version('torch'),numpy_version=version('numpy'),pruner='median12_warm8_min5_epoch9',sampler='per_ask_seed_TPE12_v1') for m in models}
    print(json.dumps(dict(storage=str(args.study_storage),studies=[prefix+'_'+m for m in models],
                          target=args.target_completed_per_model,budget_minutes=args.time_budget_minutes,contracts=contracts),indent=2))
    if args.dry_run:
        return
    from mlflow.tracking import MlflowClient
    from wildfire_recovery_forecasting.cnn_training import PatchDataset
    tracking_uri = 'sqlite:///'+((args.study_storage.parent/'mlflow.sqlite') if args.synthetic_smoke else ROOT/'mlflow.db').as_posix()
    with exclusive_lock(args.study_storage.with_suffix('.lock')):
        storage = optuna.storages.RDBStorage('sqlite:///'+args.study_storage.resolve().as_posix(),engine_kwargs={'connect_args':{'timeout':60}})
        studies = {m:open_study(storage,prefix+'_'+m,contracts[m]) for m in models}
        client = MlflowClient(tracking_uri=tracking_uri)
        for study in studies.values():
            recover(study,client)
            milestones(study)
        if args.synthetic_smoke:
            from manual_phase6_train import SyntheticDataset
            datasets = (SyntheticDataset('train',args.seed),SyntheticDataset('validation',args.seed+1))
        else:
            datasets = (PatchDataset(args.data_root,'train'),PatchDataset(args.data_root,'validation'))
        result = schedule(studies,make_objective(args,datasets,contracts,tracking_uri,args.study_storage.parent/'trials'),
                          target=args.target_completed_per_model, minutes=max(0.0,args.time_budget_minutes-(time.monotonic()-invocation_started)/60),
                          sampler_seed=args.sampler_seed,max_new=args.max_new_completed_per_model, max_attempts=1 if is_smoke else None)
        result['storage'] = str(args.study_storage)
        result['invocation_seconds'] = time.monotonic()-invocation_started
        result['target_completed_per_model'] = args.target_completed_per_model
        report = args.study_storage.parent/('session_'+uuid.uuid4().hex[:10]+'.json')
        report.write_text(json.dumps(result,indent=2))
        print('FINAL',json.dumps(result,indent=2),flush=True)
        if result['reason'].startswith('error'):
            raise SystemExit(1)


if __name__=='__main__':
    main()
