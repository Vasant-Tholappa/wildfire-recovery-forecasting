"""Shared local MLflow lifecycle for standalone training and Optuna trials."""
import hashlib
import json
import os
from pathlib import Path
import time

import pandas as pd


def tracked_fit(model, train_loader, val_loader, device, output, *, metadata,
                tracking_uri, dataset_root=None, fit_function=None, epoch_hook=None,
                run_hook=None, tags=None, **fit_kwargs):
    import mlflow
    from wildfire_recovery_forecasting.cnn_training import fit
    if os.environ.get('MLFLOW_RUN_ID') or mlflow.active_run() is not None:
        raise ValueError('A new independent MLflow run is required')
    fit_function = fit_function or fit
    output = Path(output)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment('P1 - Wildfire Recovery - Phase 6')
    run = mlflow.start_run(run_name=metadata['model'])
    run_id = run.info.run_id
    status = 'FAILED'
    started = time.monotonic()
    records = []
    result = None
    try:
        if run_hook:
            run_hook(run_id)
        mlflow.log_params(metadata)
        for key, value in (tags or {}).items():
            mlflow.set_tag(key, str(value))
        mlflow.log_dict(model.configuration, 'training/model_configuration.json')
        mlflow.log_text(str(model), 'training/architecture.txt')
        mlflow.log_dict(metadata, 'training/configuration.json')
        if dataset_root is not None:
            for name in ('manifest.json', 'tabular_preprocessing.json'):
                mlflow.log_artifact(str(Path(dataset_root)/name), artifact_path='dataset_metadata')

        def log_epoch(record):
            records.append(record)
            mlflow.log_metrics({k:v for k,v in record.items() if k != 'epoch'}, step=record['epoch'])
            print(json.dumps(dict(run_id=run_id, **record)), flush=True)
            if epoch_hook:
                epoch_hook(record)

        result = fit_function(model, train_loader, val_loader, device, output,
                              log_epoch=log_epoch, **fit_kwargs)
        status = 'COMPLETE'
        return dict(result, run_id=run_id)
    except BaseException as exc:
        status = 'PRUNED' if type(exc).__name__ == 'TrialPruned' else 'FAILED'
        mlflow.set_tag('error', f'{type(exc).__name__}: {exc}'[:4000])
        if status == 'PRUNED':
            mlflow.set_tag('pruned_epoch', records[-1]['epoch'] if records else 'unknown')
        raise
    finally:
        try:
            # Preserve best-so-far evidence even on pruning/interruption.
            if result is None and records:
                best = min(records, key=lambda r:r['val_mae'])
                result = dict(best_epoch=best['epoch'], epochs_completed=len(records),
                              best_validation={k[4:]:v for k,v in best.items() if k.startswith('val_')})
            summary = dict(fit_seconds=time.monotonic()-started,
                           epoch_seconds_total=sum(r['seconds'] for r in records),
                           epochs_completed=len(records))
            if result:
                summary.update(best_epoch=result['best_epoch'],
                               **{'best_val_'+k:v for k,v in result['best_validation'].items()})
                if status == 'COMPLETE':
                    summary['objective_value'] = result['best_validation']['mae']
            mlflow.log_metrics(summary)
            mlflow.set_tag('outcome', status)
            if output.exists():
                (output/'configuration.json').write_text(json.dumps(metadata, indent=2))
                (output/'model_configuration.json').write_text(json.dumps(model.configuration, indent=2))
                (output/'architecture.txt').write_text(str(model))
                if records:
                    pd.DataFrame(records).to_csv(output/'history.csv', index=False)
                if result:
                    (output/'selection.json').write_text(json.dumps(dict(result, outcome=status), indent=2))
                checkpoint = output/'best_state.pt'
                if checkpoint.exists():
                    mlflow.set_tag('best_checkpoint', 'training/best_state.pt')
                    mlflow.set_tag('best_checkpoint_sha256', hashlib.sha256(checkpoint.read_bytes()).hexdigest())
                mlflow.log_artifacts(str(output), artifact_path='training')
        except BaseException:
            status = 'FAILED'
            mlflow.set_tag('outcome', status)
            raise
        finally:
            mlflow.end_run(status='FINISHED' if status in ('COMPLETE', 'PRUNED') else 'FAILED')
