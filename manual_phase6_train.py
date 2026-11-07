"""Explicit local Phase 6 training. --smoke-test uses only tiny synthetic data."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import uuid

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from wildfire_recovery_forecasting.cnn import CNNRegressor
from wildfire_recovery_forecasting.cnn_training import PatchDataset, seed_everything, make_loaders, fit

ROOT = Path(__file__).resolve().parent


class SyntheticDataset(Dataset):
    """16 train / 8 validation samples; no frozen dataset files accessed."""
    def __init__(self, split, seed):
        n = 16 if split == 'train' else 8
        rng = np.random.default_rng(seed)
        self.images = rng.uniform(-1, 1, (n, 2, 15, 15)).astype('float32')
        self.tabular = rng.normal(size=(n, 70)).astype('float32')
        self.targets = self.images.mean(axis=(1, 2, 3)).astype('float32')
        self.index = pd.DataFrame({'fire_id': [f'{split}_{i//4}' for i in range(n)],
                                   'patch_id': [f'{split}_{i}' for i in range(n)]})

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        return dict(image=torch.from_numpy(self.images[i]), tabular=torch.from_numpy(self.tabular[i]),
                    target=torch.tensor(self.targets[i]), fire_id=self.index.fire_id.iloc[i], index=i)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['cnn_only', 'fusion'], required=True)
    parser.add_argument('--data-root', type=Path, default=ROOT/'data/processed/phase6_cnn_dataset_v1')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--max-epochs', type=int, default=30)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--min-delta', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--smoke-test', action='store_true', help='One synthetic epoch, 16/8 samples, batch size <=8')
    args = parser.parse_args()
    if (args.batch_size < 1 or args.max_epochs < 1 or args.patience < 1 or args.threads < 1
            or args.learning_rate <= 0 or args.weight_decay < 0 or args.min_delta < 0
            or not 0 <= args.seed < 2**32
            or not all(np.isfinite(v) for v in (args.learning_rate,args.weight_decay,args.min_delta))):
        parser.error('Invalid numeric training settings')
    seed_everything(args.seed, args.threads)
    device = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else 'cpu' if args.device == 'auto' else args.device
    if device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable; use CPU or auto')
    if args.smoke_test:
        train, val = SyntheticDataset('train', args.seed), SyntheticDataset('validation', args.seed+1)
        args.max_epochs, args.batch_size = 1, min(args.batch_size, 8)
        provenance = {'synthetic_only': True}
    else:
        train, val = PatchDataset(args.data_root, 'train'), PatchDataset(args.data_root, 'validation')
        raw = (args.data_root/'manifest.json').read_bytes()
        provenance = {'dataset_manifest_sha256': hashlib.sha256(raw).hexdigest(),
                      'dataset_version': json.loads(raw)['version']}
    train_loader, val_loader = make_loaders(train, val, args.batch_size, args.seed)
    model = CNNRegressor(args.model)
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    output = ROOT/'data/processed/phase6_training'/(
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+args.model+'_'+uuid.uuid4().hex[:8])
    metadata = dict(vars(args), data_root=str(args.data_root), device=device,
                    train_rows=len(train), validation_rows=len(val), parameters=count,
                    torch_version=str(torch.__version__), numpy_version=np.__version__,
                    deterministic_algorithms=True, num_workers=0, **provenance)
    try:
        metadata['git_commit'] = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,stderr=subprocess.DEVNULL).strip()
        metadata['git_dirty'] = bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True,stderr=subprocess.DEVNULL).strip())
    except (OSError, subprocess.CalledProcessError):
        metadata['git_commit'] = 'unavailable'
    import mlflow
    mlflow.set_tracking_uri('sqlite:///' + (ROOT/'mlflow.db').as_posix())
    mlflow.set_experiment('P1 - Wildfire Recovery - Phase 6')
    with mlflow.start_run(run_name=('synthetic_smoke_' if args.smoke_test else '')+args.model):
        mlflow.log_params(metadata)
        mlflow.set_tag('synthetic_smoke', str(args.smoke_test))
        def log_epoch(record):
            mlflow.log_metrics({k:v for k,v in record.items() if k!='epoch'},step=record['epoch'])
            print(json.dumps(record),flush=True)
        result = fit(model, train_loader, val_loader, device, output,
                     epochs=args.max_epochs, lr=args.learning_rate, weight_decay=args.weight_decay,
                     patience=args.patience, min_delta=args.min_delta, log_epoch=log_epoch)
        (output/'configuration.json').write_text(json.dumps(metadata,indent=2))
        (output/'architecture.txt').write_text(str(model))
        if not args.smoke_test:
            # Small provenance/schema files only; no datasets or test artifacts.
            for name in ('manifest.json','tabular_preprocessing.json'):
                mlflow.log_artifact(str(args.data_root/name),artifact_path='dataset_metadata')
        mlflow.log_metrics({'best_epoch':result['best_epoch'],
                           **{'best_val_'+k:v for k,v in result['best_validation'].items()}})
        mlflow.log_artifacts(str(output),artifact_path='training')
        print(json.dumps(dict(output=str(output),run_id=mlflow.active_run().info.run_id,**result),indent=2))


if __name__ == '__main__':
    main()
