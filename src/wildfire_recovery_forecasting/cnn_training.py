"""Memory-mapped development datasets and small, deterministic training loop."""
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from wildfire_recovery_forecasting.evaluation import evaluate_model


def seed_everything(seed, threads=4):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class PatchDataset(Dataset):
    """Read only train/validation; row index binds image, tabular, label and fire.

    Validity masks are preserved in the frozen dataset, not extra model channels.
    Images already contain approved clipping and training-only median filling.
    """
    def __init__(self, root, split):
        if split not in ('train', 'validation'):
            raise ValueError('Only train/validation allowed; test access forbidden')
        root = Path(root)
        manifest = json.loads((root / 'manifest.json').read_text())
        if manifest['status'] != 'VALIDATED' or manifest['image_shape_per_row'] != [2, 15, 15]:
            raise ValueError('Expected validated frozen image dataset')
        self.split = split
        self.index = pd.read_parquet(root / split / 'index.parquet',
                                     columns=['fire_id', 'patch_id', 'split', 'array_index'])
        self.images = np.load(root / split / 'images.npy', mmap_mode='r', allow_pickle=False)
        self.tabular = np.load(root / split / 'tabular_fusion.npy', mmap_mode='r', allow_pickle=False)
        self.targets = np.load(root / split / 'targets.npy', mmap_mode='r', allow_pickle=False)
        n = manifest['counts'][split]
        if (len(self.index) != n or n < 2 or not self.index.split.eq(split).all()
                or not self.index.patch_id.is_unique or self.index.isna().any().any()
                or not np.array_equal(self.index.array_index, np.arange(n))
                or self.images.shape != (n, 2, 15, 15) or self.images.dtype != np.float32
                or self.tabular.shape != (n, 70) or self.tabular.dtype != np.float32
                or self.targets.shape != (n,)):
            raise ValueError('Frozen row/shape/dtype alignment mismatch')

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        image = np.array(self.images[i], dtype=np.float32, copy=True)
        tabular = np.array(self.tabular[i], dtype=np.float32, copy=True)
        target = np.float32(self.targets[i])
        if not np.isfinite(image).all() or not np.isfinite(tabular).all() or not np.isfinite(target):
            raise ValueError('Nonfinite development sample')
        return dict(image=torch.from_numpy(image), tabular=torch.from_numpy(tabular),
                    target=torch.tensor(target), fire_id=str(self.index.fire_id.iloc[i]), index=int(i))


def make_loaders(train, validation, batch_size=128, seed=42):
    if batch_size < 1:
        raise ValueError('Batch size must be positive')
    if (set(train.index.fire_id) & set(validation.index.fire_id)
            or set(train.index.patch_id) & set(validation.index.patch_id)):
        raise ValueError('Training/validation fire or patch overlap')
    generator = torch.Generator().manual_seed(seed)
    # Zero workers avoids Windows spawn/memmap duplication; no dropped examples.
    return (DataLoader(train, batch_size=batch_size, shuffle=True, generator=generator, num_workers=0),
            DataLoader(validation, batch_size=batch_size, shuffle=False, num_workers=0))


def forward_batch(model, batch, device):
    prediction = model(batch['image'].to(device), batch['tabular'].to(device))
    target = batch['target'].to(device)
    if prediction.shape != target.shape or not torch.isfinite(prediction).all():
        raise ValueError('Invalid model prediction shape/values')
    return prediction, target


def train_epoch(model, loader, optimizer, device):
    model.train()
    squared = absolute = 0.0
    n = 0
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        prediction, target = forward_batch(model, batch, device)
        loss = nn.functional.mse_loss(prediction, target)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite training loss')
        loss.backward()
        optimizer.step()
        n += len(target)
        squared += loss.item() * len(target)
        absolute += (prediction.detach() - target).abs().sum().item()
    if not n:
        raise ValueError('Empty training loader')
    return dict(loss=squared / n, mae=absolute / n)


@torch.no_grad()
def evaluate(model, loader, device):
    previous_mode = model.training
    model.eval()
    actual, predicted, fires = [], [], []
    try:
        for batch in loader:
            prediction, target = forward_batch(model, batch, device)
            actual.append(target.cpu().numpy())
            predicted.append(prediction.cpu().numpy())
            fires.extend(batch['fire_id'])
    finally:
        model.train(previous_mode)
    if not actual:
        raise ValueError('Empty validation loader')
    actual, predicted = np.concatenate(actual), np.concatenate(predicted)
    metrics = evaluate_model(actual, predicted, fires)
    metrics['loss'] = float(np.mean((actual.astype(float) - predicted) ** 2))
    return metrics


class EarlyStopping:
    """Save every lower MAE; min_delta controls patience resets only."""
    def __init__(self, path, patience=5, min_delta=1e-4):
        if patience < 1 or min_delta < 0:
            raise ValueError('Invalid early stopping settings')
        self.path = Path(path)
        self.patience, self.min_delta = patience, min_delta
        self.best = self.progress_best = float('inf')
        self.bad_epochs = 0
        self.best_epoch = None
        self.metrics = None

    def update(self, model, epoch, metrics):
        score = metrics['mae']
        if not np.isfinite(score):
            raise ValueError('Nonfinite validation MAE')
        if score < self.best:
            self.best, self.best_epoch, self.metrics = score, epoch, dict(metrics)
            temporary = self.path.with_suffix('.tmp')
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, temporary)
            os.replace(temporary, self.path)
        if score < self.progress_best - self.min_delta:
            self.progress_best, self.bad_epochs = score, 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience

    def restore(self, model, device):
        if self.best_epoch is None:
            raise ValueError('No selected checkpoint')
        model.load_state_dict(torch.load(self.path, map_location=device, weights_only=True))


def fit(model, train_loader, val_loader, device, output, *, epochs=30, lr=1e-3,
        weight_decay=1e-4, patience=5, min_delta=1e-4, log_epoch=None):
    if epochs < 1 or lr <= 0 or weight_decay < 0:
        raise ValueError('Invalid training settings')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    stopping = EarlyStopping(output / 'best_state.pt', patience, min_delta)
    history = []
    import time
    for epoch in range(1, epochs + 1):
        start = time.monotonic()
        train = train_epoch(model, train_loader, optimizer, device)
        val = evaluate(model, val_loader, device)
        stop = stopping.update(model, epoch, val)
        record = dict(epoch=epoch, seconds=time.monotonic()-start,
                      **{'train_'+k: v for k, v in train.items()}, **{'val_'+k: v for k, v in val.items()})
        history.append(record)
        pd.DataFrame(history).to_csv(output / 'history.csv', index=False)
        if log_epoch is not None:
            log_epoch(record)
        if stop:
            break
    stopping.restore(model, device)
    result = dict(best_epoch=stopping.best_epoch, best_validation=stopping.metrics,
                  epochs_completed=len(history), stopped_early=len(history)<epochs)
    (output / 'selection.json').write_text(json.dumps(result, indent=2))
    return result
