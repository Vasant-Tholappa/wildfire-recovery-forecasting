# Phase 6: compact CNN and fusion training infrastructure

This checkpoint implements models and runs synthetic tests only. No epoch on the
frozen imagery dataset, hyperparameter search or test-data access was performed.
PyTorch is pinned by uv to the official CPU wheel index, explicit for torch only.
No torchvision, torchaudio or CUDA toolkit is required.

## Architecture

Both models share the same encoder architecture (separate learned weights per run):

| Operation | Output excluding batch dimension |
|---|---|
| Input, prefire and Season-1 NBR | 2 x 15 x 15 |
| Conv 3x3, padding 1, 2 -> 16; ReLU | 16 x 15 x 15 |
| MaxPool 2x2, stride 2 | 16 x 7 x 7 |
| Conv 3x3, padding 1, 16 -> 32; ReLU | 32 x 7 x 7 |
| MaxPool 2x2, stride 2 | 32 x 3 x 3 |
| Flatten; Linear 288 -> 64; ReLU | 64 |

CNN-only: Linear 64 -> 32, ReLU, Linear 32 -> 1: **25,553 parameters**.
Fusion: tabular Linear 70 -> 32, ReLU; concatenate with 64 image features;
Linear 96 -> 32, ReLU, Linear 32 -> 1: **28,849 parameters**.
Outputs are unbounded scalar NBR5 predictions; no sigmoid, clipping or classification.
Two pooling steps preserve a 3x3 feature map. No pretrained network, batch norm,
augmentation, dropout or extra mask channels are introduced.

Fusion retains the frozen 18 original no-climate features, including NBRpre, NBR1
and their exact patch-level difference. The saved 70-column representation is used
unchanged. Original validity masks remain in the dataset; images already contain
approved clipping and training-only median filling. No preprocessing is refitted.

## Data, reproducibility and selection

`PatchDataset` opens train/validation arrays as read-only NumPy memory maps and
copies only the sampled rows to writable float32 tensors. It checks manifest status,
shapes, split membership, unique patch IDs and sequential array_index. Targets are
cast to float32 per sample. Dataset construction rejects `test` before any file read.
The loaders check fire/patch separation, shuffle training with a seeded generator,
and preserve validation order. Zero workers avoids Windows spawn overhead and
memory-map duplication. No examples are dropped at epoch boundaries.

Seed 42, four CPU threads, deterministic PyTorch algorithms and seeded Python/NumPy
are defaults. CUDA is selected automatically only if available. Determinism is not
promised across different PyTorch versions/hardware; unsupported deterministic CUDA
operations fail instead of silently relaxing the setting.

Defaults: batch 128, AdamW learning rate 0.001, weight decay 0.0001, MSE loss,
maximum 30 epochs, patience 5, meaningful MAE improvement 0.0001. These are modest
starting settings, not tuned conclusions. Batch 128 bounds memory; small dense heads
and mild weight decay limit complexity. No Optuna or architecture search is included.

Validation follows every complete epoch, using existing `evaluate_model()` for MAE,
RMSE, R2 and equal-fire MAE; MSE validation loss is also recorded. Training loss/MAE
are example-weighted online measurements from minibatches during optimization,
not a second full training-set evaluation. Validation is no-grad and cannot update
weights. Save every strictly lower validation MAE; min_delta controls patience
resets, not which checkpoint is best. Restore the best state after stopping or
reaching max epochs. Checkpoints contain tensor state dictionaries and are loaded
with `weights_only=True`; automatic training resumption is not implemented.

MLflow uses the existing root `mlflow.db`, experiment
`P1 - Wildfire Recovery - Phase 6`. Log architecture, parameters, versions, seeds,
dataset-manifest hash, Git revision/dirty status, epoch metrics/timing, best metrics,
and checkpoint/schema artifacts. Local run outputs use unique subdirectories of
`data/processed/phase6_training/`. All model weights, curves and MLflow artifacts
remain Git-excluded. A crash may leave partial artifacts; only completed runs with
selection.json indicate a successfully restored best model.

## Commands (PowerShell, repository root)

Synthetic plumbing check only (16 training, 8 validation examples, one tiny epoch):

```powershell
uv run --offline python manual_phase6_train.py --model fusion --smoke-test
```

NEXT authorized step: a one-epoch speed benchmark on real development data:

```powershell
uv run --offline python manual_phase6_train.py --model cnn_only --max-epochs 1 --batch-size 128 --threads 4 --seed 42
uv run --offline python manual_phase6_train.py --model fusion --max-epochs 1 --batch-size 128 --threads 4 --seed 42
```

Run sequentially; do not launch both concurrently. These commands do real training
and have **not** been executed for this checkpoint. Inspect logged seconds/epoch and
validation results before choosing longer runs. `--device cpu` forces CPU;
`--learning-rate`, `--weight-decay`, `--patience`, `--min-delta` remain configurable.
Do not use the synthetic smoke metrics for scientific comparisons.

The common CNN comparison population remains 61,073 train / 19,561 validation
patches. Classical models must be evaluated on identical available patches when
compared. Splits are chronological and fire-disjoint but not fully spatially
independent: preserve the documented 48-train/53-validation context overlap and
optional sensitivity IDs. No 2020 inputs or targets enter this training interface.
