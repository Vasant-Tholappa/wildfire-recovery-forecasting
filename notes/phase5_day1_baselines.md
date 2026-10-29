# Phase 5 Day 1: validation baselines

Input: `data/processed/patch_model_eligible_terrain_v2.parquet`, with 111,257
eligible patches and 34 approved predictors. Training comprises 61,090 patches
from 78 fires (2015-2017); validation comprises 19,561 patches from 36 fires
(2018-2019). The 2020 test outcomes remain untouched. Notebook target loading
is filtered to training/validation only.

| Baseline | MAE | RMSE | R-squared | Equal-fire MAE |
|---|---:|---:|---:|---:|
| Training mean | 0.1624 | 0.1932 | -0.0255 | 0.1421 |
| NBR1 persistence | 0.1839 | 0.2423 | -0.6139 | 0.1585 |

The mean baseline predicts the training mean NBR5 (0.3122), not the validation
mean. Persistence predicts each patch's NBR1. Equal-fire MAE averages per-fire
MAEs rather than weighting fires by patch counts. The training-mean baseline
performs better on all four reported validation metrics. Negative R-squared
means worse squared error than predicting the validation target mean; that
reference is not a deployable fitted baseline.

Evidence: saved executions 4, 5 and 9 in `notebooks/phase5_ml.ipynb`.
Execution 8 records both runs in local MLflow SQLite (`mlflow.db`), experiment
`P1 - Wildfire Recovery - Phase 5`, as `baseline_training_mean` and
`baseline_nbr1_persistence`. Local MLflow data and artifacts are Git-ignored.

Reusable metrics now live in
`wildfire_recovery_forecasting.evaluation.evaluate_model`. The notebook imports
that function; historical outputs were preserved, not regenerated. Focused
pytest tests use synthetic values, not held-out labels.

Next session: train-only preprocessing, fire-grouped cross-validation and Ridge
regression. Preserve chronological splits and the approved feature allowlist;
fit transformations inside training folds. No models were retrained for this
handoff, and no new model selection or preprocessing was implemented.
