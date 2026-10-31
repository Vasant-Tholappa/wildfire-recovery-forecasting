# Phase 5 Day 3: initial Random Forest checkpoint

## Evidence and method

Saved notebook executions 19 and 21 in `notebooks/phase5_ml.ipynb` record the
initial RF evaluation and successful local MLflow logging. This is a fixed
initial configuration, **not an optimized or hyperparameter-searched forest**.
Input remains corrected terrain v2, with 61,090 training patches / 78 fires and
19,561 chronological validation patches / 36 fires. No 2020 test outcomes were
used. All 34 approved predictors are retained: 32 numerical inputs pass through
unchanged and two categorical inputs (`dominant_evt`, `ignition_month`) are
one-hot encoded with `handle_unknown="ignore"`. No numerical scaling or new
imputation is used; the notebook asserts complete development inputs.

A Random Forest averages predictions from many regression trees. Trees make
threshold splits and can represent nonlinear and conditional relationships.
Bootstrap samples and random feature subsets diversify trees; averaging reduces
variance relative to individual trees. Feature scaling is unnecessary for these
threshold comparisons. Categories still require encoding: EVT class IDs are not
meaningfully ordered measurements. Unseen categories map to an all-zero block.

## Configuration

| Parameter | Value | Meaning / provenance |
|---|---:|---|
| n_estimators | 100 | Explicit tree count (also sklearn default) |
| max_depth | 16 | Explicit depth cap; default would be unlimited |
| min_samples_leaf | 10 | Explicit minimum leaf size; default is 1 |
| max_features | 0.8 | Explicit fraction of transformed columns considered per split; default is 1.0 |
| bootstrap | True | Notebook used sklearn default; constructor makes it explicit |
| random_state | 42 | Explicit seed for reproducibility |
| n_jobs | -1 | Explicit use of available CPUs |
| min_samples_split | 2 | Unchanged sklearn default; also logged |
| criterion | squared_error | Unchanged sklearn default |

One-hot encoding expands the column count, so max_features is not simply 80%
of the 34 original predictors. Bootstrap resamples training rows, not whole
fires; fire isolation is enforced by the outer CV split, not bootstrap itself.
Other estimator settings remain sklearn defaults in the locked environment.

The reusable constructor is `modeling.build_rf_pipeline(predictors)`. It returns
an unfitted pipeline, retains NBR drop and drops columns outside the supplied
approved allowlist. The notebook now imports it without regenerating outputs.
Ridge and Ridge2 cells/results were preserved.

## Fire-grouped evaluation and saved results

Three-fold GroupKFold uses training `fire_id` groups. Each fold clones the
entire pipeline, fitting categorical encoding and the forest only on that
fold's training fires. This avoids preprocessing leakage and shared fires across
folds. It does not eliminate dependence between nearby different fires.
After CV, the forest was refitted on all training fires and evaluated on the
separate 2018-2019 validation cohort. CV is grouped, not chronological internally.

Fold patch MAEs: **0.06016, 0.07166, 0.06079**; mean **0.06420** (computed from
unrounded scores). CV averages fold patch MAEs, not equal-fire MAEs.

| Chronological validation | Ridge1 | Initial RF |
|---|---:|---:|
| MAE | 0.11308 | **0.09114** |
| RMSE | 0.14032 | **0.11746** |
| R-squared | 0.45881 | **0.62073** |
| Equal-fire MAE | 0.11276 | **0.09327** |

RF reduces validation MAE by approximately **19.4%** relative to Ridge1 using
the displayed rounded scores. Equal-fire MAE averages each fire's patch MAE,
so larger patch counts do not dominate that metric. Validation error exceeds
training-fire CV error; cohort differences are plausible contributors, not a
proven explanation. No significance claim or test-performance claim follows.

MLflow run: `7df9006fa70647f3ae3acb01880401fe`, name `random_forest_initial`,
experiment `P1 - Wildfire Recovery - Phase 5`. The saved logging cell records
parameters, CV/validation metrics and fitted preprocessing + RF as `rf_pipeline`
in the local SQLite/artifact setup. Serialization explicitly trusts
`sklearn.tree._tree.Tree`; this is not authorization to trust arbitrary artifacts.
The saved warning leaves pip unpinned in MLflow's generated environment spec;
logging completed, and the project uv lock remains available. No new run or
production-data fitting was performed while preparing this checkpoint.

## Phase 4 connections and interpretation

- Severity showed weak marginal association but stronger conditional patterns
  among higher-NBRpre patches. Trees can represent such patterns without one
  manually prescribed product term. This RF result does not establish that a
  specific interaction caused its improvement.
- All 34 inputs include `nbr_drop_patch`, unlike Ridge's 33. Exact linear
  dependence does not cause tree-design rank deficiency; the difference can
  offer useful split geometry despite adding no new raw information. This is
  therefore not a perfectly feature-identical algorithm comparison with Ridge.
- Strong climate/vegetation collinearity remains relevant to interpretation:
  correlated predictors can substitute for one another, making feature
  importance shared or unstable. Predictive improvement is not causality.
- Corrected terrain is complete in the eligible dataset, so no terrain
  imputation was added. Data quality correction and model improvement are
  distinct claims.
- The 2020 climate covariate shift remains a risk. Tree averaging does not
  reliably extrapolate response trends outside training support. Keep the
  held-out outcomes untouched and retain fire-grouped/chronological evaluation.

This checkpoint establishes a stronger development benchmark, not a final model
selection or a reason to expand the tuning scope. No splits, eligibility rules,
scientific feature definitions or datasets changed.
