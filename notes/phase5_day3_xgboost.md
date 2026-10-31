# Phase 5 Day 3: XGBoost and bounded Optuna tuning

## Data and evaluation contract

Saved `notebooks/phase5_ml.ipynb` executions 22-28 record this checkpoint.
Corrected terrain v2 contains 111,257 eligible patches and 34 approved predictors.
Training uses 61,090 patches / 78 fires from 2015-2017; chronological validation
uses 19,561 patches / 36 fires from 2018-2019. **2020 test outcomes remain
untouched.** No datasets, eligibility rules or splits changed.

All 34 predictors, including `nbr_drop_patch`, are retained. The 32 numerical
inputs pass through unchanged; EVT and ignition month use one-hot encoding with
unknown categories ignored. Scaling is unnecessary for threshold-based trees.
The entire pipeline is cloned/fitted inside each training-fire CV fold, so
category vocabularies are learned without holdout inputs. Three-fold GroupKFold
uses fire_id; tuning reuses the existing training-only `cv_splits`. Fire grouping
does not eliminate all geographic dependence between different fires.

`modeling.build_xgb_pipeline(predictors)` now constructs the same initial,
unfitted pipeline. It expects the approved allowlist and drops unrelated input
columns. Optuna changes `model__...` parameters on clones. This extraction did
not rerun experiments or regenerate saved outputs.

## Why boosting, and how it differs from Random Forest

Random Forest grows diverse trees using bootstrap samples and feature subsets,
then averages their predictions. Gradient boosting builds an additive model
sequentially: each new tree follows the current loss gradients to improve the
existing prediction. With squared error, these directions correspond to residual
errors. A learning rate scales each tree's contribution; many small corrections
can represent complex response shapes. XGBoost also uses curvature information,
regularization and efficient histogram-based split finding.

XGBoost was a reasonable tabular-regression candidate because the inputs mix
continuous environmental measurements and encoded categories, and Phase 4
suggested nonlinear/conditional relationships. This is motivation, not proof of
superiority. No claim is made that boosting always beats bagging.

## Two different objectives

XGBoost fits each training fold using **`objective="reg:squarederror"`** plus its
regularization. Optuna evaluates that fitted model using **MAE on held-out
training fires**, then minimizes the mean of the three fold MAEs. Thus training
loss is squared error, while hyperparameter selection targets CV absolute error.
The CV score is a mean of fold patch MAEs, not equal-fire MAE. Equal-fire MAE is
reported separately on chronological validation. Validation/test labels do not
appear in the Optuna objective functions. No early stopping was used; each fit
uses the proposed number of trees.

## Parameters and search design

Initial settings: 300 trees, depth 4, learning rate 0.05, min_child_weight 5,
subsample 0.8, colsample_bytree 0.8 and reg_lambda 5.0. All stages fix histogram
training, seed 42 and four model threads. Optuna runs trials sequentially.

| Parameter | Purpose | Initial search | Expanded continuation | Final focused study |
|---|---|---|---|---|
| n_estimators | Number of boosting rounds | 150-600, step 50 | 100-600, step 50 | 150-750, step 50 |
| max_depth | Limit tree complexity | 2-7 | 2-9 | 7-12 |
| learning_rate | Shrink each tree's update | 0.02-0.15, log | Same | 0.009-0.075, log |
| min_child_weight | Minimum child sum of Hessian weights | 1-15 | Same | 6-25 |
| subsample | Fraction of training rows sampled per round | 0.7-1.0 | Same | 0.65-0.90 |
| colsample_bytree | Fraction of encoded columns sampled per tree | 0.7-1.0 | Same | 0.75-1.0 |
| reg_lambda | L2 penalty on leaf weights | 0.1-20, log | Same | 0.02-4, log |

Min_child_weight is not generally a literal minimum row count (though with
unweighted squared-error fits it is closely related). Column subsampling acts
on the post-encoding columns, not just the 34 original feature names. Row
subsampling is not fire sampling; the outer CV enforces fire isolation.

The initial bounds offered a bounded comparison of shallow-to-moderate trees,
shrinkage, sampling and regularization without an exhaustive grid. Log sampling
covers multiplicative differences in learning rate/penalty. This is the practical
rationale for the design, not evidence that these are globally optimal bounds.
The pilot winner reached depth 7 and 150 trees, motivating exploration beyond
those boundaries. The expanded results favored deeper trees and low learning
rates. The final study examined that region, allowing greater depth and slower
learning while also varying child support and penalties. Boundary winners are
clues for a bounded search, not an instruction to tune indefinitely.

### How TPE learns from trials

Optuna's Tree-structured Parzen Estimator begins with startup sampling, then
models parameter values associated with better versus worse completed scores.
It proposes promising combinations based on those distributions while retaining
exploration. It learns from CV scores, not the raw targets directly. It neither
guarantees the optimum nor establishes the statistical certainty of small gains.
Seeds aid repeatability; library versions, execution order and time limits also
matter. No pruning/reporting mechanism was added to these objectives.

## The 15 -> 40 -> 90 completed-trial journey

| Stage | Completed evaluations | Best CV MAE | Detail |
|---|---:|---:|---|
| Pilot | 15 | 0.06304 | TPE seed 42; 25-minute timeout |
| Expanded continuation | 25 more, 40 cumulative | 0.06234 | Same study; 20-minute timeout |
| Focused study | 50 more, 90 total | 0.06230 | New TPE seed 43; 30-minute timeout |

These are maximum requested time budgets, not measured elapsed runtimes. Saved
outputs confirm all requested trials completed. The focused study enqueued and
re-evaluated the three best previous configurations first. Therefore **90 trial
evaluations do not mean 90 unique configurations**, and the final study did not
inherit the previous TPE's full trial history. Selection compared the best scores
from both studies. The selected result came from focused trial 42 (one-based).

Final selected settings:

```text
n_estimators = 300
max_depth = 9
learning_rate = 0.021130142327879537
min_child_weight = 13
subsample = 0.7691388361657923
colsample_bytree = 0.8678661189120203
reg_lambda = 0.8658195608411128
```

Initial XGBoost fold MAEs were 0.06289 / 0.07279 / 0.05972, mean 0.06513.
The final stage improved the previous best CV MAE by only about 0.00004
(using rounded scores). We stop tuning because marginal gains have become
small relative to the project scope and remaining interpretation work. Reusing
CV folds for many trials can itself favor those folds. This does not establish
a performance ceiling for classical ML, prove global optimality, or justify
further searches now.

## Chronological validation comparison

After training-only selection, the chosen pipeline was fitted on all training
fires and evaluated on the same separate validation fires as previous models.

| Model | CV MAE | Validation MAE | RMSE | R-squared | Equal-fire MAE |
|---|---:|---:|---:|---:|---:|
| Ridge1 | 0.07425 | 0.11308 | 0.14032 | 0.45881 | 0.11276 |
| Initial RF | 0.06420 | 0.09114 | **0.11746** | **0.62073** | 0.09327 |
| Initial XGBoost | 0.06513 | 0.09893 | 0.12431 | 0.57520 | 0.09918 |
| Tuned XGBoost | **0.06230** | **0.09084** | 0.11825 | 0.61566 | **0.09142** |

Tuned XGBoost narrowly leads RF on MAE (about 0.33% improvement) and equal-fire
MAE, while RF leads on RMSE and R-squared. RMSE gives larger errors more weight;
on the same targets, lower squared error also means higher R-squared. These
are metric tradeoffs, not a conclusive winner or a significance test. Different
models also received different tuning budgets. Ridge uses 33 predictors while
the tree models retain the redundant NBR difference, so that comparison is not
perfectly feature-identical.

Phase 4's conditional severity/NBRpre patterns motivated flexible trees, but
improvement does not identify which interaction caused it. Correlated climate
and vegetation predictors can substitute for one another, complicating feature
importance. Near-constant within-fire Daymet values and the between-fire VPD
association reinforce the need for fire grouping. The substantial 2020 climate
input shift remains unresolved: boosting is not guaranteed to extrapolate beyond
training support. A larger chronological validation error than CV error is
consistent with differing cohorts, but the cause has not been isolated.

Remaining Phase 5 questions concern per-fire/subgroup errors, interpretability,
bounded feature ablations and generalization robustness under the existing plan,
not another open-ended tuning cycle. Keep 2020 outcomes reserved for final
assessment. Validation has already been viewed across experiments and is a
development resource, not an untouched final performance estimate.

## Local tracking and reproducibility

Saved execution 28 confirms successful local MLflow logging:

- Initial run: `0976467a033444d8ab67c3b01fdc2e2c` (`xgboost_initial`), parameters
  and scores; the logging cell does not save an initial-model artifact.
- Tuned run: `0b48af653b9542faa7ba2bf327544c02` (`xgboost_optuna_90_trials`),
  selected parameters, scores and fitted `xgb_pipeline`.
- Local histories: `mlruns/xgb_optuna_trials.csv` (40 trials) and
  `mlruns/xgb_optuna_final_trials.csv` (50 trials), also offered to the tuned
  MLflow run as artifacts when present. These are not uploaded to Git.

Tracking uses local `mlflow.db`. The tuned fitted pipeline uses **cloudpickle**;
load only trusted artifacts because deserialization can execute code. The
serialization warning and unresolved pip-version warning are nonfatal, not
failed logging. Pip is unpinned in that generated environment specification;
the project dependencies/uv lock separately record XGBoost 3.4.1 and Optuna 5.0.0.
CSV histories preserve results but are not a persisted resumable Optuna storage
backend; the study objects were in-memory.

This checkpoint preserves executed results, introduces only the reusable
constructor and synthetic/static tests, and does not rerun fits, tuning,
MLflow logging or any cloud operations. Prior Ridge/Ridge2/RF outputs remain.
