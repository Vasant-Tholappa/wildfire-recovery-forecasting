# Phase 5: Interpretation, error analysis, ablations and closure

## Evidence and evaluation boundary

This development checkpoint uses the saved executions 30–38 in
`notebooks/phase5_ml.ipynb`, the local MLflow records below, and
`notes/phase4_eda_findings.md`. Earlier methods are documented in
`phase5_day1_baselines.md`, `phase5_day2_ridge.md`, `phase5_day3_rf.md` and
`phase5_day3_xgboost.md`. Reusable construction and scoring live in
`modeling.py` and `evaluation.py` under `src/wildfire_recovery_forecasting/`.

The corrected terrain v2 dataset has 111,257 eligible patches and 34 approved
predictors. Training is 61,090 patches / 78 fires (2015–2017); validation is
19,561 patches / 36 fires (2018–2019). Three-fold training CV groups by fire_id,
with preprocessing fitted inside each fold. **2020 outcomes remain untouched.**
No new fits, searches or dataset changes were performed for this closure review.

## What the interpretation methods tell us

| Method | What is measured | Main limitation |
|---|---|---|
| Ridge numerical coefficient | Signed prediction change per training-standard-deviation change in that input, conditional on the fitted linear model | Correlation and regularization affect coefficients; categorical coefficients have a different scale; neither is causal |
| RF split importance | Normalized training impurity reduction credited to splits, summed across trees | No sign; favors variables with more splitting opportunities and can redistribute importance among correlated inputs; not held-out usefulness |
| SHAP | Additive attribution of an individual model prediction relative to its model-specific baseline | Depends on model and attribution assumptions; correlated inputs complicate credit; not an intervention or causal effect |

Ridge's largest standardized numerical coefficients are NBR1 **+0.08497** and
NBRpre **+0.08137**. Climate signs can look counterintuitive: for example,
pre-S1 precipitation is −0.01502 while historical precipitation is +0.00987.
These are conditional, regularized associations, not evidence that changing
rainfall would cause those outcomes. RF assigns **0.53817** split importance to
NBR1 and **0.25340** to NBRpre. Dominance by these measurements is consistent
with Phase 4, but the importance scales are not interchangeable.

### Global and local SHAP

Global mean absolute SHAP averages the magnitude of contributions, not their
direction. The notebook explains a reproducible **400-patch validation sample**
(random_state=42), not an equal-fire sample or the whole validation population.
XGBoost's `pred_contribs=True` returns contributions plus a bias term; the
notebook checks their sum reconstructs predictions to `atol=1e-4`. One-hot
contributions are summed with their signs within each original categorical
variable before taking absolute values and averaging.

Full XGBoost is dominated by NBR1 (**0.08771**) and NBRpre (**0.06262**).
For one validation patch, the baseline is **0.3123**, prediction **0.3495**,
observed NBR5 **0.4076**, and absolute error **0.0581**. The baseline is the
model's SHAP bias, not necessarily the empirical training-target mean. Local
contributions explain the prediction's departure from that bias; they do not
explain why nature produced the observed target or establish causation.

| No-climate XGBoost input | Mean absolute SHAP |
|---|---:|
| NBR1 | 0.08487 |
| NBRpre | 0.06191 |
| Elevation | 0.01610 |
| Fire size | 0.01020 |
| Ignition month | 0.00850 |
| Tree height | 0.00763 |
| Tree cover | 0.00611 |

After removal of climate inputs, attribution shifts toward remaining variables.
This is a newly fitted model, and correlated predictors can substitute for one
another. Increased SHAP credit does not prove an input acquired greater causal
importance, or that the removed variables contained no useful information.

## Error analysis: fires and subgroups

Among all **36 validation fires**, RF wins on per-fire MAE for **19**, XGBoost
for **17**. Equal-fire MAE is RF **0.09327**, XGBoost **0.09142**. Restricting the
*diagnostic display* to the **28 fires with at least 25 eligible patches** gives
RF **18** wins versus XGBoost **10**, and equal-fire MAE **0.08495** versus
**0.08649**. Each fire contributes equally to this average, even with only one
or two eligible patches; such small samples can affect rankings substantially.
This diagnostic does **not** redefine the evaluation population or justify
discardarding small fires. Patch-weighted and equal-fire metrics answer
complementary questions and both remain required.

| Validation subgroup | Patches / fires | RF MAE | XGBoost MAE |
|---|---:|---:|---:|
| Negative pre-S1 precipitation anomaly | 6,240 / 12 | 0.09619 | 0.09187 |
| Nonnegative anomaly | 13,321 / 25 | 0.08877 | 0.09036 |
| High-severity fraction >=0.50 | 2,024 / 19 | 0.13762 | 0.13480 |
| Lower high-severity fraction | 17,537 / 36 | 0.08577 | 0.08576 |
| Negative anomaly AND high severity | 147 / 8 | 0.12697 | 0.12559 |

High-severity patches have substantially greater errors. The dry/high-severity
intersection is small, and patches within a fire are not independent replicates.
Groups differ in size, environment and target distribution; fire counts can
overlap. These results describe observed model errors, not causal effects of
drought or severity. They warrant cautious subgroup reporting, not broad claims
about performance on all extreme fires.

## Controlled feature ablations

Ablation removes an input group and **retrains** the model. SHAP instead explains
an already fitted model. Simply zeroing inputs at prediction time would not test
what a model could learn without them and could create unrealistic inputs.

Each ablation independently refitted preprocessing and the estimator on the same
three fire-grouped training folds, then on all training data for chronological
validation. Existing full-model hyperparameters were retained; no reduced model
was independently retuned. This controls one source of variation but does not
establish each feature set's best attainable performance. With fractional
feature sampling, removing columns also changes the number of available sampled
features even when the hyperparameter value is unchanged.

| Model | CV MAE | Validation MAE | Validation RMSE | Validation R² | Equal-fire MAE |
|---|---:|---:|---:|---:|---:|
| Full RF | 0.06420 | 0.09114 | 0.11746 | 0.62073 | 0.09327 |
| Full XGBoost | 0.06230 | 0.09084 | 0.11825 | 0.61566 | 0.09142 |
| NBR-only XGBoost | 0.06716 | 0.08438 | 0.11150 | 0.65824 | 0.08630 |
| No-climate XGBoost | 0.06055 | 0.07510 | 0.10172 | 0.71561 | 0.08352 |
| No-climate RF | 0.06496 | 0.07399 | 0.09974 | 0.72657 | 0.08259 |

NBR-only uses exactly `nbr_pre_mean` and `nbr_1_mean`. No-climate removes all
**16** approved inputs beginning `hist_`, `pre_s1_` or `s1_`, retaining **18**:
NBRpre, NBR1, patch NBR drop, EVT and its dominant fraction, tree cover/height,
elevation/slope/northness/eastness, fire size, ignition month, and five MTBS
severity fractions. EVT and month remain categorical. The exact NBR-drop
identity, NBRpre minus NBR1, remains relevant to linear-model collinearity;
Ridge excluded it, whereas the tree models retain it.

### Reasoning from EDA to the result

1. Phase 4 found substantial climate shifts across fire-year cohorts, using
   permitted inputs rather than held-out test outcomes.
2. Full-model SHAP emphasized NBR1 and NBRpre. We hypothesized that the other
   32 predictors added meaningful predictive value beyond these two.
3. NBR-only XGBoost unexpectedly improved chronological validation despite
   worse training-only CV. Generalizing among training fires and across years
   are different tests; the CV ranking need not transfer.
4. Removing the 16 climate inputs produced an 18-feature XGBoost model that
   outperformed both full and NBR-only models on validation.
5. The no-climate RF comparison also showed a large validation improvement,
   despite slightly worse CV than full RF. This is corroboration in another
   algorithm, **not independent external replication**, because validation is
   shared.
6. Together these support concern about the current climate features'
   out-of-period predictive generalization. They do not prove climate is
   ecologically irrelevant or identify the precise cause. Cohort differences,
   coarse climate information, correlated inputs and model behavior remain
   possible contributors rather than established explanations.

Scientific relevance is different from improving prediction under a particular
training/evaluation distribution. Repeated feature decisions informed by
2018–2019 are exploratory, so these observed advantages may be optimistically
selected. Neither validation nor the tuning plateau establishes a scientific
ceiling for classical ML. We stop further feature searches and retuning to keep
the comparison bounded and preserve the final test, not because no better model
could exist. The earlier Ridge2 interaction likewise failed to improve
validation meaningfully; that failed predictive hypothesis does not erase the
conditional severity pattern observed in EDA.

## Local experiment provenance

| Ablation | Successful MLflow run |
|---|---|
| NBR-only XGBoost | `cb37cc8997bf404985ed30eab5f4ffd0` |
| No-climate XGBoost | `2cb3267c3dac4d4c80fe118e274637be` |
| No-climate RF | `b175c6f10eeb4ebfa8c0930192468d39` |

Read-only SQLite inspection confirmed `FINISHED`, the recorded CV/validation
metrics (matching the rounded table), feature-list JSONs, training/validation
years and `hyperparameters_retuned=False`. Both XGBoost ablations retain 300
trees, depth 9, learning rate 0.021130142327879537, min_child_weight 13,
subsample 0.7691388361657923, colsample_bytree 0.8678661189120203 and lambda
0.8658195608411128. RF retains 100 trees, depth 16, leaf minimum 10,
max_features 0.8, bootstrap=True and seed 42. XGBoost also uses seed 42.

Model metadata and nonempty artifact files exist locally: XGBoost cloudpickle
files and RF skops serialization (recorded trusted type
`sklearn.tree._tree.Tree`). **Deserialization was deferred**: presence and
provenance checks are not a successful reload test. Cloudpickle can execute
code; only locally produced, verified artifacts should ever be loaded in the
matching environment. Skops trust decisions also require review, not blanket
approval. Nonfatal MLflow serialization/dependency warnings do not negate the
successful logging records. MLflow databases, models, trial CSVs and private
notes remain Git-excluded; they are not portable artifacts in this commit.

## Phase 5 handoff and stopping point

**No-climate RF and no-climate XGBoost are the two primary classical candidates**
for eventual comparison with the Phase 6 CNN. Preserve full-feature models,
NBR-only XGBoost, Ridge and the simple baselines as development references.
No conclusive final winner is declared from these exploratory comparisons.

Before examining any 2020 target outcomes, lock the final protocol, candidate
models, frozen hyperparameters, metrics and common evaluation population. If
CNN availability imposes a subset, define it and report exclusions before target
inspection so comparisons use identical rows/fires. Refit selected classical
pipelines on 2015–2019 with frozen settings only under that protocol, fitting
preprocessing on those development years. This refit has **not** happened here.
Retain patch MAE/RMSE/R² and equal-fire MAE, and predefine limited subgroup
reporting. Do not tune using the final test.

Limitations remain: one selected tile per fire; retrospective eligibility;
nonindependent patches; chronological and geographic distribution shift;
repeated validation-based selection; no independent reduced-feature retuning;
and uncertain extrapolation to extreme 2020 conditions. Fire grouping helps
but cannot remove every environmental dependence. Phase 4 observed test-input
shift; Phase 5 has not measured test predictive performance.

## Sunday consolidation checklist (not the private Q&A document)

- Baselines, NBR5 versus ecological recovery, chronological/fire-grouped splits,
  train-only preprocessing, patch versus equal-fire scoring.
- Linear regression, Ridge/Lasso, scaling, alpha, exact NBR-drop redundancy;
  Ridge2: EDA observation → hypothesis → tuned comparison → no supported gain.
- RF bagging, bootstrap/feature sampling, depth/leaf control; boosting versus
  bagging and nonlinear/conditional relationships from Phase 4.
- XGBoost loss versus Optuna CV objective, TPE and search ranges, 15→40→90
  trials, diminishing returns and why a tuning plateau is not a ceiling.
- Coefficients versus impurity importance versus local/global SHAP; correlated
  feature credit, signed dummy aggregation, additivity and noncausal meaning.
- Small-fire influence, severity/dryness subgroups, unequal sample sizes and
  observations versus causal explanations.
- NBR-only and no-climate hypotheses, retraining versus attribution, fixed
  hyperparameters, CV/chronological disagreement and validation-selection bias.
- Final classical/CNN protocol, frozen refit plan, common population, untouched
  test outcomes, local artifact provenance and safe reload limitations.
