# Phase 5 Day 2: Ridge regression

## Method and saved evidence

`notebooks/phase5_ml.ipynb` executions 8-15 record the completed experiment.
Input remains corrected terrain v2: 111,257 eligible patches, 34 approved
predictors. Development uses 61,090 training patches from 78 fires (2015-2017)
and 19,561 validation patches from 36 fires (2018-2019). **2020 test outcomes
remain untouched**; target reads are filtered to development splits.

Ridge excludes only `nbr_drop_patch = nbr_pre_mean - nbr_1_mean`, leaving
31 numerical features and two categorical features: `dominant_evt` and
`ignition_month`. All numerical inputs use StandardScaler; categorical inputs
use OneHotEncoder(handle_unknown="ignore"). Unseen categories produce an
all-zero block, not a newly learned category. No missing training inputs were
reported; the notebook also asserts complete validation inputs. No imputer
was added.

The reusable unfitted constructor is
`wildfire_recovery_forecasting.modeling.build_ridge_pipeline(predictors, alpha=1.0)`.
Pass the approved predictor allowlist, not all dataset columns. It drops other
columns. The notebook now imports this constructor; saved outputs and execution
counts were preserved rather than rerunning tuning or fitting.

Three-fold GroupKFold holds out entire training fires. Held-out patch counts
were 20,363 / 20,364 / 20,363 and fire counts 27 / 25 / 26; all folds had zero
fire overlap. Each fit clones the entire pipeline, so scaling and categories
are learned from that fold's training fires only. This CV is fire-grouped,
not itself chronological; the separate 2018-2019 validation tests later fires.

The search expanded from 0.1, 1, 10, 100 to 300, 1000, 3000, then 10000,
30000, 100000 using training CV. The final ten-alpha table selects **3000**:
mean CV MAE **0.07425**, fold MAEs 0.07256 / 0.07983 / 0.07038, standard
deviation 0.00404. Alpha 1000 was close (0.07469); this is not evidence of a
statistically decisive difference. Selection averaged patch MAE across folds,
not equal-fire MAE. The fold standard deviation is not a confidence interval.

## Chronological validation

After selection, the pipeline was fitted to all training fires once.

| Model | MAE | RMSE | R-squared | Equal-fire MAE |
|---|---:|---:|---:|---:|
| Training mean | 0.16244 | 0.19316 | -0.02555 | 0.14215 |
| NBR1 persistence | 0.18389 | 0.24230 | -0.61385 | 0.15851 |
| Ridge, alpha 3000 | **0.11308** | **0.14032** | **0.45881** | **0.11276** |

Validation patch MAE improves approximately **30.4%** relative to the
training-mean baseline: (0.16244 - 0.11308) / 0.16244. Equal-fire MAE is the
average of per-fire patch MAEs. Similar aggregate MAE and equal-fire MAE do
not establish uniformly good performance across fires.

Validation error exceeds training-fire CV error. Different fire environments
and cohort distributions are plausible contributors, but these results do not
isolate the cause. No test performance or causal effect is established.

Saved execution 15 reports MLflow run **a3d6e79143654e55bfa5e603ffc9e227**,
name `ridge_alpha_3000`, under `P1 - Wildfire Recovery - Phase 5`. It logs alpha,
feature exclusion, CV and validation metrics, and the fitted pipeline. Tracking
uses the notebook's local SQLite setup. The saved warning says pip's installed
version could not be resolved for the model environment specification; logging
completed, but that artifact's pip requirement is unpinned. The project uv lock
remains the reproducible project environment. No new MLflow run was created
while extracting the constructor. Nested MLflow artifact folders are ignored.

## Concepts and connections to Phase 4

- **Linear regression** predicts an intercept plus weighted inputs. It fits
  coefficients by minimizing squared residuals. One-hot encoding allows nominal
  categories without assuming numerical class distances.
- **Ridge** adds an L2 (squared coefficient) penalty. **Lasso** uses L1 and can
  set coefficients exactly to zero. Ridge was a simple regularized baseline,
  not proof it outperforms Lasso, which was not tested here.
- **Alpha** controls penalty strength: larger alpha shrinks coefficients more,
  trading flexibility for stability. Its scale depends on preprocessing and
  the loss convention; 3000 is not a universal recommended value. Standardizing
  numerical features makes their penalties comparable. One-hot columns retain
  the notebook's unscaled encoding.
- Phase 4 found strongly correlated climate inputs (historical Tmax/VPD about
  +0.960) and NBRpre/tree cover (+0.868). Ridge can stabilize predictions with
  correlated inputs; coefficients are not independent causal contributions.
  Ridge could accommodate exact NBR redundancy numerically, but removing the
  difference gives a cleaner representation without losing raw information.
- Phase 4's conditional severity/drop patterns motivate later nonlinear models;
  this additive Ridge has no explicit interaction terms. It is a benchmark,
  not a claim that those patterns are linear or irrelevant.
- GroupKFold prevents related patches from the same fire appearing on both
  sides of CV. It does not remove all spatial dependence between different
  fires. Keep chronological validation and the final 2020 target lock.
- The observed 2020 climate input shift remains a generalization concern;
  neither this validation improvement nor the selected alpha resolves it.

The next modeling work should build on this saved baseline, using train-only
preprocessing, fire-grouped development evaluation and the existing feature
roles. No datasets, eligibility rules, splits or scientific definitions changed.
