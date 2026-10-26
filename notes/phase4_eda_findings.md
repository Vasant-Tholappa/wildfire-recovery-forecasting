# Phase 4 EDA findings — California wildfire recovery

**Status:** Phase 4 EDA complete, including corrected-terrain validation and analysis.
This report records saved, executed outputs in
[`phase4_eda.ipynb`](../notebooks/phase4_eda.ipynb), through execution 32.
Numbers below are rounded as displayed there. No analyses were rerun for this
report and no held-out test NBR5 outcomes were inspected. Modeling decisions
below are plans, not measured predictive results.

## 1. Dataset, scientific scope and validation

The original nonterrain EDA input was `data/processed/patch_model_eligible_v1.parquet`.
Use `data/processed/patch_model_eligible_terrain_v2.parquet` for subsequent EDA
and modeling; the terrain-only refresh preserves the eligible rows, nonterrain
values, labels and splits: **111,257
eligible 90 m × 90 m patches from 151 fires**, with 129 columns. Extraction
covered 308,048 candidates from 175 selected California fires, one deterministic
upper-median-occupancy tile per fire. This is neither full-fire coverage nor a
spatially random sample; eligibility further conditions the historical cohort.

| Frozen split | Fire years | Eligible patches | Fires |
|---|---|---:|---:|
| Training | 2015–2017 | 61,090 | 78 |
| Validation | 2018–2019 | 19,561 | 36 |
| Test | 2020 | 30,606 | 37 |

The role audit reconciled saved `roles.json` with `pipeline.dataset_roles()`:
**34 predictors, one target, 15 metadata/identifier columns and 79 QA columns**;
no unmapped, conflicting, missing-declared or drifting roles. Predictors cover
NBRpre/NBR1/patch NBR drop, vegetation, terrain, fire size/month, focal severity
and 16 Daymet climate fields. `nbr_5_mean` is the fifth-season target, not an
input. Prediction information ends September 30 of Season 1 (fire year + 1).
Use the explicit feature allowlists; numeric dtype does not imply predictor status.

The frozen [eligibility decisions](phase3_eligibility_decisions.md) and
[eligibility SQL](../sql/phase3_ca175_tile1_v1_eligibility.sql) require forest and
fire-overlap fractions ≥0.50; meaningful-burn fraction ≥0.25; NBRpre/NBR1/NBR5
valid fractions ≥0.80; `prefire_selected_year_count_mean ≥ 1` for NBRpre and
`season1_obs_count_mean ≥ 1` / `season5_obs_count_mean ≥ 1` for NBR1/NBR5; observed
prior-fire and reburn overlap exactly zero; pre-10-year and post-Y..Y+5 LCMS clear
disturbance each <0.05; and both LCMS complete-coverage fractions ≥0.80.
No thresholds or splits were changed during EDA. Raster masks determine valid
measurement support; eligibility filters whole patch rows.
The NBRpre count is contributing **years**, not acquisition dates: its underlying
pixel rule still requires three usable pre-fire seasons, each with at least
three valid acquisition dates. The NBR1/NBR5 screens use patch-mean valid-date
counts; they do not require every pixel to meet the patch-level minimum.

**Executed quality findings:** unique patch IDs, consistent fire-grouped splits,
canonical patch area/tile IDs; zero infinities in development numeric fields or
test predictors; **94 range/consistency checks, zero flagged checks**. There were
no constant/all-missing training predictors. All nonterrain predictors have zero
missingness in each split. Training/validation targets have zero missingness;
test target missingness was deliberately not examined. Development QA columns
have no nulls, but both MTBS history coverage statuses are `unverified` for all
80,651 development rows: zero observed overlap does **not** prove absence of fire.

These are structural and numerical checks, not proof of source completeness,
ecological validity or predictive skill. Postfire LCMS/reburn screening and
target-availability screening select a retrospective cohort. They are not model
features. LCMS annual labels cannot determine within-year ordering relative to
ignition or September 30 of Season 5. LF2014 vegetation is a common historical
baseline, and Daymet remains coarse climate information despite patch-row storage.

## 2. Predictor distributions and the 2020 covariate shift

Saved split comparisons use approved predictors only. The table reports
patch-weighted means, not independent-fire averages:

| Predictor | Train | Validation | Test |
|---|---:|---:|---:|
| Pre-S1 precipitation anomaly (mm) | +172.89 | +255.28 | −300.69 |
| Pre-S1 VPD anomaly (kPa) | −0.05 | −0.05 | +0.09 |
| S1 VPD anomaly (kPa) | +0.08 | −0.17 | +0.32 |
| High-severity fraction | 0.176 | 0.117 | 0.236 |
| NBR1 | 0.180 | 0.168 | 0.081 |
| Patch NBR drop | 0.292 | 0.270 | 0.375 |
| Tree cover (%) | 50.71 | 47.02 | 46.00 |

**86.69%** of test patches fall outside the training minimum–maximum range for
pre-S1 precipitation anomaly; **55.34%** do so for S1 VPD anomaly. Training ranges
were approximately −101.62 to +896.80 mm and −0.29 to +0.29 kPa, respectively.
For precipitation anomaly, 99.89% of test patches fall below the training first
percentile. “2020 shift” refers to the **2020 fire cohort**: its pre-S1/S1 windows
occur in 2021, not in the ignition year.

Jointly dry and severe patches—negative pre-S1 precipitation anomaly **and**
high-severity fraction ≥0.50—constitute **2.55% of training, 0.75% of validation
and 22.45% of test observations**. This combination is much more prevalent in
the held-out 2020 cohort and represents an important generalization risk.

Weighting matters. Mean fire acreage is 55,144 train versus 85,253 test when
repeated across patches, and 22,105 versus 48,741 when each fire counts once.
The test population therefore has larger fires under both summaries. However,
equal-fire mean high-severity fraction is **0.139 train, 0.066 validation and
0.125 test**. Validation fires are generally milder on this measure;
the larger patch-weighted test value does not mean every test fire is more severe.
Equal-fire precipitation anomalies (+187.84 versus −204.97 mm) and S1 VPD
anomalies (+0.07 versus +0.28 kPa) still show a climate shift.

Vegetation composition also differs. The six most frequent training EVT classes
leave an “Other” share of 27.3% train, 40.2% validation and 47.0% test patches.
“Other” means outside those six classes, **not necessarily unseen in training**.

**Interpretation:** substantial covariate shift creates extrapolation risk,
especially for tree ensembles that cannot be assumed to extrapolate beyond
training conditions. It does not establish target shift or test performance.
Keep the predefined split; do not tune models, preprocessing or feature choices
against test outcomes. Patch counts overstate the number of independent climate
environments and should not be read as independent evidence for significance.

## 3. Development target distributions

| NBR5 summary | Training | Validation |
|---|---:|---:|
| Mean / median | 0.312 / 0.297 | 0.343 / 0.344 |
| Standard deviation | 0.185 | 0.191 |
| Minimum / maximum | −0.273 / 0.811 | −0.109 / 0.813 |
| Mean NBR5 − NBR1 | +0.133 | +0.174 |
| Median NBR5 − NBR1 | +0.094 | +0.136 |

These distributions describe absolute spectral condition and spectral change,
not a validated proportional ecological recovery measure. Higher absolute NBR5
need not mean more complete recovery relative to initial vegetation condition.
No recovery-ratio target or test-target summary was added.

## 4. Predictor screening and dependence

Training-only screening covered all **32 numerical and two categorical**
predictors. Uncorrected terrain results from that screen are superseded by the
corrected-terrain section below. No features were removed.

| Predictor versus training NBR5 | Pearson | Spearman |
|---|---:|---:|
| NBRpre | +0.795 | +0.807 |
| NBR1 | +0.766 | +0.699 |
| Tree cover | +0.731 | +0.726 |
| Pre-S1 precipitation | +0.497 | +0.460 |
| Pre-S1 VPD anomaly | +0.424 | +0.424 |
| Pre-S1 Tmax anomaly | +0.423 | +0.380 |
| S1 VPD anomaly | +0.394 | +0.386 |
| S1 precipitation anomaly | −0.375 | −0.334 |
| High-severity fraction | −0.105 | −0.099 |
| Patch NBR drop | −0.090 | −0.054 |

There are 44 training EVT classes and 10 ignition-month levels. Illustrative
training EVT means differ substantially: class 3028 has NBR5 mean 0.339
(12,425 patches), 3114 has 0.208 (7,472), and 3043 has 0.604 (2,786).
No ecological class names are inferred from IDs. July/August account for 43.113%
and 23.762% of training patches; February has just one observation. Category
means are descriptive, potentially fire-confounded and unreliable for rare levels.

Notable training feature–feature Pearson correlations: historical Tmax versus
historical S1 VPD **+0.960**; pre-S1 Tmax versus pre-S1 VPD **+0.882**; pre-S1
Tmax versus Tmin **+0.873**; NBRpre versus tree cover **+0.868**; high-severity
fraction versus patch NBR drop **+0.686**. Weak marginal association does not
exclude useful conditional/nonlinear prediction; strong correlation alone is
not a reason for automatic removal.

There is also an **exact algebraic dependency**:
`nbr_drop_patch = nbr_pre_mean - nbr_1_mean`. These patch means can use different
valid pixel supports; this is not the retired mean of pixelwise differences on
common support. Including all three in an unregularized linear design creates
rank deficiency, regardless of their pairwise correlations.

## 5. Two focused training investigations

**Pre-S1 VPD anomaly and NBR5.** Across 61,090 training patches/78 fires,
Pearson correlation is **+0.424 overall**, **+0.584 between fire averages**, and
**+0.099 within fires after demeaning both variables**. Median within-fire VPD
anomaly standard deviation is **0.0047 kPa**; three fires meet the notebook's
near-zero-variation rule (<1e-8, with missing standard deviations filled with zero).

The association is much stronger between fires than within them, consistent
with differences between fire environments contributing substantially. Limited
within-fire variation restricts interpretation; demeaning is not a causal
adjustment for all confounding. This does **not** show atmospheric dryness
benefits vegetation. Retain VPD initially and assess its predictive value through
development-only ablations and robustness checks.

**Severity/drop conditional on pre-fire NBR.** Training patches were divided
at the training NBRpre median, approximately **0.495** (higher group includes
the cutoff). Fixed high-severity bins were 0, (0,.25], (.25,.50], (.50,.75],
(.75,1]; drop used six training-wide quantile bins, shared across both groups.

| Increasing damage bins | Higher pre-fire NBR: mean NBR5 sequence | Lower pre-fire NBR: mean NBR5 sequence |
|---|---|---|
| High-severity fraction | .506, .408, .354, .317, .310 | .187, .187, .183, .179, .213 |
| Patch NBR drop | .618, .553, .484, .423, .368, .307 | .175, .189, .170, .182, .224, .199 |

The higher-NBRpre group shows a clear declining pattern; the lower group is
comparatively flat and nonmonotonic. Marginal correlations therefore obscure
conditional patterns. These are binned observational means, not causal damage
effects or established interactions after adjustment. NBR drop itself contains
NBRpre, and fire composition/bin counts can affect the patterns. Retain severity
and drop; let tree models represent nonlinearities and optionally evaluate one
development-specified interaction in a linear model.

## 6. Phase 5 decisions and corrected terrain

- **Preprocessing:** fit imputers, scalers, encoders and any category pooling on
  training data only (inside each CV fold). Treat EVT as nominal; encode ignition
  month categorically or with an explicitly evaluated cyclic representation.
  Handle unseen categories without learning from held-out labels.
- **Missing values:** never turn absent measurements into zero. No terrain
  imputation is currently necessary for the corrected eligible dataset. Retain
  all four corrected terrain predictors initially and the approved feature roles;
  use the corrected v2 Parquet for modeling.
- **Models:** compare training-mean and NBR1 persistence baselines, Ridge,
  Random Forest and gradient boosting. Remove one algebraically redundant NBR
  column for an unregularized linear baseline; regularization does not make
  redundant coefficients independently interpretable.
- **Evaluation:** keep whole fires together in development CV, use 2018–2019
  chronological validation, and preserve 2020 for final evaluation. Report
  patch MAE, mean per-fire MAE (equal fire weight), RMSE and R², with a small
  set of meaningful subgroups. Do not treat correlated patches as independent
  validation units.
- **Bounded checks:** use limited climate/feature-group ablations and, if useful,
  the approved historical-only early-ablation list. Evaluate climate-range
  robustness on development data; do not optimize to the observed test feature
  distribution. No predictive scores or causal effects have yet been established.
- **Test lock:** retain 2020 NBR5 outcomes untouched until final model evaluation.

### Corrected terrain - complete

Saved notebook executions **31 and 32** verify corrected-terrain recovery,
split distributions and training-only correlations. No analyses were rerun or
held-out test outcomes inspected for this report update.

The corrected 3DEP composition resolves the single-image selection defect
([recovery notes](phase4_terrain_recovery.md)). Across all **111,257 eligible
patches**, all **15,984** previously missing elevation measurements and **16,004
each** for slope, northness and eastness were recovered. Corrected missingness is
**0% in train, validation and test**, with no previously valid measurements lost.
Elevation spans **38.415-3,091.014 m**, slope **0-59.457 degrees**, and northness/eastness
remain within **[-1, 1]**. These plausible ranges and resolved missingness replace
the old terrain findings; the original dataset remains the rollback reference.

| Corrected terrain mean / weighting | Train | Validation | Test |
|---|---:|---:|---:|
| Elevation (m), patch-weighted | 1,252.188 | 1,362.022 | 1,193.918 |
| Elevation (m), equal-fire-weighted | 1,064.118 | 1,086.573 | 1,059.962 |
| Slope (degrees), patch-weighted | 22.788 | 21.738 | 21.588 |
| Slope (degrees), equal-fire-weighted | 21.458 | 19.002 | 19.789 |

Equal-fire weighting first averages within each fire, then gives each fire equal
weight. Its much closer elevation means show that patch-versus-fire weighting
explains some apparent split differences; patches are not independent fires.

| Corrected terrain versus training NBR5 | Pearson | Spearman |
|---|---:|---:|
| Elevation | -0.288 | -0.230 |
| Slope | +0.229 | +0.239 |
| Northness | +0.068 | +0.070 |
| Eastness | -0.025 | -0.026 |

Training terrain-to-terrain Pearson correlations are weak (maximum absolute
correlation approximately **0.126**); correlations with NBRpre/tree cover are
modest. Terrain may contribute useful geographic information, but these marginal
associations neither establish independent predictive value nor causal effects.
Retain all four predictors initially and assess predictive contribution using
fire-grouped development validation. Test comparisons above concern terrain
inputs only; 2020 NBR5 remains reserved for final evaluation.

**Evidence references:** notebook executions 10–15 (structure/roles/quality),
19–25 (predictor distributions), 26 (development target), 27–28 (screening),
29 (VPD decomposition), 30 (conditional damage patterns),
31 (corrected-terrain integrity/recovery), 32 (corrected-terrain distributions
and training-only correlations);
[production contract](../docs/phase3_production_contract.md),
[population scope](phase3_ca175_tile1_population.md),
[eligibility decisions](phase3_eligibility_decisions.md), and
[`pipeline.dataset_roles()`](../src/wildfire_recovery_forecasting/pipeline.py).
