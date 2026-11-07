# Phase 6 CNN-ready dataset v1

Built locally with `manual_phase6_build_dataset.py`; no cloud operations or model training.
Artifacts: `data/processed/phase6_cnn_dataset_v1/` (Git-excluded).
The directory must not exist before a build. Use a new version for deliberate rebuilds;
never overwrite frozen data. A dataset is accepted only when its final manifest says
`VALIDATED`; a partial directory from an interrupted build is not usable.

## Frozen population and preprocessing

| Split | Patches | Fires | Targets |
|---|---:|---:|---|
| Train, 2015-2017 | 61,073 | 78 | Included |
| Validation, 2018-2019 | 19,561 | 36 | Included |
| Test, 2020 | 30,606 | 37 | Never loaded/exported |

All four coverage fractions (two channels, full 15x15 and central 3x3) must be
at least 0.80. Exactly 17 training patches are excluded from CNN data only and
retained in `exclusions.parquet`; Phase 5 inputs remain unchanged.

Channel order: `nbrpre`, `season1`. Clip valid observations to [-1,1], never
clip nodata into observations. Preserve original boolean masks. Missing pixels
are filled with medians of valid, clipped **retained training-chip pixel
occurrences**, including repeated occurrences in overlapping chips:

- Prefire: 0.48748868703842163
- Season 1: 0.1320200264453888

Missing occurrences filled, prefire/Season 1: training 978/0; validation 13/0;
test 9/1. Clipped occurrences: training 0/0; validation 0/5; test 0/5. Raw rasters
remain unchanged. No additional image scaling is fitted.

## Array contract

Each split directory contains:
- `images.npy`: float32 [N,2,15,15], suitable for NumPy memory mapping.
- `validity.npy`: original bool [N,2,15,15].
- `index.parquet`: ordered identifiers, grid coordinates, coverage and array_index.
- `tabular.parquet`: ordered keys and the original 18 no-climate predictors.
- `tabular_raw.npy`: float64 [N,18], exact established Phase 5 predictor order.
- `tabular_fusion.npy`: float32 [N,70], 16 standardized numerical features followed
  by one-hot categorical features, fitted only on retained training rows.
- `targets.npy`: original float64 scalar NBR5 labels, **train/validation only**.

`tabular_preprocessing.json` saves input/output order, numerical means/scales,
category lists and fitting population. Categories follow Phase 5 conventions:
`dominant_evt`, `ignition_month`, unknown categories encode to all zeros.
Unseen EVT values affect 120 validation rows and one test row; no rows are dropped
and no validation/test categories enter fitting. No unknown ignition months occur.
The 70 encoded columns still represent the approved 18 predictors, not new sources.
For PyTorch, cast development target batches to float32 when needed; no test target
placeholder exists. Align all arrays by array_index; never independently shuffle them.

## Validation and spatial limitation

All 302 raster hashes were rechecked against the completed QA evidence. The builder
rechecks masks, grid alignment, every retained central footprint and coverage count,
unique keys, exact tabular joins/order, finite image/fusion values and saved fill
values. Provenance and artifact SHA256 checksums are in `manifest.json`. A streaming
checksum of the source Parquet does not decode its test target column. Target reads
use explicit train/validation column projection and split predicates, matching Phase 5.

Splits are fire-disjoint and chronological, **not fully spatially independent**.
48 training chips from CA3779911892220160804 overlap 53 validation chips from
CA3782611887220190726: 233 positive-area context-window pairs. Central target
footprints do not overlap; no test context overlap was found. This approved limited
shared context is not demonstrated target leakage. The 53 validation patch IDs
are in `shared_context_validation_ids.csv` for optional later sensitivity analysis.
No additional spatial exclusions were applied.

Six focused synthetic tests passed. The completed build and acceptance checks took
36.485 seconds. Artifacts total approximately 310 MB plus the small final manifest.
No 2020 outcomes, model training, cloud operations, commits or pushes were involved.
