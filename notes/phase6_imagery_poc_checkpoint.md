# Phase 6 imagery POC checkpoint

Validated on 5 November 2026. Production imagery extraction has **not** started.
The scope remains a two-channel CNN-only baseline and required lightweight fusion
with the existing 18 no-climate tabular predictors. The scalar target and frozen
fire-level chronological splits are unchanged; no 2020 outcomes were accessed.

## Design and implementation

| Fire | Split | Selected batch | Eligible chips | Raster columns x rows |
|---|---|---|---:|---:|
| CA3839112224120171009 | 2017 train | r187_c-209 | 104 | 84 x 204 |
| CA3839611945120180904 | 2018 validation | r180_c-186 | 215 | 162 x 138 |

Four separate exports (prefire NBR and Season 1 NBR per fire) completed successfully;
operator-reported completion was approximately seven minutes, not a population
runtime estimate. Outputs are under
`gs://ee-wildfire-recovery-vasant-data/imagery/phase6_nbr_context_poc_v1/`.

`landsat.build_fire_inputs()` reuses the established seasonal processing without
constructing Season 5. `build_fire_nbr()` preserves the Phase 3 target/QA contract.
Prefire uses the latest three usable seasons in the 2013-2017 search pool for both
POC fires; Season 1 is June-September 2018 or 2019. QA, reflectance scaling,
same-date/seasonal compositing, and late-ignition handling are unchanged.

The source reference is selected over the original exact retained-patch batch,
not the larger context rectangle. Verified references are
`LC08_044033_20180601` (EPSG:32610) and `LC08_042033_20190606` (EPSG:32611), both
native 30 m grids. Inputs cover the eligible-square bounding rectangle plus 180 m
on each side. Exports use explicit north-up EPSG:5070 30 m affine transforms.
Each 15x15 input chip is centered on a canonical 90 m patch, whose footprint is
exactly the central 3x3 pixels. Context outside the fire/tile is retained.

## Day 1F: downloaded raster and chip QA

All four downloaded GeoTIFFs matched dimensions, CRS, transforms and bounds;
channels align exactly within each fire. Each has float32 NBR and `valid` bands,
with nodata metadata -9999. All values are finite and within [-1,1].

| Output | Bytes | Valid NBR range |
|---|---:|---:|
| 2017 prefire | 93,535 | -0.143 to 0.762 |
| 2017 Season 1 | 93,705 | -0.163 to 0.750 |
| 2018 prefire | 120,948 | -0.147 to 0.649 |
| 2018 Season 1 | 123,615 | -0.429 to 0.679 |

All 319 chip windows passed containment, indexing and central-patch geometry.
Both channels have 100% coverage in every full chip and central 3x3: all 104
training and 215 validation chips pass the proposed >=80% criterion. No patches
were dropped and that threshold is **not frozen**. A real zero-NBR prefire pixel
remained valid. Raster/chip plots were inspected; no obvious alignment defect was
observed. No missing source pixels occurred in this POC.

## Day 1G: bounded source-pixel comparison

Two reference metadata checks and four small point queries compared the raw
composites and their masks against the downloaded rasters. Samples used exact
output-grid pixel centers, not patch averages: four boundary pixels, three
eligible centers and three surrounding-context pixels per fire, plus the observed
zero-NBR location for 2018. Absolute tolerance was fixed at 1e-6 NBR units (rtol=0).

| Output | Pixels | Maximum absolute difference | Mean absolute difference | Values / masks passing |
|---|---:|---:|---:|---:|
| 2017 prefire | 10 | 1.99e-8 | 9.13e-9 | 10 / 10 |
| 2017 Season 1 | 10 | 2.83e-8 | 7.54e-9 | 10 / 10 |
| 2018 prefire | 11 | 7.00e-9 | 3.33e-9 | 11 / 11 |
| 2018 Season 1 | 11 | 4.22e-9 | 1.51e-9 | 11 / 11 |

All 42 value and mask comparisons passed without tolerance changes. The zero-NBR
pixel was exactly zero with source mask 1. This bypassed the export adapter for
comparison but reused the scientific compositing functions: it establishes sampled
export consistency, **not independent validation of the scientific algorithm**,
all-pixel equality, or equivalence to source-grid area-weighted tabular means.

## Reproducibility and operational boundaries

- `manual_phase6_imagery.py`: offline frozen-key configuration and image adapter.
- `src/wildfire_recovery_forecasting/imagery.py`: local geometry/window helpers.
- `manual_phase6_submit.py`: allowlisted two-fire launcher; default local dry-run,
  explicit submission, both references verified before any task, duplicate/object
  checks, four-active-task inventory guard, durable intents/IDs, no retry or polling.
- `manual_phase6_inspect.py`: four-file read/download and local raster/chip QA.
- `manual_phase6_verify_pixels.py`: explicit bounded read-only pixel verification.
- `notes/phase6_imagery_poc.json`: small frozen configuration and key/input hashes.
  Its absolute local provenance paths are informational, not portable dataset copies.

The existing POC has already run: **do not repeat submission**. Local-only review:

```powershell
.\.venv\Scripts\python.exe -B manual_phase6_submit.py --fires both
```

Submission records remain under `data/processed/phase6_nbr_context_poc_v1_2017_submission.json`
and the corresponding `2018` path. Downloaded GeoTIFFs, hashes/object generations,
`qa_summary.json`, `chip_coverage.csv`, plots and `source_pixel_verification.json`
remain in `data/processed/phase6_nbr_context_poc_v1_inspection/`. These generated
artifacts and private technical notes are excluded from Git; preserve them locally.

## Limitations and next gate

Real missing-pixel serialization was not exercised; synthetic tests cover nodata
and validity inconsistencies. Full coverage in two small regions does not establish
population-wide coverage, compute cost, runtime or reliable scaling. Sampled
agreement is consistent with the intended output sampling but does not independently
validate every reprojection/compositing operation. Budget alerts are not hard caps.

Before separately authorizing production extraction, review the input-only coverage
policy, cross-fire/cross-split context overlap, current quota/budget and a bounded
operating plan using the existing frozen population. Preserve common evaluation
keys for classical/CNN comparisons. Do not add sources, access test outcomes or
start population exports as part of this checkpoint.

## Public archive portability note

In the presentation branch, `phase6_imagery_poc.json` uses repository-relative source-path labels instead of machine-specific checkout prefixes. Source-file digest values and scientific fields are unchanged. This is a sanitized presentation of provenance, not a claim that the JSON file itself has its original byte hash. The original file and runnable layout remain in commit `cbd13f167393098e4a7735ecb2e5fdcef4d1a199` (designated `research-v1-working` snapshot). Historical operator commands in this note apply to that original layout.
