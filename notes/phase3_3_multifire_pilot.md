# Phase 3.3 — Multi-Fire Generalization Pilot

## Purpose

Phase 3.2 validated the reusable production pipeline on a 24-patch Highline sample. Phase 3.3 tests whether the same pipeline generalizes across fires with different years, locations, sizes, and dataset splits before population-scale execution.

The production manifest contains 1,904 focal fires from 2015–2020.

## Pilot-fire selection

Four fires were deliberately selected rather than sampled randomly.

### MONUMENT — Washington, 2015
- `fire_id`: `WA4731911979920150630`
- Size: 2,076 acres
- Split: train
- Reason: exercises the oldest focal cohort and the Landsat 7 / Landsat 8 transition. Its 2010–2014 pre-fire search window includes Landsat 7 years before 2013 and Landsat 8 years from 2013 onward.
#### Pilot result
- Status: PASS
- Candidate grid: 1,153 patches; 24-patch deterministic smoke sample.
- All seven source tables returned 24 unique patch IDs with zero duplicates.
- Final joined table: 24 rows × 160 columns.
- The 2010–2014 pre-fire window successfully exercised the Landsat 7 / Landsat 8 transition; all sampled patches obtained the required three usable pre-fire seasons.
- All 24 sampled patches had `forest_fraction = 0`. EVC and EVT source coverage were complete, confirming that missing tree-cover, tree-height, dominant-EVT, and forest-supported terrain values reflected genuine absence of forest support rather than a retrieval failure.

### COLUMBUS — Oregon, 2018
- `fire_id`: `OR4293812266120180715`
- Size: 11,687 acres
- Split: validation
- Reason: medium-sized Oregon validation-cohort fire in a different geography from Highline and MONUMENT. Provides a normal Landsat-8-era cross-region generalization case.

#### Pilot result
- Status: PASS
- Candidate grid: 6,288 patches; 24-patch deterministic smoke sample.
- All seven source tables returned 24 unique patch IDs with zero duplicates.
- Final joined table: 24 rows × 160 columns.
- The pilot validated the reusable pipeline on a medium-sized Oregon fire with a 2013–2017 pre-fire search window, Season 1 in 2019, and Season 5 in 2023.
- During the first live run, Daymet failed because annual derived images entering an Earth Engine `ImageCollection` had identical band names but slightly different inferred numeric ranges for `prcp_mm`. Earth Engine therefore rejected the collection as non-homogeneous.
- The fix was deliberately narrow: annual Daymet summary images and annual daily-coverage ratio images were explicitly cast to a common `Double` type before multi-year `ImageCollection` construction. Climate formulas, temporal windows, units, masks, and spatial behavior were unchanged.
- Two regression tests were added for the affected Daymet collection boundaries. The full automated test suite increased from 71 to 73 tests and passed.
- After the type-normalization fix, the live COLUMBUS rerun completed successfully, confirming that the original failure was an Earth Engine type-consistency issue rather than a methodological or data-quality problem.
- All 24 sampled patches had `forest_fraction = 1.0` with complete EVC, EVT, tree-cover, and tree-height validity. This complements the nonforest MONUMENT sample and confirms correct behavior for fully forest-supported vegetation and terrain features.
- Landsat coverage was healthy: all 24 patches had valid `NBR_pre`, `NBR_1`, `NBR_drop`, and `NBR_5`, and every patch obtained the required three usable pre-fire seasons.
- One sampled boundary patch had only ~0.41% focal-fire overlap and `mtbs_source_valid_fraction = 0`. Its MTBS severity fractions were correctly retained as missing rather than being interpreted as zero severity. This was expected QA behavior rather than a pipeline failure.

### WOODBURY — Arizona, 2019
- `fire_id`: `AZ3340811121020190608`
- Size: 130,242 acres
- Split: validation
- Reason: large Southwestern fire selected to test the reusable pipeline under substantially different geography, ecology, terrain, and spatial extent from Highline, MONUMENT, and COLUMBUS.

#### Pilot result
- Status: PASS
- Candidate grid: 66,782 patches; 24-patch deterministic smoke sample.
- All seven source tables returned 24 unique patch IDs with zero duplicates.
- Final joined table: 24 rows × 160 columns.
- Landsat successfully produced `NBR_pre`, `NBR_1`, `NBR_drop`, and `NBR_5` for all sampled patches, and every patch obtained the required three usable pre-fire seasons.
- Season-1 and Season-5 Landsat observation counts were lower than in COLUMBUS, generally about 5–6 valid dates per patch, but all sampled patches retained valid coverage.
- Eighteen of the 24 sampled patches had `forest_fraction = 0`; six had partial forest support ranging up to approximately 0.42. Tree-cover, tree-height, dominant-EVT, and forest-supported terrain values were present only where forest support existed, confirming expected missingness behavior.
- MTBS severity coverage was structurally complete for the 24-patch sample, with no sampled patch missing severity-fraction outputs.
- The large fire exposed a performance issue in the current pilot workflow: the 66,782-patch canonical grid was constructed once to select smoke-test IDs and then reconstructed inside `run_fire_pipeline()` before subsetting to those 24 IDs. The long initial pause occurred before source outputs began, while the subsequent source evaluations completed normally.
- This is recorded as a Phase 3.4 scaling requirement: construct each fire's canonical grid once and reuse it for spatial batching rather than repeatedly regenerating the full grid.

### TABBY CANYON — Utah, 2020
- `fire_id`: `UT4078711251920200530`
- Size: 14,330 acres
- Split: test
- Reason: latest focal-fire cohort and held-out test split, exercising Season 1 in 2021 and the project's most recent Season-5 target year, 2025.

#### Pilot result
- Status: PASS
- Candidate grid: 7,470 patches; 24-patch deterministic smoke sample.
- All seven source tables returned 24 unique patch IDs with zero duplicates.
- Final joined table: 24 rows × 160 columns.
- The pilot successfully exercised the latest focal-fire cohort and confirmed that the pipeline can retrieve and construct the 2025 Season-5 target.
- Landsat produced valid `NBR_pre`, `NBR_1`, `NBR_drop`, and `NBR_5` values for all 24 sampled patches. Every patch obtained the required three usable pre-fire seasons.
- Season-1 Landsat coverage was approximately 11–12 valid dates per patch, while Season-5 coverage was approximately 9–13 valid dates per patch.
- Twenty sampled patches had no LANDFIRE forest support, while four had positive forest fractions ranging from approximately 0.03 to 0.89. Forest-supported vegetation and terrain measurements were present where forest support existed and correctly missing otherwise.
- MTBS severity outputs were structurally complete for all 24 sampled patches.

## Coverage of the pilot

Together the four fires provide variation across:

- focal years: 2015, 2018, 2019, 2020;
- splits: train, validation, test;
- geography: WA, OR, AZ, UT;
- fire size: ~2k to ~130k acres;
- Landsat sensor history;
- ecological and terrain contexts.

The purpose is pipeline generalization testing, not statistical representativeness of the full wildfire population.


## Multi-fire pilot conclusion

The Phase 3.3 pilot exercised four deliberately different fires across focal years, geographic regions, fire sizes, dataset splits, forest-support conditions, and Landsat histories:

- MONUMENT (WA, 2015, train): PASS; exercised Landsat 7 / Landsat 8 pre-fire transition.
- COLUMBUS (OR, 2018, validation): PASS after a narrow Daymet type-normalization fix; also validated fully forest-supported patches and expected MTBS boundary missingness.
- WOODBURY (AZ, 2019, validation): PASS; exercised a very large Southwestern fire and exposed redundant full-grid construction as a production scaling concern.
- TABBY CANYON (UT, 2020, test): PASS; exercised the latest focal cohort and 2025 Season-5 target.

The reusable per-fire scientific and tabular pipeline therefore generalizes sufficiently across the pilot cases to proceed to population-scale engineering.

Two implementation lessons carry forward:
1. Daymet annual derived images must be explicitly type-normalized before multi-year Earth Engine `ImageCollection` construction.
2. Population-scale execution should build each fire's canonical grid once and reuse it for spatially compact processing batches rather than regenerating the full grid.

These pilot fires were selected as deliberate generalization/stress cases and are not intended to be statistically representative of the full 1,904-fire population.