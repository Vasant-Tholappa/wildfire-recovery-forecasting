# Phase 3 Production Contract

This document freezes the production rules for building the multi-fire patch-level dataset. It generalizes the validated Phase 2 workflows while resolving known production inconsistencies before pipeline refactoring.

Production contract version: `v2`.

## 1. Study population and modeling unit

- Focal events: MTBS `Wildfire` events with ignition dates from 2015-01-01 through 2020-12-31.
- Study region: WA, OR, CA, ID, NV, MT, WY, UT, CO, AZ, NM.
- One modeling observation is one fixed 90 m × 90 m patch associated with one focal wildfire.
- Primary target: fifth-season Landsat NBR (`NBR_5`).
- Interpretive recovery metric:

  `Recovery_5 = (NBR_5 - NBR_1) / (NBR_pre - NBR_1)`

- Prediction cutoff: end of the first complete post-fire growing season.
- Growing season: June 1–September 30.
- Patches from the same fire must remain grouped by `fire_id`.

## 2. Fire timing

Use the exact MTBS ignition date (`Ig_Date`).

- `fire_year = year(Ig_Date)`
- Season 1 year = `fire_year + 1`
- Season 5 year = `fire_year + 5`

The `fire_year + 1` rule remains standardized even for late-year fires because MTBS does not provide a reliable fire-end date.

### Fully pre-fire growing seasons

`last_fully_prefire_growing_season_year` means the latest June–September growing season that finished before the focal ignition date.

Examples:

- July 2015 ignition → latest fully pre-fire growing season = 2014.
- November 2015 ignition → latest fully pre-fire growing season = 2015.

This temporal definition is separate from Landsat imagery usability.

## 3. Landsat temporal contract

### NBRpre

For each pixel:

1. Identify the five most recent fully pre-fire June–September growing seasons.
2. A season is Landsat-usable if it contains at least 3 valid acquisition dates after QA masking.
3. Use the three most recent usable seasons.
4. `NBR_pre` is the pixel-wise median of the three selected usable annual growing-season NBR composites. Each annual composite is itself the median of that season's valid acquisition-date NBR observations.
5. If fewer than three usable seasons exist within the five-season search pool, `NBR_pre` is missing for that pixel.
6. Move farther back only because imagery is unusable, never to avoid a real ecological disturbance.

Prior ecological disturbance is handled through MTBS/LCMS eligibility QA rather than by selectively choosing cleaner NBR years.

### Sensors

- Use Landsat 7 ETM+ where older pre-Landsat-8 imagery is required.
- Use Landsat 8 OLI for the Landsat-8 era.
- Map sensor-specific bands into common NIR/SWIR2 concepts.
- Apply the same conceptual workflow: QA mask → surface reflectance → NBR → same-date compositing → seasonal median → validity/count QA.
- Do not add elaborate cross-sensor harmonization unless the multi-fire pilot demonstrates a concrete problem.

### Spatial support

Landsat NBR is summarized over the valid Landsat-observed portion of the full 90 m patch.

Do not additionally forest-mask Landsat. Pre-fire forest eligibility is represented separately by LANDFIRE `forest_fraction`.

Retain Landsat valid-area and observation-count QA.

## 4. Primary model and early ablation

The primary model may use information available through the end of Season 1.

The early ablation uses the same `NBR_5` target but excludes:

- `NBR_1`
- `nbr_drop_patch`
- other observed Season-1 spectral/recovery information
- pre-Season-1 climate
- Season-1 climate

It may retain:

- pre-fire Landsat information
- static vegetation and terrain
- focal-fire characteristics
- historical pre-fire climate

## 5. MTBS wildfire-history QA

### Prior wildfire

For each patch, retain only `prior_fire_fraction_10yr`: wildfire overlap during
the ten years before the exact focal ignition date. Include the ten-year
anniversary and exclude the focal ignition instant; preserve time of day.

Only MTBS `Wildfire` events count.

Exclude the focal wildfire by event/fire ID.

The ten-year lookback is locked. Phase 3.5 still selects the patch-level overlap
threshold; no exclusion threshold is locked here.

### Reburn

A post-focal wildfire candidate is another MTBS `Wildfire` event that:

- has a different event/fire ID,
- has an ignition date strictly later than the focal ignition date,
- overlaps the patch, and
- ignites on or before September 30 of Season 5.

Do not exclude the entire focal calendar year.

Because a reliable focal-fire end date is unavailable, a later ignition from a different event is treated as a post-focal wildfire QA candidate.

## 6. LANDFIRE contract

Use LF 2014 / v1.4.0 as the common pre-fire vegetation baseline for every retained 2015–2020 focal fire.

Store `landfire_condition_year = 2014` and `landfire_version = "v1.4.0"` explicitly in every fire-manifest row.

### Phase 3.2.3C amendment and rationale

The v2 contract excludes 2013–2014 focal fires because a consistent, readily accessible pre-fire LANDFIRE vegetation product could not be obtained through the chosen production stack without disproportionate archive/version-specific ingestion work. Using LF2014/v1.4.0 for all retained fires trades temporal freshness for consistent coverage, taxonomy, reproducibility, and implementation simplicity. LCMS/MTBS disturbance QA remains responsible for identifying major intervening disturbances between the baseline and focal fire.

That focal-population amendment did not change historical source lookbacks.
The later disturbance-QA scaling amendment below retains only the ten-year
MTBS/LCMS prior window; Daymet and Landsat lookbacks remain unchanged. The focal-population start year does not truncate historical source data: retained fires may require data before 2015, and the Landsat 7/8 transition at 2013 remains unchanged.

Other rules:

- Forest support and `forest_fraction` are based on official EVC tree-lifeform classes, not the Highline-specific EVT class list. Missing EVC remains missing and its coverage QA must be retained.
- EVT remains the categorical vegetation/ecosystem feature; EVC remains the canopy/tree-cover source; EVH remains the height source.
- Categorical EVT IDs are never numerically averaged.
- EVC and EVH decoding must be centralized and based on the appropriate product metadata.
- The open-ended EVH ≥99 m category may use 99 m as a documented lower-bound approximation unless implementation review provides a better justified treatment.

## 7. Daymet temporal contract

Use three climate periods:

- Historical climate: the ten complete calendar years `fire_year - 10` through `fire_year - 1`.
- Pre-Season-1 climate: January 1–May 31 of `fire_year + 1`.
- Season-1 climate: June 1–September 30 of `fire_year + 1`.

Historical summaries are calendar-matched to the period they support:

- January–May historical summaries provide the baseline for pre-Season-1 conditions and anomalies.
- June–September historical summaries provide the baseline for Season-1 conditions and anomalies.

Use the term `pre-Season-1 climate`, not `antecedent climate`, because a late-year focal fire could theoretically remain active during part of early `fire_year + 1`.

The early ablation uses historical climate only.

Preserve Daymet's meaningful native ~1 km spatial information before associating derived climate surfaces with 90 m patches.

## 8. Production grid

Use one fixed 90 m × 90 m reference grid in EPSG:5070 across the full study region.

- Use a deterministic global grid alignment/origin.
- Patch IDs must be deterministic from `fire_id` plus grid location/index.
- Do not create independently aligned grids for individual fires.

Retain every candidate grid cell intersecting the focal-fire perimeter and calculate `fire_overlap_fraction`.

Do not apply the final perimeter-overlap eligibility threshold during grid construction.

## 9. Area and missingness rules

Canonical full patch area is 8,100 m².

Whole-patch fractions use an explicit full-patch denominator when appropriate.

Continuous means use the valid area of that specific variable:

`mean = sum(value × valid_area) / sum(valid_area)`

If valid area is zero:

- validity fraction = 0
- measurement = missing/null

Missing data must never silently become numerical zero.

For subset-supported quantities, such as tree-cover validity within forested area, use the relevant support area as the denominator.

Zero disturbance must remain distinguishable from unavailable disturbance data.

## 10. Source spatial behavior

EPSG:5070 defines the vector modeling grid, not the native resolution of every source.

Preserve meaningful source-scale information:

- Landsat: ~30 m
- MTBS: ~30 m
- LANDFIRE: ~30 m
- LCMS: ~30 m
- 3DEP: ~10 m
- Daymet: ~1 km

Summarize or associate those sources with the common 90 m vector patches.

Derived Earth Engine rasters must retain appropriate projection metadata before spatial reduction.

Do not add research-grade exact Daymet native-cell intersection machinery unless later validation demonstrates a concrete need.

## 11. Required QA retention

Production exports must retain enough QA to make eligibility decisions without rerunning Earth Engine.

### LCMS window and coverage QA

Pre-fire LCMS QA retains one ten-year window: `fire_year - 10` through
`fire_year - 1`. Summarize clear competing disturbance and Other Loss separately
using spatial unions across the annual observations.

Post-fire LCMS QA audits annual products from the focal `fire_year` through `fire_year + 5`, inclusive.

Because LCMS is an annual product, focal-year and Year-5 detections do not imply precise within-year timing. In particular:

- a focal-year detection cannot always be precisely ordered before or after the focal wildfire, and
- a `fire_year + 5` detection cannot always be proven to occur before the June–September Year-5 target period.

LCMS is therefore used as broad retrospective competing-disturbance QA rather than exact event-date logic.

Each audited LCMS window must retain sufficient temporal/spatial coverage information to distinguish observed zero disturbance from incomplete source coverage.

### Grid / focal fire
- patch area
- fire overlap fraction

### MTBS
- valid severity coverage
- non-mapping/invalid severity coverage as appropriate
- 10-year prior-fire fraction
- reburn fraction through Season 5

### Landsat
- NBRpre valid-area fraction
- NBR1 valid-area fraction
- NBR5 valid-area fraction
- compact valid-acquisition-count summaries
- sufficient pre-fire contributing-year information to verify NBRpre usability

### LANDFIRE
- forest fraction
- EVT validity/coverage information
- tree-cover valid fraction
- tree-height valid fraction

### 3DEP
- elevation valid fraction
- slope valid fraction
- aspect valid fraction

### Daymet
- Daymet valid coverage
- preserve additional validity fields only if different climate feature families genuinely require them

### LCMS
- pre-fire clear competing-disturbance fraction for the ten-year lookback
- pre-fire ambiguous/Other Loss fraction for the ten-year lookback
- post-fire clear competing-disturbance fraction from focal fire year through Season 5
- post-fire ambiguous/Other Loss fraction from focal fire year through Season 5
- no earliest-disturbance-year or timing-minimum outputs; eligibility uses extent within each predefined window
- LCMS valid coverage for each audited window, sufficient to distinguish complete observed absence from incomplete temporal/spatial coverage

These QA fields do not automatically become model predictors.

Final numerical eligibility thresholds are chosen later from population-level evidence.

### Phase 3 scaling amendment: disturbance QA for v1

The original 5/7/10-year sensitivity comparisons were retained to choose a
lookback empirically. For v1, this is unnecessary methodological overhead for
auxiliary QA screens. A single conservative ten-year prior-disturbance and
prior-wildfire lookback is sufficient and easier to defend.

LCMS is retrospective contamination QA, not a primary predictor. Eligibility
uses disturbance extent/fraction within the predefined pre-fire and post-fire
windows, so exact disturbance year within a window is not needed. Remove LCMS
earliest-year calculations and outputs while retaining source/coverage QA and
the annual-window uncertainty described above. MTBS still filters by exact
ignition dates internally, and its reburn window is unchanged.

This is a deliberate Phase 3 scaling/scope decision: it reduces repeated
computation, code complexity and unnecessary output columns without weakening
the core fifth-season NBR prediction problem. Phase 3.5 still determines how
much disturbance or wildfire overlap excludes a patch. Historical Phase 2
notebooks, notes and earlier pilot outputs remain records of the previous design.

## 12. Fire manifest

Create one canonical GeoParquet fire manifest with one row per focal wildfire.

Include approximately:

- fire ID
- fire name
- ignition date
- fire year
- fire size
- state/source metadata
- focal-fire geometry
- last fully pre-fire growing-season year
- five-season NBRpre search-window start/end years
- Season 1 year
- Season 5 year
- LANDFIRE condition year/version
- fixed train/validation/test split

Splits:

- 2015–2017 → train
- 2018–2019 → validation
- 2020 → test

The canonical manifest is written as GeoParquet using GeoPandas/PyArrow. `pyarrow` is therefore a required Phase 3 dependency and must be present before manifest generation.

Patch-level measurements and run/batch metadata do not belong in the fire manifest.

## 13. Lightweight provenance and materialization

Each production dataset build must retain a small machine-readable run-metadata record containing enough information to identify:

- run timestamp / run ID
- Git commit
- production-contract version
- output/schema version
- EPSG:5070 / 90 m grid version
- source/product versions
- fire-manifest version

Do not build a large lineage framework.

Materialized Earth Engine/GCS/intermediate outputs must also be version-aware.

An existing output may be reused only when its processing/output version is compatible with the current pipeline. Do not reuse an asset merely because an object with the same name already exists.

## 14. Deferred population-level decisions

Do not finalize these in the production contract:

- minimum focal-fire overlap
- minimum forest fraction
- minimum Landsat coverage/observation quality
- minimum EVC/EVH coverage
- final prior-fire overlap threshold (the ten-year lookback is already locked)
- reburn exclusion threshold
- LCMS competing-disturbance threshold

Retain the necessary QA fields and select these rules from the population later in Phase 3.

## 15. Scope guardrail

Phase 3 should generalize and scale validated Phase 2 behavior, fix known correctness issues, and produce a trustworthy modeling table.

Do not add research-grade complexity, optional feature expansion, or unrelated sophistication unless a concrete production failure demonstrates the need.


## Phase 3.4.7A amendment: patch-level NBR drop (Candidate A)

Approved production definition: `nbr_drop_patch = nbr_pre_mean - nbr_1_mean`.
The operands are separately area-weighted patch means over each season's valid
Landsat support. This is NOT the old mean of pixelwise differences on common
valid pixels. Changing coverage can affect the new difference; even disjoint
supports can yield a value. Retain both original means and their valid fractions.

Phase 3.6 BigQuery assembly must add:

```sql
nbr_pre_mean - nbr_1_mean AS nbr_drop_patch
```

Ordinary SQL subtraction propagates NULL if either operand is NULL; do not use
COALESCE to replace missing means with zero. The new field is a primary predictor,
never an early-ablation predictor. `nbr_5_mean` remains the target. No SQL assembly
pipeline exists yet; current source tables intentionally do not require this
future derived column. If present, local dataset validation checks its definition.

Retire the `nbrdrop` export and `nbr_drop_sum`, `nbr_drop_area`, `nbr_drop_mean`,
and `nbr_drop_valid_fraction`. Marginal validity cannot reconstruct common-support
coverage. Historical artifacts retain their original meaning and are not renamed.

Production now has 13 raster products: Landsat 4, LANDFIRE 4, terrain 1,
Daymet 2, LCMS 1, focal severity 1. Combined Daymet is still experimental;
promoting it separately would reduce this count to 12. Tile size remains 120.

The interpretive `Recovery_5` expression in section 1 is unchanged. No active
recovery-ratio implementation exists. Phase 3.6 must explicitly resolve its spatial
support and zero/near-zero denominator policy before implementing it; do not
silently substitute `nbr_drop_patch` into a former common-support ratio or add
Recovery_5 to predictors (it uses the Season-5 target).

This is an approved scientific scope amendment: simpler patch-summary change in
exchange for losing matched-pixel change information, saving the dedicated
pixelwise-drop export. Phase 2 notebooks and earlier pilot results remain history.
