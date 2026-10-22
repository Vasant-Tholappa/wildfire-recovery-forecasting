# California eligible_v1: frozen Phase 3 decisions

The reproducible rules are in `sql/phase3_ca175_tile1_v1_eligibility.sql`.
Source: `phase3_ca175_tile1_v1.patch_candidates_v1`; destination:
`phase3_ca175_tile1_v1.patch_model_eligible_v1`. The table already exists.
The SQL is a reproduction record, not an instruction to recreate it now; plain
CREATE TABLE deliberately fails if the destination exists. All source columns,
including separate MTBS coverage statuses, are retained; only eligibility_status
is replaced with `eligible_v1`. Required comparisons reject NULLs naturally.

These are the chosen v1 screening rules, not empirically optimal or universal
ecological thresholds. No sequential exclusion counts were measured here.

| Rule | Scientific purpose | Limitation |
|---|---|---|
| forest_fraction >= 0.50 | Focus on majority forest/tree-supported patches. | Mixed vegetation remains; LF2014 is a fixed, potentially stale baseline. |
| fire_overlap_fraction >= 0.50 | Require most of the full patch to lie within the focal perimeter. | Full-square measurements can include outside-perimeter area. |
| mtbs_meaningful_burn_fraction >= 0.25 | Require a material mapped burn footprint. | This is a minimum footprint screen, not proof that the entire patch burned or a severity cutoff. |
| NBR pre, Season 1 and Season 5 valid fractions each >= 0.80 | Require substantial observed support for predictors and target. | Up to 20% missing support remains, and the three supports need not coincide. Target availability also selects the modeled cohort. |
| prefire_selected_year_count_mean, season1_obs_count_mean, season5_obs_count_mean each >= 1 | Reject absent/zero observation support at patch level. | Patch means do not require every pixel to have that count. These screens do not replace the existing per-pixel three-date and latest-three-usable-prefire-season rules. |
| prior_fire_fraction_10yr = 0; reburn_fraction = 0 | Remove observed prior wildfire and post-focal reburn overlap. | MTBS coverage is unverified: zero means no observed overlap, NOT proven absence. Exact equality allows no observed positive overlap. |
| LCMS pre-10-year and post-Y..Y+5 clear-disturbance fractions each < 0.05 | Limit mapped competing disturbance to less than 5% in each audit window. | Small contamination remains; Other Loss is retained as QA but is not an exclusion rule here. |
| Both LCMS complete_coverage_fraction fields >= 0.80 | Require broad spatial support with complete annual coverage in each window. | This does not guarantee complete coverage everywhere or perfect disturbance detection. |

LCMS labels are annual: focal-year detections cannot establish ordering relative
to ignition, and Season-5-year detections cannot establish whether disturbance
preceded September 30. Postfire LCMS and reburn are retrospective cohort QA, not
predictors. Season-5 NBR remains the target; predictors retain the end-of-Season-1
cutoff. Do not feed the full eligible table indiscriminately into a model; use
the existing primary/early feature lists. No recovery-ratio definition is added.

## Observed final result

User-reported completed BigQuery result, recorded without rerunning cloud queries:
308,048 candidates -> **111,257 eligible patches across 151 fires**.

| Original chronological split | Eligible patches | Fires |
|---|---:|---:|
| Train: 2015-2017 | 61,090 | 78 |
| Validation: 2018-2019 | 19,561 | 36 |
| Test: 2020 | 30,606 | 37 |
| Total | 111,257 | 151 |

Entire fires retain their original split. Eligibility can remove all patches from
a fire; it does not reassign fires or rebalance the splits. No intermediate or
per-rule removal counts are inferred from these totals.

## EVT parsing reproducibility

The assembly generator now uses SAFE.PARSE_JSON with
`wide_number_mode => 'round'` after the existing key=value normalization.
High-precision area values can be rounded to a representable JSON number rather
than making the entire parsed value NULL. Class codes remain categorical;
dominant class selection, area/support fraction, tie-breaking and parsing/member
QA remain unchanged. Malformed input still surfaces through QA. Offline tests
verify the generated SQL and the reported high-precision fixture; they do not
claim to execute BigQuery's parser locally.
