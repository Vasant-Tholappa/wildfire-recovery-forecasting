# Terrain coverage correction and recovery plan

## Implemented correction

`terrain.build_terrain_images()` retains the legacy first intersecting 3DEP DEM
as the primary image. `compose_elevation()` fills only masked pixels, using
alternatives in ascending `system:index` order and extending the footprint.
Unsupported pixels stay masked. Valid primary elevations win even when
overlapping assets disagree; there is no averaging.

Alternatives are selected within a 100 m geometry buffer for derivative
neighbors; images themselves are not clipped. One explicit reprojection to
the primary image's exact native projection/transform establishes the grid
before `ee.Terrain.products`. Default nearest-neighbor sampling handles the
slightly differing source transforms. This is not a 90 m DEM. Forest masking,
patch reductions, output fields and scientific eligibility are unchanged.
The primary selection intentionally retains the existing collection-first
behavior; it is not a permanent pin of a mutable upstream asset catalogue.

## Small read-only verification

No exports or cloud writes were performed. Only development-fire predictors
were loaded; no test outcomes were inspected. Measurements below are small
native-grid probes, not regenerated production patch tables.

| Fire / patch suffix | Old elevation support | New elevation/slope/aspect support | New mean elevation / slope |
|---|---:|---:|---:|
| CA4143512316220170626 / r26308_c-24535 | 0% | 100% / 100% / 100% | 1831.22 m / 24.31 degrees |
| CA4130712105620170725 / r25550_c-22856 | 0% | 100% / 100% / 100% | 1618.42 m / 22.58 degrees |
| CA3388411764320150419 / r15882_c-21869 | 100% | 100% / 100% / 100% | 156.20 m / 0.44 degrees |

Two additional previously supported patches and the reference had zero
pixelwise elevation and slope differences on common support. In both affected
fires, elevation and slope projections exactly matched the primary EPSG:4269
affine transform. Small overlap probes also had zero primary-value differences.
Edge-probe slopes were 29–34 and 21–26 degrees, respectively. Their forest-
masked elevation support increased from 6.70% to 17.25% and 36.08% to 95.79%;
new slope/aspect support matched new elevation support. These edge percentages
include forest masking and must not be interpreted as unmasked DEM coverage.

Earlier overlapping-asset checks found nonidentical elevations (approximately
-1.079 to +0.038 m in one probe), motivating primary priority. The sampled
seams show no obvious anomaly, not a guarantee about all seams. Derivatives
near newly filled boundaries may legitimately change. Genuine source gaps
and forest-mask exclusions remain missing; recovery of all missing rows is
not yet established.

## Smallest safe recovery (requires separate authorization)

1. Freeze the corrected code revision and original outputs. Use an isolated
   terrain-recovery version, e.g. `phase3_ca175_tile1_terrain_v2`; do not reuse
   original COMPLETE status keys or overwrite original terrain Parquets.
2. Run terrain only on the existing 175 selected tiles with the existing
   builder, forest mask, reductions, controller and CSV-to-Parquet validation.
   Validate a small affected/reference subset before continuing. This is up to
   **175 exports**, not 2,100. Refresh whole tiles, including partially supported
   observations, rather than patching only existing NULLs. No new population.
3. Require exact `(fire_id, patch_id)` coverage for all 308,048 candidates,
   unique keys, bounded support fractions and plausible terrain values. Compare
   old/new support and values, including seams; do not promise identical slopes
   at filled boundaries. Preserve unrecoverable missing values.
4. Prepare a terrain-only BigQuery join against the validated existing
   candidates, replacing elevation/slope/northness/eastness means and their
   existing validity fractions. Override the terrain URI even for the reused
   earlier-version tile. Leave the other 11 raster products and vector QA alone.
   Dry-run the SQL and review cost before execution; retain old tables.
5. Refresh eligible terrain columns by joining the **existing eligible keys**,
   not by changing or reinterpreting eligibility. Require exactly 111,257 rows,
   151 fires and unchanged splits (61,090 train / 19,561 validation / 30,606 test),
   labels, IDs and every nonterrain value. Then refresh/version the local
   Parquet and repeat terrain missingness checks without exploring test targets.

## Exposure and remaining limits

Historical terrain benchmark EECU-seconds for 3,600 / 8,100 / 14,400 patches:
18.886 / 51.980 / 119.822. Applying the existing occupancy forecast scenarios
to these 175 tiles (308,048 candidates) gives about **0.411 EECU-hours central**
and **0.976 hours with the small-batch floor**, before retries. A 20% allowance
on the latter is about 1.17 hours. These are **old single-image graph estimates**,
not measurements or a cap for the corrected composition. Extra DEMs and the
explicit native-grid derivative evaluation can increase work. Small synchronous
checks do not establish export runtime or full-tile memory behavior.

No new full-population computation has been authorized or executed. Inspect
actual initial recovery-task usage before the remaining terrain tasks. EECU
hours are not a GCS/BigQuery dollar estimate; current billing headroom under
the $50 ceiling must be checked separately before writes/queries. The recovery
runner/SQL URI override and actual artifact refresh are subsequent execution
steps, not implemented by this source correction.

## Prepared two-tile pilot (manual execution only)

`manual_terrain_recovery.py` is fixed to the following frozen terrain work items:

| Fire | Selected batch | Candidate patches |
|---|---|---:|
| CA4143512316220170626 | r219_c-205 | 3,963 |
| CA4009112093120200817 | r197_c-195 | 5,110 |

Only these two terrain tasks can be submitted. The namespace is
`phase3_ca175_tile1_terrain_pilot_v2`, project `ee-wildfire-recovery-vasant`,
bucket `ee-wildfire-recovery-vasant-data`. Original production paths, tables
and the modeling dataset are not updated. Run only one pilot process at a
time; do not run another submission controller concurrently.

From the repository root in PowerShell:

```powershell
uv run --offline python -m unittest discover -s tests -p test_terrain_recovery.py -v
uv run --offline python manual_terrain_recovery.py preflight
# STOP here until quota and billing checks below are satisfactory.
uv run --offline python manual_terrain_recovery.py submit
# Run later manually; one inspection/advance per item, no polling loop:
uv run --offline python manual_terrain_recovery.py check
# Once both statuses are COMPLETE:
uv run --offline python manual_terrain_recovery.py compare
```

`preflight` is entirely local: frozen manifest/occupancy hashes, exact canonical
patch construction, counts, and isolated URI reporting. It cannot verify live
credentials, remaining quota or billing. Before `submit`, inspect the project's
current Earth Engine EECU usage/quota and Google Cloud Billing actual costs and
pending usage. The previously reported 1,000-hour allocation and old consumed
balance are not a current balance. Keep headroom under the total $50 ceiling;
alerts and the two-task limit are not financial caps. The historical terrain
curve suggests roughly 0.014 EECU-hours for these two occupancies before retries,
but the corrected multi-DEM graph has not been export-benchmarked. No automatic
quota/dollar enforcement is provided. Submission checks visible active tasks
and refuses if adding the missing pilot tasks would exceed four active tasks.

`submit` builds only terrain and its existing LANDFIRE forest support, not the
12-source graph. Saved work is resumed/skipped through the existing controller;
there are no automatic retries. `check` refuses missing status and blocks export
creation defensively, including a raced/disappearing status. It constructs only
local batch polygons/IDs, never source graphs. READY/RUNNING returns WAITING;
EE COMPLETED proceeds through existing validated CSV-to-Parquet persistence.
FAILED/NEEDS_REVIEW stops for operator review. Ctrl+C is safe for later check;
an ambiguous saved submission intent needs inspection, not blind resubmission.

Each mode saves its own local JSON under `data/processed/` named
`phase3_ca175_tile1_terrain_pilot_v2_report_<mode>.json`. Check records include
the EE task metadata (state, timestamps, errors and EECU where available) and
persisted row count/output URI. Unexpected API errors also appear in the terminal;
saved cloud status remains authoritative. Reports are last-pass snapshots.

Comparison downloads only the original and corrected terrain raw Parquets.
It verifies schemas, exact IDs, unique keys, validity fractions and basic numeric
ranges, then uses the existing area/support summarizer. It reports null counts,
recovered/lost measurements, support decreases, mean/max changes on rows with
both measurements and corrected ranges. These are **patch-mean** comparisons;
changed means can reflect recovered pixel support and are not automatically
errors. It does not access test NBR5, re-screen eligibility, or alter data.

Share the check and compare JSONs plus any terminal errors. Successful structure
means COMPLETE/COMPLETE for both items, 3,963 and 5,110 exact patch rows and no
duplicates. Review recovered support, remaining nulls and any lost support or
large value changes before authorizing wider recovery. Remaining nulls alone
are not failure: genuine gaps and forest exclusions may remain. No wider
recovery or BigQuery/local modeling refresh is authorized by this pilot.

## Population recovery preparation (2026-10-25)

`manual_terrain_population.py` adapts the existing `manual_population` wave,
session, status, local lock and `BatchProducts` machinery. Its only graph builder
is the pilot's terrain product. Reconciliation never builds source graphs.
The frozen population remains 175 tiles / 308,048 candidate patches. Exactly two
products reference the pilot namespace in place; the other 173 use
`phase3_ca175_tile1_terrain_v2`. No original Parquet or status is overwritten.
The local preflight writes the 175 resolved URIs and hashes of canonical candidate patch-ID sets from the existing validated
supplemental QA Parquet to `data/processed/phase3_ca175_tile1_terrain_v2_preflight.json`.

Read-only inspection of the two old and two corrected terrain Parquets found:

- On unchanged valid support, measurements were identical to floating-point
  precision (largest slope difference about 3.6e-15 degrees).
- On originally >=99%-supported patches, maximum changes were 0.252 m elevation,
  0.111 degrees slope and 0.00254 for either aspect component.
- All changes exceeding diagnostic review levels of 1 m, 1 degree or 0.05
  aspect-component units coincided with increased support; none remained
  unexplained on unchanged support. These are review levels, NOT eligibility
  thresholds. Filling boundary support can change patch means and derivative
  neighborhoods. The evidence does not separately identify every seam effect.
- Pilots consumed 40.481 and 30.980 EECU-seconds (71.461 total, ~0.01985 hours).
  The old single-image forecast remains only a scale indication; neither these
  pilots nor the forecast guarantee a dollar/runtime cap for all tiles.

The detailed local safeguard is `data/processed/terrain_recovery_pilot_safeguard.json`.
No test target or nonterrain data was accessed. The existing candidate key set,
not the smaller eligible set, defines each recovery export.

Run from the repository root in PowerShell, with only one submission controller:

```powershell
# Local only; verifies frozen inputs and existing candidate keys for every tile.
uv run --offline python manual_terrain_population.py preflight
# Read-only current completion counts, including the two validated reused pilots.
uv run --offline python manual_terrain_population.py status

# After quota/billing review: fixed first five non-pilot fires, max four active.
uv run --offline python manual_terrain_population.py session --initial-five --session-new-cap 5 --max-minutes 120 --poll-seconds 60
# Drain those five with zero new submissions; then validate downloaded raw Parquets.
uv run --offline python manual_terrain_population.py session --initial-five --session-new-cap 0 --max-minutes 120 --poll-seconds 60
uv run --offline python manual_terrain_population.py verify --initial-five

# Only after reviewing the five: continue the other 168, still at max-active=4.
uv run --offline python manual_terrain_population.py session --session-new-cap 168 --max-minutes 720 --poll-seconds 60
# Drain any last tasks and validate all 175 resolved Parquets (read-only downloads).
uv run --offline python manual_terrain_population.py session --session-new-cap 0 --max-minutes 120 --poll-seconds 60
uv run --offline python manual_terrain_population.py status
uv run --offline python manual_terrain_population.py verify
```

`--initial-five` always means the same five sorted non-pilot fires, even on
restart. The first session submits four, reconciles completions, then submits
the fifth. Session caps count NEW tasks; a cap/time limit can leave remote tasks
running. Rerun to resume, or use `check` for one zero-submission reconciliation
pass (up to four completions). `status` makes no cloud writes. `verify` checks
raw schemas, exact IDs, row counts, finite measurements and support/range rules;
it is a structural validation, not a population-wide old/new scientific comparison.
Final expectation: 175 COMPLETE, zero running/failed/pending, verified row totals
308,048; two Parquet URIs retain the pilot version. All per-variable nulls/ranges
remain visible in the verification report. Genuine source gaps may remain.

The progress/verification JSONs live under the recovery version in local
`data/processed/`. Counts are pre-action snapshots; `status` refreshes them.
Failed records halt submission, with no automatic retries. Unknown task IDs
reserve capacity; ambiguous submission intents block new work. Ctrl+C releases
the local lock and leaves remote tasks/status intact. After forced termination,
remove a stale recovery lock ONLY after confirming no controller is running.
Do not run the original population controller concurrently. Session limits and
billing alerts do not enforce the $50 budget; check actual billing and remaining
monthly EECU quota before the five and before the larger session.

### Later refresh, not executed here

Use the preflight's exact 175 terrain URI mapping, including the two pilot URIs,
for terrain-only BigQuery staging. Dry-run/cost-check first. Validate unique
`(fire_id, patch_id)` and exact candidate coverage; summarize with the existing
terrain sum/valid-area/support formulas. Join to a new version of the existing
candidate table, replacing only four terrain means and existing terrain validity
columns. Preserve all nonterrain values including labels and retrospective QA.
Refresh the existing eligible rows by their frozen keys, preserving 111,257
patches / 151 fires and 61,090 train / 19,561 validation / 30,606 test rows.
Verify nonterrain equality and no row multiplication before publishing a new
local modeling Parquet. Retain the old files/tables for rollback. No eligibility
rerun, split changes, CNN extraction or nonterrain exports are needed.

Preflight checks the existing candidate IDs against frozen per-fire occupancies,
canonical ID syntax and tile coordinates without rebuilding all geometries.
Submission/reconciliation reconstructs each needed canonical tile and checks its
exact IDs against that same candidate key set. No target columns are loaded.
