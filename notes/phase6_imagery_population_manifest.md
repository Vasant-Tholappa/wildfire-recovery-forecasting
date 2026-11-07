# Phase 6 Day 2A: offline imagery production manifest

Namespace: `phase6_nbr_context_v1`. No exports or cloud calls were made.
Generator: `manual_phase6_manifest.py`. Output:
`data/processed/phase6_nbr_context_v1_manifest.json` (local, Git-ignored).
Scale audit: `data/processed/phase6_nbr_context_v1_manifest_audit.json`.

| Split | Years | Fires | Eligible patches | Products |
|---|---|---:|---:|---:|
| Train | 2015-2017 | 78 | 61,090 | 156 |
| Validation | 2018-2019 | 36 | 19,561 | 72 |
| Test | 2020 | 37 | 30,606 | 74 |
| Total | | 151 | 111,257 | 302 |

Only fire/patch/batch IDs, grid coordinates, year and split are loaded from the
eligible Parquet. No target/measurement columns are decoded. Frozen Phase 3 file
hashes are verified by `load_population()`. Keys are unique, associations and
chronological dates match, and every eligible square has positive-area intersection
with its original fire. All 111,257 full 15x15 windows and central 3x3 footprints
pass exact grid arithmetic. No patches or fires are added, dropped or resampled.

Entries are sorted by fire ID and serialized with sorted JSON keys, without a
runtime timestamp. They contain key hashes, input provenance, dates, original batch
occupancy, output bounds/grid, two stable product identities and storage estimates.
Scene selection remains earliest Season-1 L8 SR_B5 over ALL retained squares in the
original frozen batch. It must be reconstructed with `batch_patches()` using that
fire's frozen perimeter, not the eligible subset or buffered rectangle. Scene IDs
and native projections for the other 149 fires remain unresolved offline. Never
hardcode the two POC UTM zones as a population-wide projection rule.

## Geometry and scale

All output grids are north-up EPSG:5070 at 30 m, with 180 m context beyond the
eligible patch-square rectangle. Width min/median/max: 15/138/372 pixels.
Height min/median/max: 15/126/372 pixels (marginal medians, not one typical raster).
Total uncompressed payload: **63,278,208 bytes (60.35 MiB)** for both NBR products,
each carrying float32 NBR plus float32 validity. This excludes GeoTIFF overhead,
compression effects, separate chip arrays, backups and duplicate downloads; reused
files are counted once.

| Largest fire region | Dimensions | Bytes for both products |
|---|---:|---:|
| CA3646112190120160722 | 363 x 372 | 2,160,576 |
| CA3442911910020171205 | 372 x 330 | 1,964,160 |
| CA4065012263020180723 | 321 x 372 | 1,910,592 |
| CA4009112093120200817 | 372 x 300 | 1,785,600 |
| CA3687411890520150801 | 345 x 318 | 1,755,360 |

No dimension exceeds the full tile-plus-context bound of 372 pixels. Thirty-five
fires have eligible area below 10% of their unbuffered eligible-square bounding
rectangle; this is a reporting flag, not a filter. Examples: CA3792911898320200817
(7 patches, 0.88%), CA3805012095020180629 (40, 1.08%), CA3334911719120171207
(21, 1.27%). Sparse rectangles include intervening landscape context intentionally.

## Reuse

Four files qualify based on matching eligible and original-batch keys, geometry,
export semantics, submission identities, saved raster/source-pixel QA, and actual
local GeoTIFF hashes. They remain at their existing POC locations:

- `CA3839112224120171009/r187_c-209/nbrpre.tif`
- `CA3839112224120171009/r187_c-209/season1.tif`
- `CA3839611945120180904/r180_c-186/nbrpre.tif`
- `CA3839611945120180904/r180_c-186/season1.tif`

Base: `gs://ee-wildfire-recovery-vasant-data/imagery/phase6_nbr_context_poc_v1/`.
The manifest distinguishes `planned_uri` in the new namespace from `effective_uri`
pointing to the reused object; action REUSE means do not submit/copy it. Object
versions and local hashes are recorded. Current remote availability was not queried;
check the recorded generation before consuming remotely. **298 new exports remain.**

## Proposed first wave and safety gate (not executed)

Four tasks: both channels for these two non-test fires, excluding reused POCs:

- CA4083912297820170808 / r212_c-207: 714 patches, 219 x 75. Chosen nearest the
  population median raster pixel count, ties broken by fire ID.
- CA3646112190120160722 / r166_c-210: 3,794 patches, 363 x 372. Largest available
  non-test rectangle, deterministic fire-ID tie break.

Future orchestration should reuse exclusive controller locking, durable intents and
IDs, task/object duplicate checks, reference metadata verification before submission,
and no automatic retries. Maximum four visible active tasks, counting other work;
run no concurrent controller. Reconcile ambiguous/failed submissions before retrying.
A production launcher is not implemented in this manifest-only step.

Review current EE quota and remaining cloud budget before authorization. Neither
storage estimates nor the seven-minute POC establish population runtime/EECU cost;
five-season prefire graph cost and queues remain uncertain. No new standalone
benchmark campaign is required by this manifest. Review cross-fire/cross-split
context overlap before training and freeze input-only imagery coverage handling
separately. Real missing-pixel behavior and population-wide image coverage remain
unverified. No scientific thresholds or model scope were changed.
