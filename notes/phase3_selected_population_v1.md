# Budget-sized whole-fire population — 2026-10-18

Proposed frozen selection: **643 fires**, 2,259 tile-120 batches and 5,838,002
canonical patches before eligibility screening. No source calculations, target,
chronological splits or patch eligibility thresholds changed.

Run locally with `.venv\Scripts\python.exe manual_select_population.py`.
Inputs are the existing 1,904-fire GeoParquet, exact tile-120 occupancy table,
and Candidate A + combined-Daymet forecast. There are no cloud calls.

Within each recorded event-state code × ignition year × acreage category,
rank fires by SHA256 of `phase3_fire_sample_v1:{fire_id}`. Acreage categories are
[0,5000), [5000,20000), [20000,100000), and [100000,infinity).
Keep `max(1, floor(fraction * stratum_count))` fires from each ranked stratum.
Choose the largest fraction on a fixed 0.001 grid whose total forecast including
20% retries fits 700 EECU-hours. The result uses fraction 0.363 across 211 strata.
Cost sizes the common fraction; fires are not sorted or individually rejected
by cost. All patches of each selected fire are retained before later QA.
Sparse strata are oversampled: this is not a self-weighted probability sample.
Retain the available/selected stratum counts for later reporting or weighting.
No target values or observed recovery outcomes enter selection.

All eleven western state codes remain represented. The original spatial-intersection
population also includes one TX-coded event; it remains selected, separately
labelled, rather than silently changing the geographic inclusion rule. State
labels here are event-ID prefixes, not newly calculated perimeter-state assignments.

Forecast uses the existing product-wise occupancy curves, excluding nbrdrop and
using validated combined Daymet (not Candidate C). Central: **356.929 h**;
original planning scenario: **577.585 h**. Budget uses the larger of central and
planning per batch, **578.004 h**, then adds 20% (**115.601 h**) for retries:
**693.605 h**. This is a planning scenario, not a guaranteed maximum. With the
reported 1.538 h already consumed, the monthly total would be approximately
695.143 h, leaving about 304.857 h against the stated 1,000 h allowance.
There is no reliance on November quota.

Split counts: train 337; validation 186; test 120. Years 2015–2020:
87, 95, 155, 117, 69, 120. Size categories: 374, 151, 87, 31 respectively;
largest retained fire is 432,180 acres. These are candidate-fire counts;
forest/disturbance/measurement QA may later reduce usable fires and patches.

Artifacts under `data/processed/phase3_selected_fire_population_v1`:

- `.parquet`: unchanged canonical manifest columns/geometries for selected fires.
- `.json`: totals, margins, input hashes, selected-ID hash and assumptions.
- `_selection.csv`: all input fires, selection flag, rank hash, stratum and cost.
- `_strata.csv`: joint state/year/size available and selected counts.

The original canonical manifest and historical benchmark artifacts are untouched.
These outputs are local/gitignored. The frozen ID hash is
`144260f1aea28220426bfe2b08461a3c138a6307673b616b7a82760cdda0db53`.

Launch dependency: combined Daymet must be promoted separately; current production
still uses separate Daymet products. With promotion this sample implies 27,108
raster tasks (12 per batch); vector-local MTBS history is not an EE raster task.
The cost model is based on three WOODBURY measurements, not cross-fire replicates.
Small-batch costs and geographic/scene-complexity transfer remain uncertain; the
planning curve holds the smallest benchmark cost constant below 3,600 patches.
No launch, benchmark, Daymet promotion or cloud operation is authorized by this note.
