# Final California scope: ca_portfolio_175_v1

**Current extraction scope:** [one representative tile per fire](phase3_ca175_tile1_population.md)
supersedes the all-tile workload below. This document preserves the unchanged
175-fire selection and historical all-tile counts.

This supersedes the 643-fire launch population. The original 1,904-fire manifest
and former 643-fire artifacts remain historical and unchanged. Retain full original
perimeters of CA-coded events, not state-clipped geometries.

Exactly 175 fires: years 2015..2020 have 25,25,41,27,16,41 fires;
train/validation/test = 91/43/41. Tile size 120; 729 batches; 2,334,795 candidate
patches; 12 products per batch = 8,748 exports. Candidate A and combined Daymet
remain approved; Candidate C is excluded. Forecast: 152.957 central hours and
262.573 conservative hours including 20% retries, neither a guaranteed cost cap
nor a seven-day elapsed-time guarantee.

Reproduce locally: `uv run --offline python manual_freeze_ca175.py`.
Use original CA event codes; strata are year and [0,5000), [5000,20000),
[20000,100000), [100000,infinity) acres. Allocate floor(175*n/323), minimum one
per nonempty stratum. Add slots by largest ideal-minus-current allocation
(reverse for excess, never below one); ties follow sorted year/size-label order.
Rank each stratum by SHA256(`ca_portfolio_175_v1:{fire_id}`), then fire_id.
The mandatory ID hash is
`3fc06786066229a08fbd63016826f6fd49beac1c6b43fbf09989b92e4bc5659f`.
This is the previously compared sample, not a new draw. Sparse strata are slightly
oversampled. Keep fire-held-out splits and report California-specific generalization.

## Inputs, namespace, preservation

Frozen manifest: data/processed/phase3_ca175_population_v1.parquet.
Companion JSON and _selection.csv retain selection metadata; phase3_ca175_batches_v1.parquet
contains exact occupancy. notes/phase3_ca175_inputs_v1.json records IDs/input/output
hashes and the new ZIP hash. Back up data/processed/phase3_ca175_launch_inputs_v1.zip
independently before launch. The earlier Drive backup does not contain this new sample.
The original full manifest remains necessary to rerun selection; preserve its existing
local copy/backup as well. Frozen geometries can be recovered directly from the new ZIP.

manual_population.py explicitly uses production version phase3_ca175_v1 for EVERY
work-item path, controller call and run metadata operation. Former phase3_prod_v1
Arizona outputs remain untouched and are not California completion records. Other
visible active EE tasks still conservatively consume concurrency capacity.
Local progress and lock filenames are California-specific. Core config defaults
and historical benchmark tools are unchanged; use this launcher for the new build.

CNN prerequisites remain: patch_id encodes signed global EPSG:5070 row/column;
90m squares are reconstructable. Frozen ignition dates, prefire search years,
Season-1/5 years and splits are unchanged. nbr5 is the label, never a predictor.
Future CNN comparisons must use the same final eligible patches and fire splits.

## Initial real throughput checks (authorization required)

Use the existing launcher, no benchmark framework. Start with the 500-patch batch
and all 12 products; after completion inspect usage/errors, then repeat on medium
and full batches. These are retained production outputs, not throwaway experiments.

```powershell
uv run --offline python manual_population.py dry-run --max-new 12 --max-active 4 --only-batch CA3574511858420160816/r152_c-185
uv run --offline python manual_population.py session --session-new-cap 12 --max-active 4 --max-progress 12 --poll-seconds 60 --max-minutes 120 --only-batch CA3574511858420160816/r152_c-185
uv run --offline python manual_population.py session --session-new-cap 0 --max-active 4 --max-progress 12 --poll-seconds 60 --max-minutes 120 --only-batch CA3574511858420160816/r152_c-185
uv run --offline python manual_population.py status
```

For subsequent approved checks, repeat the session commands with:
- CA4118512343320200728/r215_c-210 (3,602 patches)
- CA3612312160220200819/r161_c-208 (14,400 patches)

The session stops at its new-submission cap; use zero-cap sessions to collect
remaining tasks. Repeat bounded sessions if the time limit expires. Do not assume
12 tasks were submitted just because the command ended: inspect reported actions.
Ctrl+C pauses locally; remote tasks continue. No automatic failed-item retries.
Quota and the $50 budget require operator checks. For broader production omit
--only-batch and choose a bounded session cap after reviewing real throughput.
No launch or cloud operation occurred while preparing this population.
