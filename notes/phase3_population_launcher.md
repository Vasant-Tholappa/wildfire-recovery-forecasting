# Minimal population waves

`manual_population.py` reads the frozen selected manifest and existing tile-120
occupancy table. It checks 643 fires / 2,259 batches / 5,838,002 patches and
27,108 unique source/product/fire/batch status paths. It does not resample.
Daymet is `combined`; Landsat has four products; Candidate C is absent.

## Operator commands (repository root)

```powershell
# Local-only enumeration; cloud counts deliberately unknown.
uv run --offline python manual_population.py dry-run --max-new 2 --max-active 4

# Read-only cloud snapshot; actual persisted counts and proposed next wave.
uv run --offline python manual_population.py status --max-new 2 --max-active 4

# AFTER authorization: first wave, at most two new submissions.
uv run --offline python manual_population.py run --max-new 2 --max-active 4 --max-progress 2

# Progress completed exports into Parquet; no new submissions.
uv run --offline python manual_population.py run --max-new 0 --max-active 4 --max-progress 12

# Continue after checking monthly EECU and billing in the consoles.
uv run --offline python manual_population.py run --max-new 4 --max-active 4 --max-progress 12
```

`--offline` prevents uv dependency downloads; only `dry-run` prevents cloud
access. `status` reads EE/GCS and writes only a local snapshot. `run` performs
cloud mutations through the existing controller and persistence helpers.
The `run` command exits after one bounded pass. The `session` command repeats
these same operations at a configurable interval (minimum 30 seconds), bounded
by total new submissions and elapsed minutes. This is a foreground process,
not a background service. Limits are operational safety settings, not tuning.

Run one operator/process only, including across machines. The exclusive local
`data/processed/population_driver.lock` prevents concurrent local invocations.
After a hard process kill, verify no other driver remains before manually removing
that stale lock. Do not delete status records to force a retry.

Pause by issuing no further `run` commands. Ctrl+C interrupts an invocation;
already submitted remote tasks continue. Submission intent/status makes uncertain
submissions review-required rather than silently resubmitted. A missing saved task
ID blocks new work. Tasks omitted from the EE snapshot reserve capacity. Counts
are snapshots, not transactional global concurrency guarantees if other operators
submit simultaneously.

Completed and failed records are skipped; failed items require explicit review
and the existing controller's `retry_failed=True` mechanism, not an automatic retry
loop. Existing raw shards may require review before retry. No retry-all option is
provided. An unexpected controller result stops new submissions in that wave.

The current fire grid and batch graphs are cached within an invocation. Grids are
built only for fires needed by that invocation using existing canonical logic;
no population grids or raster graphs are built during dry-run. Each actual batch
must match its saved occupancy and the builder must return exactly the locked
12-product inventory. Existing raw schemas go directly to the existing controller.

Run metadata freezes the manifest hash, tile size and inventory under the existing
production version. Conflicting existing metadata/status identities stop the run.
Commit approved code before launch so recorded Git HEAD identifies the executed
implementation. No commits or launch were performed while preparing this driver.

`data/processed/population_progress.json` stores the latest read/advance report.
Reported EECU is only available metadata from visible task history, not a complete
monthly usage meter. Check EE quota/usage and Google Cloud billing between waves.
The 693.605-hour planning forecast (including retry allowance) is not a hard cap;
the stated $50 spending limit is not automatically enforced by this script.
Pause new waves if usage or cost approaches operator limits. No November reset
is assumed. No monitoring infrastructure or automatic quota claims are added.

An offline dry-run cannot prove production has no existing cloud objects: its
completed/active/pending counts are null, not fabricated zeros. Use `status` for
actual counts. `run` always rereads persisted statuses rather than trusting the
local report. EE task states are inspected once per invocation; completion uses
the existing controller and CSV-to-Parquet flow, never object existence alone.

## Bounded sessions (2026-10-18)

After the two-task first wave is verified, run foreground sessions:

```powershell
# Collect the first wave automatically, with ZERO new submissions.
uv run --offline python manual_population.py session --session-new-cap 0 --max-active 4 --max-progress 12 --poll-seconds 60 --max-minutes 60

# First broader session: up to 24 new tasks, at most 4 active.
uv run --offline python manual_population.py session --session-new-cap 24 --max-active 4 --max-progress 12 --poll-seconds 60 --max-minutes 120

# After review, larger bounded sessions without thousands of commands.
uv run --offline python manual_population.py session --session-new-cap 500 --max-active 4 --max-progress 12 --poll-seconds 60 --max-minutes 480
```

A session stops as soon as its new-submission cap is reached; remaining remote
tasks continue and are collected by the next session. The duration is checked
between operations, not a hard timeout on an in-progress API call. Zero-cap
sessions only collect existing work. Errors/failures stop a session; no automatic
retry loop. Ctrl+C releases the local lock normally, without cancelling EE tasks.
Counts are explicitly pre-action snapshots; session newly-submitted count includes
this pass. Run `status` for a refreshed final count. Keep this terminal/computer
running; restarting the command resumes using GCS status, not a local cursor.

## CNN prerequisites and artifact preservation

PASS for reconstructability (not a claim that CNN imagery or labels already exist):
patch_id = fire_id + global row/column; EPSG:5070 square bounds are
[col*90, row*90, (col+1)*90, (row+1)*90]. Geometry can remain separate from tables.
The selected manifest retains exact ignition dates, the five-year prefire search
pool, Season-1/Season-5 years and chronological fire-level splits. Future CNN
features must use the approved prefire/Season-1 windows only, through September 30
of Season 1; Season-5 NBR is a label, never an imagery predictor. Use the SAME final
eligible patch IDs and fire splits for tabular/CNN comparison. Archived code retains
Landsat QA/scaling/compositing rules. Production nbr5 raw sums/areas support the
same nbr_5_mean label after local/SQL assembly; missing labels remain missing.

The selected manifest, occupancy table, selection audit and results are gitignored.
A local `data/processed/phase3_launch_inputs_v1.zip` preserves the exact frozen
inputs, with an entry hash/selected-ID record in `notes/phase3_launch_inputs_v1.json`.
Include that small JSON in the planned Git checkpoint. Copy the ZIP to a durable
independent backup before launch; a same-disk ZIP is NOT disaster protection.
No cloud backup was performed. Do not rely on refetching a potentially updated
MTBS asset to recreate exact geometries. Original local files remain untouched.

Recommended checkpoint: selected-population utility/note; combined-Daymet code,
contracts and tests; population launcher/tests/operator note plus the input-hash
record. Exclude review diffs, credentials, logs and large generated artifacts.
No commit/push has been performed. Read-only preflight on 2026-10-18 found zero
production status records and zero visible active EE tasks. This does not prove
future write permissions or current monthly billing/quota; review consoles.
