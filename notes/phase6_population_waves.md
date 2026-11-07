# Phase 6 Day 2D: bounded population imagery waves

No cloud operations were executed while implementing this controller. The frozen
manifest and scientific adapter remain unchanged. This extends the same
`manual_phase6_production.py`; legacy pilot commands are preserved.

## State and acceptance

Offline baseline: 151 fires, 111,257 eligible patches, 302 products; four original
POC products plus four QA-accepted pilot products complete, 294 products pending.
The pinned manifest supplies POC task IDs/generations. Pilot acceptance is checked
against its existing submission record and saved raster/source-pixel QA. These
historical files are read, never rewritten. Population state is separate:
`data/processed/phase6_nbr_context_v1_population_submission.json`.
Back up that file alongside the frozen manifest and historical acceptance records.

A live reconciliation checks both versioned GCS prefixes, exact single GeoTIFF
objects and recorded generations. New EE COMPLETED tasks must have a nonempty,
unambiguous object; this establishes export completion, **not raster/chip QA**.
A missing object stops submissions for later reconciliation; it never causes a
retry. Every new raster still needs local inspection before CNN use.

## Operator commands (PowerShell, repository root)

Offline first-wave preview (no state writes or cloud access):

```powershell
.\.venv\Scripts\python.exe -B manual_phase6_production.py --population --max-new 16
```

Only after authorization and manual EE usage/billing review:

```powershell
.\.venv\Scripts\python.exe -B manual_phase6_production.py --population --submit --max-new 16 --max-minutes 60 --poll-seconds 60
```

Remote read-only status, including task states and object inventory (no local writes):

```powershell
.\.venv\Scripts\python.exe -B manual_phase6_production.py --population --status
```

Reconcile after interruption or after tasks finish (remote reads, atomic local-state
updates only; no submissions/retries):

```powershell
.\.venv\Scripts\python.exe -B manual_phase6_production.py --population --reconcile
```

Preview again, then repeat the bounded submit command to continue. Each invocation
has its own NEW-submission cap, not a cumulative budget. Completed and recorded
active products are skipped. A partially submitted fire's pending channel can
resume without repeating its successful channel. Default selection excludes 2020;
when separately authorized, add `--include-test-inputs` to preview/submit remaining
2020 inputs. The same identifier-only reads apply: no targets are loaded.

## First wave: 16 products / eight non-test fires

Ordering is deterministic, using only manifest year, raster area and fire ID.
Round-robin 2015-2019; within each year choose the lower median remaining raster
area, then largest, then remaining ascending areas (ties by ID). This provides a
modest range of years and occupancy, not a statistically representative sample.
Each selected fire contributes nbrpre and season1; intact pairs are not split to
fill an odd cap. Interrupted single-channel remainders are permitted.

| Fire | Batch | Year/split | Eligible patches | Raster |
|---|---|---|---:|---|
| CA4049212319820150803 | r210_c-209 | 2015 train | 1,138 | 165x96 |
| CA3622311855820161021 | r157_c-185 | 2016 train | 428 | 123x81 |
| CA3334911719120171207 | r125_c-181 | 2017 train | 21 | 177x102 |
| CA3664611822120180708 | r161_c-181 | 2018 validation | 135 | 90x177 |
| CA3703411834520190904 | r164_c-181 | 2019 validation | 712 | 138x123 |
| CA3687411890520150801 | r165_c-186 | 2015 train | 4,418 | 345x318 |
| CA3453012005420160615 | r143_c-201 | 2016 train | 1,018 | 201x192 |
| CA3442911910020171205 | r142_c-195 | 2017 train | 2,557 | 372x330 |

The ordered selected products are atomically persisted before the first start in
`data/processed/phase6_nbr_context_v1_population_submission.waves.json`. The ledger
also records fire IDs, manifest checksum, expected count, task IDs and derived
completed/submitted/pending membership. Submission state remains authoritative;
membership is refreshed when reading, reconciling and exiting a wave.

Continuation prioritizes the same wave's pending product IDs, even with an odd
remaining count or a smaller `--max-new` cap. Once all wave products are submitted or completed, the same invocation selects
the next deterministic pending wave. Previously submitted tasks continue to count
toward the four-task limit; completion is not required to advance. One global
new-submission cap and deadline cover every wave in the invocation. Include `--include-test-inputs` both to
select and to resume a wave containing unfinished 2020 inputs.

The interrupted original 16-product wave was recovered locally from the documented
eight fires and existing 11 unique completed task IDs. Recovery requires exactly
19 completed/283 pending, no issues, and precisely the expected five pending
products; discrepancies abort. The separate ledger preserves the original
submission-file SHA256 and does not rewrite historical submission records.

Resume preview and execution (execution only when operator authorized):

```powershell
.\.venv\Scripts\python.exe -B manual_phase6_production.py --population --max-new 5
.\.venv\Scripts\python.exe -B manual_phase6_production.py --population --submit --max-new 5 --max-minutes 60 --poll-seconds 60
```

With `--max-new 5`, this submits at most those five pending products. A larger cap
permits automatic continuation into subsequent waves within the same deadline.
Back up both the submission JSON and `.waves.json`; never delete either to force
retries. The existing bounded-session and four-outstanding-task protections remain.

## Safety and cloud-call behavior

- `--submit` is mandatory; `--population` explicitly selects the full-state mode.
- Foreground only: finite cap (1-302), finite duration (up to 24 hours), 30-300-second
  polling interval. Defaults: 16 products, 60 minutes, 60 seconds. No unlimited mode.
- At most four outstanding tasks, counting the union of visible EE active tasks
  (including unrelated clients) and recorded unfinished IDs. Refresh immediately
  before every start, after potentially slow image preparation. No account-wide
  atomic quota lock exists: operators must not start other EE workloads concurrently.
- Population and legacy pilot exclusive locks prevent simultaneous local controllers.
  Status can run concurrently because records are atomically replaced.
- Resolve one small exact-original-batch reference metadata request per newly
  prepared fire; reuse its verified projection for both unchanged input composites.
  No reference selection over the buffered rectangle.
- Each reconciliation: one EE task-list request, one task-status batch when recorded
  nonaccepted IDs exist, and two GCS prefix listings (pagination may add requests).
  Listings are bounded at 1,000 objects each; hitting the bound fails closed.
  No image downloads or cloud writes except explicitly authorized EE export starts.
- Reconcile at entry, before each submission and after each capacity wait. With no
  waits the 16-product wave uses 17 reconciliation passes and eight reference checks;
  waits add passes. No runtime/EECU estimate is inferred from the pilot.
- Durable intent precedes start; returned IDs are immediately saved. Ctrl+C or a lost
  response during start records ambiguity and stops. Resolve only a uniquely matching
  remote identity through explicit reconciliation. Never delete state to force retry.
- Stop on failures, cancellations, missing/conflicting objects, unknown identities,
  quota/permission exceptions. No automatic retries. EE retries disabled; GCS listing
  retries disabled. Review errors before continuing; no failed-task reset command.
- Exit immediately when cap is reached, or when duration expires; do not wait for
  already-submitted tasks to finish. They continue remotely without the controller.
  Duration is a submission-admission deadline, not a hard process-kill deadline:
  in-flight requests/graph construction can overrun it. HTTP timeouts are bounded.
- On a hard kill, inspect both `.lock` files, confirm no process is running, then
  remove only stale locks and reconcile. Never remove task records.
- Reporting distinguishes completed, running, queued, failed, pending, remaining,
  new submissions and issues. Remote status reports do not imply imagery QA acceptance.

Neither the task cap nor billing alerts enforce the $50 project budget or EE quota.
Check current usage and remaining budget manually between waves. No population-wide
coverage promise is made: the pilot had 16 chips below provisional full-context 80%
coverage. That threshold is not frozen and does not change export selection.

## Active-task output propagation grace

An exact, single, nonempty GeoTIFF appearing while a uniquely identified recorded
task is READY/RUNNING is provisional, not a completion or an immediate conflict.
The controller persists its first-observed wall-clock time and object generation,
keeps the task in the outstanding count, and monitors through the existing bounded
poll loop. Completion requires EE COMPLETED and the same verified object generation.
`--sync-grace-seconds 600` is the default (configurable 30-3600 seconds). The timer
survives interruption; monitoring does not reset it. Expiry is checked on each poll.

Duplicate/unexpected objects, changed generations, disappearing provisional outputs,
unverified identities, failures/cancellations and permission/quota errors still stop
immediately. No task is resubmitted. Status remains read-only; reconciliation saves
the timer locally. Object identity checks do not independently prove raster contents.

If a known synchronization delay remains when the submission cap is reached, the
foreground process continues read-only monitoring until resolution, grace expiry,
or the existing wave-duration deadline. It never submits beyond the cap. If the
wave deadline is reached first, the process exits with the provisional state saved;
resume/reconcile later. Ordinary active tasks without this condition retain the
existing exit-at-cap behavior. Check `synchronizing` and `issues` in progress output.
The grace duration is an operational allowance, not a measured EE propagation bound.

## Explicit 2020 input authorization

`--include-test-inputs` now propagates through `inputs_for()` and
`image_export_arguments()` to `validate_candidate(allow_test_inputs=True)`.
The former POC-only unconditional test-fire rejection is removed only with this
explicit permission. Default rejection, canonical occupancy, temporal fields,
frozen keys, geometry and reference selection checks remain intact. The adapter
still calls only `landsat.build_fire_inputs()` and returns nbrpre/season1; no
Season-5 target imagery is built. Eligible Parquet reads remain identifier-only.
Offline wave previews now run the same local input/geometry validator as live
submission, rather than only listing manifest entries. Persisted waves and task
IDs are unchanged; reconcile before resuming an interrupted authorized wave.
