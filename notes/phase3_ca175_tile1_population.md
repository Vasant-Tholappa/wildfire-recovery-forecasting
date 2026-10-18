# Compact California population: phase3_ca175_tile1_v1

## Automatic bounded continuation (Windows PowerShell)

From the repository root, after manually checking quota/billing:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_population_continuous.ps1 -MaxHours 12 -MaxNew 400 -SessionMinutes 15 -NoProgressMinutes 120
```

This runs the existing controller sequentially in 15-minute slices, always with
max-active=4, and automatically continues until all 2,100 products are persisted,
12 total wall-clock hours or 400 total new status records have been reached, or an
error/intervention/no-progress stop occurs. Durable new status intents count
conservatively toward the total cap, even if submission was ambiguous. No retries.
No overlapping wrappers (exclusive local lock) or controllers (existing driver
lock) are permitted. Do not run a manual submitting controller alongside it.

The total limits span all child sessions in this wrapper invocation. Restarting
the wrapper is an explicit operator authorization for a fresh allowance, after
reviewing usage. Existing persisted work is skipped/resumed, not resubmitted.
Limits are checked between synchronous operations: an in-flight preparation,
network call or conversion may run past a time boundary. They are not hard task
timeouts. The no-progress limit is checked at slice boundaries, allowing up to one
slice plus in-flight work before stopping. Remote tasks continue after a stop.

Logs are `data/processed/phase3_ca175_tile1_v1_unattended_*.log`; console output
shows the wrapper PID and completed/running/failed/pending/new counts between
slices. Native session output, including throughput, is appended after each
slice. In another terminal:

```powershell
uv run --offline python manual_population.py status
Get-Content (Get-ChildItem .\data\processed\phase3_ca175_tile1_v1_unattended_*.log | Sort-Object LastWriteTime | Select-Object -Last 1).FullName -Tail 30
Get-Process -Id <PID-shown-by-wrapper>
```

Press Ctrl+C in the wrapper terminal to stop; wait for the Python child to exit
before resuming. If Windows/process termination leaves locks, confirm no wrapper
or Python controller remains before removing the corresponding continuation and
driver lock files. Never delete a live controller's lock. Read-only status is safe
alongside the wrapper. After inspecting any failure, resume with the same command;
failed work still requires explicit review, not automatic resubmission.

Before leaving overnight, plug in the laptop, disable sleep while plugged in,
keep the terminal/network active, and check remaining EE monthly EECU allowance
and current plus delayed Google Cloud charges against the $50 total budget.
There is **no verified automatic quota/billing enforcement**. Alerts and workload
forecasts are not caps; smaller unattended limits are necessary if budget headroom
is uncertain. Inspect consumption again before authorizing another allowance.

This approved scope reduction retains the exact 175 frozen fires and full original
perimeters, but restricts extraction to one nonempty tile per fire. Within each
fire, sort tiles by retained patch count, then batch ID; choose index `floor(n/2)`
(upper median). No imagery, target, future outcomes, or eligibility thresholds
enter selection. This avoids choosing only sparse edge tiles or only the densest
tiles. It is an occupancy-based sample, not a random or spatially exhaustive
representation of each fire; report that limitation in modeling results.

- 175 tiles, 308,048 candidate patches, 2,100 required product tables.
- 2015..2020: 25, 25, 41, 27, 16, 41 fires; train/validation/test: 91/43/41.
- Patch occupancy: min 43, median 1,262, mean 1,760.27, p75 2,328,
  p90 3,957, max 6,265. Candidate counts are not post-QA usable counts.
- All 12 products, combined Daymet, NBR5 target, 90 m origin-aligned lattice,
  tile side 120, timing, missingness, and fire-held-out splits are unchanged.
- Patch IDs still encode global row/column; full manifest geometries and timing
  remain available for future imagery/CNN extraction with the same splits.

## Frozen local inputs and reproducibility

Run `uv run --offline python manual_freeze_ca175_tile1.py` using the preserved
`phase3_ca175_population_v1.parquet`, companion JSON, and
`phase3_ca175_batches_v1.parquet`. This writes only new local files:

- `data/processed/phase3_ca175_tile1_v1.parquet` (identical approved fire manifest)
- `data/processed/phase3_ca175_tile1_v1_batches.parquet` (the actual tile restriction)
- `data/processed/phase3_ca175_tile1_v1.json`
- `notes/phase3_ca175_tile1_inputs_v1.json` (tracked hashes and all 175 tile IDs)

Back up the three generated inputs before launch; keep the existing input archive.
Both manifests are necessary: the fire manifest alone does not specify this subset.
The launcher validates both hashes, exact counts, and reuse identities, and records
them in run metadata. Previous population artifacts remain untouched.

## Reuse and recovery

The medium tile `CA4118512343320200728/r215_c-210` is selected naturally by the
rule. All 12 original outputs in `phase3_ca175_v1` were read back and checked for
exact IDs, unique rows, required raw columns, and COMPLETE status (3,602 rows each).
Four interrupted reconciliations were finished from existing CSVs without new
exports: Landsat nbrpre, combined Daymet, LCMS, focal MTBS severity.

The launcher reads those original COMPLETE records and references their original
`output_uri` in place, failing closed if status/row count/URI disagree. It does not
copy objects or write replacement COMPLETE records. **Final assembly must use the
reuse mapping, not merely list the new namespace.** There are 2,088 new exports.
The previously completed 500-patch tile is not selected; its 12 verified outputs
remain intact, outside this modeling subset. Arizona outputs are untouched.

Reconciliation now creates only exact batch patch IDs and raw column schemas;
it never builds source graphs. Submission uses the same batch-local canonical
geometry and caches the existing 12-product graph descriptors for that batch.

## Operator commands (do not launch until approved)

```powershell
uv run --offline python manual_population.py dry-run --max-new 4 --max-active 4
uv run --offline python manual_population.py run --max-new 4 --max-active 4 --max-progress 12
uv run --offline python manual_population.py status
uv run --offline python manual_population.py session --session-new-cap 48 --max-active 4 --max-progress 12 --poll-seconds 60 --max-minutes 120
```

The last command is a bounded continuation, not an unlimited population launch.
Ctrl+C pauses local work; remote tasks continue and are reconciled on restart.
No automatic retries. Quota/billing checks remain operator responsibilities.

## Launch and overnight operation

Use one operator session at a time. For the first 8–12 hours, after the initial
four-export wave, use:

```powershell
uv run --offline python manual_population.py session --session-new-cap 400 --max-active 4 --max-progress 12 --poll-seconds 60 --max-minutes 720
```

The session stops at its submission cap, time limit, or a new failure/ambiguity;
it does not retry indefinitely. Remote tasks may still run when it exits. Keep
the terminal open and computer awake. Resume with the same command after checking
usage. Network errors stop safely; inspect and rerun, rather than blindly restarting.
If a hard process kill leaves the local lock, remove that lock only after confirming
no other writer is running. Normal Ctrl+C removes it in `finally`.

Read-only `status` is allowed while a session runs. Do not run a second `run` or
`session` writer. After pausing, reconcile without submitting using:

```powershell
uv run --offline python manual_population.py run --max-new 0 --max-active 4 --max-progress 12
```

Status reports new persisted COMPLETE products divided by wall-clock hours since
run metadata creation (before source/grid preparation). Reused completions are
excluded from the rate. Preparation, local/GCS work, idle time, and pauses are
included. A session report is the pre-action snapshot; the next iteration or a
separate status call reflects newly persisted results. Unknown/zero throughput
produces no ETA. EE COMPLETED alone does not count.

Assess at 8 hours (at least 200 new persisted products) and 12 hours (at least 300)
for the 25/hour target. At exactly that rate, 2,088 new products take about 83.5
wall-clock hours; this is conditional arithmetic, not a runtime guarantee. Ensure
completed work spans all 12 products and small (<1,000), middle (1,000–2,500), and
larger (>2,500) selected-tile occupancies before extrapolating. These bins describe
operations only, not scientific eligibility. If the early sample lacks a bin,
continue ordinary selected production rather than launching a benchmark. Review
failure counts and EE monthly usage/billing before each subsequent bounded session;
neither the forecast nor billing alerts are hard caps.

Backup archive: `data/processed/phase3_ca175_tile1_launch_inputs_v1.zip` contains
exactly the three frozen inputs listed above. Archive and member hashes are in
`notes/phase3_ca175_tile1_backup_v1.json`. Generated data/ZIP are excluded from Git.
