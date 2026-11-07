# Phase 6 Day 2B: controlled production imagery pilot

Historical pilot interface below remains available. Day 2D adds explicit bounded
`--population` waves; see [population operation](phase6_population_waves.md).

`manual_phase6_production.py` is restricted to exactly four products:

| Fire | Batch | Eligible patches | Raster |
|---|---|---:|---:|
| CA4083912297820170808 | r212_c-207 | 714 | 219 x 75 |
| CA3646112190120160722 | r166_c-210 | 3,794 | 363 x 372 |

Each exports `nbrpre` and `season1`, under
`gs://ee-wildfire-recovery-vasant-data/imagery/phase6_nbr_context_v1/<fire>/<batch>/<product>.tif`.
The four completed POC products are excluded. There is no arbitrary-fire option,
population submission mode, polling loop or automatic retry.

The launcher pins the exact Day 2A manifest SHA256. It verifies frozen local keys,
dates, original-batch occupancy and reconstructed geometry before submission.
It reuses `image_export_arguments()` and changes only the task description/bucket/
prefix to the frozen production destinations. No scientific image logic changes.
Both references are resolved from the original complete retained batch before any
export starts: Season-1 L8 scene, SR_B5, valid native UTM CRS and north-up 30 m grid.
Unlike the POC, these two scene IDs are not known in advance; the selector and
returned metadata are checked and the actual projection is saved/passed onward.
Metadata checks do not establish successful image materialization.

## Commands (PowerShell, repository root)

Offline dry-run (default; no cloud access or submission-state writes):

```powershell
.\.venv\Scripts\python.exe -B manual_phase6_production.py
```

After explicit authorization, submit at most the four pilot products:

```powershell
.\.venv\Scripts\python.exe -B manual_phase6_production.py --submit
```

Read-only status (queries recorded IDs once; no local/cloud writes or submissions):

```powershell
.\.venv\Scripts\python.exe -B manual_phase6_production.py --status
```

After interruption, reconcile existing tasks (EE reads, local JSON update only):

```powershell
.\.venv\Scripts\python.exe -B manual_phase6_production.py --reconcile
```

Only after reconciliation succeeds, the same `--submit` command can submit remaining
PENDING products. Existing successful/active/completed IDs are skipped. FAILED,
CANCELLED, ambiguous or missing remote records block submission. No automatic reset
or retry command exists. An ambiguous intent can be resolved only by exactly one
matching task; no remote match is not proof that retrying is safe.

## Records and recovery boundaries

Durable local state:
`data/processed/phase6_nbr_context_v1_pilot_submission.json`.
Exclusive lock: same basename with `.lock`. Atomic JSON replacement and fsync are
reused from the POC. Intent is saved before every start; returned IDs are immediately
saved. Ctrl+C stops the one-shot command and retains records; remote tasks continue.
A hard process kill can leave a lock: confirm no controller is running and review
records before manually removing only that lock. Never delete the state JSON to
force retries. The launcher exits without waiting for EE completion.

Before each wave, all visible active tasks plus pending submissions must fit within
four slots. This is an inventory guard, not an account-wide scheduler; do not run a
second controller concurrently. Existing destination objects or unknown task states
fail closed. Previously submitted pilot tasks must remain uniquely identifiable.
`COMPLETED` means EE task completion, not validated GeoTIFF acceptance. Inspect the
finished rasters before approving any wider extraction.

Check actual EE usage and remaining cloud budget before launch. The historical
$50 budget and billing alerts are not automatic spending enforcement. No cost or
runtime extrapolation from the POC is claimed. This step prepares commands only;
no pilot exports or live metadata queries were performed by Codex.
