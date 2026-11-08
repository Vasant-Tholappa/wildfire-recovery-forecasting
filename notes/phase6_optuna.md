# Phase 6: bounded, resumable CNN hyperparameter tuning

## Scope and commands

`manual_phase6_optuna.py` uses the frozen train (61,073) and validation (19,561) rows only. It never opens test arrays or refits preprocessing. MSE trains each model; the lowest chronological-validation patch MAE selects its checkpoint and is the Optuna objective. Each trial starts from scratch with training seed 42. Existing classical models and four historical Phase 6 runs are preserved.

Run from the repository root in PowerShell, using the existing environment (no installation or dependency synchronization):

```powershell
# Optional repeat of the isolated integration check: ONE epoch per model.
.\.venv\Scripts\python.exe -B manual_phase6_optuna.py --models both --smoke-test --time-budget-minutes 10 --threads 4 --seed 42

# Offline configuration/identity preview; creates no study or training run.
.\.venv\Scripts\python.exe -B manual_phase6_optuna.py --models both --dry-run --time-budget-minutes 540

# Nine-hour unattended session; do not run a second controller concurrently.
.\.venv\Scripts\python.exe -B manual_phase6_optuna.py --models both --target-completed-per-model 75 --time-budget-minutes 540 --threads 4 --seed 42

# Sunday continuation: same storage/study names; choose the available shared budget.
.\.venv\Scripts\python.exe -B manual_phase6_optuna.py --models both --target-completed-per-model 75 --time-budget-minutes 540 --threads 4 --seed 42
```

Smoke mode always creates new `phase6_tpe_v1_smoke_<unique>_<model>` studies in an isolated subdirectory. Both must show one COMPLETE, zero FAIL, one epoch, an MLflow run ID and checkpoint hash. Smoke mode permits only one attempt per family, including failures; it never retrains a failed smoke automatically. Smoke runs are tagged and never count toward production. `--synthetic-smoke` uses tiny generated development samples and isolated tracking instead. Neither smoke mode changes production's 30-epoch contract.

## Search space and model compatibility

| Setting | Search |
|---|---|
| Learning rate | Log-uniform 0.0001–0.003 |
| Batch size | 64, 128, 256 |
| AdamW weight decay | Log-uniform 0.000001–0.01 |
| Image embedding dropout | Uniform 0–0.4 |
| Two convolution widths | (16,32) or (32,64) |
| Image embedding width | 32, 64, 128 |
| Prediction head width | 16, 32, 64 |
| Fusion-only tabular branch width | 16, 32, 64 |
| Fusion-only tabular branch dropout | Uniform 0–0.4 |

Both preserve two 3x3 convolutions, ReLU and two modest pooling operations, 2x15x15 input, and unbounded scalar output. Dropout acts on the image embedding; the separate fusion dropout acts on its tabular representation. No irrelevant fusion parameters are sampled for CNN-only. All sampled parameters affect computation. Default parameter-bearing layer names and shapes are preserved, so original default state dictionaries remain loadable. `model_configuration.json` reconstructs each sampled architecture; weights remain separate.

Each new study queues the original configuration first: widths 16/32, embedding 64, head 32, fusion branch 32, zero dropout, batch 128, learning rate 0.001 and weight decay 0.0001. This is a new controlled trial, not a rewrite or reinterpretation of historical runs. Fixed settings: CPU, four threads, training seed 42, MSE/AdamW, 30 epochs maximum, early-stop patience 5, min_delta 0.0001. Every strictly lower validation MAE is checkpointed even when improvement is below min_delta.

## Persistence, scheduling and stopping

Production storage is Git-excluded `data/processed/phase6_optuna/studies.sqlite`, with stable study names `phase6_tpe_v1_cnn_only` and `phase6_tpe_v1_fusion`. SQLite persists each trial and intermediate metrics. A local OS-held lock prevents overlapping invocations using that storage and releases on process death; never manually bypass a live lock.

Resume rejects mismatched dataset manifest hash/version, search space, model family, objective, training settings, library versions or relevant implementation hashes. Do not change source/settings halfway through the studies. Study contracts are stored in SQLite. TPE uses all stored trial history; its per-ask RNG seed is deterministically derived from sampler seed (default 2026), study name and stored attempt index. This avoids requiring unsafe pickle deserialization of sampler state and produces the same sequence across ordinary session boundaries. Interrupted attempts remain in history and can change subsequent proposals. Exact reproducibility across hardware/library changes is not promised.

Scheduling is sequential and prefers the family with fewer COMPLETE trials, using attempted counts as a tie-breaker. If the last two attempts were from one family, the other eligible family goes next to prevent starvation from pruning. At most 75 COMPLETE trials per model; PRUNED and FAIL do not count. `--max-new-completed-per-model` optionally bounds additional completions in one invocation. After one family reaches its limit, the other continues. Milestones 25/50/75 are durable study attributes (`milestone_25`, etc.), not pauses. They include state counts, best configuration/run ID, cumulative trial time and improvement since the preceding milestone. Session JSON reports include recent 12 outcomes and median duration.

The single invocation deadline includes setup time and prevents scheduling another trial after expiry. A current trial finishes and saves; overrun may be as long as the remainder of a 30-epoch trial, not necessarily a few seconds. Fifty completions each is the desired minimum effort, not a guarantee within nine hours; wider networks and trial lengths affect throughput. The driver never expands the cap to compensate for pruning/failures.

Ctrl+C marks the interrupted attempt FAIL and preserves best-so-far evidence where Python can finalize it; completed trials are not repeated. On abrupt termination, the next locked invocation reconciles RUNNING Optuna trials against their linked MLflow records. A verified completed MLflow objective is committed to Optuna without retraining; verified pruning is retained; incomplete attempts become FAIL without automatic retry. Missing/unreadable or conflicting identities block recovery. A tiny crash window before linking a newly created MLflow run can leave an orphan run; no completed Optuna trial is duplicated or deleted. Check available disk space, keep the laptop powered/awake, and avoid changing files while tuning.

Three consecutive FAIL trials across both models halt unattended scheduling. This history survives restart; simply relaunching does not clear it. Investigate the recorded tracebacks before any recovery decision. No automatic failed-trial retries are performed. Ctrl+C is practical, not a guarantee against forced process termination or power loss during writes. SQLite and saved artifacts preserve the evidence already written.

## Pruning and tracking

A median pruner requires 12 COMPLETE trials, eight warm-up epoch steps and at least five comparable observations. The callback explicitly prohibits pruning before epoch 9. It reports validation MAE at 1-based full-epoch steps. Early stopping can legitimately complete a trial before pruning begins. PRUNED is distinct from FAIL and COMPLETE; MLflow uses a FINISHED lifecycle for pruned runs with an explicit `outcome=PRUNED` tag and pruning epoch/reason.

`cnn_tracking.tracked_fit()` is shared by standalone training and Optuna. Every trial has a fresh MLflow run, bidirectional study/trial/run linkage, actual parameters, architecture/configuration, dataset hash and small provenance sidecars, epoch train MSE/MAE, validation MSE/MAE/RMSE/R2/equal-fire MAE, epoch durations, best metrics, checkpoint/hash and outcome/error tags. `trial_seconds` covers objective setup and artifact logging; `fit_seconds` covers the fitting/tracking block before final artifact upload. Completed objective values are checked against MLflow's best validation MAE. Pruned/interrupted trials retain available curves and best-so-far weights without counting as completed results. Prior runs are not resumed or overwritten.

All SQLite stores, model weights, reports and datasets stay in ignored local paths. Keep `mlflow.db`, `mlruns/`, frozen data and Optuna storage in private backups. This work does not start a production study, retune classical models or inspect 2020. Repeated selection on the same 2018–2019 cohort is exploratory; freeze the final comparison protocol before any test evaluation.

## Validation

Focused tests exercise both widths/families, checkpoint-key compatibility, sampling, cumulative resume/counting, fairness under pruning, global deadline, systemic failure stop, milestones, contract mismatch, exclusive lock, interruption recovery and source-level test protection. A tiny real SQLite/MLflow synthetic test simulates interruption between MLflow completion and Optuna tell, then resumes without rerunning the completed trial. Separate partial-trial tests verify pruning/failure histories and checkpoint evidence. Standalone synthetic commands remain covered by regression tests. Actual real-data smoke results are recorded below after verification.

Real-data integration smoke verified: two COMPLETE one-epoch trials, zero PRUNED/FAIL, about 76.6 seconds including setup. CNN-only run `6b47079bbb054dec993bbb9218902978` had MAE 0.1057750490; fusion run `30e878fafc114098beeaad472d5f039c` had MAE 0.0830462022. These match the earlier one-epoch benchmarks exactly and are integration checks, not new tuned model claims. Both have complete 1-based epoch histories, matching Optuna/MLflow objectives, nonempty checkpoint files with matching hashes and reconstruction metadata. Smoke storage: `data/processed/phase6_optuna/phase6_tpe_v1_smoke_9221baf586/studies.sqlite`. All four prior runs' rows, parameters, metric histories and tags were unchanged. Production study storage was not created.

Final focused validation: **35 tests passed**. Static syntax, offline production dry-run and Git whitespace checks passed. One nonfatal SQLAlchemy deprecation warning originates in MLflow's ORM; successful metric/artifact readback was verified. Optuna 5.0.0 was already installed and declared, so dependencies/lockfile were unchanged. No production optimization was launched.
