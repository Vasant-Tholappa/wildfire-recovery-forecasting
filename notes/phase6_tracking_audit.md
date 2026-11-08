# Phase 6 MLflow tracking audit

Audit date: 2026-11-07. PASS for the four completed runs' recorded training evidence. No training, model deserialization, test-data access or cloud operations were performed.

| Run | Run ID | Epochs / best epoch | Best validation MAE | Epoch time sum / MLflow run elapsed (s) |
|---|---|---:|---:|---:|
| CNN-only benchmark | `52740c13e775462c9c2ba68689a456bd` | 1 / 1 | 0.10577505 | 32.485 / 33.357 |
| Fusion benchmark | `b893bfef0f684f0b970b3e7e1735ab82` | 1 / 1 | 0.08304620 | 34.813 / 35.032 |
| CNN-only initial | `0cbb94a71bd648d1b2c45e4b37bdf124` | 10 / 5 | 0.07577765 | 338.390 / 339.457 |
| Fusion initial | `867d9787ba8f4278a456ad37ffccc617` | 10 / 5 | 0.07597226 | 302.050 / 302.436 |

## Evidence checked

All runs are FINISHED. Each saved `training/history.csv` agrees with MLflow's complete, consecutive 1-based epoch histories: training MAE/MSE, validation MAE/RMSE/R2/MSE/equal-fire MAE and epoch seconds. MSE is named `train_loss` / `val_loss`; it was already present, not missing. `selection.json` agrees with the lowest validation-MAE epoch and all logged best metrics.

Each run retains `training/architecture.txt`, `configuration.json`, `selection.json`, `history.csv`, and a nonempty `best_state.pt`, plus `dataset_metadata/manifest.json` and `tabular_preprocessing.json`. Saved configurations match MLflow parameters. The logged dataset-manifest bytes match recorded SHA256 `1d7019a466e459a625b0e6ee00a73d92219dda846c1cbddce142285221d1b063` (dataset `phase6_cnn_dataset_v1`). All four record clean source revision `880cde5e46b6fead703257aef21907d33c744bf2`, seed 42, CPU, four threads, deterministic algorithms, package versions, actual hyperparameters, and 61,073/19,561 train/validation rows.

Initial full runs used batch 128, AdamW learning rate 0.001 and weight decay 0.0001, maximum 30 epochs, patience 5 and minimum improvement 0.0001. Both stopped after epoch 10 and selected epoch 5. Benchmark maximum epochs was one. Optimizer implementation/defaults are recoverable from the recorded source and PyTorch version; architecture and parameter counts (25,553 / 28,849) are saved.

## Minimal tracking additions

`manual_phase6_train.py` now explicitly labels AdamW, MSE and validation-MAE selection. It logs completed epochs, cumulative epoch seconds and elapsed `fit_seconds`, plus best checkpoint artifact path and SHA256. `fit_seconds` includes the fit function's history/checkpoint I/O and best-weight restoration, but excludes later artifact logging. Epoch seconds and MLflow start/end duration are different measurements and should not be conflated.

Existing runs received only additive `epochs_completed`, `epoch_seconds_total`, `run_elapsed_seconds` metrics and checkpoint path/hash/source tags, derived from saved histories, existing timestamps and checkpoint bytes. Original metrics, statuses, timestamps and checkpoints were preserved. A SQLite backup and the exact local backfill script are retained under Git-excluded `data/processed/phase6_tracking_audit/`. Historical `fit_seconds` cannot be reconstructed precisely and was not invented.

Future trial invocations may pass paired `--study-name NAME --trial-number N` arguments. Every invocation creates a fresh run, logs its actual CLI configuration and best validation MAE, and tags study/trial identity. `MLFLOW_RUN_ID` is rejected to prevent accidental historical-run resumption. A future Optuna driver must invoke this tracked entrypoint per trial (with actual sampled arguments), or explicitly integrate equivalent tracking; calling bare `cnn_training.fit()` does not create an MLflow run. No Optuna study/driver was implemented or executed here.

## Reproducibility limits and readiness

The four experiments have consistent source/configuration/data identities, complete curves and selected-weight artifacts. This audit verifies recorded reproducibility evidence, not bitwise replay or checkpoint inference equivalence: models were not loaded or retrained. Keep the frozen dataset, local SQLite database and `mlruns/` artifacts together in private backups; none belong in Git. Hardware/package differences can affect future replay.

Five focused mocked tests cover metric completeness and steps, fresh run creation across trial invocations, trial tags/configuration, checkpoint references, invalid trial identities and rejection of historical-run resumption. They perform no fitting. No tracking blocker remains for a separately authorized bounded study; its trial driver still needs to use the documented tracking contract. Repeated selection on chronological validation remains exploratory, and 2020 stays untouched.
