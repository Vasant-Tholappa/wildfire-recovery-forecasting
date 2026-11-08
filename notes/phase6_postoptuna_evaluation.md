# Phase 6 post-Optuna audit and validation checkpoint

## Status and scope

The frozen studies passed audit: CNN-only **75 COMPLETE / 12 PRUNED / 0 FAIL** (87 attempts); fusion **75 COMPLETE / 3 PRUNED / 0 FAIL** (78 attempts). All 165 attempts have unique, correctly linked MLflow runs. Complete and pruned histories, 1-based epoch steps, sampled configurations, architecture metadata, best epochs/metrics, Optuna intermediate values, dataset/preprocessing identities, checkpoint paths and SHA256 hashes agree. Pruned runs retain partial histories and explicit pruning reasons/epochs. No bookkeeping repair or historical overwrite was needed. All four selected checkpoints reconstructed with strict state-dictionary loading and finite synthetic predictions.

No further Optuna trials or search changes were made. The matched validation comparison below has been executed. **Three-seed stability is prepared but not yet complete:** four verified seed-42 results are reusable; eight new fits (seeds 123 and 456) remain for the operator command below. Existing durations suggest roughly 48 minutes if repeated fits take similar time, but early stopping and system load can change this. No stability means/standard deviations are invented.

## Frozen top-two distinct configurations

Selections use COMPLETE trials only, sort by validation MAE (trial number breaks ties), and deduplicate the full parameter dictionary. Original model/training source hashes, dataset manifest, seed 42, 30-epoch cap, patience 5 and four-thread execution agree, supporting seed-42 reuse.

| Family / trial | Seed-42 MAE | RMSE | R2 | Equal-fire MAE | Best / completed epoch | Duration (s) |
|---|---:|---:|---:|---:|---:|---:|
| CNN-only 17 | 0.07310426 | 0.09693876 | 0.74169228 | 0.07965209 | 14 / 19 | 389.83 |
| CNN-only 84 | 0.07368615 | 0.09826616 | 0.73456977 | 0.08035714 | 10 / 15 | 513.33 |
| Fusion 61 | 0.06640042 | 0.09045971 | 0.77506718 | 0.07793929 | 5 / 10 | 284.06 |
| Fusion 67 | 0.06643529 | 0.09011452 | 0.77678057 | 0.07746832 | 5 / 10 | 249.80 |

All use convolution widths 16/32. CNN 17: batch 128, learning rate 0.000988835423723774, weight decay 0.00006271865329967662, image dropout 0.022770208420461152, embedding/head 64/32. CNN 84: batch 128, learning rate 0.0027147292741852413, weight decay 0.0081720131463388, dropout 0.0034867146166319843, embedding/head 64/64.

Fusion 61: batch 64, learning rate 0.0004580416514938281, weight decay 0.00008473978283146536, image dropout 0.28187745744729453, embedding/head/tabular widths 128/64/32, tabular dropout 0.0016703546020509566. Fusion 67: batch 64, learning rate 0.0004466244131484898, weight decay 0.00000200777595186038, image dropout 0.20329141498541387, widths 128/64/32, tabular dropout 0.0015763723616229013. Exact machine-readable configurations and all run IDs are in the local audit artifact.

## Matched validation comparison — executed

All four predict the **same 19,561 patch keys in the frozen order, across 36 validation fires (2018–2019)**. No test arrays or 2020 targets are read. Neural inference uses the saved best checkpoint in evaluation mode, not the last epoch. The classical pipelines are the original no-climate RF (`b175c6f10eeb4ebfa8c0930192468d39`) and original no-climate XGBoost (`2cb3267c3dac4d4c80fe118e274637be`), not the later retuned XGBoost.

| Model | MAE | RMSE | R2 | Equal-fire MAE |
|---|---:|---:|---:|---:|
| Frozen no-climate RF | 0.07399106 | 0.09973618 | 0.72656889 | 0.08258693 |
| Frozen no-climate XGBoost | 0.07510489 | 0.10171563 | 0.71560770 | 0.08352082 |
| CNN-only trial 17, seed 42 | 0.07310426 | 0.09693876 | 0.74169228 | 0.07965209 |
| Fusion trial 61, seed 42 | 0.06640042 | 0.09045971 | 0.77506718 | 0.07793929 |

RF and XGBoost exactly reproduced their historical validation metrics. Local MLmodel run linkage, recorded model sizes, fitted hyperparameters, exact 18-feature order, identity numerical transformation and fitted OneHotEncoder were verified. Phase 6 raw validation inputs exactly match frozen Phase 5 source values. RF used skops with only the reviewed `sklearn.tree._tree.Tree` trust allowance; XGBoost used its allowlisted, locally produced cloudpickle artifact. No arbitrary external pickle is accepted. SHA256 fingerprints are recorded now; absent historical cryptographic hashes, this establishes current provenance/reproduction rather than cryptographic proof that bytes never changed.

Common metrics use frozen float64 targets for every model. Original CNN epoch scoring used float32 targets; recomputed checkpoint MAEs differ by about 5e-11, within the explicit 1e-8 verification tolerance. This is a precision difference, not a change in labels or selection.

At the selected frozen models/checkpoints, inference training MAEs on the same 61,073 coverage-eligible training rows are RF 0.02925809, XGBoost 0.02933173, CNN-only 0.05398456 and fusion 0.04110684. All validation MAEs are higher. These are fixed-model, dropout-disabled inference diagnostics, not online minibatch training-loss values. Classical models originally fitted 61,090 training rows; this diagnostic uses the common subset and does not refit them. Gaps alone do not distinguish overfitting from temporal/cohort shift.

## Per-fire pairing and shared-context sensitivity

Per-fire MAEs and every pairwise fire-level MAE difference are saved locally. Fusion beats RF on **23/36 fires**, with paired patch-MAE improvement 0.00759064 and equal-fire improvement 0.00464765. Fusion beats CNN-only on **19/36 fires**, improving patch MAE by 0.00670384 and equal-fire MAE by 0.00171280. CNN-only beats RF on **20/36 fires**; RF beats XGBoost on 22/36. Fire-level wins and patch-weighted gains measure different things because fire patch counts differ. No patch-independent significance or causal claim is made.

Removing only the preidentified 53 validation patches with training-context overlap leaves **19,508** rows:

| Model | Sensitivity MAE | Equal-fire MAE |
|---|---:|---:|
| RF | 0.07411834 | 0.08260144 |
| XGBoost | 0.07524609 | 0.08353758 |
| CNN-only | 0.07318937 | 0.07965431 |
| Fusion | 0.06651213 | 0.07794753 |

The ranking remains unchanged. This narrow sensitivity does not establish full spatial independence; the splits remain fire-disjoint and chronological with documented limited shared context. Primary results retain all frozen validation observations.

## Seed workflow and remaining work

Run from the repository root, after reviewing this checkpoint:

```powershell
.\.venv\Scripts\python.exe -B manual_phase6_postoptuna.py --train-seeds --evaluate
```

This first re-audits the frozen studies, then runs/resumes the eight missing seed fits, each with unchanged selected hyperparameters, training-only fitting, the original validation split, maximum 30 epochs and patience 5. It does not invoke Optuna optimization. Every new run is separately tracked with a parent Optuna run ID and `seed_stability` role. Each seed uses its best validation-MAE checkpoint. It writes individual MAE/RMSE/R2/equal-fire MAE, best epochs and durations, and configuration-specific mean MAE and **sample** standard deviation (ddof=1) once seeds 42/123/456 are available. It never pools different configurations into one seed estimate or chooses the lowest new seed as a winner.

The local job ledger records new run IDs and identity before fitting; an exclusive lock prevents competing invocations. Rerunning the same command reuses validated finished jobs, including a run completed before the ledger update. Failed, running or ambiguous jobs stop for review rather than being silently retrained. Atomic ledger saves retry transient Windows permission errors briefly and preserve temporary evidence if saving fails. Ctrl+C preserves MLflow/checkpoint evidence where finalization is possible; an interrupted fit is not automatically restarted.

Audit only: ` .\.venv\Scripts\python.exe -B manual_phase6_postoptuna.py `.
Existing-model comparison only: ` .\.venv\Scripts\python.exe -B manual_phase6_postoptuna.py --evaluate `.

Local outputs, all Git-excluded, live under `data/processed/phase6_postoptuna_v1/`: `audit.json`, `comparison.json`, `validation_predictions.parquet`, `per_fire.csv`, and `paired_differences.csv`. The seed command additionally creates `seed_jobs.json`, `seed_results.csv`, `seed_summary.json` and isolated checkpoint directories. MLflow remains in the existing local SQLite/artifact store. No historical runs or weights are overwritten.

These best-of-75 results are validation-selected estimates, not unbiased generalization performance. Seed stability addresses initialization variability conditional on selected configurations, not the optimism caused by repeated validation reuse. Keep 2020 sealed; freeze model identities and final-fitting/evaluation protocol before the eventual holdout comparison. Comprehensive consolidation and interview Q&A remain separate.

Validation: 26 focused tests passed (post-Optuna selection/audit tamper detection, key alignment, sealed-test guard, paired/sensitivity metrics, seed statistics, completed-job recovery and refusal of unfinished-job duplication, plus existing evaluation tests). Static syntax and CLI checks passed. The full 165-trial audit and the frozen-model comparison were executed locally; no additional real seed training, Optuna trial, cloud operation or test evaluation was performed.
