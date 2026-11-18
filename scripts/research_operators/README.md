# Archived research operators

These 30 files preserve the original research source **byte-for-byte**. They were relocated for repository navigation, not refactored into a new runnable package. The public Streamlit app and local FastAPI/Docker service do not import them.

## Original working snapshot

- Verified pre-reorganization commit: `cbd13f167393098e4a7735ecb2e5fdcef4d1a199`.
- Annotated local tag: `research-v1-working`, created at the verified pre-reorganization commit using the repository-local GitHub noreply identity. The tag has not been pushed or published.
- This snapshot preserves the code layout, not the private data, fitted models, MLflow/Optuna records or Python environment.

For historical workflows, use a **separate** checkout at the commit above. Do not switch or overwrite a working directory that contains valuable local artifacts. Reuse excluded artifacts only after checking the original recorded paths and environment requirements.

Root-level commands in historical notes, notebooks and `run_population_continuous.ps1` apply to that snapshot. The PowerShell launcher remains historical evidence and is not operational in this presentation layout. Many research tests also require the original root imports.

## Why these files are not directly runnable here

Sibling imports, repository-root calculations, recorded source-file locations and Phase 7 preservation checks assume the original layout. Moving the files does not change their bytes, but does change their paths. Original hashes and verification records have **not** been regenerated or bypassed. Do not treat a missing-file verification error here as changed scientific results, and do not edit a frozen record to make it pass.

Ordinary current application checks are:

```powershell
.\.venv\Scripts\python.exe -B -m pytest tests/test_explorer.py tests/test_api.py -q -p no:cacheprovider
```

Do not run every historical operator or the entire research suite to launch the dashboard. Research commands can perform cloud operations or access sealed artifacts; use their original authorization and execution contracts.

## Inventory

- `manual_daymet_combined_benchmark.py`
- `manual_daymet_qa_benchmark.py`
- `manual_daymet_qa_experiment.py`
- `manual_freeze_ca175.py`
- `manual_freeze_ca175_tile1.py`
- `manual_phase6_build_dataset.py`
- `manual_phase6_download.py`
- `manual_phase6_imagery.py`
- `manual_phase6_inspect.py`
- `manual_phase6_local_qa.py`
- `manual_phase6_manifest.py`
- `manual_phase6_optuna.py`
- `manual_phase6_pilot_qa.py`
- `manual_phase6_postoptuna.py`
- `manual_phase6_production.py`
- `manual_phase6_submit.py`
- `manual_phase6_train.py`
- `manual_phase6_verify_pixels.py`
- `manual_phase7_evaluate.py`
- `manual_phase7_refit.py`
- `manual_planb_benchmark.py`
- `manual_planb_eecu_forecast.py`
- `manual_planb_population_counts.py`
- `manual_population.py`
- `manual_prepare_bigquery.py`
- `manual_prepare_terrain_refresh.py`
- `manual_select_population.py`
- `manual_terrain_population.py`
- `manual_terrain_recovery.py`
- `manual_vector_qa.py`
