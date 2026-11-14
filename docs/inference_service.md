# Local FastAPI and Docker inference

FastAPI serves one prepared patch record using the frozen **Option A original no-climate XGBoost** pipeline. Docker packages the application and its serving environment; it is not a cloud-hosting platform. This is local inference, not satellite retrieval, retraining or evaluation. No Streamlit application is included.

## Model and input contract

The trusted local artifact is `notebooks/mlruns/1/models/m-a4ba121dd1884ea6a02f0911fe21f021/artifacts/model.pkl`, from run `2cb3267c3dac4d4c80fe118e274637be`. SHA-256 is `c4e14763ef726627a5b1e16b354fc71b80d5f41fc800307321c4cddc36aca488`, matching the Phase 7 freeze. The pipeline already includes fitted numerical passthrough and categorical one-hot encoding; no external scaler or new fitting is needed. The loader verifies the fixed hash **before** cloudpickle deserialization and then verifies feature/encoded order, categories and unknown-ignore behavior. A hash is an identity check, not permission to load arbitrary pickle files. Never mount an untrusted artifact or edit the expected hash merely to make another file load.

`WILDFIRE_MODEL_PATH` is configured by the operator, never supplied through a request. Missing, corrupt or incompatible artifacts leave `/health` and `/predict` unavailable (503); responses expose no local paths. The application remains available for diagnostics, but a 503 is not a healthy deployment.

- `GET /health`: 200 with ready status/model identity only after successful loading.
- `POST /predict`: exactly one JSON object containing the 18 fields in the synthetic example below. Response: finite `nbr_5_mean` and `model_id`.
- `/docs`: automatic interactive schema and request documentation.

The API explicitly restores the trained feature order, regardless of JSON key order. Extra/missing fields, nulls, booleans/strings in numeric fields, fractional categorical codes, and nonfinite values are rejected. EVT is a positive integer category ID; ignition month is an integer 1–12. Unseen positive EVT codes and unseen but valid months preserve the frozen encoder's all-zero unknown-category behavior rather than inventing new columns. This is not validation that an unknown EVT code exists in LANDFIRE; the upstream data producer must supply legitimate source codes.

Fractions are 0–1, tree cover is percent 0–100, tree height/elevation are meters, slope is degrees 0–90, northness/eastness are -1–1 and fire area is positive acres. Fraction/directional boundaries permit 1e-12 floating-point reduction roundoff, without altering values. Tabular NBR inputs accept finite values without imposing imagery clipping or training-range bounds; `nbr_drop_patch` must be supplied upstream as NBRpre minus NBR1. The service validates field types/ranges, not the upstream temporal provenance or scientific eligibility. Inputs must use the project's source definitions and Season 1 cutoff. No target, fire ID, patch ID, climate feature or future observation belongs in the input matrix.

Predictions are not clipped and are **spectral vegetation condition, not ecological recovery percentage**. No endpoint accepts batch requests, models, targets or arbitrary locations.

## Run locally — Windows PowerShell

Run from the repository root in the existing project `.venv` (managed by uv). Its installed serving/model versions were used for verification; no package changes are required in that environment. The separate `requirements-serving.txt` pins the serving dependencies without installing the full geospatial/training stack or modifying `uv.lock`. XGBoost 3.4.1 is pinned separately in Docker to omit unused Linux CUDA dependencies. The public repository does not distribute the private model.

```powershell
$env:WILDFIRE_MODEL_PATH = (Resolve-Path 'notebooks/mlruns/1/models/m-a4ba121dd1884ea6a02f0911fe21f021/artifacts/model.pkl').Path
.\.venv\Scripts\python.exe -B -m uvicorn wildfire_recovery_forecasting.api:app --host 127.0.0.1 --port 8000 --workers 1
```

Leave that terminal open. In another repository-root PowerShell terminal:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Start-Process 'http://127.0.0.1:8000/docs'
$payload = Get-Content -Raw examples/predict_request.json
$native = Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict -ContentType 'application/json' -Body $payload
$native
```

`examples/predict_request.json` is a complete **hypothetical** patch using real feature names, not a released test record. It is suitable for both native/container parity checks. Stop native Uvicorn with Ctrl+C. Local loopback binding avoids exposing this minimal unauthenticated service publicly.

## Build and run Docker — Windows PowerShell

Prerequisite: Docker Desktop with a working Linux-container engine, available image/package network access, and the verified local artifact. The container runs as UID 10001 and reads the mounted file; ensure Docker file sharing permits access. Only the API/package initializer and dependency file are allowed into the build context. Private data, Git metadata, credentials, notes and model files are excluded by `.dockerignore` and never copied into the image.

```powershell
docker version
docker build -t wildfire-nbr5:phase7 .
if ($LASTEXITCODE -ne 0) { throw 'Docker build failed; do not continue' }
$modelPath = (Resolve-Path 'notebooks/mlruns/1/models/m-a4ba121dd1884ea6a02f0911fe21f021/artifacts/model.pkl').Path
docker run --rm --name wildfire-nbr5 -p 127.0.0.1:8001:8000 --mount "type=bind,source=$modelPath,target=/model/model.pkl,readonly" wildfire-nbr5:phase7
```

Keep that terminal open; no background process is required. In a separate repository-root PowerShell terminal, with native Uvicorn still running:

```powershell
Invoke-RestMethod http://127.0.0.1:8001/health
Start-Process 'http://127.0.0.1:8001/docs'
$payload = Get-Content -Raw examples/predict_request.json
$native = Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict -ContentType 'application/json' -Body $payload
$container = Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8001/predict -ContentType 'application/json' -Body $payload
if ($native.model_id -ne $container.model_id -or [Math]::Abs($native.nbr_5_mean - $container.nbr_5_mean) -gt 1e-7) { throw 'Native/container parity failed; investigate without changing artifacts' }
$native
$container
docker stop wildfire-nbr5
```

The absolute tolerance is a cross-platform smoke-check tolerance, not a change to predictions or evaluation metrics. Do not assume a Windows pickle is portable simply because dependencies are pinned: actual Linux reload/prediction parity must pass. Docker was unavailable during initial implementation. The user subsequently reported a successful Linux image build, ready health response and identical native/container prediction (0.3534964323043823) for the synthetic example. These are manual verification results, not a Docker rerun by Codex; repeat the parity check after environment changes. The base Python image and OS package repository are not digest/snapshot pinned; application dependency versions are fixed, but byte-identical image rebuilds are not claimed. No CUDA toolkit, PyTorch, MLflow or cloud client is needed for inference.

## Focused tests

Synthetic tests do not need the model:

```powershell
.\.venv\Scripts\python.exe -B -m pytest tests/test_api.py -q -p no:cacheprovider
```

Explicit single-record real-model smoke (no dataset or targets read):

```powershell
$env:WILDFIRE_MODEL_PATH = (Resolve-Path 'notebooks/mlruns/1/models/m-a4ba121dd1884ea6a02f0911fe21f021/artifacts/model.pkl').Path
$env:WILDFIRE_REAL_SMOKE = '1'
.\.venv\Scripts\python.exe -B -m pytest tests/test_api.py -q -p no:cacheprovider
Remove-Item Env:WILDFIRE_REAL_SMOKE
```

Tests cover readiness, successful predictions, ordering, invalid input, model-load failure, hash rejection before deserialization, nonfinite output and real-pipeline agreement. Existing HTTPX TestClient emits a deprecation warning in this environment; it does not indicate a model or inference failure. No dependencies were upgraded to silence it. No training/evaluation commands are part of service startup.
