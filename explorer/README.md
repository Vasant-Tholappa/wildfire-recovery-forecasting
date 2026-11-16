# After the Fire — public presentation app

Single-page Streamlit explorer, independent of FastAPI, Docker, model weights and private research files. All values are precomputed. **Prepared for review; not published or deployed.**

## Run locally (PowerShell, repository root)

```powershell
.\.venv\Scripts\python.exe -B -m streamlit run explorer/streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

Open `http://127.0.0.1:8501`. Stop with Ctrl+C. The implementation environment had Plotly/pandas/numpy already; six missing Streamlit-related packages were installed without upgrading existing packages. On a fresh presentation-only environment, install `explorer/requirements.txt` (Python 3.12). Do not sync the full research stack merely to run this dashboard.

Local review: inspect the hero at desktop and mobile widths; click a statewide fire marker and verify the selector/details change; use the accessible selector; change model and map layer; confirm scatter plots update; inspect the four-model performance expander and source notes. Marker selection is wired through Streamlit's supported PyDeck callback, with a stable layer ID. Automated tests exercise its event-handling helper and other widgets, but cannot substitute for actual browser/WebGL click testing. No screenshot or visual rendering review was performed during implementation.

## What visitors see

- Hero with attributed NASA Earth Observatory wildfire imagery (illustrative, not a project prediction), 151 studied fires, 51.45% relative MAE reduction and 30,606 held-out patches/37 fires.
- Genuine MTBS fire locations and all sampled patch footprints for the chosen fire, with a selectable frozen model and three views: predicted fifth-season vegetation, observed fifth-season vegetation, and prediction error.
- Actual saved point-level observations/predictions, saved per-fire metrics, four-model comparison with both fixed baselines and a validation/test reversal view.
- Optional four-model performance table for XGBoost, Random Forest, CNN, and CNN + tabular fusion, with concise scientific limitations. NBR is not percentage ecological recovery.

Saved scientific experiments retain eight model records across two training strategies; the dashboard deliberately displays only the original four fitted models. These presentation choices do not remove or change experimental evidence.

This is **Option C**: named fire geography and canonical patch geometry were already available. Full MTBS perimeters are not shipped. Statewide markers are interior representative points computed in EPSG:5070 from saved MTBS geometry and transformed to EPSG:4326, not ignition points. Every 90 m patch corner comes from the existing zero-anchored EPSG:5070 grid. No population reconstruction or satellite extraction occurred. Patch coverage is one selected tile per fire, not whole-fire coverage.

## Deploy later to Streamlit Community Cloud

After local review and separately authorized publication:

1. Publish only reviewed app code and `presentation/` assets with their provenance. Keep research datasets, models, notes and databases excluded.
2. In Community Cloud, select the approved GitHub repository/branch and **`explorer/streamlit_app.py`** as the entrypoint; choose Python **3.12**.
3. Keep `explorer/requirements.txt` next to that entrypoint. Community Cloud searches the entrypoint directory before repository root; this avoids selecting the root research `uv.lock` and installing torch/EE/MLflow. No secrets, model mount, FastAPI URL or OS packages are needed.
4. Confirm the build uses the small serving requirements, then review health, maps and controls in the hosted browser. Do not mistake a successful build for visual acceptance.

The root `.streamlit/config.toml` supplies the dark theme and disables Streamlit usage statistics. Core code/data can also run from a copy of the `explorer` directory alone; copy the theme configuration if desired. Basemap display requires public CARTO/OpenStreetMap tile access and browser WebGL. No Earth Engine, GCS, BigQuery or restricted APIs are used. Public basemap requests can expose ordinary browser/network metadata to their providers. If tiles are blocked, selectors, charts and tabular results remain available, but geographic background display will be degraded.

Free hosting may sleep/restart, has limited resources and requires checking current Python/package support. The application caches a ~3 MB presentation bundle and renders only the selected fire's polygons (largest observed fire view 4,116 patches), not all statewide patch polygons simultaneously. Hosted dependency installation has not yet been tested. No account sign-in, push or deployment was performed.

References: [Streamlit dependency precedence](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies), [PyDeck selection API](https://docs.streamlit.io/develop/api-reference/charts/st.pydeck_chart).

## Tests and preservation

```powershell
.\.venv\Scripts\python.exe -B -m pytest tests/test_explorer.py -q -p no:cacheprovider
```

The focused tests check bundle hashes/counts/coordinates, all-fire/model views, selection event interpretation, exact saved metric agreement when private sources exist, and Streamlit AppTest widget changes. They do not load a model or score new predictions.

`scripts/build_explorer_presentation.py` is a **local maintainer-only, offline** packager, not an app dependency. It requires private sources and existing geospatial libraries; refuses to overwrite a presentation directory; verifies exact keys and original artifact hashes; and never trains or infers. Do not run it to start or deploy the app. The public bundle is already prepared. No new scientific model rankings are calculated: all reported MAE/RMSE/R² values come from saved evaluation tables. Pointwise absolute differences are calculated solely for the requested map layer.

## Final UI revision review

The map keeps a stable pickable fire layer and a separate non-pickable mint highlight; only a single valid fire event changes the shared dropdown state. Empty, ambiguous, unknown and overlay events retain the current choice. This removes a rerun instability risk; it does not establish the cause of every missed click. Overlapping markers can still require zooming or the searchable dropdown. Automated event checks do not replace a browser click test.

Manually review desktop/mobile layout, map selection and dropdown synchronization, all three map views, equal-axis scatterplots, and attribution before approving publication. The responsive hero adapts its text and satellite image to narrow screens; Streamlit columns stack on narrow layouts. No new hosting or scientific dependencies were added. The app remains independent of the private inputs used only by the offline packager.
