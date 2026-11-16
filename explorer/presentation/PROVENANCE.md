# Curated presentation data — review before public release

This deliberately small extract is intended for the portfolio app. It is not the private research dataset or a new evaluation. Full precision model outputs are preserved in the compressed CSV; display formatting and map colors do not change them. All 30,606 evaluated 2020 patch keys are retained, across 37 fires. No display subsampling is applied. The broader eligible study contains 151 fires/111,257 patches.

## Mapping to authoritative local evidence

Paths below are repository-relative descriptions, not runtime dependencies. No absolute machine paths or private database identifiers are packaged.

| Public file | Authoritative local source | Transformation |
|---|---|---|
| `fires.csv` | `data/processed/fire_manifest.parquet`; `phase6_cnn_dataset_v1/test/index.parquet` | Only 37 matched fire IDs, names, area, evaluated counts and interior representative-point coordinates; no full perimeter dump |
| `patches.csv.gz` | `data/processed/phase7_test_v1/predictions.parquet`; `phase6_cnn_dataset_v1/test/index.parquet` | Exact ordered join; retain keys, saved observed NBR5 and four Option A predictions; transform canonical patch centers/corners from EPSG:5070 to EPSG:4326 |
| `per_fire.csv` | `data/processed/phase7_test_v1/per_fire_metrics.csv` | Saved eight-model MAE/RMSE and patch counts; no rescore |
| `summary.json` | `phase7_test_v1/comparison.csv`; `supplemental_baselines_v1/{baseline_metrics,model_mae_reductions}.csv`; `phase6_nbr_context_v1_manifest.json` | Saved comparisons/baselines/reductions and verified population totals; serialized metrics agree within 1e-14 |
| `provenance.json` | Above sources and saved evaluation/supplemental audits | Source-file hashes, exact ordered-key identity, output hashes and packaging policy; no raw targets outside the deliberately curated patch observations |

Coordinates use `(col + 0.5)*90, (row + 0.5)*90` for centers, and lattice cell edges for corners. Signed row/column identifiers match the existing `grid.build_fire_grid` contract. Latitude/longitude is not inferred by parsing the fire ID. Polygon corners are transformed with `always_xy=True`. These are actual sampled patch footprints, not approximated fire areas. Statewide markers are within MTBS geometry, not asserted incident ignition locations or geometric centroids.

The packager verified the original prediction-file digest, target digest and ordered-key digest against the PASS evaluation audit; baseline file hashes and audit identities also match. Original files were rehashed afterward and unchanged. Only saved outputs were selected; no model was loaded. The app has no private model/data/cloud runtime dependency. Observations are intentionally authorized derived presentation records, not a release of input rasters, fitted preprocessing or the original modeling Parquet.

## Source rights and attribution reviewed 14 November 2026

- **MTBS:** USDA Forest Service / US Geological Survey. Monitoring Trends in Burn Severity Thematic Burn Severity. Salt Lake City, Utah / Sioux Falls, South Dakota. The [catalog terms](https://developers.google.com/earth-engine/datasets/catalog/USFS_GTAC_MTBS_burned_area_boundaries_v1) permit use without additional permissions or fees and request attribution. They disclaim accuracy/fitness warranties and prohibit treating the maps as legal boundaries. Names and representative locations derive from that saved source.
- **Landsat:** USGS/NASA Landsat supplies the spectral observations underlying NBR. [USGS reuse guidance](https://www.usgs.gov/faqs/are-there-any-restrictions-use-or-redistribution-landsat-data) states no use/redistribution restrictions. No scientific input raster is republished in this data bundle. The separate illustrative NASA Landsat hero image is documented in [the asset provenance](../assets/README.md).
- Predictions and summary metrics are this project's outputs. No LANDFIRE/Daymet/DEM raw raster or feature table is redistributed in this presentation bundle. The project models used those documented inputs, but they are not provided as public source layers here.
- The browser basemap is attributed to CARTO and OpenStreetMap contributors; tiles are requested from the provider, not downloaded/repackaged. The hero combines original CSS with an attributed NASA Earth Observatory image; its source, credit and reuse guidance are recorded in [the asset provenance](../assets/README.md).

Privacy review: no personal information, credentials, absolute filesystem paths, serialized models, experiment databases, run artifacts or private notes. Fire IDs/names are public-source incident attributes. Precise sampled-patch coordinates and observed/predicted NBR5 are intentional presentation disclosures and should be reviewed by the owner before the future public checkpoint. Do not automatically license the entire research archive on the basis of these source-specific terms.

Scientific caveats: one tile per fire, retrospective eligibility, clustered patches, chronological shift, unequal development/tuning procedures and validation-selection optimism. Post-test subgroup inspection is exploratory. A XGBoost's primary-MAE win does not imply superiority for every fire or independent post-selection validation. No ecological recovery percentage, causal mechanism or operational savings is established.

## Final presentation revision

Year 1 adds only `nbr_1_mean` from `data/processed/phase6_cnn_dataset_v1/test/tabular.parquet`, matched in exact unique `(fire_id, patch_id)` order for all 30,606 rows. Its source hash is recorded. Existing CSV fields were preserved byte-for-byte when appending this column. No target, prediction, geometry or metric changed.

The frozen population manifest verifies 78 training, 36 validation and 37 test fires (151 total). The fire manifest confirms first/fifth complete seasons 2021/2025 for the 2020 cohort, and fire years 2015–2020 with fifth-season observations through 2025.

Historical presentation provenance (superseded): an earlier hero used explicit **2024** context rather than a typical-year average: [CAL FIRE's 2024 archive](https://www.fire.ca.gov/incidents/2024) reports 8,110 wildfires and 1,077,711 acres. Verified 14 November 2026. That earlier design used the existing 37 MTBS representative locations as its hero graphic. These historical checks are retained as provenance, not descriptions of the current UI.

The current hero uses an attributed NASA Earth Observatory Bobcat Fire Landsat image, clearly labeled illustrative rather than a project prediction. Its opening reads “California wildfires have burned millions of acres in recent years.” The CAL FIRE statistics link retains the dated, comparable year-to-date basis of the source in its tooltip; no full-year average is asserted. See [asset provenance](../assets/README.md) for the image source and reuse terms.
