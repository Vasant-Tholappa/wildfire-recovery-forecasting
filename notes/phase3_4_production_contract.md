# Phase 3.4.3 production execution contract

This records the locked storage contract. Scientific methodology remains defined
by [the Phase 3 production contract](../docs/phase3_production_contract.md).

| Setting | Value |
| --- | --- |
| GCP project | `ee-wildfire-recovery-vasant` |
| GCS bucket | `ee-wildfire-recovery-vasant-data` |
| Region | `us-central1` |
| production_version | `phase3_prod_v1` |
| contract_version | `v2` |
| grid_version | `v1` |
| schema_version | `v1` |

The unit of work is `(production_version, fire_id, source, product, batch_id)`.
`product` is optional for single-product sources; their existing paths and status
records remain unchanged when it is omitted.
Batch IDs are deterministic strings supplied by later batching logic. Path helpers
preserve them exactly, including leading zeros; no batch size or ID-generation
algorithm is chosen here.

## Object layout

All names below are relative to the locked bucket:

```text
production/{production_version}/
  raw_exports/{source}/fire_id={fire_id}/batch_{batch_id}
  parquet/{source}/fire_id={fire_id}/batch_{batch_id}.parquet
  status/{source}/fire_id={fire_id}/batch_{batch_id}.json
  metadata/run.json
```

The raw-export path is an extension-free prefix; actual format and any export
suffixes/shards belong to the later export implementation. Helpers return object
names using forward slashes, not local filesystem paths or `gs://` URIs. A full
URI is `gs://{bucket}/{object_name}`. Metadata describes one production version;
this step does not implement writing or overwriting it.

## Work-item status record

Use a plain JSON object with these fields:

| Field | Representation |
| --- | --- |
| production_version, fire_id, source, batch_id | Strings identifying the work item |
| status | `RUNNING`, `FAILED`, or `COMPLETE` |
| stage | Required: `EE_EXPORT`, `RAW_OUTPUT_CHECK`, `CSV_TO_PARQUET`, `STRUCTURAL_VALIDATION`, `PARQUET_UPLOAD`, or `COMPLETE` |
| ee_task_id | String or null when not applicable/known |
| row_count | Nonnegative integer or null when unknown |
| output_uri | GCS URI string or null when unknown |
| started_at | UTC ISO 8601 timestamp |
| completed_at | UTC ISO 8601 timestamp or null before completion |
| error_message | String or null when not applicable |

`MISSING` is inferred from absence of a status record, not a written status.
`COMPLETE` status requires stage `COMPLETE`. For `RUNNING` or `FAILED`, the caller
supplies the current or failing operation's stage; transition logic is deferred.
Failure to read a record is not evidence that it is absent.

## Run metadata

Use a plain JSON object containing `production_version`, `contract_version`,
`grid_version`, `schema_version`, `git_commit`, `gcp_project`, `gcs_bucket`,
`gcs_region`, `batching_configuration`, and `created_at`.
Version, commit and location fields are strings; `created_at` is a UTC ISO 8601
timestamp. `batching_configuration` is a JSON object (empty until settings are specified).
`contract_version` uses the existing `PRODUCTION_CONTRACT_VERSION` constant.

This step provides constants and pure path construction only. It does not perform
batching, Earth Engine exports, Parquet conversion, cloud writes, or orchestration.


## Spatial batching

Production batches are fixed 2D square tiles over the canonical global grid using
`grid_row` and `grid_col`; they are not sequential row-major chunks.

Each retained patch belongs to exactly one tile. Irregular fire boundaries may
therefore produce sparse batches with fewer patches than the tile maximum; patches
from other tiles are never pulled in to fill them.

Batch IDs are derived deterministically from tile coordinates, so the same grid
and tile configuration always produce the same assignments.

`tile_side_cells` remains configurable until production benchmarking. Once chosen,
it is recorded in the run batching configuration and must remain fixed within that
production version. A changed official batching configuration requires a new
production configuration/version.

## Resume/orchestration mental model

Each call advances one source/fire/batch; it never waits for EE to finish.

- No saved status → save submission intent → submit EE → `RUNNING / EE_EXPORT`
  with the task ID.
- `RUNNING / EE_EXPORT` → EE `READY`/`RUNNING`: wait; EE `COMPLETED`: continue
  downstream; EE `FAILED`/`CANCELLED`: `FAILED / EE_EXPORT`.
- Downstream → check raw GCS output → CSV/Parquet processing and structural
  validation → Parquet upload → `COMPLETE / COMPLETE`. Stage callbacks mark the
  actual operation, including repeated validation/encoding steps. Failures retain
  the failing stage and error; interrupted downstream work can replay from raw CSV.
- Previously `COMPLETE` → skip entirely after reading status.
- Previously `FAILED` → do nothing by default; retry only on explicit request.
  Check the stored EE task first: reuse active work or a completed export rather
  than resubmitting it.

Earth Engine supplies task state. GCS supplies persisted files/objects. Python
combines those facts and writes our status JSON; file existence alone never means
COMPLETE. Restarting Python must not duplicate active or completed work. Missing
task IDs/unknown states require review; failed exports with leftover raw objects
also require review before resubmission. Calls for the same work item must be
serialized; this is not a concurrent-worker locking system.

## Landsat production products

Landsat persists four independent products per spatial batch under the single
logical source `landsat`:

| Product | Existing reduction fields |
| --- | --- |
| `nbrpre` | NBRpre and prefire selected-year count |
| `season1` | NBR1 and Season-1 observation count |
| `nbr5` | NBR5 |
| `season5_count` | Season-5 observation count |

For a supplied product, paths insert `product={product}/` after the source:
`production/{version}/{raw_exports|parquet|status}/landsat/product={product}/fire_id={fire_id}/batch_{batch_id}`
with the existing file suffixes. Status JSON includes `product`; readers verify
it matches the requested identity. Each product has its own task ID, status,
raw CSVs and intermediate Parquet, so completion/retry applies to that product
only. Omitted products keep the old path and omit the JSON field.

This preserves the memory-safe independent graphs validated earlier. The shared
`LANDSAT_PRODUCTS` definitions and `build_area_sums()` power both synchronous
reductions and `build_landsat_products()`. Production outputs remain raw sums,
valid areas and the same-grid full-area denominator. Later final assembly will reuse
the validated local summarization and one-to-one joins; the graphs are never
recombined into one Earth Engine export. One completed product does not establish
completion of the whole Landsat source.

**Phase 3.7 consolidation note:** Include this interview/debugging story: the
initial one-source/one-task assumption conflicted with Landsat's five products;
we preserved the memory-safe independent graphs and extended production identity
instead of recombining them.

## Disturbance-QA scaling amendment for v1

The current scientific contract retains only ten-year prior LCMS/MTBS QA.
LCMS keeps pre-fire Y-10..Y-1 and post-fire Y..Y+5 union fractions and coverage, with no
earliest-year outputs. MTBS keeps exact-date prior-fire overlap and unchanged
reburn logic. Final exclusion thresholds still belong to Phase 3.5.

The previous 5/7/10 sensitivity design was unnecessary overhead for auxiliary
contamination screens in v1. Extent within these predefined windows is sufficient;
LCMS is retrospective QA, not a primary predictor. This deliberate scaling/scope
decision preserves the core prediction problem while reducing repeated work.
See the rationale in the scientific contract. Earlier notebooks and pilot results
are historical, not the current output schema.

LCMS now has two independent raster reductions instead of eight (one extent/
coverage reduction per window, no timing reductions). At this amendment, total full-source raster
boundaries per unsplit batch fell from 22 to 16; Landsat's five products are
unchanged. MTBS history remains vector retrieval/local intersection work and
adds no raster export. These boundary counts are not a finalized export task plan.
Previously persisted LCMS/MTBS-history tables use an obsolete schema and must
not be silently reused as current results; final dataset validation rejects the
removed output columns. No existing cloud objects or production versions were
changed by this amendment.

## Plan B production products

Plan B is the implemented execution target: **12 independent raster-export
products per unsplit spatial batch**. This is an execution optimization, not a
scientific feature change or a choice of Phase 3.5 eligibility thresholds.

| Source | Product identifiers | Count |
| --- | --- | --- |
| landsat | nbrpre, season1, nbr5, season5_count | 4 |
| landfire | evc, height, evt_support, evt_categories | 4 |
| terrain | None (single product; existing path) | 1 |
| daymet | combined | 1 |
| lcms | None (single product; existing path) | 1 |
| mtbs_severity | None (single product; existing path) | 1 |

Forest and tree cover share the EVC source grid and sum reducer, so `evc`
contains forest, tree_cover and support sums/valid areas in one reduction.
Known nonforest pixels remain in EVC coverage. Each measurement mask and its
valid-area/forest-support denominator remain independent. Height retains its
EVH grid. EVT support and categorical groups retain separate reductions because
their reducer semantics differ; EVT class IDs are never averaged.

LCMS pre_10yr and post_y_to_y5 share the same LCMS reference grid and sum reducer.
They now use one reduction with window-prefixed variables, separate masks and
valid areas, and one unmasked same-grid full-area reference. Expected years
remain local assembly metadata (10 and 6), exposed in the product descriptor.
No disturbance timing fields return. Validated Daymet primary and coverage now
share one `combined` product, retaining all climate and temporal QA fields. Terrain and severity stay consolidated as
before. Landsat retains four independent graphs after Candidate A; only the retired drop graph is removed.

`pipeline.build_raster_products(fire, patches, reference_projection=...)` returns
source -> product -> descriptor for an existing canonical batch, without fetching,
submitting or rebuilding its grid. Descriptors contain `collection` and `columns`;
LCMS also includes `expected_years`. Submit each collection independently using
the existing controller's source/product identity, expected patch IDs and batch ID.
Unnamed single-product sources use product=None, preserving their existing path layout;
Daymet explicitly uses product="combined".
MTBS prior/reburn history remains separate vector/local work and is not enumerated.

Synchronous reducers use the same raw builders and retain the existing analytical
columns and local summarization. Persisted tables remain raw/intermediate sums,
areas or EVT groups; export-ready does not mean already submitted or assembled.
The new bundled raw schemas must not be confused with earlier intermediate
outputs. Live numerical equivalence, memory/runtime and categorical CSV round-trip
validation remain required before production; no live benchmark was part of this
implementation.


## Phase 3.4.7A Candidate A: current inventory supersedes historical counts

The dedicated `nbrdrop` product is retired. Candidate A reduced the inventory to 13;
subsequent approved combined-Daymet promotion makes 12 per unsplit tile-120 batch.
Other products and identities are unchanged.
The controller/export API reject retired nbrdrop requests before cloud work.
Historical status/raw/Parquet objects remain readable and are not deleted or
required for completion of the current inventory. Earlier five-product Landsat
and 14-product benchmark descriptions above are historical execution evidence.

Phase 3.6 derives `nbr_drop_patch` as `nbr_pre_mean - nbr_1_mean`, with SQL NULL
propagation and independently valid supports. There is no drop valid-fraction
field. See the scientific amendment in docs/phase3_production_contract.md.
The old manual Plan B harness is retained for historical check/collection only;
its original inventory is not a current production submission contract.

## Combined Daymet production promotion ? 2026-10-18

Use source=`daymet`, product=`combined` with the existing deterministic path/status
helpers. Earlier primary/coverage status records cannot mark this new identity
complete. The raw schema contains 32 sum/area pairs (16 climate measurements,
8 historical daily-coverage summaries and 8 current valid-day summaries), plus
one `raster_full_area_m2`: 65 numeric columns, plus patch_id and fire_id.
The validated EPSG:5070 30 m geographic association, independent masks and
valid-area denominators are unchanged. Synchronous processing fetches once
and retains its original analytical schema. Candidate C is rejected. Historical
benchmark artifacts remain historical; no cloud data is migrated or overwritten.


## Final California population scope

The approved modeling population is the frozen 175-fire CA-coded sample from
the original eligible manifest, seed `ca_portfolio_175_v1`: 91 train, 43 validation,
41 test fires; 729 tile-120 batches and 8,748 exports. Full original geometries
and all source/target semantics remain unchanged. This supersedes the former
643-fire production selection without deleting its outputs. Production launcher
namespace is `phase3_ca175_v1`; historical `phase3_prod_v1` is separate.
See `notes/phase3_ca175_population.md` for exact selection and preservation rules.
