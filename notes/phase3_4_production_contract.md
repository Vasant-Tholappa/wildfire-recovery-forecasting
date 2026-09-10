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

The unit of work is `(production_version, fire_id, source, batch_id)`.
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
