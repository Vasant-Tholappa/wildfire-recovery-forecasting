# Phase 3.4 export and persistence design

The production data flow is:

    local Python
    → submit asynchronous Earth Engine computation/export
    → Earth Engine performs geospatial computation
    → raw CSV export is written directly to GCS
    → local Python later reads one completed batch
    → converts it to canonical Parquet
    → writes the Parquet back to GCS

The raw export unit is one `source × fire × spatial batch`.

CSV is the raw Earth Engine export format because Earth Engine does not natively export these tables as Parquet. The raw CSV remains persisted in GCS so completed Earth Engine work is not lost if local Python is offline.

Parquet is the canonical analytical format used downstream. CSV-to-Parquet conversion is lightweight local processing performed one completed batch at a time; the expensive raster computation remains in Earth Engine.

BigQuery is not part of the format-conversion step. It will be introduced later for cloud-side SQL querying, validation, filtering, and final dataset assembly.

Exact export/sharding and retry mechanics are finalized only after the live production path is tested.

## 3.4.5 live-path validation

A live smoke test using a small MONUMENT terrain batch successfully proved the end-to-end path:

    Earth Engine computation
    → asynchronous CSV export to GCS
    → local CSV-to-Parquet conversion
    → Parquet upload to GCS
    → read-back validation

The test completed on the first attempt and produced one CSV file. The Parquet read-back matched the expected patch IDs, fire ID, row count, required columns, and CSV values with no duplicate patch IDs.

Smoke-test outputs were written under `phase3_smoke_v1`; `phase3_prod_v1` remains reserved for the final production configuration.

The following remain intentionally unproven and must be addressed in Phase 3.4.6:

1. Real Earth Engine blank/null CSV serialization and its Parquet round-trip.
2. Multi-shard export behavior and shard recombination.
3. Final production `tile_side_cells` / batch size, selected through live benchmarking.

Phase 3.4.6 will also add production status/provenance metadata and resume/skip/failure handling before the population launch.