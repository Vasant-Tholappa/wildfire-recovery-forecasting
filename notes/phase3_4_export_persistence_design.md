# Phase 3.4 export and persistence design

The production data flow is:

```text
local Python
→ submit asynchronous Earth Engine computation/export
→ Earth Engine performs geospatial computation
→ raw CSV export is written directly to GCS
→ local Python later reads one completed batch
→ converts it to canonical Parquet
→ writes the Parquet back to GCS