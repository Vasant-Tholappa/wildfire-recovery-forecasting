# Terrain-only BigQuery and local refresh (prepared, not executed)

Do not start until recovery has **175 COMPLETE**, no running/failed/pending
items, and `manual_terrain_population.py verify` has passed for all 175 products.
Do not change or interrupt the running controller. The following steps are for
the operator after recovery completes and budget review authorizes execution.

`manual_prepare_terrain_refresh.py` is a separate local-only preparation utility.
It uses the frozen manifest and existing URI helpers to reference **173 population
recovery Parquets + two pilot Parquets**, exactly once each. It reuses
`manual_prepare_bigquery.area_fields()` for the existing valid-area means and
forest-support validity denominators. No original terrain file is overwritten.

Only these seven fields change: `elevation_mean`, `slope_mean`, `northness_mean`,
`eastness_mean`, `elevation_valid_fraction`, `slope_valid_fraction`,
`aspect_valid_fraction`. Every other column is selected from the existing eligible
table. No eligibility rerun, threshold changes, resampling, deduplication or split
reassignment occurs. Original candidate and eligible tables stay untouched.

## Outputs and gates

In the existing `ee-wildfire-recovery-vasant.phase3_ca175_tile1_v1` dataset:

1. `terrain_recovery_raw_v2`: one native materialization of the 175 external files.
2. `patch_model_eligible_terrain_staging_v2`: UNVALIDATED replacement of terrain
   columns on the existing eligible key set.
3. `patch_model_eligible_terrain_v2`: publish only after every QA count is zero
   and the terrain-only missingness/recovery summary is reviewed.

Raw QA checks 308,048 rows, unique/nonmissing keys, exact candidate key coverage,
finite sums, valid area/support denominators. Native QA checks 111,257 rows,
151 fires, exact eligible keys, split counts/chronology, finite terrain values,
bounded validity, mask consistency and broad diagnostic terrain ranges
(elevation -500..5000 m, slope 0..90 degrees, components -1..1). These ranges are
review guards, not scientific eligibility. Any failure stops publication.
No previously valid measurement may become NULL; valid fractions may not
decrease beyond 1e-6 numerical tolerance. Investigate failures rather than
clamping, deduplicating or loosening thresholds to pass.

A complete nonterrain row comparison checks labels, predictors, QA, metadata,
eligibility and splits unchanged. It emits an issue count only, never target
values/statistics. This is an integrity check, not test-outcome EDA. Terrain
summary reports old/new missingness, recovered counts and corrected ranges.
Remaining genuine terrain NULLs are allowed; missing is not zero.

The local corrected file will be `data/processed/patch_model_eligible_terrain_v2.parquet`,
with `.metadata.json` and `.roles.json` sidecars. The original
`patch_model_eligible_v1.parquet` remains the rollback file. The finalizer refuses
an existing corrected destination, verifies exact keys/schema and unchanged
nonterrain values/types, rechecks terrain quality/support, and records hashes.
Roles are copied unchanged from the approved original roles file. Point later
terrain EDA/Phase 5 to the corrected file explicitly; this workflow does not
edit the notebook or model code. Split totals remain 61,090/19,561/30,606.

## Operator commands — PowerShell from repository root

Preparation below is local-only and may be run before completion:

```powershell
uv run --offline python manual_prepare_terrain_refresh.py prepare
```

After recovery finishes, confirm all records and Parquets using the existing
`status` and `verify` commands, then review current Cloud Billing and EE usage.
The $50 total budget is not enforced by alerts or these commands. BigQuery
`maximum_bytes_billed=1073741824` is a one-GiB **per-query** scan guard, not a
total-spend/storage/slot-time cap. Do not automatically raise it. External dry
runs can underestimate scans. No live SQL dry run was performed during preparation.

```powershell
$dir = Join-Path $PWD 'data/processed/bigquery_terrain_refresh_v2'
$dataset = 'ee-wildfire-recovery-vasant:phase3_ca175_tile1_v1'
$common = @('--project_id=ee-wildfire-recovery-vasant','--location=us-central1','--format=json','query','--use_legacy_sql=false','--maximum_bytes_billed=1073741824','--max_rows=1000')
$definition = '--external_table_definition=corrected_terrain::' + (Join-Path $dir 'corrected_terrain.json')
function Assert-ZeroIssues($lines) {
  $rows = ($lines -join "`n") | ConvertFrom-Json
  if (-not $rows -or @($rows | Where-Object { $null -eq $_.issues -or [long]$_.issues -ne 0 }).Count) {
    $rows | Format-Table
    throw 'QA failure: STOP; do not publish'
  }
}

# 1. Dry run, then one external scan into a NEW native table.
Get-Content -Raw "$dir/load_raw.sql" | bq @common $definition --dry_run
if ($LASTEXITCODE -ne 0) { throw 'Dry run failed' }
# Review estimated bytes before executing the next command.
Get-Content -Raw "$dir/load_raw.sql" | bq @common $definition --destination_table="${dataset}.terrain_recovery_raw_v2"
if ($LASTEXITCODE -ne 0) { throw 'Raw staging failed' }

# 2. Raw keys/coverage/denominators must pass before joining.
Get-Content -Raw "$dir/raw_qa.sql" | bq @common --dry_run
if ($LASTEXITCODE -ne 0) { throw 'Raw QA dry run failed' }
$qa = Get-Content -Raw "$dir/raw_qa.sql" | bq @common
if ($LASTEXITCODE -ne 0) { throw 'Raw QA failed' }
Assert-ZeroIssues $qa

# 3. Exact eligible-key join; original table remains untouched.
Get-Content -Raw "$dir/assemble.sql" | bq @common --dry_run
if ($LASTEXITCODE -ne 0) { throw 'Assembly dry run failed' }
Get-Content -Raw "$dir/assemble.sql" | bq @common --destination_table="${dataset}.patch_model_eligible_terrain_staging_v2"
if ($LASTEXITCODE -ne 0) { throw 'Eligible staging failed' }

# 4. Native QA plus terrain-only diagnostics.
Get-Content -Raw "$dir/qa.sql" | bq @common --dry_run
if ($LASTEXITCODE -ne 0) { throw 'QA dry run failed' }
$qa = Get-Content -Raw "$dir/qa.sql" | bq @common
if ($LASTEXITCODE -ne 0) { throw 'QA query failed' }
Assert-ZeroIssues $qa
Get-Content -Raw "$dir/summary.sql" | bq @common
if ($LASTEXITCODE -ne 0) { throw 'Terrain summary failed' }

# STOP and review the zero-issue QA and terrain summary before publication.
Get-Content -Raw "$dir/publish.sql" | bq @common --destination_table="${dataset}.patch_model_eligible_terrain_v2"
if ($LASTEXITCODE -ne 0) { throw 'Publication failed' }
```

No `--replace` or `--append` is used. If a destination exists, STOP and inspect;
do not overwrite it to rerun the recipe. Native staging is not considered valid
until QA passes. All QA outputs are single result sets, so no earlier statement's
failures are hidden by the final result. No 12-source assembly is repeated.

After successful publication, extract/download to a new isolated prefix and folder:

```powershell
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$prefix = "gs://ee-wildfire-recovery-vasant-data/production/phase3_ca175_tile1_terrain_v2/model_refresh/$stamp"
$download = Join-Path $PWD "data/processed/terrain_refresh_download_$stamp"
New-Item -ItemType Directory -Path $download -ErrorAction Stop | Out-Null
bq --project_id=ee-wildfire-recovery-vasant --location=us-central1 extract --destination_format=PARQUET "${dataset}.patch_model_eligible_terrain_v2" "$prefix/part-*.parquet"
if ($LASTEXITCODE -ne 0) { throw 'Extract failed' }
gcloud storage cp "$prefix/part-*.parquet" "$download/"
if ($LASTEXITCODE -ne 0) { throw 'Download failed' }
uv run --offline python manual_prepare_terrain_refresh.py finalize --download-dir "$download"
if ($LASTEXITCODE -ne 0) { throw 'Local integrity validation failed; keep original dataset' }
```

The download may contain multiple shards; the finalizer concatenates them without
deduplication and requires the exact original row/key set. It validates labels
by equality only, with no printed or aggregated test outcomes. Local metadata
and generated SQL/provenance are under `data/processed/bigquery_terrain_refresh_v2`.
Keep the successful cloud QA results with the refresh record. No extraction,
upload, query execution or table creation has been performed by preparation.
