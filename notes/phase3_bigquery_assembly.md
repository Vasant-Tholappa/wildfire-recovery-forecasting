# BigQuery candidate assembly: staged workflow

Version: phase3_ca175_tile1_v1. This supersedes the earlier external-source QA
commands. Do not run a stale copy of qa.sql: regenerate locally first.

## Why this changes execution, not science

The earlier QA reused the full joined model CTE in hundreds of UNION ALL branches.
CTEs are not a materialization boundary: repeated branches can repeat external
Parquet scans, joins, EVT parsing, and distinct/key-set aggregations. This is a
likely bottleneck, not a measured query-plan diagnosis of the cancelled job.

The new sequence is:
1. source_keys.sql reads only fire_id/patch_id from each of 13 external definitions
   once into a temporary key table (2,100 raster files plus one vector supplement).
   It checks missing/blank keys, duplicates, exact key-set agreement across ALL 13
   inputs, unexpected fires, and frozen per-fire counts. No deduplication occurs.
2. assemble.sql computes the existing features once and writes an UNVALIDATED
   native staging table. All joins retain both IDs. Three internal diagnostic
   columns retain invalid raw denominator and EVT parsing/member counts.
3. qa.sql aggregates all candidate numerical/cardinality/chronology/vector checks
   in one native-table scan. nulls.sql similarly aggregates missingness once.
4. Only after zero QA issues, publish.sql selects the staging columns without the
   three internal diagnostics into the candidate table. No scientific field is
   removed. primary.sql/early.sql use native staging, not external files, and must
   not be used until QA passes. roles.json still defines leakage-safe feature lists.

Expected rows: 308,048. Exact cross-source keys, per-fire counts, patch identity and
selected tile are checked, with no filtering or silent duplicate removal. The local
supplement is structurally validated and hash-recorded during preparation. Existing
raw artifacts must remain immutable between key checks and staging. Both separate
MTBS history coverage statuses remain unverified. Eligibility is still pending
Phase 3.5; candidate-table validation is not training-eligibility approval.

## Operator steps (not executed by the code changes)

First confirm the cancelled job is no longer running and review current project
spend. The TOTAL project budget remains $50. A one-GiB maximum_bytes_billed below
is a per-query on-demand scan guard, NOT a slot-time, storage, or total-dollar cap.
Do not automatically increase it. External-data dry-run estimates can be incomplete;
script dry runs may stop validation at temporary-table DDL. Runtime is not guaranteed.

Regenerate and dry-run assembly (read-only cloud request):

```powershell
uv run --offline python manual_prepare_bigquery.py prepare
uv run --offline python manual_prepare_bigquery.py dry-run --query assemble
uv run --offline python manual_prepare_bigquery.py dry-run --query source_keys
```

The previously uploaded supplement must be at:
`gs://ee-wildfire-recovery-vasant-data/production/phase3_ca175_tile1_v1/supplemental/vector_qa_v1.parquet`.
No new upload is required if that unchanged object already exists.

Run key checks, and STOP unless every issues count is zero:

```powershell
$dir = Join-Path $PWD 'data/processed/bigquery_ca175_tile1_v1'
$defs = @(Get-ChildItem "$dir/raw_*.json" | ForEach-Object { '--external_table_definition=' + $_.BaseName + '::' + $_.FullName })
$common = @('--project_id=ee-wildfire-recovery-vasant','--location=us-central1','--format=json','query','--use_legacy_sql=false','--maximum_bytes_billed=1073741824','--max_rows=1000')
function Assert-ZeroIssues($lines) {
  $rows = ($lines -join "`n") | ConvertFrom-Json
  if (-not $rows -or @($rows | Where-Object { $null -eq $_.issues -or [long]$_.issues -ne 0 }).Count) {
    $rows | Where-Object { [long]$_.issues -ne 0 } | Format-Table
    throw 'QA issues or missing results: STOP'
  }
}
$keyResult = Get-Content -Raw "$dir/source_keys.sql" | bq @common @defs
if ($LASTEXITCODE -ne 0) { throw 'Key-check query failed: STOP' }
Assert-ZeroIssues $keyResult
```

Use the existing dataset. If it does not exist, create it separately only after
approval; none of these commands creates a dataset. Write staging without replacing
an existing table (stop and inspect if it already exists):

```powershell
Get-Content -Raw "$dir/assemble.sql" | bq @common @defs --destination_table=ee-wildfire-recovery-vasant:phase3_ca175_tile1_v1.patch_candidates_staging_v1
if ($LASTEXITCODE -ne 0) { throw 'Staging failed: STOP' }
uv run --offline python manual_prepare_bigquery.py dry-run --query qa
if ($LASTEXITCODE -ne 0) { throw 'Native QA dry run failed: STOP' }
$qaResult = Get-Content -Raw "$dir/qa.sql" | bq @common
if ($LASTEXITCODE -ne 0) { throw 'Native QA failed: STOP' }
Assert-ZeroIssues $qaResult
Get-Content -Raw "$dir/nulls.sql" | bq @common
if ($LASTEXITCODE -ne 0) { throw 'Missingness query failed: STOP' }
```

Review null counts; do not invent exclusion thresholds or fill missing values.
Then publish only the successfully validated candidate snapshot:

```powershell
Get-Content -Raw "$dir/publish.sql" | bq @common --destination_table=ee-wildfire-recovery-vasant:phase3_ca175_tile1_v1.patch_candidates_v1
if ($LASTEXITCODE -ne 0) { throw 'Publication failed: STOP' }
```

No replace/append flags. If either destination exists, stop and inspect rather
than overwrite. If key or stage QA fails, staging remains UNVALIDATED: do not
publish or train. If even the bounded key pass is slow, inspect its job plan before
launching more queries; do not repeatedly rerun external assembly.

## Offline verification

Seven focused assembly tests pass, including actual local supplemental validation,
arithmetic equivalence, feature roles, all-source key checks and native-only QA.
Generated SQL was inspected locally. Updated BigQuery dry runs were NOT executed;
no cloud operations were authorized for this implementation. Performance gains
remain to be measured: external joins now occur in one assembly query, while QA
and later inspection use a single 308,048-row native table.
