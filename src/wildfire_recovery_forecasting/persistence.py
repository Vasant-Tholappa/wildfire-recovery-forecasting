"""Batch Parquet and small production JSON records in GCS."""

from collections.abc import Mapping
from datetime import datetime, timezone
from io import BytesIO
import json
import re

from google.api_core.exceptions import NotFound
from google.cloud import storage
import pandas as pd

from wildfire_recovery_forecasting import config, production
from wildfire_recovery_forecasting.validation import validate_patch_table


def _timestamp(value):
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise ValueError("Timestamps must be timezone-aware datetimes or ISO 8601 strings")
    return value.astimezone(timezone.utc).isoformat()


def _json_blob(path, client):
    if client is None:
        client = storage.Client(project=config.GCP_PROJECT)
    return client.bucket(config.GCS_BUCKET).blob(path)


def _write_json(path, record, client):
    payload = json.dumps(record, allow_nan=False, sort_keys=True)
    _json_blob(path, client).upload_from_string(payload, content_type="application/json")
    return json.loads(payload)


def _read_json(path, client):
    try:
        payload = _json_blob(path, client).download_as_bytes()
    except NotFound:
        return None
    record = json.loads(payload)
    if not isinstance(record, dict):
        raise ValueError("Production JSON must contain an object")
    return record


def _validate_stage(status, stage):
    if not isinstance(stage, str) or stage not in production.WORK_ITEM_STAGES:
        raise ValueError("Invalid or missing work-item stage")
    if (status == "COMPLETE") != (stage == "COMPLETE"):
        raise ValueError("COMPLETE status and stage must occur together")


def write_status(source, fire_id, batch_id, *, status, stage, started_at,
                 ee_task_id=None, row_count=None, output_uri=None,
                 completed_at=None, error_message=None, client=None,
                 production_version=config.PRODUCTION_VERSION):
    """Write the caller's explicit status; no task/output inspection or retries."""
    path = production.status_path(source, fire_id, batch_id, production_version=production_version)
    if status not in production.WORK_ITEM_STATUSES:
        raise ValueError("Status must be RUNNING, FAILED, or COMPLETE")
    _validate_stage(status, stage)
    if row_count is not None and (type(row_count) is not int or row_count < 0):
        raise ValueError("row_count must be a nonnegative integer or None")
    for value in (ee_task_id, output_uri, error_message):
        if value is not None and not isinstance(value, str):
            raise ValueError("Task ID, output URI, and error message must be strings or None")
    return _write_json(path, {
        "production_version": production_version, "fire_id": fire_id,
        "source": source, "batch_id": batch_id, "status": status, "stage": stage,
        "ee_task_id": ee_task_id, "row_count": row_count, "output_uri": output_uri,
        "started_at": _timestamp(started_at),
        "completed_at": None if completed_at is None else _timestamp(completed_at),
        "error_message": error_message,
    }, client)


def read_status(source, fire_id, batch_id, *, client=None,
                production_version=config.PRODUCTION_VERSION):
    """Return stored status, or None only for GCS NotFound; other failures propagate."""
    record = _read_json(production.status_path(
        source, fire_id, batch_id, production_version=production_version), client)
    if record is not None:
        if record.get("status") not in production.WORK_ITEM_STATUSES:
            raise ValueError("Invalid stored status")
        _validate_stage(record["status"], record.get("stage"))
        expected = {"production_version": production_version, "source": source,
                    "fire_id": fire_id, "batch_id": batch_id}
        if any(record.get(key) != value for key, value in expected.items()):
            raise ValueError("Stored status identity does not match requested work item")
    return record


def write_run_metadata(*, git_commit, batching_configuration, created_at,
                       client=None, production_version=config.PRODUCTION_VERSION):
    """Write compact run provenance with explicit caller-supplied batching settings."""
    if not isinstance(batching_configuration, Mapping):
        raise ValueError("batching_configuration must be a JSON-serializable mapping")
    if not isinstance(git_commit, str):
        raise ValueError("git_commit must be a string")
    return _write_json(production.run_metadata_path(production_version=production_version), {
        "production_version": production_version,
        "contract_version": config.PRODUCTION_CONTRACT_VERSION,
        "grid_version": config.GRID_VERSION, "schema_version": config.SCHEMA_VERSION,
        "git_commit": git_commit, "gcp_project": config.GCP_PROJECT,
        "gcs_bucket": config.GCS_BUCKET, "gcs_region": config.GCS_REGION,
        "batching_configuration": dict(batching_configuration),
        "created_at": _timestamp(created_at),
    }, client)


def read_run_metadata(*, client=None, production_version=config.PRODUCTION_VERSION):
    """Read run provenance unchanged; None means the object is absent."""
    record = _read_json(production.run_metadata_path(production_version=production_version), client)
    if record is not None and record.get("production_version") != production_version:
        raise ValueError("Stored run metadata version does not match requested version")
    return record


def csv_to_parquet(source, fire_id, batch_id, *, expected_patch_ids,
                   expected_columns, client=None,
                   production_version=config.PRODUCTION_VERSION):
    """Validate completed raw CSV shards and upload one in-memory Parquet file.

    Caller must establish EE task completion first; object presence alone is
    not a task-completion signal. Accept prefix.csv or prefix-N[...-N].csv
    numeric shard names, never neighboring batch prefixes. Required columns
    must exist in every shard; additional raw columns are retained. Output rows
    follow expected_patch_ids. No scientific transformation or null filling.
    This writes the deterministic destination and leaves raw objects intact.
    """
    prefix = production.raw_export_prefix(
        source, fire_id, batch_id, production_version=production_version)
    destination = production.parquet_path(
        source, fire_id, batch_id, production_version=production_version)
    expected = list(expected_patch_ids)
    validate_patch_table(pd.DataFrame({"patch_id": expected}))
    if not expected:
        raise ValueError("Expected batch patch IDs must be nonempty")
    required = {"patch_id", "fire_id", *expected_columns}
    if client is None:
        client = storage.Client(project=config.GCP_PROJECT)
    pattern = re.compile(re.escape(prefix) + r"(?:-\d+(?:-\d+)*)?\.csv")
    blobs = sorted(
        (blob for blob in client.list_blobs(config.GCS_BUCKET, prefix=prefix)
         if pattern.fullmatch(blob.name)), key=lambda blob: blob.name)
    if not blobs:
        raise ValueError(f"No completed CSV output found for gs://{config.GCS_BUCKET}/{prefix}")
    shards = []
    for blob in blobs:
        try:
            table = pd.read_csv(BytesIO(blob.download_as_bytes()),
                                dtype={"patch_id": "string", "fire_id": "string"},
                                keep_default_na=False, na_values=[""])
        except pd.errors.EmptyDataError as error:
            raise ValueError(f"Empty CSV output: {blob.name}") from error
        missing = required - set(table.columns)
        if missing:
            raise ValueError(f"Missing required columns in {blob.name}: {sorted(missing)}")
        if table.fire_id.isna().any():
            raise ValueError(f"Missing fire_id in {blob.name}")
        if not table.empty:
            validate_patch_table(table, fire_id=fire_id)
        shards.append(table)
    combined = pd.concat(shards, ignore_index=True)
    if combined.empty:
        raise ValueError("Raw CSV output contains no rows")
    validate_patch_table(combined, expected_ids=expected, fire_id=fire_id)
    combined = combined.set_index("patch_id").loc[expected].reset_index()
    buffer = BytesIO()
    combined.to_parquet(buffer, engine="pyarrow", index=False)
    client.bucket(config.GCS_BUCKET).blob(destination).upload_from_string(
        buffer.getvalue(), content_type="application/octet-stream")
    return {"row_count": len(combined), "output_uri": f"gs://{config.GCS_BUCKET}/{destination}"}
