"""Convert one completed batch's raw GCS CSV output to its Parquet object."""

from io import BytesIO
import re

from google.cloud import storage
import pandas as pd

from wildfire_recovery_forecasting import config, production
from wildfire_recovery_forecasting.validation import validate_patch_table


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
