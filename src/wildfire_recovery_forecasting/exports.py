"""Asynchronous raw CSV submission for one already-built EE patch table."""

import re

import ee

from wildfire_recovery_forecasting import config, production
from wildfire_recovery_forecasting.reductions import without_geometry


def start_table_export(collection, columns, *, source, fire_id, batch_id, product=None,
                       production_version=config.PRODUCTION_VERSION):
    """Start one GCS CSV task and return it immediately, without polling.

    EE must already be initialized with the configured GCP project. columns
    lists raw properties in addition to patch_id/fire_id (always included).
    The caller supplies the pre-materialization FeatureCollection; this does
    not run local summaries, download rows, or assemble multiple source graphs.
    Submission errors propagate to the caller. No retry or persistence occurs.
    """
    prefix = production.raw_export_prefix(
        source, fire_id, batch_id, product=product, production_version=production_version)
    description = f"{production_version}_{source}_{fire_id}_batch_{batch_id}"
    if product is not None:
        description += f"_product_{product}"
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", description):
        raise ValueError("Task description must be at most 100 letters, digits, underscores or hyphens")
    task = ee.batch.Export.table.toCloudStorage(
        collection=without_geometry(collection), description=description,
        bucket=config.GCS_BUCKET, fileNamePrefix=prefix, fileFormat="CSV",
        selectors=list(dict.fromkeys(["patch_id", "fire_id", *columns])),
    )
    task.start()
    return task
