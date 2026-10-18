"""One non-polling work-item step; callers serialize calls for the same item."""

from datetime import datetime, timezone

import ee

from wildfire_recovery_forecasting import config, exports, persistence, production


def lookup_task_status(task_id):
    """Inspect the stored EE task ID once, including after Python restarts."""
    return ee.data.getTaskStatus(task_id)[0]


def advance_work_item(collection, columns, *, source, fire_id, batch_id,
                      expected_patch_ids, client=None, task_status_lookup=None, product=None,
                      retry_failed=False, production_version=config.PRODUCTION_VERSION):
    """Submit, wait, or finish one item without polling or automatic retries.

    COMPLETE is trusted only from saved status. Interrupted downstream work is
    replayed from raw CSVs; no EE resubmission. Missing/unknown task identity
    requires review, not a guess. Concurrent callers for one item are unsupported.
    Status persistence/read failures propagate instead of pretending success.
    """
    if source == "landsat" and product == "nbrdrop":
        raise ValueError("nbrdrop is retired; derive nbr_drop_patch in Phase 3.6 SQL")
    key = dict(source=source, fire_id=fire_id, batch_id=batch_id,
               production_version=production_version, client=client, product=product)
    saved = persistence.read_status(**key)
    if saved and saved["status"] == "COMPLETE":
        return {"action": "SKIPPED_COMPLETE", "status": saved}
    if saved and saved["status"] == "FAILED" and not retry_failed:
        return {"action": "NEEDS_RETRY", "status": saved}

    started = saved["started_at"] if saved else datetime.now(timezone.utc)
    task_id = saved.get("ee_task_id") if saved else None
    current_stage = "EE_EXPORT"

    def write(status, *, error=None, result=None):
        return persistence.write_status(**key, status=status, stage=current_stage,
            started_at=started, ee_task_id=task_id,
            completed_at=datetime.now(timezone.utc) if status in ("FAILED", "COMPLETE") else None,
            error_message=error, row_count=result["row_count"] if result else None,
            output_uri=result["output_uri"] if result else None)

    def fail(error):
        return {"action": "FAILED", "status": write("FAILED", error=str(error))}

    submit = saved is None
    if saved:
        if not task_id:
            return {"action": "NEEDS_REVIEW", "status": saved,
                    "reason": "No saved task ID; submission outcome is ambiguous. Do not resubmit."}
        lookup = task_status_lookup or lookup_task_status
        task = lookup(task_id)  # Lookup failures must not authorize a new export.
        state = task.get("state")
        if state in ("READY", "RUNNING", "CANCEL_REQUESTED"):
            return {"action": "WAITING", "ee_task_id": task_id, "ee_state": state}
        if state in ("FAILED", "CANCELLED"):
            if saved["status"] != "FAILED":
                return fail(task.get("error_message") or f"Earth Engine task {state}")
            # Failed exports may have left shards. Do not mix old and new output.
            storage_client = client if client is not None else persistence.storage.Client(project=config.GCP_PROJECT)
            prefix = production.raw_export_prefix(source, fire_id, batch_id,
                                                  production_version=production_version, product=product)
            if next(iter(storage_client.list_blobs(config.GCS_BUCKET, prefix=prefix)), None) is not None:
                return {"action": "NEEDS_REVIEW", "status": saved,
                        "reason": "Failed export left raw-prefix objects; review before resubmission."}
            submit = True
        elif state != "COMPLETED":
            return {"action": "NEEDS_REVIEW", "status": saved,
                    "reason": f"Unresolved EE task state: {state}"}

    if submit:
        started = datetime.now(timezone.utc)
        task_id = None
        write("RUNNING")  # Durable intent before the potentially ambiguous submission.
        try:
            task = exports.start_table_export(collection, columns, source=source,
                fire_id=fire_id, batch_id=batch_id, production_version=production_version, product=product)
            task_id = task.id
            if not task_id:
                raise ValueError("Export returned no task ID; inspect EE before retrying")
        except Exception as error:
            return fail(error)
        return {"action": "SUBMITTED", "status": write("RUNNING")}

    persistence_error = False

    def progress(stage):
        nonlocal current_stage, persistence_error
        current_stage = stage
        try:
            write("RUNNING")
        except Exception:
            persistence_error = True
            raise

    try:
        result = persistence.csv_to_parquet(source, fire_id, batch_id,
            expected_patch_ids=expected_patch_ids, expected_columns=columns,
            client=client, production_version=production_version, on_stage=progress, product=product)
    except Exception as error:
        if persistence_error:
            raise
        return fail(error)
    current_stage = "COMPLETE"
    return {"action": "COMPLETE", "status": write("COMPLETE", result=result)}
