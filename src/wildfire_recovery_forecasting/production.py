"""Production status vocabulary and GCS object names; no cloud operations."""

from wildfire_recovery_forecasting.config import PRODUCTION_VERSION

WORK_ITEM_STATUSES = frozenset({"RUNNING", "FAILED", "COMPLETE"})
WORK_ITEM_STAGES = frozenset({
    "EE_EXPORT", "RAW_OUTPUT_CHECK", "CSV_TO_PARQUET",
    "STRUCTURAL_VALIDATION", "PARQUET_UPLOAD", "COMPLETE",
})


def _component(value: str) -> str:
    """Require a single nonempty path component, preserving supplied IDs."""
    if (not isinstance(value, str) or not value or value.strip() != value
            or value in {".", ".."} or "/" in value or "\\" in value):
        raise ValueError("Path identifiers must be nonempty strings without path separators")
    return value


def _work_path(kind, source, fire_id, batch_id, production_version):
    version, source, fire_id, batch_id = map(
        _component, (production_version, source, fire_id, batch_id))
    return f"production/{version}/{kind}/{source}/fire_id={fire_id}/batch_{batch_id}"


def raw_export_prefix(source, fire_id, batch_id, *, production_version=PRODUCTION_VERSION):
    """Return an extension-free object prefix for a work item's raw export."""
    return _work_path("raw_exports", source, fire_id, batch_id, production_version)


def parquet_path(source, fire_id, batch_id, *, production_version=PRODUCTION_VERSION):
    """Return the work item's Parquet object name."""
    return _work_path("parquet", source, fire_id, batch_id, production_version) + ".parquet"


def status_path(source, fire_id, batch_id, *, production_version=PRODUCTION_VERSION):
    """Return the work item's JSON status object name."""
    return _work_path("status", source, fire_id, batch_id, production_version) + ".json"


def run_metadata_path(*, production_version=PRODUCTION_VERSION):
    """Return the production version's run-metadata JSON object name."""
    return f"production/{_component(production_version)}/metadata/run.json"
