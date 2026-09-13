"""Manual Plan B benchmark: submit once, then check/collect in a later invocation."""

import argparse
from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path
import re
import subprocess

import ee
import geopandas as gpd
from google.cloud import storage
import pandas as pd

from wildfire_recovery_forecasting import config, persistence, pipeline, production, work_items
from wildfire_recovery_forecasting.grid import assign_spatial_batches, build_fire_grid
from wildfire_recovery_forecasting.validation import validate_patches, validate_patch_table

FIRE_ID = "AZ3340811121020190608"
VERSION = "phase3_planb_bench_v1"
RESULTS = Path("data/processed/planb_benchmark_results.json")
INVENTORY = {
    "landsat": {"nbrpre", "season1", "nbr5", "season5_count", "nbrdrop"},
    "landfire": {"evc", "height", "evt_support", "evt_categories"},
    "terrain": {None}, "daymet": {"primary", "coverage"},
    "lcms": {None}, "mtbs_severity": {None},
}


def record_git_commit(results):
    """Record this invocation's HEAD while preserving the initial benchmark commit."""
    try:
        commit = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "--verify", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"Cannot determine benchmark Git commit: {getattr(error, 'stderr', '') or error}") from error
    if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit):
        raise RuntimeError("Cannot determine benchmark Git commit: invalid git rev-parse output")
    results.setdefault("git_commit", commit)
    results["last_invocation_git_commit"] = commit


def groups_diagnostics(table, expected_ids):
    """Describe the received values without parsing or transforming EVT groups."""
    present = "groups" in table.columns
    covered = "patch_id" in table and set(expected_ids).issubset(set(table.patch_id))
    result = {"column_present": present, "all_expected_rows_have_column": bool(present and covered)}
    if present:
        observed = table["groups"].dropna()
        value = observed.iloc[0] if len(observed) else None
        result.update(pandas_dtype=str(table["groups"].dtype),
                      non_null_row_count=len(observed), null_row_count=int(table["groups"].isna().sum()),
                      python_value_type=None if value is None else f"{type(value).__module__}.{type(value).__qualname__}",
                      representative_value=value if isinstance(value, str) else repr(value) if value is not None else None,
                      representative_encoding="string" if isinstance(value, str) else "repr" if value is not None else None)
    return result


def benchmark_version(tile):
    if type(tile) is not int or tile not in (60, 90):
        raise ValueError("Only tile sizes 60 and 90 are supported")
    return f"{VERSION}_t{tile}"


def select_batch(grid, tile):
    """Densest tile; ties nearest median grid location, then numeric tile row/col."""
    benchmark_version(tile)
    patches = assign_spatial_batches(grid, tile_side_cells=tile)
    if patches.empty:
        raise ValueError("Empty fire grid")
    patches["tile_row"] = patches.grid_row // tile
    patches["tile_col"] = patches.grid_col // tile
    counts = patches.groupby(["tile_row", "tile_col"]).size().rename("count").reset_index()
    counts["distance"] = ((counts.tile_row * tile + tile / 2 - patches.grid_row.median()) ** 2
                          + (counts.tile_col * tile + tile / 2 - patches.grid_col.median()) ** 2)
    chosen = counts.sort_values(["count", "distance", "tile_row", "tile_col"],
                                ascending=[False, True, True, True]).iloc[0]
    return patches.loc[(patches.tile_row == chosen.tile_row) & (patches.tile_col == chosen.tile_col)] \
        .drop(columns=["tile_row", "tile_col"]).sort_values(["grid_row", "grid_col"]).reset_index(drop=True)


def product_entries(products):
    if {source: set(parts) for source, parts in products.items()} != INVENTORY:
        raise ValueError("Expected exactly the 14 Plan B raster products")
    entries = [(source, name, descriptor) for source, parts in products.items()
               for name, descriptor in parts.items()]
    if len(entries) != 14 or len({(source, name) for source, name, _ in entries}) != 14:
        raise ValueError("Duplicate or missing product identity")
    for _, _, descriptor in entries:
        if "collection" not in descriptor or not descriptor.get("columns"):
            raise ValueError("Missing raw collection/columns")
    return entries


def identity(batch, item):
    return dict(source=item["source"], product=item["product"], fire_id=FIRE_ID,
                batch_id=batch["batch_id"], production_version=benchmark_version(batch["tile_side_cells"]))


def paths(batch, item):
    key = identity(batch, item)
    return {"raw_prefix": production.raw_export_prefix(**key),
            "status_path": production.status_path(**key), "parquet_path": production.parquet_path(**key)}


def save(path, results):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def submit(args, results, client):
    # A local canonical manifest is required; never rebuild the EE population here.
    manifest = gpd.read_parquet(args.manifest)
    selected = manifest.loc[manifest.fire_id.eq(FIRE_ID)]
    if len(selected) != 1 or manifest.crs is None:
        raise ValueError("Manifest must contain exactly one WOODBURY record and a CRS")
    fire = selected.iloc[0].to_dict()
    grid = build_fire_grid(fire["geometry"], FIRE_ID, perimeter_crs=manifest.crs)
    for tile in args.tiles:
        batch_patches = select_batch(grid, tile)
        validate_patches(batch_patches, FIRE_ID)
        batch_id = batch_patches.batch_id.iloc[0]
        ids = batch_patches.patch_id.tolist()
        products = product_entries(pipeline.build_raster_products(fire, batch_patches))
        batch = results["batches"].setdefault(str(tile), {
            "tile_side_cells": tile, "batch_id": batch_id, "patch_count": len(ids),
            "expected_patch_ids": ids, "production_version": benchmark_version(tile), "products": []})
        if batch["batch_id"] != batch_id or batch["expected_patch_ids"] != ids:
            raise ValueError("Saved benchmark batch differs from current selection; review before submission")
        print(f"tile={tile} batch={batch_id} patches={len(ids)}", flush=True)
        for source, name, descriptor in products:
            item = next((r for r in batch["products"] if (r["source"], r["product"]) == (source, name)), None)
            if item is None:
                item = {"source": source, "product": name, "expected_columns": descriptor["columns"]}
                item.update(paths(batch, item))
                if "expected_years" in descriptor:
                    item["expected_years"] = descriptor["expected_years"]
                batch["products"].append(item)
            if item["expected_columns"] != descriptor["columns"]:
                raise ValueError("Saved raw schema differs; review benchmark identity before reuse")
            save(args.results, results)  # Record intent before the controller may submit.
            try:
                action = work_items.advance_work_item(descriptor["collection"], descriptor["columns"],
                    **identity(batch, item), expected_patch_ids=ids, client=client)
                item["action"] = action["action"]
                status = action.get("status") or persistence.read_status(**identity(batch, item), client=client)
                if status:
                    item.update(task_id=status.get("ee_task_id"), submission_timestamp=status["started_at"],
                                status=status)
                item.pop("harness_error", None)
            except Exception as error:
                item["harness_error"] = f"{type(error).__name__}: {error}"
            save(args.results, results)
            print(source, name, item.get("task_id"), item.get("harness_error", item.get("action")), flush=True)


def collect(args, results, client):
    for batch in results["batches"].values():
        for item in batch["products"]:
            try:
                key = identity(batch, item)
                saved = persistence.read_status(**key, client=client)
                if not saved or not saved.get("ee_task_id"):
                    raise ValueError("Missing persisted task ID; check mode will never submit")
                task_id = saved["ee_task_id"]
                if item.get("task_id") not in (None, task_id):
                    raise ValueError("Recorded and persisted task IDs differ")
                task = work_items.lookup_task_status(task_id)  # Exactly one lookup this invocation.
                item.update(task_id=task_id, submission_timestamp=saved["started_at"], ee_task=task,
                            checked_at=datetime.now(timezone.utc).isoformat())
                terminal = task.get("state") in ("COMPLETED", "FAILED", "CANCELLED")
                start, end = task.get("start_timestamp_ms"), task.get("update_timestamp_ms")
                item["execution_seconds"] = (end - start) / 1000 if terminal and start and end else None
                created = task.get("creation_timestamp_ms")
                item["elapsed_seconds"] = (end - created) / 1000 if terminal and created and end else None
                item["eecu_seconds"] = task.get("batch_eecu_usage_seconds")
                # Never give check mode a missing-status path through to submission.
                if saved["status"] == "RUNNING" and terminal:
                    action = work_items.advance_work_item(None, item["expected_columns"], **key,
                        expected_patch_ids=batch["expected_patch_ids"], client=client,
                        task_status_lookup=lambda _: task)
                    item["action"] = action["action"]
                    saved = action.get("status") or persistence.read_status(**key, client=client)
                else:
                    item["action"] = {"COMPLETE": "SKIPPED_COMPLETE", "FAILED": "NEEDS_RETRY"}.get(
                        saved["status"], "WAITING")
                item["status"] = saved
                item["row_count"] = saved.get("row_count")
                prefix = paths(batch, item)["raw_prefix"]
                pattern = re.compile(re.escape(prefix) + r"(?:-\d+(?:-\d+)*)?\.csv")
                raw = [blob for blob in client.list_blobs(config.GCS_BUCKET, prefix=prefix)
                       if pattern.fullmatch(blob.name)]
                item["raw_objects"] = [{"name": blob.name, "size_bytes": blob.size} for blob in raw]
                item["raw_shard_count"] = len(raw)
                item["raw_size_bytes"] = sum(blob.size or 0 for blob in raw)
                if saved["status"] == "COMPLETE":
                    blob = client.bucket(config.GCS_BUCKET).get_blob(paths(batch, item)["parquet_path"])
                    if blob is None:
                        raise ValueError("COMPLETE status but Parquet object is missing")
                    table = pd.read_parquet(BytesIO(blob.download_as_bytes()))
                    if (item["source"], item["product"]) == ("landfire", "evt_categories"):
                        item["groups_diagnostics"] = groups_diagnostics(table, batch["expected_patch_ids"])
                    validate_patch_table(table, expected_ids=batch["expected_patch_ids"], fire_id=FIRE_ID)
                    if not set(item["expected_columns"]).issubset(table.columns):
                        raise ValueError("Parquet missing required columns")
                    item["parquet_size_bytes"] = blob.size
                    item["row_count"] = len(table)
                    item["parquet_null_counts"] = {name: int(n) for name, n in table.isna().sum().items() if n}
                    item["parquet_null_value_count"] = sum(item["parquet_null_counts"].values())
                item.pop("harness_error", None)
            except Exception as error:
                item["harness_error"] = f"{type(error).__name__}: {error}"
            save(args.results, results)
            print(batch["tile_side_cells"], item["source"], item["product"],
                  item.get("ee_task", {}).get("state"), item.get("harness_error", item.get("action")), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["submit", "check"])
    parser.add_argument("--manifest", type=Path, help="Existing local canonical fire manifest GeoParquet")
    parser.add_argument("--tiles", type=int, nargs="+", choices=[60, 90], default=[60, 90])
    parser.add_argument("--results", type=Path, default=RESULTS)
    args = parser.parse_args()
    if args.mode == "submit" and (args.manifest is None or not args.manifest.is_file()):
        parser.error("submit requires --manifest pointing to an existing local GeoParquet")
    results = json.loads(args.results.read_text(encoding="utf-8")) if args.results.exists() else {
        "benchmark": VERSION, "fire_id": FIRE_ID, "fire_name": "WOODBURY",
        "bucket": config.GCS_BUCKET, "batches": {}}
    if results["benchmark"] != VERSION or results["fire_id"] != FIRE_ID:
        parser.error("Results file belongs to a different benchmark")
    if args.mode == "check" and not results["batches"]:
        parser.error("No saved submissions to check")
    record_git_commit(results)  # Fail before initializing any cloud clients.
    ee.Initialize(project=config.GCP_PROJECT)
    client = storage.Client(project=config.GCP_PROJECT)
    (submit if args.mode == "submit" else collect)(args, results, client)


if __name__ == "__main__":
    main()
