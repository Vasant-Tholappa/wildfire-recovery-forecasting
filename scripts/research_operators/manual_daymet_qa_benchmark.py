"""Single WOODBURY tile-60 Candidate C benchmark; each invocation takes one step."""

import argparse
from io import BytesIO
import json
from pathlib import Path
import re

import ee
import geopandas as gpd
from google.cloud import storage
import numpy as np
import pandas as pd
import shapely

import manual_daymet_qa_experiment as experiment
from wildfire_recovery_forecasting import config, daymet, persistence, pipeline, production, work_items
from wildfire_recovery_forecasting.validation import validate_patches

combined = experiment.combined
baseline = combined.baseline
RESULTS = Path("data/processed/daymet_qa4_tile60_benchmark_results.json")
REFERENCE = combined.RESULTS
BATCH_ID = "r235_c-259"


def identity(batch, item):
    if (batch["tile_side_cells"], batch["batch_id"], item["source"], item["product"]) != (
            60, BATCH_ID, "daymet", "qa4"):
        raise ValueError("Only the single approved Candidate C tile-60 identity is allowed")
    return experiment.experiment_identity(60, BATCH_ID)


def paths(batch, item):
    key = identity(batch, item)
    return {"raw_prefix": production.raw_export_prefix(**key),
            "status_path": production.status_path(**key),
            "parquet_path": production.parquet_path(**key)}


def load_reference():
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    if reference["benchmark"] != combined.VERSION or reference["fire_id"] != baseline.FIRE_ID:
        raise ValueError("Wrong combined-Daymet reference")
    batch = reference["batches"]["60"]
    item, = batch["products"]
    if (batch["batch_id"] != BATCH_ID or batch["patch_count"] != 3600
            or len(batch["expected_patch_ids"]) != 3600
            or len(set(batch["expected_patch_ids"])) != 3600
            or (item["source"], item["product"]) != ("daymet", "combined")
            or item["status"]["status"] != "COMPLETE"
            or item["ee_task"]["state"] != "COMPLETED"
            or not batch["comparison"]["equivalent"]
            or not batch["comparison"]["null_patterns_identical"]):
        raise ValueError("Reference must be the completed, validated 3600-patch combined benchmark")
    return batch


def load_fixture(manifest_path, reference):
    """Rehydrate saved full canonical squares, avoiding a redundant full-fire grid."""
    manifest = gpd.read_parquet(manifest_path)
    selected = manifest.loc[manifest.fire_id.eq(baseline.FIRE_ID)]
    if len(selected) != 1 or manifest.crs is None:
        raise ValueError("Require one canonical WOODBURY record and CRS")
    fire = selected.iloc[0].to_dict()
    if (fire["fire_year"], fire["season1_year"]) != (2019, 2020):
        raise ValueError("Unexpected WOODBURY timing")
    pattern = re.compile(re.escape(baseline.FIRE_ID) + r"_r(-?\d+)_c(-?\d+)")
    indices = []
    for patch_id in reference["expected_patch_ids"]:
        match = pattern.fullmatch(patch_id)
        if match is None:
            raise ValueError("Unexpected reference patch ID")
        indices.append(tuple(map(int, match.groups())))
    rows, cols = np.array(indices).T
    if not (np.all(rows // 60 == 235) and np.all(cols // 60 == -259)):
        raise ValueError("Reference patches leave the approved tile")
    if len(set(indices)) != 3600:
        raise ValueError("Reference does not cover the full tile")
    size = config.PATCH_SIZE_M
    cells = shapely.box(cols * size, rows * size, (cols + 1) * size, (rows + 1) * size)
    perimeter = selected.to_crs(config.GRID_CRS).geometry.iloc[0]
    area = shapely.area(shapely.intersection(cells, perimeter))
    if not np.all(area > 0):
        raise ValueError("A reference cell is not a retained positive-area fire patch")
    patches = gpd.GeoDataFrame({"patch_id": reference["expected_patch_ids"],
        "fire_id": baseline.FIRE_ID, "grid_row": rows, "grid_col": cols,
        "patch_area_m2": size**2, "fire_overlap_fraction": area / size**2,
        "batch_id": BATCH_ID}, geometry=cells, crs=config.GRID_CRS)
    validate_patches(patches, baseline.FIRE_ID)
    return fire, patches


def submit(fire, patches, results, client):
    collection = pipeline.patches_to_ee(patches)
    images = daymet.build_fire_climate_images(fire, collection.geometry().bounds(maxError=1))
    descriptor = experiment.build_qa4_product(images, collection, fire)
    if len(descriptor["columns"]) != 41 or len(set(descriptor["columns"])) != 41:
        raise ValueError("Expected 41 raw numeric fields")
    batch = results["batches"]["60"]
    if not batch["products"]:
        item = {"source": "daymet", "product": "qa4", "expected_columns": descriptor["columns"]}
        item.update(paths(batch, item))
        batch["products"].append(item)
    item, = batch["products"]
    if item["expected_columns"] != descriptor["columns"]:
        raise ValueError("Saved Candidate C schema differs")
    baseline.save(RESULTS, results)
    try:
        action = work_items.advance_work_item(descriptor["collection"], descriptor["columns"],
            **identity(batch, item), expected_patch_ids=batch["expected_patch_ids"], client=client)
        saved = action.get("status") or persistence.read_status(**identity(batch, item), client=client)
        item.update(action=action["action"], status=saved, task_id=saved.get("ee_task_id"),
                    submission_timestamp=saved["started_at"])
        item.pop("harness_error", None)
    except Exception as error:
        item["harness_error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        baseline.save(RESULTS, results)
    print(item["action"], item["task_id"])


def compare(results, reference, client):
    item, = results["batches"]["60"]["products"]
    old, = reference["products"]
    def read(record):
        if record.get("status", {}).get("status") != "COMPLETE":
            raise ValueError("Run check first; both Parquet outputs must be COMPLETE")
        return pd.read_parquet(BytesIO(client.bucket(config.GCS_BUCKET)
                              .blob(record["parquet_path"]).download_as_bytes()))
    comparison = experiment.compare_climate_tables(read(item), read(old), reference["expected_patch_ids"])
    new_cost = item["ee_task"]["batch_eecu_usage_seconds"]
    old_cost = old["ee_task"]["batch_eecu_usage_seconds"]
    comparison.update(candidate_eecu_seconds=new_cost, combined_reference_eecu_seconds=old_cost,
                      reduction_percent=100 * (1 - new_cost / old_cost))
    results["comparison"] = comparison
    baseline.save(RESULTS, results)
    print(json.dumps(comparison, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["preflight", "submit", "check", "compare"])
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/fire_manifest.parquet"))
    args = parser.parse_args()
    reference = load_reference()
    if args.mode in ("preflight", "submit"):
        fire, patches = load_fixture(args.manifest, reference)
    if args.mode == "preflight":
        print(json.dumps({"fire_id": fire["fire_id"], "batch_id": BATCH_ID, "patch_count": len(patches),
                          "reference_parquet": reference["products"][0]["parquet_path"],
                          "reference_verified": "local records only; no live GCS read"}, indent=2))
        return
    results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {
        "benchmark": experiment.VERSION, "fire_id": baseline.FIRE_ID,
        "batches": {"60": {"tile_side_cells": 60, "batch_id": BATCH_ID, "patch_count": 3600,
                            "expected_patch_ids": reference["expected_patch_ids"], "products": []}}}
    if (results["benchmark"] != experiment.VERSION or results["fire_id"] != baseline.FIRE_ID
            or set(results["batches"]) != {"60"}
            or results["batches"]["60"]["expected_patch_ids"] != reference["expected_patch_ids"]):
        raise ValueError("Saved experimental results differ from the approved fixture")
    if args.mode != "submit" and not results["batches"]["60"]["products"]:
        raise ValueError("No saved Candidate C submission")
    baseline.record_git_commit(results)
    if args.mode != "compare":
        ee.Initialize(project=config.GCP_PROJECT)
    client = storage.Client(project=config.GCP_PROJECT)
    if args.mode == "submit":
        submit(fire, patches, results, client)
    elif args.mode == "check":
        baseline.collect(argparse.Namespace(results=RESULTS), results, client,
                         identity_fn=identity, paths_fn=paths)
    else:
        compare(results, reference, client)


if __name__ == "__main__":
    main()
