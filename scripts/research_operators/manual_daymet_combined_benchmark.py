"""Experimental Daymet-only submit/check/compare; production inventory is unchanged."""

import argparse
from io import BytesIO
import json
from pathlib import Path

import ee
import geopandas as gpd
from google.cloud import storage
import numpy as np
import pandas as pd

import manual_planb_benchmark as baseline
from wildfire_recovery_forecasting import config, daymet, persistence, pipeline, production, reductions, work_items
from wildfire_recovery_forecasting.validation import validate_patch_table

VERSION = "phase3_daymet_combined_bench_v1"
RESULTS = Path("data/processed/daymet_combined_benchmark_results.json")


def build_combined_product(images, patches):
    """One area-sum boundary using exactly the existing per-variable input images."""
    primary, coverage, projection = reductions.daymet_area_inputs(images)
    if primary.keys() & coverage.keys():
        raise ValueError("Daymet variable names overlap")
    return reductions.build_area_product({**primary, **coverage}, patches, projection)


def identity(batch, item):
    tile = batch["tile_side_cells"]
    if tile not in (60, 90, 120) or (item["source"], item["product"]) != ("daymet", "combined"):
        raise ValueError("Only experimental combined Daymet tiles 60/90/120 are supported")
    return dict(source="daymet", product="combined", fire_id=baseline.FIRE_ID,
                batch_id=batch["batch_id"], production_version=f"{VERSION}_t{tile}")


def paths(batch, item):
    key = identity(batch, item)
    return {"raw_prefix": production.raw_export_prefix(**key),
            "status_path": production.status_path(**key),
            "parquet_path": production.parquet_path(**key)}


def old_products(batch):
    parts = {p["product"]: p for p in batch["products"] if p["source"] == "daymet"}
    if set(parts) != {"primary", "coverage"}:
        raise ValueError("Baseline must contain primary and coverage Daymet products")
    return parts


def submit(args, results, old, client):
    manifest = gpd.read_parquet(args.manifest)
    selected = manifest.loc[manifest.fire_id.eq(baseline.FIRE_ID)]
    if len(selected) != 1 or manifest.crs is None:
        raise ValueError("Require one canonical WOODBURY record and CRS")
    fire = selected.iloc[0].to_dict()
    grid = baseline.build_fire_grid(fire["geometry"], baseline.FIRE_ID, perimeter_crs=manifest.crs)
    for tile in args.tiles:
        prior = old["batches"][str(tile)]
        patches = baseline.select_batch(grid, tile)
        ids = patches.patch_id.tolist()
        if (patches.batch_id.iloc[0] != prior["batch_id"]
                or ids != prior["expected_patch_ids"] or len(ids) != tile**2):
            raise ValueError("Current selection differs from the full baseline benchmark batch")
        collection = pipeline.patches_to_ee(patches)
        images = daymet.build_fire_climate_images(fire, collection.geometry().bounds(maxError=1))
        descriptor = build_combined_product(images, collection)
        parts = old_products(prior)
        expected = set().union(*(set(p["expected_columns"]) for p in parts.values()))
        if set(descriptor["columns"]) != expected or len(descriptor["columns"]) != len(expected):
            raise ValueError("Combined raw schema differs from baseline union")
        batch = results["batches"].setdefault(str(tile), {
            "tile_side_cells": tile, "batch_id": prior["batch_id"], "patch_count": len(ids),
            "expected_patch_ids": ids, "products": []})
        if batch["expected_patch_ids"] != ids or batch["batch_id"] != prior["batch_id"]:
            raise ValueError("Saved experimental batch differs")
        if not batch["products"]:
            item = {"source": "daymet", "product": "combined", "expected_columns": descriptor["columns"]}
            item.update(paths(batch, item))
            batch["products"].append(item)
        item, = batch["products"]
        if item["expected_columns"] != descriptor["columns"]:
            raise ValueError("Saved experimental schema differs")
        baseline.save(args.results, results)
        action = work_items.advance_work_item(descriptor["collection"], descriptor["columns"],
            **identity(batch, item), expected_patch_ids=ids, client=client)
        status = action.get("status") or persistence.read_status(**identity(batch, item), client=client)
        item.update(action=action["action"], status=status, task_id=status.get("ee_task_id"),
                    submission_timestamp=status["started_at"])
        baseline.save(args.results, results)
        print(tile, item["action"], item["task_id"], flush=True)


def compare_tables(combined, primary, coverage, ids):
    """Strict IDs/schema/null comparison; tight tolerance for floating sum order."""
    tables = []
    for table in (combined, primary, coverage):
        validate_patch_table(table, expected_ids=ids, fire_id=baseline.FIRE_ID)
        if not table.columns.is_unique:
            raise ValueError("Duplicate columns")
        tables.append(table.set_index("patch_id").sort_index())
    actual, first, second = tables
    shared = {"fire_id", reductions.RASTER_FULL_AREA}
    if set(first.columns) & set(second.columns) != shared:
        raise ValueError("Unexpected overlapping baseline columns")
    if set(actual.columns) != set(first.columns) | set(second.columns):
        raise ValueError("Combined columns differ from baseline union")
    diagnostics = {}
    for label, reference in (("primary", first), ("coverage", second)):
        for name in reference.columns:
            left, right = actual[name], reference[name]
            if not left.isna().equals(right.isna()):
                raise ValueError(f"Null pattern changed: {label}/{name}")
            if name == "fire_id":
                continue
            a, b = left.to_numpy(dtype=float), right.to_numpy(dtype=float)
            if not np.allclose(a, b, rtol=1e-9, atol=1e-6, equal_nan=True):
                raise ValueError(f"Numerical mismatch: {label}/{name}")
            valid = ~np.isnan(a)
            diagnostics[f"{label}/{name}"] = float(np.max(np.abs(a[valid] - b[valid]))) if valid.any() else None
    return {"equivalent": True, "rows": len(combined), "columns": len(combined.columns),
            "rtol": 1e-9, "atol": 1e-6, "null_patterns_identical": True,
            "max_absolute_differences": diagnostics}


def compare(args, results, old, client):
    def read(item):
        if item.get("status", {}).get("status") != "COMPLETE":
            raise ValueError("Run check first; comparison requires completed Parquet outputs")
        return pd.read_parquet(BytesIO(client.bucket(config.GCS_BUCKET)
                              .blob(item["parquet_path"]).download_as_bytes()))
    for tile, batch in results["batches"].items():
        prior = old["batches"][tile]
        if (batch["batch_id"] != prior["batch_id"]
                or batch["expected_patch_ids"] != prior["expected_patch_ids"]):
            raise ValueError("Baseline and experimental populations differ")
        item, = batch["products"]
        parts = old_products(prior)
        result = compare_tables(read(item), read(parts["primary"]), read(parts["coverage"]),
                                batch["expected_patch_ids"])
        combined_cost = item["ee_task"]["batch_eecu_usage_seconds"]
        separate_cost = sum(p["ee_task"]["batch_eecu_usage_seconds"] for p in parts.values())
        result.update(combined_eecu_seconds=combined_cost, separate_eecu_seconds=separate_cost,
                      eecu_reduction_percent=100 * (1 - combined_cost / separate_cost))
        batch["comparison"] = result
        baseline.save(args.results, results)
        print(tile, {k: v for k, v in result.items() if k != "max_absolute_differences"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["submit", "check", "compare"])
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/fire_manifest.parquet"))
    parser.add_argument("--tiles", type=int, nargs="+", choices=[60, 90, 120], default=[60, 90, 120])
    parser.add_argument("--baseline", type=Path, default=baseline.RESULTS)
    parser.add_argument("--results", type=Path, default=RESULTS)
    args = parser.parse_args()
    if args.results.resolve() == args.baseline.resolve():
        parser.error("Experimental results must not overwrite baseline results")
    old = json.loads(args.baseline.read_text(encoding="utf-8"))
    if old["benchmark"] != baseline.VERSION or old["fire_id"] != baseline.FIRE_ID:
        parser.error("Wrong baseline benchmark")
    results = json.loads(args.results.read_text(encoding="utf-8")) if args.results.exists() else {
        "benchmark": VERSION, "fire_id": baseline.FIRE_ID, "batches": {}}
    if results["benchmark"] != VERSION or results["fire_id"] != baseline.FIRE_ID:
        parser.error("Wrong experimental results namespace")
    if args.mode != "submit" and not results["batches"]:
        parser.error("No saved experimental submissions")
    baseline.record_git_commit(results)
    if args.mode != "compare":
        ee.Initialize(project=config.GCP_PROJECT)
    client = storage.Client(project=config.GCP_PROJECT)
    if args.mode == "submit":
        submit(args, results, old, client)
    elif args.mode == "check":
        baseline.collect(args, results, client, identity_fn=identity, paths_fn=paths)
    else:
        compare(args, results, old, client)


if __name__ == "__main__":
    main()
