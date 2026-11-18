"""Two fixed terrain-only recovery tiles; preflight is offline, other modes manual."""
import argparse
from datetime import datetime, timezone
from io import BytesIO
import json

import ee
import numpy as np
import pandas as pd
from google.cloud import storage

import manual_population as population
from wildfire_recovery_forecasting import config, landfire, persistence, pipeline
from wildfire_recovery_forecasting import production, reductions, terrain, validation, work_items

VERSION = "phase3_ca175_tile1_terrain_pilot_v2"
FIRES = ("CA4143512316220170626", "CA4009112093120200817")
REPORT = population.DATA / (VERSION + "_report.json")
NAMES = ("elevation", "slope", "northness", "eastness")


def pilot_inputs():
    manifest, all_items = population.load_population()  # Existing frozen hash checks.
    items = [i for i in all_items if i["fire_id"] in FIRES and i["source"] == "terrain"]
    if len(items) != 2 or {i["fire_id"] for i in items} != set(FIRES):
        raise ValueError("Require exactly the two approved terrain tiles")
    return manifest, items


def identity(item, version=VERSION):
    return dict(source="terrain", product=None, fire_id=item["fire_id"],
                batch_id=item["batch_id"], production_version=version)


def paths(item, version=VERSION):
    key = identity(item, version)
    return {name: f"gs://{config.GCS_BUCKET}/{fn(**key)}" for name, fn in (
        ("raw_prefix", production.raw_export_prefix), ("status_uri", production.status_path),
        ("parquet", production.parquet_path))}


def prepare_patches(manifest, item):
    row = manifest.set_index("fire_id").to_crs(config.GRID_CRS).loc[item["fire_id"]]
    patches = population.batch_patches(row.geometry, item["fire_id"], item["batch_id"])
    validation.validate_patches(patches, item["fire_id"])
    if len(patches) != item["patch_count"]:
        raise ValueError("Frozen occupancy mismatch")
    return patches


def terrain_product(fire, patches):
    collection = pipeline.patches_to_ee(patches)
    region = collection.geometry().bounds(maxError=1)  # Identical production geometry.
    vegetation = landfire.build_landfire_images(fire, region)
    return reductions.build_terrain_product(
        terrain.build_terrain_images(region), vegetation["forest_mask"], collection)


def advance(mode, item, fire, patches, client):
    """Check cannot submit, even if saved state changes before the controller reads it."""
    key = identity(item)
    saved = persistence.read_status(**key, client=client)
    columns = population.raw_columns("terrain", None)
    task = None
    collection = None
    if saved:
        if saved.get("ee_task_id"):
            task = work_items.lookup_task_status(saved["ee_task_id"])
    elif mode == "submit":
        product = terrain_product(fire, patches)
        if product["columns"] != columns:
            raise ValueError("Terrain raw schema changed")
        collection = product["collection"]
    else:
        return {"action": "NOT_SUBMITTED", "status": None}

    # A missing/raced status must never initiate export during reconciliation.
    def lookup(task_id):
        if not task or task_id != saved.get("ee_task_id"):
            raise ValueError("Saved task identity changed; inspect before continuing")
        return task

    if collection is None:
        from unittest.mock import patch
        with patch.object(work_items.exports, "start_table_export",
                          side_effect=RuntimeError("Reconciliation cannot submit exports")):
            result = work_items.advance_work_item(None, columns, **key, client=client,
                expected_patch_ids=patches.patch_id, task_status_lookup=lookup)
    else:
        result = work_items.advance_work_item(collection, columns, **key, client=client,
            expected_patch_ids=patches.patch_id, task_status_lookup=lookup)
    return {**result, "ee_task": task}


def compare_tables(old, new, item, expected_ids):
    summaries = []
    required = {"patch_id", "fire_id", *population.raw_columns("terrain", None)}
    for raw in (old, new):
        if not required.issubset(raw.columns):
            raise ValueError("Missing terrain raw columns")
        validation.validate_patch_table(raw, expected_ids=expected_ids, fire_id=item["fire_id"])
        table = reductions.summarize_area_sums(raw, NAMES, support="support")
        validation.validate_measurements(table)
        if np.isinf(table.select_dtypes(include="number").to_numpy()).any():
            raise ValueError("Infinite terrain measurement")
        for name, low, high in (("slope", 0, 90), ("northness", -1, 1), ("eastness", -1, 1)):
            if not table[name + "_mean"].dropna().between(low-1e-6, high+1e-6).all():
                raise ValueError(f"Implausible {name}")
        summaries.append(table.set_index("patch_id").sort_index())
    a, b = summaries
    report = {"rows": len(b), "exact_ids": True, "duplicates": 0, "variables": {}}
    for name in NAMES:
        x, y = a[name + "_mean"], b[name + "_mean"]
        delta = (y-x).dropna()
        av, bv = a[name + "_valid_fraction"], b[name + "_valid_fraction"]
        report["variables"][name] = dict(
            old_missing=int(x.isna().sum()), new_missing=int(y.isna().sum()),
            recovered=int((x.isna() & y.notna()).sum()), lost=int((x.notna() & y.isna()).sum()),
            old_valid_fraction_mean=float(av.mean()), new_valid_fraction_mean=float(bv.mean()),
            support_decreased=int((bv < av-1e-6).sum()),
            common_support_mean_change=float(delta.mean()) if len(delta) else None,
            common_support_max_abs_change=float(delta.abs().max()) if len(delta) else None,
            new_min=float(y.min()) if y.notna().any() else None,
            new_max=float(y.max()) if y.notna().any() else None)
    return report


def read_parquet(client, item, version):
    blob = client.bucket(config.GCS_BUCKET).blob(production.parquet_path(**identity(item, version)))
    return pd.read_parquet(BytesIO(blob.download_as_bytes()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "submit", "check", "compare"))
    args = parser.parse_args()
    manifest, items = pilot_inputs()
    records = []
    client = None
    if args.mode != "preflight":
        client = storage.Client(project=config.GCP_PROJECT)
    if args.mode in ("submit", "check"):
        ee.Initialize(project=config.GCP_PROJECT)
    if args.mode == "submit":
        # Conservative account-wide visible task capacity, not a quota/budget check.
        active = sum(t.get("state") in population.ACTIVE for t in ee.data.getTaskList())
        missing = sum(persistence.read_status(**identity(i), client=client) is None for i in items)
        if active + missing > 4:
            raise RuntimeError("Insufficient capacity under max-active=4; submit later")
    for item in items:
        patches = prepare_patches(manifest, item)
        entry = {**item, **paths(item)}
        if args.mode in ("submit", "check"):
            fire = manifest.set_index("fire_id", drop=False).loc[item["fire_id"]].to_dict()
            entry.update(advance(args.mode, item, fire, patches, client))
        elif args.mode == "compare":
            saved = persistence.read_status(**identity(item), client=client)
            if not saved or saved["status"] != "COMPLETE" or saved["row_count"] != len(patches):
                raise ValueError("Recovery must be COMPLETE with correct row count before comparison")
            old_version = population.REUSED_BATCHES.get((item["fire_id"], item["batch_id"]), population.VERSION)
            entry["comparison"] = compare_tables(read_parquet(client, item, old_version),
                read_parquet(client, item, VERSION), item, patches.patch_id)
        records.append(entry)
        report = dict(version=VERSION, project=config.GCP_PROJECT, bucket=config.GCS_BUCKET,
                      mode=args.mode, timestamp=datetime.now(timezone.utc).isoformat(), items=records,
                      quota_and_billing="Live balances NOT verified; check manually before submit.")
        output = REPORT.with_name(REPORT.stem + "_" + args.mode + ".json")
        output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps(entry, indent=2), flush=True)
        if entry.get("action") in ("FAILED", "NEEDS_RETRY", "NEEDS_REVIEW"):
            raise RuntimeError("Stopped for operator review; no automatic retry")
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
