"""Freeze one upper-median-occupancy tile per approved CA fire, entirely locally."""
import hashlib
import json
from pathlib import Path
import shutil

import geopandas as gpd
import pandas as pd

DATA = Path(__file__).resolve().parent / "data/processed"
VERSION = "phase3_ca175_tile1_v1"


def select_tiles(batches):
    if batches.duplicated(["fire_id", "batch_id"]).any() or not batches.patch_count.gt(0).all():
        raise ValueError("Expected unique, positive-occupancy batches")
    ordered = batches.sort_values(["fire_id", "patch_count", "batch_id"])
    return pd.concat([g.iloc[[len(g)//2]] for _, g in ordered.groupby("fire_id")]).reset_index(drop=True)


def main():
    source = DATA / "phase3_ca175_population_v1.parquet"
    occupancy = DATA / "phase3_ca175_batches_v1.parquet"
    manifest = gpd.read_parquet(source)
    prior = json.loads((DATA / "phase3_ca175_population_v1.json").read_text())
    assert hashlib.sha256(source.read_bytes()).hexdigest() == prior["manifest_sha256"]
    batches = select_tiles(pd.read_parquet(occupancy))
    assert set(batches.fire_id) == set(manifest.fire_id) and len(batches) == 175
    assert batches.patch_count.sum() == 308048
    destination = DATA / (VERSION + ".parquet")
    batch_file = DATA / (VERSION + "_batches.parquet")
    # Preserve the exact approved fire rows/geometries; tile restriction is a separate manifest.
    shutil.copyfile(source, destination)
    batches.to_parquet(batch_file, index=False)
    report = {
        "production_version": VERSION,
        "selection_rule": "Sort each fire's nonempty tiles by patch_count then batch_id; choose zero-based index floor(n/2) (upper median).",
        "selected_ids_sha256": prior["selected_ids_sha256"],
        "manifest_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "batches_sha256": hashlib.sha256(batch_file.read_bytes()).hexdigest(),
        "inputs_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (source, occupancy)},
        "fire_count": 175, "spatial_batches": 175, "export_tasks": 2100,
        "candidate_patches": int(batches.patch_count.sum()),
        "year_counts": manifest.fire_year.value_counts().sort_index().to_dict(),
        "split_counts": manifest.split.value_counts().to_dict(),
        "occupancy": batches.patch_count.describe(percentiles=[.25,.5,.75,.9,.95]).to_dict(),
        "reuse": [{"fire_id":"CA4118512343320200728", "batch_id":"r215_c-210",
                   "production_version":"phase3_ca175_v1"}],
        "selected_batches": batches.to_dict("records"),
    }
    text = json.dumps(report, indent=2) + "\n"
    (DATA / (VERSION + ".json")).write_text(text)
    Path("notes/phase3_ca175_tile1_inputs_v1.json").write_text(text)
    print(json.dumps({k:v for k,v in report.items() if k not in ("selected_batches",)}, indent=2))


if __name__ == "__main__":
    main()
