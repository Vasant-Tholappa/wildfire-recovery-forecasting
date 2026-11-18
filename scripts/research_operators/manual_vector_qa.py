"""Local supplemental QA persistence; only missing tiles read MTBS vectors from EE.

Pilot: uv run --offline python manual_vector_qa.py --limit 2
Full/resume: uv run --offline python manual_vector_qa.py --all
No raster exports or cloud writes. Run only one instance at a time.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import manual_population as population
from wildfire_recovery_forecasting import config, disturbance, pipeline, reductions, validation

OUTPUT = population.DATA / (population.VERSION + "_vector_qa")
FRACTIONS = ["fire_overlap_fraction", "prior_fire_fraction_10yr", "reburn_fraction"]
STATUSES = [name + "_coverage_status" for name in FRACTIONS[1:]]
COLUMNS = ["fire_id", "patch_id", *FRACTIONS, *STATUSES]


def selected_tiles(items, limit=None):
    tiles = {}
    for item in items:
        key = (item["fire_id"], item["batch_id"])
        if key in tiles and tiles[key]["patch_count"] != item["patch_count"]:
            raise ValueError("Conflicting tile occupancy")
        tiles[key] = {k: item[k] for k in ("fire_id", "batch_id", "patch_count")}
    if limit is not None and not 1 <= limit <= len(tiles):
        raise ValueError("limit must be between 1 and the selected tile count")
    return [tiles[key] for key in sorted(tiles)][:limit]


def validate_tile(table, patches, fire_id):
    if set(table.columns) != set(COLUMNS):
        raise ValueError("Unexpected supplemental QA schema")
    validation.validate_patch_table(table, expected_ids=patches.patch_id, fire_id=fire_id)
    values = table[FRACTIONS].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Missing/nonfinite vector fractions")
    validation.validate_measurements(table)
    expected = patches.set_index("patch_id").fire_overlap_fraction
    if not np.allclose(table.fire_overlap_fraction, expected.loc[table.patch_id], rtol=0, atol=1e-12):
        raise ValueError("Fire overlap differs from canonical patches")
    if not table.fire_overlap_fraction.gt(0).all():
        raise ValueError("Retained patches require positive fire overlap")
    for name in STATUSES:
        if not table[name].eq("unverified").all():
            raise ValueError("MTBS completeness must remain explicitly unverified")


def save_parquet(table, path):
    temporary = path.with_suffix(".tmp.parquet")
    table.to_parquet(temporary, index=False)
    temporary.replace(path)


def run_tiles(manifest, tiles, output, *, history_loader, provenance, filename):
    """Validate resumed tiles, atomically save new tiles, then combine exact IDs."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    record = output / "inputs.json"
    if record.exists():
        if json.loads(record.read_text()) != provenance:
            raise ValueError("Supplement inputs changed; use a new output directory")
    else:
        temporary = record.with_suffix(".tmp")
        temporary.write_text(json.dumps(provenance, indent=2) + "\n")
        temporary.replace(record)
    parts = output / "tiles"
    parts.mkdir(exist_ok=True)
    fires = manifest.set_index("fire_id", drop=False)
    perimeters = manifest.to_crs(config.GRID_CRS).set_index("fire_id").geometry
    tables = []
    for tile in tiles:
        fire_id, batch_id = tile["fire_id"], tile["batch_id"]
        patches = population.batch_patches(perimeters.loc[fire_id], fire_id, batch_id)
        validation.validate_patches(patches, fire_id)
        if len(patches) != tile["patch_count"]:
            raise ValueError("Frozen tile occupancy mismatch")
        path = parts / f"{fire_id}_{batch_id}.parquet"
        if path.exists():
            table = pd.read_parquet(path)
            validate_tile(table, patches, fire_id)
            action = "REUSED"
        else:
            history = history_loader(fires.loc[fire_id].to_dict(), patches)
            table = reductions.reduce_wildfire_history(history, patches).merge(
                pd.DataFrame(patches[["fire_id", "patch_id", "fire_overlap_fraction"]]),
                on=["fire_id", "patch_id"], validate="one_to_one")
            table = table[COLUMNS]
            validate_tile(table, patches, fire_id)
            save_parquet(table, path)
            action = "SAVED"
        tables.append(table)
        print(f"{action} {fire_id}/{batch_id}: {len(table)} patches", flush=True)
    combined = pd.concat(tables, ignore_index=True)
    validation.validate_patch_table(combined)
    if len(combined) != sum(tile["patch_count"] for tile in tiles):
        raise ValueError("Combined occupancy mismatch")
    destination = output / filename
    save_parquet(combined, destination)
    print(f"Validated {len(tiles)} tiles / {len(combined)} rows: {destination}", flush=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--limit", type=int, help="First N frozen tiles, sorted by fire_id/batch_id")
    mode.add_argument("--all", action="store_true", help="All 175 frozen tiles; reuse completed local parts")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    manifest, items = population.load_population()
    tiles = selected_tiles(items, args.limit)
    provenance = {"production_version": population.VERSION, "qa_schema_version": 1,
                  "mtbs_source": disturbance.MTBS_SOURCE,
                  "coverage_note": "Observed overlap only; source completeness unverified. Resumed tiles retain earlier retrievals.",
                  "inputs_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in (population.MANIFEST, population.OCCUPANCY, population.REPORT)}}
    initialized = False

    def retrieve(fire, patches):
        nonlocal initialized
        import ee
        if not initialized:
            ee.Initialize(project=config.GCP_PROJECT)
            initialized = True
        # Cover full boundary patches, including portions outside the focal fire.
        region = pipeline.patches_to_ee(patches).geometry().bounds(maxError=1)
        return disturbance.build_wildfire_history(fire, region)

    filename = "supplemental.parquet" if args.all else f"pilot_{args.limit}.parquet"
    run_tiles(manifest, tiles, args.output_dir, history_loader=retrieve,
              provenance=provenance, filename=filename)


if __name__ == "__main__":
    main()
