"""Local exact tile counts; verify historical totals before counting new sizes."""

import argparse
import json
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import shapely

from wildfire_recovery_forecasting.config import GRID_CRS, PATCH_SIZE_M


REGRESSION_COUNTS = {10: 200208, 20: 62258, 30: 33285}
METHOD = (
    "Positive-area intersections with globally aligned EPSG:5070 square tiles; "
    "equivalent to nonempty retained-patch batches"
)


def count_tiles(manifest, tile_side_cells):
    """Count positive-area tile intersections, preserving holes and small parts."""
    width = tile_side_cells * PATCH_SIZE_M
    totals = []
    for geom in manifest.geometry:
        x0, y0, x1, y1 = geom.bounds
        xx, yy = np.meshgrid(
            np.arange(math.floor(x0 / width), math.ceil(x1 / width)),
            np.arange(math.floor(y0 / width), math.ceil(y1 / width)),
        )
        cells = shapely.box(
            xx.ravel() * width, yy.ravel() * width,
            (xx.ravel() + 1) * width, (yy.ravel() + 1) * width,
        )
        cells = cells[shapely.intersects(cells, geom)]
        totals.append(int(np.count_nonzero(
            shapely.area(shapely.intersection(cells, geom)) > 0
        )))
    return {
        "spatial_batches": sum(totals),
        "per_fire_median": float(np.median(totals)),
        "per_fire_max": max(totals),
    }


def verified_counts(manifest, tiles):
    """Fail before new counts/output if any historical regression disagrees."""
    manifest = manifest.to_crs(GRID_CRS)
    regressions = {}
    for tile, expected in REGRESSION_COUNTS.items():
        actual = count_tiles(manifest, tile)
        if actual["spatial_batches"] != expected:
            raise ValueError(
                f"Tile {tile} regression failed: expected {expected}, "
                f"got {actual['spatial_batches']}; new counts not trusted"
            )
        regressions[str(tile)] = actual
        print(f"Tile {tile}: regression verified ({expected})", flush=True)
    counts = {}
    for tile in tiles:
        counts[str(tile)] = count_tiles(manifest, tile)
        print(f"Tile {tile}: {counts[str(tile)]}", flush=True)
    return {"fire_count": len(manifest), "method": METHOD,
            "regression_counts": regressions, "counts": counts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=Path("data/processed/fire_manifest.parquet"))
    parser.add_argument("--tiles", type=int, nargs="+", choices=[90, 120, 150],
                        default=[90, 120, 150])
    parser.add_argument("--output", type=Path,
                        default=Path("data/processed/phase3_planb_population_counts.json"))
    args = parser.parse_args()
    manifest = gpd.read_parquet(args.manifest)
    if (len(manifest) != 1904 or "fire_id" not in manifest
            or manifest.fire_id.isna().any() or not manifest.fire_id.is_unique
            or manifest.crs is None or manifest.geometry.isna().any()
            or manifest.geometry.is_empty.any() or not manifest.geometry.is_valid.all()):
        raise ValueError("Expected 1,904 unique fires with valid, nonempty geometry and CRS")
    results = verified_counts(manifest, args.tiles)
    results["manifest_path"] = str(args.manifest.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
