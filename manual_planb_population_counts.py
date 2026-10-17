"""Local exact tile counts; verify historical totals before counting new sizes."""

import argparse
import json
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from wildfire_recovery_forecasting.config import GRID_CRS, PATCH_SIZE_M
from wildfire_recovery_forecasting.grid import assign_spatial_batches, build_fire_grid


REGRESSION_COUNTS = {10: 200208, 20: 62258, 30: 33285}
METHOD = (
    "Positive-area intersections with globally aligned EPSG:5070 square tiles; "
    "equivalent to nonempty retained-patch batches"
)


def batch_patch_counts_reference(manifest):
    """Slow canonical implementation retained as an equivalence oracle."""
    tables = []
    for fire in manifest.itertuples():
        patches = build_fire_grid(fire.geometry, fire.fire_id, perimeter_crs=manifest.crs)
        batches = assign_spatial_batches(patches, tile_side_cells=120)
        tables.append(batches.groupby(["fire_id", "batch_id"]).size()
                      .rename("patch_count").reset_index())
        print(f"{fire.fire_id}: {len(patches)} patches, {len(tables[-1])} batches", flush=True)
    return pd.concat(tables, ignore_index=True).sort_values(
        ["fire_id", "batch_id"], ignore_index=True
    )


def batch_patch_counts(manifest):
    """Exact counts using prepared predicates and vectorized boundary overlays.

    Work on at most one 120x120 tile at a time. Covered squares necessarily
    have positive area; only intersecting, not-covered squares need overlays.
    Intersect against the original projected perimeter, never a clipped version.
    """
    records = []
    for fire in manifest.to_crs(GRID_CRS).itertuples():
        geom = fire.geometry
        if (geom.geom_type not in ("Polygon", "MultiPolygon") or geom.is_empty
                or not geom.is_valid or not np.isfinite(geom.bounds).all()):
            raise ValueError("Require a valid nonempty projected Polygon/MultiPolygon")
        shapely.prepare(geom)
        x0, y0, x1, y1 = geom.bounds
        c0, c1 = math.floor(x0 / PATCH_SIZE_M), math.ceil(x1 / PATCH_SIZE_M)
        r0, r1 = math.floor(y0 / PATCH_SIZE_M), math.ceil(y1 / PATCH_SIZE_M)
        fire_patches = fire_batches = 0
        for tr in range(r0 // 120, (r1 - 1) // 120 + 1):
            for tc in range(c0 // 120, (c1 - 1) // 120 + 1):
                tile = shapely.box(tc * 120 * PATCH_SIZE_M, tr * 120 * PATCH_SIZE_M,
                                   (tc + 1) * 120 * PATCH_SIZE_M,
                                   (tr + 1) * 120 * PATCH_SIZE_M)
                if not shapely.intersects(geom, tile):
                    continue
                if shapely.covers(geom, tile):
                    count = 120**2
                else:
                    cols, rows = np.meshgrid(
                        np.arange(max(c0, tc * 120), min(c1, (tc + 1) * 120)),
                        np.arange(max(r0, tr * 120), min(r1, (tr + 1) * 120)),
                    )
                    cells = shapely.box(cols.ravel() * PATCH_SIZE_M, rows.ravel() * PATCH_SIZE_M,
                                        (cols.ravel() + 1) * PATCH_SIZE_M,
                                        (rows.ravel() + 1) * PATCH_SIZE_M)
                    covered = shapely.covers(geom, cells)
                    boundary = cells[~covered & shapely.intersects(geom, cells)]
                    count = int(covered.sum() + np.count_nonzero(
                        shapely.area(shapely.intersection(boundary, geom)) > 0))
                if count:
                    records.append((fire.fire_id, f"r{tr}_c{tc}", count))
                    fire_patches += count
                    fire_batches += 1
        print(f"{fire.fire_id}: {fire_patches} patches, {fire_batches} batches", flush=True)
    return pd.DataFrame(records, columns=["fire_id", "batch_id", "patch_count"]).sort_values(
        ["fire_id", "batch_id"], ignore_index=True)


def write_batch_patch_counts(table, path):
    """Persist only a complete tile-120 population with the known batch total."""
    if (len(table) != 6301 or table[["fire_id", "batch_id", "patch_count"]].isna().any().any()
            or table.duplicated(["fire_id", "batch_id"]).any()
            or not table.patch_count.between(1, 120**2).all()):
        raise ValueError("Expected exactly 6,301 unique nonempty tile-120 batches")
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, index=False)


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
    parser.add_argument("--batch-counts-output", type=Path,
                        help="Instead of aggregate counts, write tile-120 patch counts to Parquet")
    args = parser.parse_args()
    manifest = gpd.read_parquet(args.manifest)
    if (len(manifest) != 1904 or "fire_id" not in manifest
            or manifest.fire_id.isna().any() or not manifest.fire_id.is_unique
            or manifest.crs is None or manifest.geometry.isna().any()
            or manifest.geometry.is_empty.any() or not manifest.geometry.is_valid.all()):
        raise ValueError("Expected 1,904 unique fires with valid, nonempty geometry and CRS")
    if args.batch_counts_output is not None:
        write_batch_patch_counts(batch_patch_counts(manifest), args.batch_counts_output)
        print(args.batch_counts_output.resolve())
        return
    results = verified_counts(manifest, args.tiles)
    results["manifest_path"] = str(args.manifest.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
