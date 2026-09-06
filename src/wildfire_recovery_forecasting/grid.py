"""Common production patch grid and patch identifiers."""

from math import ceil, floor, isfinite

import geopandas as gpd
from pyproj import CRS
from shapely.geometry import MultiPolygon, Polygon, box

from wildfire_recovery_forecasting.config import GRID_CRS, PATCH_SIZE_M


def build_fire_grid(
    perimeter: Polygon | MultiPolygon,
    fire_id: str,
    *,
    perimeter_crs: str | CRS,
) -> gpd.GeoDataFrame:
    """Return full lattice squares with positive area inside one fire perimeter.

    Supply the perimeter CRS explicitly, for example from the fire manifest.
    Columns increase eastward and rows northward from EPSG:5070 (0, 0):
    col = floor(x / 90), row = floor(y / 90). Each cell extends from its
    indexed lower-left corner to the next lattice boundary in both axes.
    IDs are ``{fire_id}_r{grid_row}_c{grid_col}``, with signed integer indices.
    Output is ordered by ascending row, then column. No eligibility threshold
    is applied, and geometries are never clipped to the perimeter.
    """
    if not isinstance(fire_id, str) or not fire_id.strip():
        raise ValueError("fire_id must be a nonempty string.")
    if (
        not isinstance(perimeter, (Polygon, MultiPolygon))
        or perimeter.is_empty
        or not perimeter.is_valid
    ):
        raise ValueError("perimeter must be a valid, nonempty Polygon or MultiPolygon.")
    if perimeter_crs is None:
        raise ValueError("perimeter_crs is required.")

    projected = gpd.GeoSeries([perimeter], crs=perimeter_crs).to_crs(GRID_CRS).iloc[0]
    if (
        projected.is_empty
        or not projected.is_valid
        or not all(isfinite(value) for value in projected.bounds)
    ):
        raise ValueError("perimeter cannot be represented as a valid EPSG:5070 polygon.")

    min_x, min_y, max_x, max_y = projected.bounds
    patch_area = PATCH_SIZE_M**2
    records = []
    for row in range(floor(min_y / PATCH_SIZE_M), ceil(max_y / PATCH_SIZE_M)):
        for col in range(floor(min_x / PATCH_SIZE_M), ceil(max_x / PATCH_SIZE_M)):
            cell = box(
                col * PATCH_SIZE_M,
                row * PATCH_SIZE_M,
                (col + 1) * PATCH_SIZE_M,
                (row + 1) * PATCH_SIZE_M,
            )
            intersection_area = cell.intersection(projected).area
            if intersection_area > 0:
                records.append(
                    {
                        "patch_id": f"{fire_id}_r{row}_c{col}",
                        "fire_id": fire_id,
                        "grid_row": row,
                        "grid_col": col,
                        "patch_area_m2": patch_area,
                        "fire_overlap_fraction": intersection_area / patch_area,
                        "geometry": cell,
                    }
                )

    return gpd.GeoDataFrame(
        records,
        columns=[
            "patch_id", "fire_id", "grid_row", "grid_col", "patch_area_m2",
            "fire_overlap_fraction", "geometry",
        ],
        geometry="geometry",
        crs=GRID_CRS,
    )
