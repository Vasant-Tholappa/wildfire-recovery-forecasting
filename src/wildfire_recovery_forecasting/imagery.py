"""Local Phase 6 grid/spec preparation. No EE initialization or export calls."""
from numbers import Integral

from affine import Affine
from rasterio.windows import Window

from wildfire_recovery_forecasting.config import GRID_CRS, PATCH_SIZE_M

PIXEL_SIZE = 30
CHIP_SIDE = 15
CONTEXT_BUFFER = 180


def patch_center(grid_row, grid_col):
    """Return the canonical full-square center in EPSG:5070 metres."""
    if any(isinstance(v, bool) or not isinstance(v, Integral)
           for v in (grid_row, grid_col)):
        raise ValueError("Grid indices must be integers")
    return (PATCH_SIZE_M * (grid_col + 0.5), PATCH_SIZE_M * (grid_row + 0.5))


def prepare_region(eligible_rows, *, fire_id, batch_id):
    """Prepare one frozen tile's eligible-key rectangle, not the fire perimeter.

    Caller obtains these rows from the existing eligible dataset and verifies
    membership in the frozen batch manifest. No dataset is read here. Context
    extends beyond fire/tile edges. Returns bounds, affine and raster dimensions.
    """
    required = {"fire_id", "patch_id", "grid_row", "grid_col", "batch_id"}
    if not required.issubset(eligible_rows.columns) or eligible_rows.empty:
        raise ValueError("Nonempty eligible identifiers/grid columns required")
    if eligible_rows[list(required)].isna().any().any() or not eligible_rows.patch_id.is_unique:
        raise ValueError("Missing or duplicate patch identifiers")
    for row in eligible_rows.itertuples():
        patch_center(row.grid_row, row.grid_col)
        if (row.fire_id != fire_id or row.batch_id != batch_id
                or batch_id != f"r{row.grid_row // 120}_c{row.grid_col // 120}"
                or row.patch_id != f"{fire_id}_r{row.grid_row}_c{row.grid_col}"):
            raise ValueError("Patch identity or frozen tile mismatch")
    # Canonical 90 m bounds and 180 m buffer are already exact multiples of 30.
    xmin = int(eligible_rows.grid_col.min()) * PATCH_SIZE_M - CONTEXT_BUFFER
    xmax = (int(eligible_rows.grid_col.max()) + 1) * PATCH_SIZE_M + CONTEXT_BUFFER
    ymin = int(eligible_rows.grid_row.min()) * PATCH_SIZE_M - CONTEXT_BUFFER
    ymax = (int(eligible_rows.grid_row.max()) + 1) * PATCH_SIZE_M + CONTEXT_BUFFER
    return {"crs": GRID_CRS, "bounds": (xmin, ymin, xmax, ymax),
            "transform": Affine(30, 0, xmin, 0, -30, ymax),
            "width": (xmax-xmin)//30, "height": (ymax-ymin)//30}


def chip_window(grid_row, grid_col, region):
    """Integer 15x15 Rasterio window; its [6:9,6:9] is the original patch."""
    x, y = patch_center(grid_row, grid_col)
    col, row = ~region["transform"] * (x, y)
    if col % 1 != 0.5 or row % 1 != 0.5:
        raise ValueError("Patch center is not aligned to a pixel center")
    left, top = int(col - 0.5) - 7, int(row - 0.5) - 7
    if min(left, top) < 0 or left+15 > region["width"] or top+15 > region["height"]:
        raise ValueError("Chip is outside export region")
    return Window(left, top, CHIP_SIDE, CHIP_SIDE)


def prepare_export_specs(region):
    """Pure specifications, deliberately NOT executable EE export arguments.

    Later authorized adapter must resolve reference scene/projection using the
    ORIGINAL exact batch union (all retained patches, not eligible-only/buffered
    bounds), matching pipeline.build_raster_products. Build inputs on the wider
    region using that reference. Materialize each NBR separately as float32 with
    an explicit validity band (1 valid/0 invalid), preserving independent masks.
    Fill invalid NBR with -9999 only at serialization, with GeoTIFF nodata=-9999;
    validity is authoritative (never confuse a valid zero with missing data).
    Nearest-neighbor output sampling is proposed, not verified by this helper.
    No image, bucket, path or resolved source projection is invented here.
    """
    xmin, ymin, xmax, ymax = region["bounds"]
    return {product: {
        "band": band, "crs": region["crs"],
        "crsTransform": list(region["transform"])[:6],
        "region_epsg5070": [xmin, ymin, xmax, ymax],
        "fileFormat": "GeoTIFF", "nodata": -9999,
        "validity_band": "valid", "sampling": "nearest (POC verification required)",
        "source_reference": "UNRESOLVED: original exact batch Season-1 L8 reference",
    } for product, band in (("nbrpre", "NBR_pre"), ("season1", "NBR_1"))}
