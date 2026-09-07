"""Independent source evaluation followed by local one-to-one table assembly.

EE must already be initialized. No exports or persistent writes occur here.
Use explicit patch IDs for small validation runs; population batching and
eligibility decisions are deliberately outside this module.
"""

from datetime import datetime, timezone
from numbers import Integral

import ee
import numpy as np
import pandas as pd

from wildfire_recovery_forecasting import (
    config, daymet, disturbance, landfire, landsat, reductions, terrain,
)
from wildfire_recovery_forecasting.grid import build_fire_grid
from wildfire_recovery_forecasting.validation import (
    join_source_tables, validate_dataset, validate_measurements,
    validate_patch_table, validate_patches,
)


def patches_to_ee(patches):
    """Keep full planar EPSG:5070 vector squares and identifiers in EE."""
    validate_patches(patches, patches.fire_id.iloc[0])
    return ee.FeatureCollection([
        ee.Feature(ee.Geometry.Polygon(
            [list(row.geometry.exterior.coords)], proj=config.GRID_CRS, geodesic=False,
        ), {"patch_id": row.patch_id, "fire_id": row.fire_id})
        for row in patches.itertuples()
    ])


def smoke_patch_ids(patches, count=24):
    """Select deterministic, spatially spread full and partial-overlap examples."""
    if count < 2 or count > len(patches):
        raise ValueError("Sample size must be between two and the candidate count")
    full = patches.loc[np.isclose(patches.fire_overlap_fraction, 1, rtol=0, atol=1e-9)]
    boundary = patches.loc[~patches.index.isin(full.index)]
    selected = []
    for group, size in ((full, count // 2), (boundary, count - count // 2)):
        group = group.sort_values(["grid_row", "grid_col"])
        if len(group) < size:
            raise ValueError("Insufficient full/boundary patches for the requested sample")
        selected.extend(group.iloc[np.linspace(0, len(group) - 1, size, dtype=int)].patch_id)
    return selected


def landsat_patch_table(fire, patches, *, batch_size=None, reference_projection=None):
    """Use exact patch unions and separate requests, with optional deterministic batches.

    None runs the full requested set. A caller-selected positive batch size
    sorts by patch_id before splitting; there is no automatic retry/shrinking.
    All batches share one reference projection, selected once for the request.
    Concatenation restores the original input order and checks global coverage.
    """
    validate_patches(patches, fire["fire_id"])
    if batch_size is not None and (isinstance(batch_size, bool)
                                  or not isinstance(batch_size, Integral) or batch_size <= 0):
        raise ValueError("batch_size must be a positive integer or None")
    requested_geometry = patches_to_ee(patches).geometry()
    if reference_projection is None:
        reference_projection = ee.Image(ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
            .filterBounds(requested_geometry)
            .filterDate(f"{fire['season1_year']}-06-01", f"{fire['season1_year']}-10-01")
            .sort("system:time_start").first()).select("SR_B5").projection()
    ordered = patches if batch_size is None else patches.sort_values("patch_id")
    size = len(ordered) if batch_size is None else int(batch_size)
    tables = []
    for start in range(0, len(ordered), size):
        batch = ordered.iloc[start:start + size]
        collection = patches_to_ee(batch)
        images = landsat.build_fire_nbr(fire, collection.geometry(),
                                        reference_projection=reference_projection)
        table = reductions.reduce_landsat(images, collection, expected=batch,
                                          reference_projection=reference_projection)
        validate_patch_table(table, expected_ids=batch.patch_id, fire_id=fire["fire_id"])
        validate_measurements(table)
        tables.append(table)
    result = pd.concat(tables, ignore_index=True)
    validate_patch_table(result, expected_ids=patches.patch_id, fire_id=fire["fire_id"])
    return result.set_index("patch_id").loc[patches.patch_id].reset_index()


def source_patch_tables(fire, patches, *, landsat_batch_size=None):
    """Yield independently retrieved tables; a source failure leaves earlier results usable.

    Vegetation's forest image is reused for terrain support, but its reduced
    table is evaluated independently. Landsat is evaluated last because its
    seasonal graphs are the most demanding small-run computations.
    """
    validate_patches(patches, fire["fire_id"])
    collection = patches_to_ee(patches)
    region = collection.geometry().bounds(maxError=1)

    def checked(name, table):
        validate_patch_table(table, expected_ids=patches.patch_id, fire_id=fire["fire_id"])
        validate_measurements(table)
        return name, table

    yield checked("mtbs_severity", reductions.reduce_focal_severity(fire, collection))
    vegetation = landfire.build_landfire_images(fire, region)
    yield checked("landfire", reductions.reduce_landfire(vegetation, collection))
    yield checked("terrain", reductions.reduce_terrain(
        terrain.build_terrain_images(region), vegetation["forest_mask"], collection))
    yield checked("daymet", reductions.reduce_daymet(daymet.build_fire_climate_images(fire, region), collection))
    yield checked("mtbs_history", reductions.reduce_wildfire_history(
        disturbance.build_wildfire_history(fire, region), patches))
    yield checked("lcms", reductions.reduce_lcms(disturbance.build_lcms_history(fire, region), collection))
    yield checked("landsat", landsat_patch_table(fire, patches, batch_size=landsat_batch_size))


def dataset_roles(table):
    """Explicit predictor/target/QA separation, including historical-only ablation."""
    historical = list(daymet.HISTORICAL_BANDS)
    static = ["nbr_pre_mean", "dominant_evt", "dominant_evt_fraction",
              "tree_cover_mean", "tree_height_mean", "elevation_mean", "slope_mean",
              "northness_mean", "eastness_mean", "fire_size_acres", "ignition_month"]
    severity = ["mtbs_" + name + "_fraction" for name in reductions.SEVERITY_CLASSES if name != "nonmapping"]
    early = static + severity + historical
    primary = early + ["nbr_1_mean", "nbr_drop_mean"] + daymet.PRE_S1_BANDS + daymet.PRE_S1_ANOMALY_BANDS + daymet.S1_ANOMALY_BANDS
    metadata = {"patch_id", "fire_id", "fire_name", "ignition_date", "fire_year", "split",
                "grid_row", "grid_col", "source_dataset", "landfire_condition_year", "landfire_version",
                "last_fully_prefire_growing_season_year", "prefire_search_start_year",
                "prefire_search_end_year", "season1_year", "season5_year"}
    return {
        "target": "nbr_5_mean", "predictors": primary, "early_predictors": early,
        "historical_climate": historical,
        "qa": sorted(set(table) - set(primary) - metadata - {"nbr_5_mean"}),
    }


def run_fire_pipeline(fire, *, perimeter_crs, patch_ids=None, on_source=None,
                      git_commit=None, manifest_version=None, landsat_batch_size=None):
    """Build a grid/subset, evaluate sources independently, and return a checked dataset.

    on_source(name, table) can display progress or keep successful local tables
    if a later source fails. Run metadata records unknown versions explicitly;
    this transient runner never reuses assets or writes persistent outputs.
    """
    patches = build_fire_grid(fire["geometry"], fire["fire_id"], perimeter_crs=perimeter_crs)
    if patch_ids is not None:
        requested = list(patch_ids)
        if len(set(requested)) != len(requested):
            raise ValueError("Duplicate requested patch IDs")
        patches = patches.loc[patches.patch_id.isin(requested)].reset_index(drop=True)
        validate_patch_table(patches, expected_ids=requested, fire_id=fire["fire_id"])
    validate_patches(patches, fire["fire_id"])
    tables = {}
    for name, table in source_patch_tables(fire, patches, landsat_batch_size=landsat_batch_size):
        tables[name] = table
        if on_source is not None:
            on_source(name, table)
    joined = join_source_tables(patches, tables, fire_id=fire["fire_id"])
    for name, value in fire.items():
        if name != "geometry" and name not in joined:
            joined[name] = value
    joined["ignition_month"] = fire["ignition_date"].month
    roles = dataset_roles(joined)
    validate_dataset(joined, roles, expected_ids=patches.patch_id, fire_id=fire["fire_id"])
    metadata = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(), "git_commit": git_commit,
        "manifest_version": manifest_version,
        "production_contract_version": config.PRODUCTION_CONTRACT_VERSION,
        "schema_version": config.SCHEMA_VERSION, "grid_version": config.GRID_VERSION,
        "grid_crs": config.GRID_CRS, "patch_size_m": config.PATCH_SIZE_M,
        "sources": {
            "landfire": landfire.landfire_sources(fire), "daymet": daymet.DAYMET_SOURCE,
            "lcms": disturbance.LCMS_SOURCE, "mtbs": disturbance.MTBS_SOURCE,
            "mtbs_severity": reductions.MTBS_SEVERITY_SOURCE,
            "terrain": "USGS/3DEP/10m_collection", "landsat": "Collection 2 Tier 1 Level 2; LE07/LC08",
        },
        "mtbs_history_completeness": "unverified; observed fractions do not establish archive completeness",
        "lcms_timing": "annual retrospective QA; within-year ordering is unknown",
        "daymet_association": "native ~1 km temporal surfaces, 30 m geographic area sampling",
    }
    return {"patches": patches, "source_tables": tables, "table": joined, "roles": roles, "metadata": metadata}
