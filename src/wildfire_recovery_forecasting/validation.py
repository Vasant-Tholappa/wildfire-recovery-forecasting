"""Structural and measurement checks, without population eligibility thresholds."""

import numpy as np
import pandas as pd

from wildfire_recovery_forecasting.config import GRID_CRS, PATCH_SIZE_M


def validate_patch_table(table, *, expected_ids=None, fire_id=None):
    """Reject missing/duplicate IDs, changed populations, and wrong fire IDs."""
    if "patch_id" not in table or not table.patch_id.map(
        lambda value: isinstance(value, str) and bool(value.strip())
    ).all():
        raise ValueError("patch_id must contain nonempty strings")
    if table.patch_id.duplicated().any():
        raise ValueError("Duplicate patch_id")
    if expected_ids is not None and set(table.patch_id) != set(expected_ids):
        raise ValueError("Source table does not preserve the requested patch IDs")
    if fire_id is not None and ("fire_id" not in table or not table.fire_id.eq(fire_id).all()):
        raise ValueError("Unexpected or missing fire_id")


def validate_patches(patches, fire_id):
    """Verify canonical patch CRS, IDs, and full square area before evaluation."""
    validate_patch_table(patches, fire_id=fire_id)
    if patches.empty or patches.crs is None or patches.crs.to_string() != GRID_CRS:
        raise ValueError("Require nonempty EPSG:5070 patches")
    if patches.geometry.isna().any() or not patches.geometry.is_valid.all():
        raise ValueError("Missing or invalid patch geometry")
    for areas in (patches.geometry.area, patches.patch_area_m2):
        if not np.allclose(areas, PATCH_SIZE_M ** 2, rtol=0, atol=1e-6):
            raise ValueError("Patch area must be 8100 m2")


def validate_measurements(table, *, tolerance=1e-6):
    """Check fraction bounds and zero-support missingness; retain observed zeros."""
    for name in table:
        if "fraction" in name and not name.endswith("_coverage_status"):
            values = pd.to_numeric(table[name], errors="raise").dropna()
            if not values.between(-tolerance, 1 + tolerance).all():
                raise ValueError(f"Fraction outside [0, 1]: {name}")
        if name.endswith("_valid_fraction"):
            base = name.removesuffix("_valid_fraction")
            measurements = [base, base + "_mean"]
            if base == "aspect":
                measurements = ["northness_mean", "eastness_mean"]
            for measurement in measurements:
                if measurement in table and table.loc[table[name].eq(0), measurement].notna().any():
                    raise ValueError(f"Measurement with zero valid support: {measurement}")
    if "forest_fraction" in table:
        unsupported = table.forest_fraction.fillna(0).eq(0)
        for name in ("tree_cover_mean", "tree_height_mean", "elevation_mean", "slope_mean",
                     "northness_mean", "eastness_mean", "dominant_evt"):
            if name in table and table.loc[unsupported, name].notna().any():
                raise ValueError(f"Forest-supported measurement without forest: {name}")


def join_source_tables(patches, source_tables, *, fire_id):
    """Join exact one-to-one populations in input order, rejecting column collisions."""
    validate_patches(patches, fire_id)
    result = pd.DataFrame(patches.drop(columns="geometry"))
    for source, table in source_tables.items():
        validate_patch_table(table, expected_ids=patches.patch_id, fire_id=fire_id)
        validate_measurements(table)
        values = table.drop(columns="fire_id")
        collisions = (set(result) & set(values)) - {"patch_id"}
        if collisions:
            raise ValueError(f"Overlapping fields from {source}: {sorted(collisions)}")
        result = result.merge(values, on="patch_id", how="left", validate="one_to_one", sort=False)
    validate_patch_table(result, expected_ids=patches.patch_id, fire_id=fire_id)
    validate_measurements(result)
    return result


def validate_dataset(table, roles, *, expected_ids, fire_id):
    """Ensure the target and ablation/retrospective QA contract remains explicit."""
    validate_patch_table(table, expected_ids=expected_ids, fire_id=fire_id)
    validate_measurements(table)
    if roles["target"] != "nbr_5_mean" or roles["target"] not in table:
        raise ValueError("NBR_5 target is missing or misidentified")
    primary, early, qa = map(set, (roles["predictors"], roles["early_predictors"], roles["qa"]))
    if not (primary | early | qa).issubset(table.columns):
        raise ValueError("Declared fields missing from dataset")
    if roles["target"] in primary | early or (primary | early) & qa:
        raise ValueError("Target/QA leaked into predictors")
    if any(name.startswith("lcms_") and name not in qa for name in table):
        raise ValueError("LCMS fields must remain retrospective QA")
    if any(name.startswith(("pre_s1_", "s1_", "nbr_1", "nbr_drop")) for name in early):
        raise ValueError("Post-fire observations leaked into the early ablation")
    from wildfire_recovery_forecasting.daymet import HISTORICAL_BANDS
    if not set(HISTORICAL_BANDS).issubset(early):
        raise ValueError("Historical-only climate separation is missing")
