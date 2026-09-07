"""Independent source-to-vector reductions and small local table boundaries.

Raster sums use pixel area and Earth Engine's fractional boundary weights.
Measurements are never unmasked to zero. Only absent valid-area sums become
zero. Each source grid is explicit; no 90 m intermediate raster is created.
Whole-patch raster fractions use a full-area reference integrated on that
same grid, so boundary-pixel weights affect numerator and denominator alike.
"""

from datetime import date

import ee
import numpy as np
import pandas as pd

from wildfire_recovery_forecasting import daymet
from wildfire_recovery_forecasting.config import GRID_CRS, PATCH_SIZE_M
from wildfire_recovery_forecasting.validation import validate_patch_table, validate_measurements

PATCH_AREA = PATCH_SIZE_M ** 2
RASTER_FULL_AREA = "raster_full_area_m2"
MTBS_SEVERITY_SOURCE = "USFS/GTAC/MTBS/annual_burn_severity_mosaics/v1"
SEVERITY_CLASSES = {
    "unburned_low": 1, "low": 2, "moderate": 3, "high": 4,
    "increased_greenness": 5, "nonmapping": 6,
}


def fetch_patch_table(collection, columns):
    """Evaluate one source table, retaining null properties as explicit columns."""
    table = ee.data.computeFeatures({
        "expression": collection.map(lambda feature: feature.setGeometry(None)),
        "fileFormat": "PANDAS_DATAFRAME",
    })
    validate_patch_table(table)
    return table.reindex(columns=["patch_id", "fire_id", *columns])


def area_sums(images, patches, projection):
    """Retrieve variable sums and an unmasked full-area reference in one reduction."""
    bands = [ee.Image.pixelArea().rename(RASTER_FULL_AREA)]
    for name, image in images.items():
        area = ee.Image.pixelArea().updateMask(image.mask())
        bands.extend([image.multiply(area).rename(name + "_sum"), area.rename(name + "_area")])
    reduced = ee.Image.cat(bands).reduceRegions(
        collection=patches, reducer=ee.Reducer.sum(),
        crs=projection, tileScale=4,
    )
    return fetch_patch_table(reduced, [RASTER_FULL_AREA] + [
        name + suffix for name in images for suffix in ("_sum", "_area")])


def summarize_area_sums(raw, names, *, support=None, fractions=()):
    """Valid-area means or whole-patch fractions, preserving unobserved values.

    support names an already reduced binary support image (e.g. forest).
    Its observed area sum is the denominator for subset validity. A missing
    or zero support denominator yields zero validity and missing measurement.
    Whole-patch raster fractions use the same-grid full-area reference, not
    the exact vector area. The reference remains internal to the reduction.
    """
    result = raw[["patch_id", "fire_id"]].copy()
    full_area = pd.to_numeric(raw[RASTER_FULL_AREA])
    if not (np.isfinite(full_area) & full_area.gt(0)).all():
        raise ValueError("Rasterized full-patch area must be finite and positive")
    denominator = raw[support + "_sum"].fillna(0) if support else full_area
    for name in names:
        area = pd.to_numeric(raw[name + "_area"]).fillna(0)
        total = pd.to_numeric(raw[name + "_sum"])
        if name in fractions:
            result[name + "_fraction"] = (total / full_area).where(area.gt(0))
        else:
            result[name + "_mean"] = total.div(area.where(area.gt(0)))
        if support:
            result[name + "_valid_fraction"] = area.div(denominator.where(denominator.gt(0))).fillna(0)
        else:
            result[name + "_valid_fraction"] = area / denominator
    return result


def reduce_landsat(images, patches, *, expected, reference_projection):
    """Retrieve five logical products independently and join exact patch populations.

    Images retain their pixelwise definitions, including NBR drop and masks.
    expected is the local input patch table; no first returned subtable is
    allowed to redefine the population. The shared 30 m reference is explicit.
    """
    fire_id = expected.fire_id.iloc[0]
    validate_patch_table(expected, fire_id=fire_id)
    names = ["nbr_pre", "nbr_1", "nbr_5", "nbr_drop",
             "prefire_selected_year_count", "season1_obs_count", "season5_obs_count"]
    groups = [("nbr_pre", "prefire_selected_year_count"),
              ("nbr_1", "season1_obs_count"), ("nbr_5",),
              ("season5_obs_count",), ("nbr_drop",)]
    result = pd.DataFrame(expected[["patch_id", "fire_id"]]).copy()
    for group in groups:
        raw = area_sums({name: images[name] for name in group}, patches, reference_projection.atScale(30))
        part = summarize_area_sums(raw, group)
        validate_patch_table(part, expected_ids=expected.patch_id, fire_id=fire_id)
        validate_measurements(part)
        result = result.merge(part.drop(columns="fire_id"), on="patch_id",
                              how="left", validate="one_to_one", sort=False)
    validate_patch_table(result, expected_ids=expected.patch_id, fire_id=fire_id)
    return result[["patch_id", "fire_id"] + [
        name + suffix for name in names for suffix in ("_mean", "_valid_fraction")]]


def dominant_category(groups):
    """Return the greatest-area categorical code and its area; ties use lower code."""
    if not isinstance(groups, list) or not groups:
        return np.nan, np.nan
    valid = [group for group in groups if group["sum"] > 0]
    if not valid:
        return np.nan, np.nan
    winner = min(valid, key=lambda group: (-group["sum"], group["class"]))
    return int(winner["class"]), float(winner["sum"])


def reduce_landfire(images, patches):
    """EVC forest area, valid-forest cover/height, and categorical EVT area mode.

    EVT uses its own grid; its dominance denominator is forest support on that
    same grid. EVT/EVC coverage uses whole-patch area. Cover/height coverage
    uses forest area on each measurement's grid, never nonforest zero values.
    """
    forest = images["forest_indicator"]
    raw = area_sums({"forest": forest}, patches, forest.projection())
    result = summarize_area_sums(raw, ["forest"], fractions=["forest"]).rename(
        columns={"forest_valid_fraction": "evc_valid_fraction"})
    for name, image in (("tree_cover", images["tree_cover_pct"]),
                        ("tree_height", images["tree_height_m"])):
        raw = area_sums({name: image, "support": forest}, patches, image.projection())
        part = summarize_area_sums(raw, [name], support="support")
        result = result.merge(part, on=["patch_id", "fire_id"], validate="one_to_one")
    evt = images["evt"]
    projection = evt.projection()
    raw = area_sums({"evt_coverage": evt.mask(), "support": forest}, patches, projection)
    area_by_class = ee.Image.cat([
        ee.Image.pixelArea().updateMask(evt.mask()).updateMask(images["forest_mask"]).rename("area"),
        evt.rename("class"),
    ]).reduceRegions(
        collection=patches, reducer=ee.Reducer.sum().group(groupField=1, groupName="class"),
        crs=projection, tileScale=4,
    )
    categories = fetch_patch_table(area_by_class, ["groups"])
    raw = raw.merge(categories, on=["patch_id", "fire_id"], validate="one_to_one")
    winners = raw.groups.map(dominant_category)
    part = raw[["patch_id", "fire_id"]].copy()
    part["dominant_evt"] = pd.array([value[0] for value in winners], dtype="Int64")
    part["dominant_evt_fraction"] = pd.Series([value[1] for value in winners]).div(
        raw.support_sum.where(raw.support_sum.gt(0)))
    part["evt_valid_fraction"] = summarize_area_sums(
        raw, ["evt_coverage"], fractions=["evt_coverage"]
    )["evt_coverage_fraction"].fillna(0)
    return result.merge(part, on=["patch_id", "fire_id"], validate="one_to_one")


def reduce_terrain(images, forest, patches):
    """Forest-supported native 3DEP means and per-variable coverage."""
    names = ["elevation", "slope", "northness", "eastness"]
    values = {name: images[name].updateMask(forest) for name in names}
    raw = area_sums({**values, "support": forest}, patches, images["elevation"].projection())
    result = summarize_area_sums(raw, names, support="support")
    return result.rename(columns={"northness_valid_fraction": "aspect_valid_fraction"}).drop(
        columns="eastness_valid_fraction")


def reduce_daymet(images, patches):
    """Associate native ~1 km temporal surfaces using Phase 2's 30 m area sampling.

    This samples the piecewise native climate surface geographically; it does
    not create 90 m climate measurements. No reproject/resample is applied to
    the images. QA retains per-feature spatial coverage and each window's
    minimum daily coverage across historical years (per variable).
    """
    names = daymet.HISTORICAL_BANDS + daymet.PRE_S1_BANDS + daymet.PRE_S1_ANOMALY_BANDS + daymet.S1_ANOMALY_BANDS
    values = {name: images["primary"].select(name) for name in names}
    # Explicit geographic association grid, after all native temporal aggregation.
    projection = ee.Projection(GRID_CRS).atScale(30)
    raw = area_sums(values, patches, projection)
    result = summarize_area_sums(raw, names).rename(columns={name + "_mean": name for name in names})
    qa = {}
    for period in ("pre_s1", "s1"):
        history = images["historical_" + period]
        normalized = []
        for year, counts in history["valid_days_by_year"].items():
            start, end = daymet.climate_window(year, period)
            days = (date.fromisoformat(end) - date.fromisoformat(start)).days
            # Year-specific counts/divisors must share a type before collection min.
            normalized.append(counts.divide(days).toDouble())
        minimum = ee.ImageCollection.fromImages(normalized).min().setDefaultProjection(
            history["image"].projection())
        for band in daymet.SUMMARY_BANDS:
            qa[f"daymet_hist_{period}_{band}_daily_coverage"] = minimum.select(band)
            qa[f"daymet_{period}_{band}_valid_days"] = images[period]["valid_days"].select(band)
    raw = area_sums(qa, patches, projection)
    part = summarize_area_sums(raw, qa)
    # QA masks describe coverage counts themselves, not additional climate features.
    part = part.drop(columns=[name + "_valid_fraction" for name in qa])
    return result.merge(part, on=["patch_id", "fire_id"], validate="one_to_one")


def reduce_wildfire_history(history, patches):
    """Intersect already-unioned vectors; zero observed overlap is not completeness."""
    if patches.crs != history["union_crs"]:
        raise ValueError("MTBS unions and canonical patches must share EPSG:5070")
    result = pd.DataFrame(patches[["patch_id", "fire_id"]])
    for name, window in history["windows"].items():
        result[name] = patches.geometry.intersection(window["union_geometry"]).area.to_numpy() / PATCH_AREA
        result[name + "_coverage_status"] = window["coverage_status"]
    return result


def reduce_lcms(history, patches):
    """Reduce observed union areas, first annual detections, and coverage per window."""
    parts = []
    for window, images in history["windows"].items():
        prefix = "lcms_" + window + "_"
        projection = images["valid_years"].projection()
        categories = ["clear_disturbance", "other_loss"]
        names = categories + ["complete_coverage", "valid_years"]
        raw = area_sums({name: images[name] for name in names}, patches, projection)
        part = summarize_area_sums(raw, names, fractions=categories + ["complete_coverage"])
        part = part.rename(columns={name: prefix + name for name in part if name not in ("patch_id", "fire_id")})
        timing = ee.Image.cat([images[name + "_first_year"] for name in categories]).reduceRegions(
            collection=patches, reducer=ee.Reducer.min(), crs=projection, tileScale=4,
        )
        columns = [name + "_first_year" for name in categories]
        dates = fetch_patch_table(timing, columns).rename(columns={name: prefix + name for name in columns})
        part = part.merge(dates, on=["patch_id", "fire_id"], validate="one_to_one")
        part[prefix + "expected_years"] = images["expected_years"]
        parts.append(part)
    result = parts[0]
    for part in parts[1:]:
        result = result.merge(part, on=["patch_id", "fire_id"], validate="one_to_one")
    result["lcms_timing_resolution"] = history["timing_resolution"]
    return result


def reduce_focal_severity(fire, patches):
    """Phase 2 annual CONUS severity mosaic: classes 1..6, full-patch fractions."""
    severity = ee.Image(ee.ImageCollection(MTBS_SEVERITY_SOURCE).filter(
        ee.Filter.eq("system:index", f"mtbs_CONUS_{fire['fire_year']}" )).first()).select("Severity")
    known = severity.remap(list(range(1, 7)), list(range(1, 7)))
    values = {"mtbs_" + name: known.eq(code) for name, code in SEVERITY_CLASSES.items()}
    values["mtbs_valid"] = known.lte(5)
    values["mtbs_meaningful_burn"] = known.gte(2).And(known.lte(4))
    raw = area_sums(values, patches, severity.projection())
    result = summarize_area_sums(raw, values, fractions=values)
    # Every class shares the same known-code source support.
    coverage = result.pop("mtbs_valid_valid_fraction")
    result = result.drop(columns=[name for name in result if name.endswith("_valid_fraction") and name != "mtbs_valid_fraction"])
    result["mtbs_source_valid_fraction"] = coverage
    result["mtbs_valid_fraction"] = result["mtbs_valid_fraction"].fillna(0)
    return result
