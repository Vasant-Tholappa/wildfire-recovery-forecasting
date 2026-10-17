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
LANDSAT_PRODUCTS = {
    "nbrpre": ("nbr_pre", "prefire_selected_year_count"),
    "season1": ("nbr_1", "season1_obs_count"),
    "nbr5": ("nbr_5",),
    "season5_count": ("season5_obs_count",),
    "nbrdrop": ("nbr_drop",),
}
MTBS_SEVERITY_SOURCE = "USFS/GTAC/MTBS/annual_burn_severity_mosaics/v1"
SEVERITY_CLASSES = {
    "unburned_low": 1, "low": 2, "moderate": 3, "high": 4,
    "increased_greenness": 5, "nonmapping": 6,
}


def without_geometry(collection):
    """Keep table properties without transferring patch polygons."""
    return collection.map(lambda feature: feature.setGeometry(None))


def fetch_patch_table(collection, columns):
    """Evaluate one source table, retaining null properties as explicit columns."""
    table = ee.data.computeFeatures({
        "expression": without_geometry(collection),
        "fileFormat": "PANDAS_DATAFRAME",
    })
    validate_patch_table(table)
    return table.reindex(columns=["patch_id", "fire_id", *columns])


def area_sums(images, patches, projection):
    """Retrieve variable sums and an unmasked full-area reference in one reduction."""
    reduced = build_area_sums(images, patches, projection)
    return fetch_patch_table(reduced, [RASTER_FULL_AREA] + [
        name + suffix for name in images for suffix in ("_sum", "_area")])


def build_area_sums(images, patches, projection):
    """Build the same raw reduction FeatureCollection without evaluating rows."""
    bands = [ee.Image.pixelArea().rename(RASTER_FULL_AREA)]
    for name, image in images.items():
        area = ee.Image.pixelArea().updateMask(image.mask())
        bands.extend([image.multiply(area).rename(name + "_sum"), area.rename(name + "_area")])
    return ee.Image.cat(bands).reduceRegions(
        collection=patches, reducer=ee.Reducer.sum(),
        crs=projection, tileScale=4,
    )


def build_area_product(images, patches, projection):
    """Describe one export-ready area reduction; identifiers are supplied by patches."""
    return {"collection": build_area_sums(images, patches, projection),
            "columns": [RASTER_FULL_AREA] + [
                name + suffix for name in images for suffix in ("_sum", "_area")]}


def fetch_product(product):
    """Use the synchronous boundary for an otherwise export-ready product."""
    return fetch_patch_table(product["collection"], product["columns"])


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


def build_landsat_products(images, patches, *, reference_projection):
    """Return five independent raw collections/columns, without fetching or joining.

    images comes from landsat.build_fire_nbr using the requested exact patch
    union; patches is that same EE patch collection. Pass the shared reference
    projection used to build those images. Columns exclude the two identifier
    columns automatically included by the exporter.
    """
    return {
        product: {
            "collection": build_area_sums(
                {name: images[name] for name in group}, patches, reference_projection.atScale(30)),
            "columns": [RASTER_FULL_AREA] + [
                name + suffix for name in group for suffix in ("_sum", "_area")],
        }
        for product, group in LANDSAT_PRODUCTS.items()
    }


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
    result = pd.DataFrame(expected[["patch_id", "fire_id"]]).copy()
    for group in LANDSAT_PRODUCTS.values():
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


def build_landfire_products(images, patches):
    """Four native-grid products: combined EVC, height, EVT support and categories."""
    forest, evt = images["forest_indicator"], images["evt"]
    products = {
        "evc": build_area_product({"forest": forest, "tree_cover": images["tree_cover_pct"],
                                   "support": forest}, patches, forest.projection()),
        "height": build_area_product({"tree_height": images["tree_height_m"], "support": forest},
                                      patches, images["tree_height_m"].projection()),
        "evt_support": build_area_product({"evt_coverage": evt.mask(), "support": forest},
                                           patches, evt.projection()),
    }
    categories = ee.Image.cat([
        ee.Image.pixelArea().updateMask(evt.mask()).updateMask(images["forest_mask"]).rename("area"),
        evt.rename("class"),
    ]).reduceRegions(
        collection=patches, reducer=ee.Reducer.sum().group(groupField=1, groupName="class"),
        crs=evt.projection(), tileScale=4,
    )
    products["evt_categories"] = {"collection": categories, "columns": ["groups"]}
    return products


def reduce_landfire(images, patches):
    """Fetch Plan B products; preserve full-patch and valid-forest denominators."""
    products = build_landfire_products(images, patches)
    raw = fetch_product(products["evc"])
    result = summarize_area_sums(raw, ["forest"], fractions=["forest"]).rename(
        columns={"forest_valid_fraction": "evc_valid_fraction"})
    cover = summarize_area_sums(raw, ["tree_cover"], support="support")
    result = result.merge(cover, on=["patch_id", "fire_id"], validate="one_to_one")
    height = summarize_area_sums(fetch_product(products["height"]), ["tree_height"], support="support")
    result = result.merge(height, on=["patch_id", "fire_id"], validate="one_to_one")
    raw = fetch_product(products["evt_support"])
    categories = fetch_product(products["evt_categories"])
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


def build_terrain_product(images, forest, patches):
    """One unchanged native-3DEP reduction, ready for fetch or export."""
    values = {name: images[name].updateMask(forest)
              for name in ("elevation", "slope", "northness", "eastness")}
    return build_area_product({**values, "support": forest}, patches, images["elevation"].projection())


def reduce_terrain(images, forest, patches):
    """Forest-supported native 3DEP means and per-variable coverage."""
    names = ["elevation", "slope", "northness", "eastness"]
    raw = fetch_product(build_terrain_product(images, forest, patches))
    result = summarize_area_sums(raw, names, support="support")
    return result.rename(columns={"northness_valid_fraction": "aspect_valid_fraction"}).drop(
        columns="eastness_valid_fraction")


def daymet_area_inputs(images):
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
    return values, qa, projection


def build_daymet_products(images, patches):
    """Keep production primary and coverage as two independent reductions."""
    values, qa, projection = daymet_area_inputs(images)
    return {"primary": build_area_product(values, patches, projection),
            "coverage": build_area_product(qa, patches, projection)}


def reduce_daymet(images, patches):
    """Fetch the two unchanged Daymet products and retain existing climate/QA fields."""
    products = build_daymet_products(images, patches)
    names = daymet.HISTORICAL_BANDS + daymet.PRE_S1_BANDS + daymet.PRE_S1_ANOMALY_BANDS + daymet.S1_ANOMALY_BANDS
    result = summarize_area_sums(fetch_product(products["primary"]), names).rename(
        columns={name + "_mean": name for name in names})
    qa = [name.removesuffix("_sum") for name in products["coverage"]["columns"] if name.endswith("_sum")]
    part = summarize_area_sums(fetch_product(products["coverage"]), qa)
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


def build_lcms_product(history, patches):
    """One shared-grid sum with independent, window-prefixed masks and coverage."""
    values = {}
    expected = {}
    for window in ("pre_10yr", "post_y_to_y5"):
        images = history["windows"][window]
        expected[window] = images["expected_years"]
        for name in ("clear_disturbance", "other_loss", "complete_coverage", "valid_years"):
            values[f"lcms_{window}_{name}"] = images[name]
    # build_lcms_history gives both windows this same reference projection.
    projection = history["windows"]["pre_10yr"]["valid_years"].projection()
    product = build_area_product(values, patches, projection)
    product["expected_years"] = expected  # Local assembly metadata, not an extra raster.
    return product


def summarize_lcms_product(raw, expected_years):
    """Restore the existing window fields from the combined raw product locally."""
    parts = []
    for window, expected in expected_years.items():
        prefix = f"lcms_{window}_"
        categories = [prefix + name for name in ("clear_disturbance", "other_loss", "complete_coverage")]
        part = summarize_area_sums(raw, categories + [prefix + "valid_years"], fractions=categories)
        part[prefix + "expected_years"] = expected
        parts.append(part)
    return parts[0].merge(parts[1], on=["patch_id", "fire_id"], validate="one_to_one")


def reduce_lcms(history, patches):
    """Fetch one LCMS product; preserve both windows' existing analytical QA schema."""
    product = build_lcms_product(history, patches)
    return summarize_lcms_product(fetch_product(product), product["expected_years"])


def build_focal_severity_product(fire, patches):
    """Phase 2 annual CONUS severity mosaic: classes 1..6, full-patch fractions."""
    severity = ee.Image(ee.ImageCollection(MTBS_SEVERITY_SOURCE).filter(
        ee.Filter.eq("system:index", f"mtbs_CONUS_{fire['fire_year']}" )).first()).select("Severity")
    known = severity.remap(list(range(1, 7)), list(range(1, 7)))
    values = {"mtbs_" + name: known.eq(code) for name, code in SEVERITY_CLASSES.items()}
    values["mtbs_valid"] = known.lte(5)
    values["mtbs_meaningful_burn"] = known.gte(2).And(known.lte(4))
    return build_area_product(values, patches, severity.projection())


def reduce_focal_severity(fire, patches):
    """Fetch the unchanged severity product and summarize class/coverage fractions."""
    product = build_focal_severity_product(fire, patches)
    values = [name.removesuffix("_sum") for name in product["columns"] if name.endswith("_sum")]
    raw = fetch_product(product)
    result = summarize_area_sums(raw, values, fractions=values)
    # Every class shares the same known-code source support.
    coverage = result.pop("mtbs_valid_valid_fraction")
    result = result.drop(columns=[name for name in result if name.endswith("_valid_fraction") and name != "mtbs_valid_fraction"])
    result["mtbs_source_valid_fraction"] = coverage
    result["mtbs_valid_fraction"] = result["mtbs_valid_fraction"].fillna(0)
    return result
