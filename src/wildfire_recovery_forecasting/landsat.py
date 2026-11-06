"""Landsat seasonal NBR predictors, targets, and coverage QA."""

from datetime import date, timedelta
from numbers import Integral

import ee

from wildfire_recovery_forecasting.config import (
    GROWING_SEASON_END_MONTH_DAY,
    GROWING_SEASON_START_MONTH_DAY,
    LANDSAT_MIN_VALID_ACQUISITION_DATES,
    NBR_PRE_REQUIRED_USABLE_SEASONS,
    NBR_PRE_SEARCH_SEASONS,
)


def sensor_for_year(year: int) -> str:
    """Use Landsat 7 before 2013 and Landsat 8 for 2013+ growing seasons."""
    return "L7" if year < 2013 else "L8"


def preprocess_landsat(image: ee.Image, sensor: str) -> ee.Image:
    """Apply C2 L2 QA and scaling, returning common NIR/SWIR2 bands.

    QA definitions: developers.google.com/earth-engine/datasets/catalog/
    LANDSAT_LE07_C02_T1_L2 and LANDSAT_LC08_C02_T1_L2.
    Preserve Phase 2's lack of an aerosol exclusion or reflectance clamp.
    """
    if sensor == "L7":
        bands = ["SR_B4", "SR_B7"]
        # Fill, dilated cloud, cloud, shadow, snow; bit 2 is unused.
        pixel_bits = (1 << 0) | (1 << 1) | (1 << 3) | (1 << 4) | (1 << 5)
        # NIR/SWIR2 saturation and dropped pixels; no terrain-occlusion bit.
        saturation_bits = (1 << 3) | (1 << 6) | (1 << 9)
    elif sensor == "L8":
        bands = ["SR_B5", "SR_B7"]
        pixel_bits = 0b111111  # Phase 2 bits 0-5, including cirrus.
        saturation_bits = (1 << 4) | (1 << 6) | (1 << 11)
    else:
        raise ValueError("sensor must be 'L7' or 'L8'")

    quality = image.select("QA_PIXEL").bitwiseAnd(pixel_bits).eq(0).And(
        image.select("QA_RADSAT").bitwiseAnd(saturation_bits).eq(0)
    )
    reflectance = image.select(bands).multiply(0.0000275).add(-0.2)
    return ee.Image(
        reflectance.rename(["NIR", "SWIR2"]).updateMask(quality)
        .copyProperties(image, image.propertyNames())
    )


def calculate_nbr(image: ee.Image) -> ee.Image:
    """Calculate NBR with zero denominators masked, retaining scene metadata."""
    nir, swir2 = image.select("NIR"), image.select("SWIR2")
    denominator = nir.add(swir2)
    nbr = nir.subtract(swir2).divide(denominator).rename("NBR")
    return ee.Image(
        nbr.updateMask(denominator.neq(0)).copyProperties(image, image.propertyNames())
    )


def build_seasonal_nbr(
    year: int, geometry: ee.Geometry, *, reference_projection: ee.Projection,
) -> dict:
    """Build lazy June-September NBR and valid unique-date count images.

    The caller initializes EE and supplies a deliberate reference projection
    (e.g. a source UTM projection). Composite defaults use that projection at
    30 m, avoiding EE's generic degree-grid fallback. Later materialization
    must also specify its CRS/transform or scale explicitly. No reprojection
    is forced and no export or server evaluation is launched here.

    Return Phase 2 keys: nbr (NBR), obs_count (valid_obs_count), scene_count,
    date_count. The latter two are scene-level metadata, not per-pixel QA.
    Pixels with no valid dates have masked NBR and count zero within geometry;
    an entirely empty season has the same band schema. No usability threshold
    is applied here. Negative reflectance/NBR values remain eligible after QA.
    """
    if reference_projection is None:
        raise ValueError("An explicit reference_projection is required")
    start = date(year, *GROWING_SEASON_START_MONTH_DAY).isoformat()
    end = (date(year, *GROWING_SEASON_END_MONTH_DAY) + timedelta(days=1)).isoformat()
    sensor = sensor_for_year(year)
    collection = "LANDSAT/LE07/C02/T1_L2" if sensor == "L7" else "LANDSAT/LC08/C02/T1_L2"
    scenes = ee.ImageCollection(collection).filterBounds(geometry).filterDate(start, end)
    nbr_scenes = scenes.map(lambda image: preprocess_landsat(image, sensor)).map(calculate_nbr)
    dates = (
        ee.List(nbr_scenes.aggregate_array("system:time_start"))
        .map(lambda timestamp: ee.Date(timestamp).format("YYYY-MM-dd"))
        .distinct().sort()
    )
    projection = reference_projection.atScale(30)

    def combine_date(date_string):
        acquisition = ee.Date.parse("YYYY-MM-dd", ee.String(date_string))
        same_day = nbr_scenes.filterDate(acquisition, acquisition.advance(1, "day"))
        return same_day.median().rename("NBR").setDefaultProjection(projection).set({
            "date": acquisition.format("YYYY-MM-dd"),
            "system:time_start": acquisition.millis(),
            "source_scene_count": same_day.size(),
        })

    daily = ee.ImageCollection.fromImages(dates.map(combine_date))
    # A fully masked seed supplies a band schema for empty collections without
    # contributing an observation or replacing missing NBR with zero.
    empty = ee.Image.constant(0).rename("NBR").updateMask(ee.Image.constant(0))
    empty = empty.setDefaultProjection(projection)
    daily_with_schema = daily.merge(ee.ImageCollection.fromImages([empty]))
    seasonal = daily_with_schema.median().rename("NBR").setDefaultProjection(projection)
    count = (
        daily_with_schema.count().rename("valid_obs_count")
        .unmask(0, sameFootprint=False).setDefaultProjection(projection).clip(geometry)
    )
    return {
        "nbr": seasonal,
        "obs_count": count,
        "scene_count": scenes.size(),
        "date_count": daily.size(),
    }


def build_fire_inputs(
    fire, geometry: ee.Geometry, *, reference_projection: ee.Projection,
) -> dict[str, ee.Image]:
    """Build prefire/Season-1 images and their existing QA only; no Season 5.

    Uses authoritative manifest search years, including late-ignition windows.
    Caller supplies computation geometry and the original batch reference grid.
    This constructs lazy graphs only; it does not initialize EE or evaluate them.
    nbr_pre_valid is 0/1; selected-year count remains 0-3 even where prefire
    NBR is masked. Season-1 count is valid unique dates, without a new threshold.
    Only NBR_pre and NBR_1 are scientific image inputs; the other outputs are QA.
    """
    fields = (
        "prefire_search_start_year", "prefire_search_end_year",
        "season1_year",
    )
    for field in fields:
        if field not in fire or isinstance(fire[field], bool) or not isinstance(fire[field], Integral):
            raise ValueError(f"Manifest requires an integer {field}")
    start, end = int(fire[fields[0]]), int(fire[fields[1]])
    if end - start + 1 != NBR_PRE_SEARCH_SEASONS:
        raise ValueError("Manifest prefire search pool must contain five consecutive years")
    if reference_projection is None:
        raise ValueError("An explicit reference_projection is required")
    projection = reference_projection.atScale(30)
    selected_count = (
        ee.Image.constant(0).rename("prefire_selected_year_count")
        .setDefaultProjection(projection).clip(geometry)
    )
    selected_seasons = []
    for year in range(end, start - 1, -1):
        season = build_seasonal_nbr(year, geometry, reference_projection=reference_projection)
        # Unmask QA only. A missing annual NBR must never be counted as usable.
        usable = (
            season["obs_count"].gte(LANDSAT_MIN_VALID_ACQUISITION_DATES)
            .And(season["nbr"].mask()).unmask(0, sameFootprint=False)
        )
        take = selected_count.lt(NBR_PRE_REQUIRED_USABLE_SEASONS).And(usable)
        selected_seasons.append(season["nbr"].updateMask(take))
        selected_count = selected_count.add(take)

    valid = selected_count.eq(NBR_PRE_REQUIRED_USABLE_SEASONS).rename("nbr_pre_valid")
    nbr_pre = (
        ee.ImageCollection.fromImages(selected_seasons).median()
        .updateMask(valid).rename("NBR_pre").setDefaultProjection(projection)
    )
    season1 = build_seasonal_nbr(int(fire["season1_year"]), geometry, reference_projection=reference_projection)
    return {
        "nbr_pre": nbr_pre,
        "nbr_1": season1["nbr"].rename("NBR_1"),
        "nbr_pre_valid": valid,
        "prefire_selected_year_count": selected_count,
        "season1_obs_count": season1["obs_count"].rename("season1_obs_count"),
    }


def build_fire_nbr(
    fire, geometry: ee.Geometry, *, reference_projection: ee.Projection,
) -> dict[str, ee.Image]:
    """Existing production inputs plus Season-5 target/QA, with unchanged science."""
    if ("season5_year" not in fire or isinstance(fire["season5_year"], bool)
            or not isinstance(fire["season5_year"], Integral)):
        raise ValueError("Manifest requires an integer season5_year")
    result = build_fire_inputs(fire, geometry, reference_projection=reference_projection)
    season5 = build_seasonal_nbr(
        int(fire["season5_year"]), geometry, reference_projection=reference_projection,
    )
    return {**result, "nbr_5": season5["nbr"].rename("NBR_5"),
            "season5_obs_count": season5["obs_count"].rename("season5_obs_count")}
