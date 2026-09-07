"""MTBS wildfire-history and LCMS competing-disturbance QA."""

from datetime import date
from numbers import Real, Integral

import ee
import geopandas as gpd
import pandas as pd

from wildfire_recovery_forecasting.config import GRID_CRS, MTBS_PRIOR_FIRE_LOOKBACK_YEARS
from wildfire_recovery_forecasting.fire_catalog import MTBS_SOURCE, normalize_fire_geometry


def wildfire_history_windows(fire) -> dict:
    """UTC windows: prior [anniversary, ignition), reburn (ignition, Oct 1).

    Preserve the manifest's time of day. Calendar-year subtraction clamps
    February 29 to February 28 when its anniversary year is not a leap year.
    """
    ignition = fire.get("ignition_date")
    fire_id = fire.get("fire_id")
    season5 = fire.get("season5_year")
    if not isinstance(fire_id, str) or not fire_id.strip():
        raise ValueError("A nonempty manifest fire_id is required")
    if not isinstance(ignition, date) or pd.isna(ignition):
        raise ValueError("Manifest ignition_date must be a non-missing date/datetime")
    ignition = pd.Timestamp(ignition)
    ignition = ignition.tz_localize("UTC") if ignition.tz is None else ignition.tz_convert("UTC")
    if (not isinstance(season5, Integral) or isinstance(season5, bool)
            or season5 != ignition.year + 5):
        raise ValueError("Manifest season5_year must match ignition year + 5")
    windows = {
        f"prior_fire_fraction_{years}yr": {
            "start": ignition - pd.DateOffset(years=years), "end": ignition,
            "start_inclusive": True,
        } for years in MTBS_PRIOR_FIRE_LOOKBACK_YEARS
    }
    windows["reburn_fraction"] = {
        "start": ignition, "end": pd.Timestamp(f"{season5}-10-01", tz="UTC"),
        "start_inclusive": False,
    }
    return windows


def history_from_mtbs(fire, events: gpd.GeoDataFrame) -> dict:
    """Validate a spatially retrieved source subset and prepare window unions.

    Input uses raw MTBS fields and epoch-millisecond Ig_Date. Keep complete
    event perimeters in their original CRS; normalize only supported geometry
    collections using the catalog helper. Union projected polygon parts in
    EPSG:5070 so later patch intersections cannot double-count repeat burns.

    Empty unions describe zero *observed* overlap, not confirmed absence.
    Source completeness cannot be inferred from local events or their latest
    date. It remains unverified for every window, particularly recent reburn
    years. Retrieval/schema failures raise rather than becoming empty results.
    No eligibility rules or final patch fractions are applied here.
    """
    windows = wildfire_history_windows(fire)
    required = {"Event_ID", "Incid_Type", "Ig_Date"}
    if events.crs is None or not required.issubset(events.columns):
        raise ValueError("MTBS history requires CRS and Event_ID, Incid_Type, Ig_Date")
    if events["Incid_Type"].isna().any():
        raise ValueError("Missing MTBS incident type requires source review")
    wildfires = events.loc[events.Incid_Type.eq("Wildfire")].copy()
    if not wildfires.Event_ID.map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError("Missing MTBS Event_ID requires source review")
    wildfires = wildfires.loc[wildfires.Event_ID.ne(fire["fire_id"])].copy()
    if not wildfires.Ig_Date.map(
        lambda value: isinstance(value, Real) and not isinstance(value, bool) and pd.notna(value)
    ).all():
        raise ValueError("MTBS Ig_Date requires non-missing numeric epoch milliseconds")
    dates = pd.to_datetime(wildfires.Ig_Date, unit="ms", utc=True, errors="raise")
    if not dates.dt.year.between(1984, 2100).all():
        raise ValueError("Implausible MTBS date; verify epoch-millisecond units")
    result = {}
    for name, window in windows.items():
        lower = dates.ge(window["start"]) if window["start_inclusive"] else dates.gt(window["start"])
        selected = wildfires.loc[lower & dates.lt(window["end"])].copy()
        selected.geometry = selected.geometry.map(normalize_fire_geometry)
        projected = selected.to_crs(GRID_CRS)
        result[name] = {
            **window, "events": selected, "union_geometry": projected.geometry.union_all(),
            "event_count": len(selected), "coverage_status": "unverified",
        }
    return {"source_dataset": MTBS_SOURCE, "union_crs": GRID_CRS, "windows": result}


def build_wildfire_history(fire, search_geometry: ee.Geometry) -> dict:
    """Retrieve only nearby MTBS vectors, then prepare history for one focal fire.

    search_geometry MUST cover full candidate patches, not just the focal
    perimeter. Nearby events outside that perimeter can overlap boundary
    patches. Fetch all dates/types locally so missing source fields are not
    silently discarded by server-side temporal filters. Do not pass the
    focal population (2015–2020) as history: older/later events are required.
    EE must already be initialized. No raster queries or writes occur.
    """
    wildfire_history_windows(fire)  # Fail before retrieval on bad manifest input.
    source = ee.FeatureCollection(MTBS_SOURCE).filterBounds(search_geometry)
    records = ee.data.computeFeatures({"expression": source, "fileFormat": "GEOPANDAS_GEODATAFRAME"})
    # EE's converter omits CRS, and an empty result may also omit properties.
    if records.empty:
        records = gpd.GeoDataFrame(columns=["Event_ID", "Incid_Type", "Ig_Date", "geometry"],
                                   geometry="geometry", crs="EPSG:4326")
    else:
        records = records.set_crs("EPSG:4326")
    return history_from_mtbs(fire, records)


LCMS_SOURCE = "projects/gtac-data-publish/assets/LCMS/Product_Version/2025-11"
LCMS_CLEAR_CODES = (1, 2, 5, 6, 8, 9, 10, 11)
LCMS_OTHER_LOSS_CODES = (13,)


def lcms_year_windows(fire) -> dict[str, list[int]]:
    """Annual retrospective windows, not exact ignition/cutoff timestamps."""
    year, season5 = fire.get("fire_year"), fire.get("season5_year")
    if (not isinstance(year, Integral) or isinstance(year, bool)
            or not 2015 <= year <= 2020 or season5 != year + 5):
        raise ValueError("Require manifest fire_year 2015–2020 and matching season5_year")
    return {
        **{f"pre_{n}yr": list(range(year - n, year)) for n in (5, 7, 10)},
        "post_y_to_y5": list(range(year, season5 + 1)),
    }


def lcms_change_flags(change: ee.Image) -> dict[str, ee.Image]:
    """Phase 2 classes: known 1–15 valid; 16/unknown/masked pixels unavailable."""
    valid_change = change.remap(list(range(1, 16)), list(range(1, 16)))
    return {
        name: valid_change.remap(list(codes), [1] * len(codes), 0).rename(name)
        for name, codes in (("clear_disturbance", LCMS_CLEAR_CODES),
                            ("other_loss", LCMS_OTHER_LOSS_CODES))
    }


def summarize_lcms_window(annual: dict, years: list[int], projection: ee.Projection) -> dict:
    """Observed spatial unions and first observed years, with explicit coverage.

    A union is 1 if detected in any observed year; 0 means no observed detection,
    and must be interpreted with valid_years/complete_coverage. No valid years
    remains masked. First years are masked when no detection exists, and may
    be later than the true first disturbance if earlier coverage is missing.
    complete_coverage is QA, not a patch eligibility threshold or pixel filter.
    """
    flags = {year: lcms_change_flags(annual[year]) for year in years}
    valid = ee.ImageCollection.fromImages([
        flags[year]["clear_disturbance"] for year in years
    ]).count().unmask(0, sameFootprint=False).rename("valid_years").setDefaultProjection(projection)
    result = {
        "years": years, "expected_years": len(years), "valid_years": valid,
        "complete_coverage": valid.eq(len(years)).rename("complete_coverage"),
    }
    for category in ("clear_disturbance", "other_loss"):
        masks = [flags[year][category] for year in years]
        result[category] = (ee.ImageCollection.fromImages(masks).max()
                            .rename(category).setDefaultProjection(projection))
        dated = [mask.multiply(0).add(year).updateMask(mask.eq(1)).rename("first_year")
                 for year, mask in zip(years, masks)]
        result[f"{category}_first_year"] = (ee.ImageCollection.fromImages(dated).min()
                                            .rename(f"{category}_first_year")
                                            .setDefaultProjection(projection))
    return result


def build_lcms_history(fire, geometry: ee.Geometry) -> dict:
    """Build native LCMS retrospective QA only; no reduction, filtering or exports.

    Focal-year labels cannot order change relative to ignition. Season-5 labels
    cannot order change relative to September 30. Neither detections nor their
    annual timing belong in normal model predictors. MTBS handles wildfire QA.

    Retain annual image counts: missing or duplicate CONUS year records yield
    masked annual data, never arbitrary first-image selection or zero change.
    Native projection comes from an available regional source image; an empty
    regional source fails on evaluation rather than inventing a reference grid.
    """
    windows = lcms_year_windows(fire)
    years = sorted({year for window in windows.values() for year in window})
    source = (ee.ImageCollection(LCMS_SOURCE).filter(ee.Filter.eq("study_area", "CONUS"))
              .filterBounds(geometry).filter(ee.Filter.gte("year", years[0]))
              .filter(ee.Filter.lte("year", years[-1])))
    projection = ee.Image(source.sort("year").first()).select("Change").projection()
    missing = ee.Image.constant(0).rename("Change").updateMask(ee.Image.constant(0)).setDefaultProjection(projection)
    annual, counts = {}, {}
    for year in years:
        candidates = source.filter(ee.Filter.eq("year", year))
        counts[year] = candidates.size()
        annual[year] = ee.Image(ee.Algorithms.If(
            counts[year].eq(1), ee.Image(candidates.first()).select("Change"), missing))
    return {
        "source_dataset": LCMS_SOURCE, "role": "retrospective_qa_only",
        "timing_resolution": "annual; focal-year and Season-5 within-year ordering uncertain",
        "annual_image_counts": counts,
        "windows": {name: summarize_lcms_window(annual, window, projection)
                    for name, window in windows.items()},
    }
