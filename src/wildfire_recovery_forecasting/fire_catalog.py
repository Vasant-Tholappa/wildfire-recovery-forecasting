"""Focal-fire catalog and manifest construction."""

from datetime import date, datetime, timezone
from math import isclose, isfinite
from numbers import Real

import ee
import geopandas as gpd
import pandas as pd
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

from wildfire_recovery_forecasting.config import (
    GROWING_SEASON_END_MONTH_DAY,
    NBR_PRE_SEARCH_SEASONS,
)

MTBS_SOURCE = "USFS/GTAC/MTBS/burned_area_boundaries/v1"
WESTERN_STATE_CODES = (
    "WA", "OR", "CA", "ID", "NV", "MT", "WY", "UT", "CO", "AZ", "NM"
)


def derive_fire_fields(ignition_date: date) -> dict[str, int | str]:
    """Derive manifest fields from a date (aware datetimes use UTC).

    Naive dates/datetimes are interpreted as UTC calendar values. Raw MTBS
    millisecond timestamps must first be decoded by ``manifest_from_mtbs``.
    """
    if not isinstance(ignition_date, date) or pd.isna(ignition_date):
        raise ValueError("ignition_date must be a non-missing date or datetime")
    if isinstance(ignition_date, datetime):
        if ignition_date.tzinfo is not None:
            ignition_date = ignition_date.astimezone(timezone.utc)
        ignition_date = ignition_date.date()

    fire_year = ignition_date.year
    if not 2015 <= fire_year <= 2020:
        raise ValueError("Focal fire year must be in 2015–2020")

    season_end = date(fire_year, *GROWING_SEASON_END_MONTH_DAY)
    last_prefire_year = fire_year if ignition_date > season_end else fire_year - 1

    split = "train" if fire_year <= 2017 else "validation" if fire_year <= 2019 else "test"
    return {
        "fire_year": fire_year,
        "last_fully_prefire_growing_season_year": last_prefire_year,
        "prefire_search_start_year": last_prefire_year - NBR_PRE_SEARCH_SEASONS + 1,
        "prefire_search_end_year": last_prefire_year,
        "season1_year": fire_year + 1,
        "season5_year": fire_year + 5,
        "landfire_condition_year": 2014,
        "landfire_version": "v1.4.0",
        "split": split,
    }


def normalize_fire_geometry(geometry: BaseGeometry) -> Polygon | MultiPolygon:
    """Preserve MTBS polygon parts, excluding only zero-area collection members.

    No repair, dissolve, simplification, or coordinate transformation occurs.
    Area is checked in the unchanged input coordinate system with relative
    tolerance 1e-12 and no absolute tolerance; it is not an acreage estimate.
    """
    if not isinstance(geometry, BaseGeometry) or geometry.is_empty:
        raise ValueError("Every focal fire requires a nonempty perimeter geometry")
    if not geometry.is_valid:
        raise ValueError("Invalid MTBS perimeter geometry requires source review")
    if isinstance(geometry, (Polygon, MultiPolygon)):
        normalized = geometry
    elif isinstance(geometry, GeometryCollection):
        polygons = []
        pending = list(reversed(geometry.geoms))
        while pending:
            part = pending.pop()
            if isinstance(part, Polygon):
                polygons.append(part)
            elif isinstance(part, (MultiPolygon, GeometryCollection)):
                pending.extend(reversed(part.geoms))
            elif part.area != 0:
                raise ValueError("Cannot discard a non-polygonal component with nonzero area")
        if not polygons:
            raise ValueError("MTBS GeometryCollection contains no polygonal components")
        normalized = polygons[0] if len(polygons) == 1 else MultiPolygon(polygons)
    else:
        raise ValueError("Unsupported MTBS perimeter geometry type: " + geometry.geom_type)
    if normalized.is_empty or not normalized.is_valid:
        raise ValueError("Normalized MTBS perimeter must be nonempty and valid")
    if not (
        isfinite(geometry.area)
        and isfinite(normalized.area)
        and isclose(geometry.area, normalized.area, rel_tol=1e-12, abs_tol=0.0)
    ):
        raise ValueError("MTBS geometry normalization did not preserve area")
    return normalized


def manifest_from_mtbs(fires: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Convert already region-filtered MTBS records to a manifest, locally.

    Input must retain MTBS field names and numeric epoch-millisecond Ig_Date.
    Wildfire type and focal-year bounds are checked rather than silently
    dropping unexpected rows. Duplicate IDs require source review; perimeters
    are never dissolved, clipped, or arbitrarily discarded here.
    """
    required = {"Event_ID", "Incid_Name", "Incid_Type", "Ig_Date", "BurnBndAc"}
    missing = required - set(fires.columns)
    if missing:
        raise ValueError(f"Missing MTBS fields: {sorted(missing)}")
    if fires.empty:
        raise ValueError("MTBS focal-fire population is empty")
    if fires.crs is None:
        raise ValueError("MTBS perimeter CRS is required")
    if not fires["Incid_Type"].eq("Wildfire").all():
        raise ValueError("Manifest input must contain only MTBS Wildfire events")
    ids = fires["Event_ID"]
    if not ids.map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError("Every focal fire requires a nonempty Event_ID")
    if ids.duplicated().any():
        raise ValueError("Duplicate MTBS Event_ID values require source review")
    if fires.geometry.isna().any() or fires.geometry.is_empty.any():
        raise ValueError("Every focal fire requires a nonempty perimeter geometry")
    geometries = [normalize_fire_geometry(geometry) for geometry in fires.geometry]

    timestamps = fires["Ig_Date"]
    if timestamps.isna().any() or not timestamps.map(
        lambda value: isinstance(value, Real) and not isinstance(value, bool)
    ).all():
        raise ValueError("Ig_Date must contain non-missing numeric epoch milliseconds")
    ignition_dates = pd.to_datetime(timestamps, unit="ms", utc=True, errors="raise")
    derived = pd.DataFrame(
        [derive_fire_fields(value) for value in ignition_dates]
    )

    # Positional assembly also handles a caller's non-default/duplicate index.
    manifest = gpd.GeoDataFrame(
        {
            "fire_id": ids.to_numpy(),
            "fire_name": fires["Incid_Name"].to_numpy(),
            "ignition_date": ignition_dates.array,
            "fire_size_acres": fires["BurnBndAc"].to_numpy(),
            "source_dataset": MTBS_SOURCE,
        },
        geometry=geometries,
        crs=fires.crs,
    )
    for field in derived:
        manifest[field] = derived[field]
    return manifest.sort_values("fire_id").reset_index(drop=True)


def build_fire_manifest() -> gpd.GeoDataFrame:
    """Fetch the focal-fire vector population and construct its manifest.

    The caller must initialize Earth Engine. Calling this function downloads
    the filtered population and its full perimeters; it does not query rasters
    or write outputs. Region membership follows Phase 2 perimeter intersection,
    not a centroid or Event_ID state-prefix filter.
    """
    western_states = ee.FeatureCollection("TIGER/2018/States").filter(
        ee.Filter.inList("STUSPS", list(WESTERN_STATE_CODES))
    )
    fires = (
        ee.FeatureCollection(MTBS_SOURCE)
        .filter(ee.Filter.eq("Incid_Type", "Wildfire"))
        .filter(ee.Filter.gte("Ig_Date", ee.Date("2015-01-01").millis()))
        .filter(ee.Filter.lt("Ig_Date", ee.Date("2021-01-01").millis()))
        .filterBounds(western_states.geometry())
        .select(["Event_ID", "Incid_Name", "Incid_Type", "Ig_Date", "BurnBndAc"])
    )
    records = ee.data.computeFeatures({
        "expression": fires,
        "fileFormat": "GEOPANDAS_GEODATAFRAME",
    })
    # computeFeatures returns EPSG:4326 coordinates, but its GeoPandas
    # converter does not attach CRS metadata. Keep the returned full geometry.
    records = records.set_crs("EPSG:4326")
    return manifest_from_mtbs(records)
