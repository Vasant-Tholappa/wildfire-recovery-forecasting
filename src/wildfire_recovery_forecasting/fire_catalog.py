"""Focal-fire catalog and manifest construction."""

from datetime import date, datetime, timezone
from numbers import Real

import ee
import geopandas as gpd
import pandas as pd

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
    if not 2013 <= fire_year <= 2020:
        raise ValueError("Focal fire year must be in 2013–2020")

    season_end = date(fire_year, *GROWING_SEASON_END_MONTH_DAY)
    last_prefire_year = fire_year if ignition_date > season_end else fire_year - 1
    if fire_year <= 2014:
        condition_year, landfire_version = 2012, "v1.3.0"
    elif fire_year <= 2016:
        condition_year, landfire_version = 2014, "v1.4.0"
    else:
        condition_year, landfire_version = 2016, "v2.0.0"

    split = "train" if fire_year <= 2017 else "validation" if fire_year <= 2019 else "test"
    return {
        "fire_year": fire_year,
        "last_fully_prefire_growing_season_year": last_prefire_year,
        "prefire_search_start_year": last_prefire_year - NBR_PRE_SEARCH_SEASONS + 1,
        "prefire_search_end_year": last_prefire_year,
        "season1_year": fire_year + 1,
        "season5_year": fire_year + 5,
        "landfire_condition_year": condition_year,
        "landfire_version": landfire_version,
        "split": split,
    }


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
    if not fires.geometry.geom_type.isin(["Polygon", "MultiPolygon"]).all():
        raise ValueError("MTBS perimeters must be Polygon or MultiPolygon geometries")
    if not fires.geometry.is_valid.all():
        raise ValueError("Invalid MTBS perimeter geometry requires source review")

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
        geometry=fires.geometry.reset_index(drop=True),
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
        .filter(ee.Filter.gte("Ig_Date", ee.Date("2013-01-01").millis()))
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
