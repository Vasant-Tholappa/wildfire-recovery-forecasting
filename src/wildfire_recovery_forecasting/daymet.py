"""Daymet historical, pre-Season-1, and Season-1 climate features."""

from collections.abc import Mapping
from numbers import Integral

import ee

from wildfire_recovery_forecasting.config import DAYMET_HISTORICAL_BASELINE_YEARS

DAYMET_SOURCE = "NASA/ORNL/DAYMET_V4"
VARIABLES = ["prcp", "tmax", "tmin", "vpd_kpa"]
SUMMARY_BANDS = ["prcp_mm", "tmax_c", "tmin_c", "vpd_kpa"]
HISTORICAL_BANDS = ["hist_prcp_mm", "hist_tmax_c", "hist_tmin_c", "hist_s1_vpd_kpa"]
PRE_S1_BANDS = ["pre_s1_" + name for name in SUMMARY_BANDS]
PRE_S1_ANOMALY_BANDS = ["pre_s1_prcp_anom_mm", "pre_s1_tmax_anom_c",
                        "pre_s1_tmin_anom_c", "pre_s1_vpd_anom_kpa"]
S1_ANOMALY_BANDS = ["s1_prcp_anom_mm", "s1_tmax_anom_c", "s1_tmin_anom_c", "s1_vpd_anom_kpa"]


def add_daymet_vpd(image: ee.Image) -> ee.Image:
    """Add Phase 2 daily VPD in kPa, using Tmax/Tmin saturation pressures."""
    saturation = []
    for band in ("tmax", "tmin"):
        temperature = image.select(band)
        saturation.append(temperature.multiply(17.27).divide(temperature.add(237.3))
                          .exp().multiply(0.6108))
    vpd = (saturation[0].add(saturation[1]).divide(2)
           .subtract(image.select("vp").divide(1000)).max(0).rename("vpd_kpa"))
    return image.addBands(vpd)


def climate_window(year: int, period: str) -> tuple[str, str]:
    """Return inclusive start/exclusive end for the two approved climate windows."""
    if period == "pre_s1":
        return f"{year}-01-01", f"{year}-06-01"
    if period == "s1":
        return f"{year}-06-01", f"{year}-10-01"
    raise ValueError("period must be pre_s1 or s1")


def aggregate_daymet_window(
    daily: ee.ImageCollection, year: int, period: str, projection: ee.Projection,
) -> dict[str, ee.Image]:
    """Sum precipitation (mm); mean Tmax/Tmin (C) and daily VPD (kPa).

    Retain per-variable valid-day counts. Partial coverage is not silently
    treated as complete: consumers must inspect counts before eligibility.
    No observations stays masked, including precipitation. A masked seed
    supplies the band schema for an empty collection without adding data.
    """
    start, end = climate_window(year, period)
    seed = (ee.Image.constant([0] * 4).rename(VARIABLES)
            .updateMask(ee.Image.constant(0)).setDefaultProjection(projection))
    observations = daily.filterDate(start, end).select(VARIABLES).merge(
        ee.ImageCollection.fromImages([seed]))
    counts = observations.count().rename(SUMMARY_BANDS).unmask(0).setDefaultProjection(projection)
    summary = ee.Image.cat([
        observations.select("prcp").sum(),
        observations.select(["tmax", "tmin", "vpd_kpa"]).mean(),
    ]).rename(SUMMARY_BANDS).updateMask(counts.gt(0)).setDefaultProjection(projection)
    return {"image": summary, "valid_days": counts}


def historical_baseline(
    daily: ee.ImageCollection, fire_year: int, period: str, projection: ee.Projection,
) -> dict:
    """Average ten matched yearly summaries, with per-year daily coverage QA."""
    years = range(fire_year - DAYMET_HISTORICAL_BASELINE_YEARS, fire_year)
    annual = {year: aggregate_daymet_window(daily, year, period, projection) for year in years}
    summaries = ee.ImageCollection.fromImages([value["image"] for value in annual.values()])
    return {
        "image": summaries.mean().setDefaultProjection(projection),
        "valid_years": summaries.count().unmask(0).setDefaultProjection(projection),
        "valid_days_by_year": {year: value["valid_days"] for year, value in annual.items()},
    }


def build_fire_climate_images(fire: Mapping, geometry: ee.Geometry) -> dict:
    """Build native ~1 km climate surfaces without evaluation or patch reduction.

    primary contains Phase 2's 16 core fields, with ante_* renamed pre_s1_*.
    historical contains only the four early-ablation predictors. Matched
    baseline and current-window structures retain four-band summaries and QA;
    these supporting images do not add predictors to the core feature list.
    Anomalies are current minus matched historical conditions. Validity is
    per feature; never use a single precipitation mask for all denominators.
    """
    year, season1 = fire.get("fire_year"), fire.get("season1_year")
    if (not isinstance(year, Integral) or isinstance(year, bool)
            or not isinstance(season1, Integral) or isinstance(season1, bool)
            or not 2015 <= year <= 2020 or season1 != year + 1):
        raise ValueError("Require a 2015–2020 manifest fire_year and its season1_year")
    source = (ee.ImageCollection(DAYMET_SOURCE).filterBounds(geometry)
              .filterDate(f"{year - DAYMET_HISTORICAL_BASELINE_YEARS}-01-01", f"{season1}-10-01"))
    projection = ee.Image(source.first()).select("tmax").projection()
    daily = source.map(add_daymet_vpd)
    hist_pre = historical_baseline(daily, year, "pre_s1", projection)
    hist_s1 = historical_baseline(daily, year, "s1", projection)
    pre = aggregate_daymet_window(daily, season1, "pre_s1", projection)
    s1 = aggregate_daymet_window(daily, season1, "s1", projection)
    historical = hist_s1["image"].rename(HISTORICAL_BANDS)
    primary = ee.Image.cat([
        historical, pre["image"].rename(PRE_S1_BANDS),
        pre["image"].subtract(hist_pre["image"]).rename(PRE_S1_ANOMALY_BANDS),
        s1["image"].subtract(hist_s1["image"]).rename(S1_ANOMALY_BANDS),
    ]).setDefaultProjection(projection)
    return {
        "primary": primary, "historical": historical,
        "primary_valid": primary.mask().gt(0).unmask(0).setDefaultProjection(projection),
        "historical_valid": historical.mask().gt(0).unmask(0).setDefaultProjection(projection),
        "historical_pre_s1": hist_pre, "historical_s1": hist_s1,
        "pre_s1": pre, "s1": s1,
    }
