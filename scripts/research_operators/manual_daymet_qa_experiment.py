"""Candidate C builders/comparison only; no initialization, submission or cloud IO.

Compare against the validated combined-Daymet benchmark, not separate coverage.
No production registry imports this experimental variant.
"""

from datetime import date

import ee
import numpy as np
import pandas as pd

import manual_daymet_combined_benchmark as combined
from wildfire_recovery_forecasting import daymet, reductions
from wildfire_recovery_forecasting.validation import validate_patch_table

VERSION = "phase3_daymet_qa4_bench_v1"
QA_NAMES = (
    "daymet_hist_pre_s1_completeness", "daymet_hist_s1_completeness",
    "daymet_pre_s1_completeness", "daymet_s1_completeness",
)


def expected_days(year, period):
    """Reuse the approved calendar windows, including leap-year Jan-May length."""
    start, end = daymet.climate_window(year, period)
    return (date.fromisoformat(end) - date.fromisoformat(start)).days


def worst_variable_coverage(counts, days, projection):
    """Pixelwise minimum over all four variables; masked inputs are not skipped.

    Existing valid-day counts distinguish no observations (observed zero) from
    masked/unknown support. Do not unmask measurements; only the mask is filled.
    """
    values = counts.select(daymet.SUMMARY_BANDS).divide(days).toDouble()
    all_valid = values.mask().unmask(0).reduce(ee.Reducer.min())
    return (values.reduce(ee.Reducer.min()).updateMask(all_valid)
            .rename("completeness").setDefaultProjection(projection))


def build_temporal_qa(images, fire):
    """Four native-grid completeness images; minima precede spatial reduction."""
    year, season1 = int(fire["fire_year"]), int(fire["season1_year"])
    if not 2015 <= year <= 2020 or season1 != year + 1:
        raise ValueError("Require canonical fire_year and season1_year")
    expected_years = set(range(year - 10, year))
    qa = {}
    for period in ("pre_s1", "s1"):
        history = images["historical_" + period]
        if set(history["valid_days_by_year"]) != expected_years:
            raise ValueError("Require every year in the ten-year historical window")
        projection = history["image"].projection()
        annual = [worst_variable_coverage(history["valid_days_by_year"][y],
                                         expected_days(y, period), projection)
                  for y in sorted(expected_years)]
        annual = ee.ImageCollection.fromImages(annual)
        # EE min ignores masked members; count ensures all ten years are known.
        worst = annual.min().updateMask(annual.count().eq(10)).setDefaultProjection(projection)
        qa[f"daymet_hist_{period}_completeness"] = worst
        qa[f"daymet_{period}_completeness"] = worst_variable_coverage(
            images[period]["valid_days"], expected_days(season1, period), projection)
    return {name: qa[name] for name in QA_NAMES}


def build_qa4_product(images, patches, fire):
    """One reduction: unchanged 16 climate inputs plus four temporal QA inputs."""
    primary, _, projection = reductions.daymet_area_inputs(images)
    # The unused full-QA graph is lazy and is not connected to this export.
    qa = build_temporal_qa(images, fire)
    return reductions.build_area_product({**primary, **qa}, patches, projection)


def experiment_identity(tile, batch_id):
    """Dedicated identities; never reuse validated combined or production paths."""
    if type(tile) is not int or tile not in (60, 90, 120):
        raise ValueError("Require an existing WOODBURY benchmark tile: 60, 90 or 120")
    return dict(source="daymet", product="qa4", fire_id=combined.baseline.FIRE_ID,
                batch_id=batch_id, production_version=f"{VERSION}_t{tile}")


def compare_climate_tables(candidate, full_qa, ids):
    """Verify retained raw fields AND derived climate means/valid fractions.

    New QA cannot be reconstructed from the old patch means. This comparison
    checks its bounds and missingness structure, not pixelwise minimum accuracy;
    a separate native-pixel audit is needed for live numerical QA verification.
    """
    names = daymet.HISTORICAL_BANDS + daymet.PRE_S1_BANDS + daymet.PRE_S1_ANOMALY_BANDS + daymet.S1_ANOMALY_BANDS
    columns = ["patch_id", "fire_id", reductions.RASTER_FULL_AREA] + [
        name + suffix for name in names for suffix in ("_sum", "_area")]
    qa_columns = [name + suffix for name in QA_NAMES for suffix in ("_sum", "_area")]
    for table in (candidate, full_qa):
        validate_patch_table(table, expected_ids=ids, fire_id=combined.baseline.FIRE_ID)
        if not table.columns.is_unique or not set(columns).issubset(table):
            raise ValueError("Missing or duplicate climate columns")
    if set(candidate) != set(columns + qa_columns):
        raise ValueError("Expected unchanged climate fields and exactly four QA sum/area pairs")
    a, b = [table.set_index("patch_id").sort_index().reset_index()
            for table in (candidate, full_qa)]
    for name in columns[2:]:
        if (not a[name].isna().equals(b[name].isna())
                or not np.allclose(a[name], b[name], equal_nan=True, rtol=1e-9, atol=1e-6)):
            raise ValueError(f"Climate field changed: {name}")
    a_climate = reductions.summarize_area_sums(a, names)
    b_climate = reductions.summarize_area_sums(b, names)
    pd.testing.assert_frame_equal(a_climate, b_climate, check_dtype=False, rtol=1e-9, atol=1e-9)
    qa = reductions.summarize_area_sums(a, QA_NAMES)
    for name in QA_NAMES:
        total, area = a[name + "_sum"], a[name + "_area"]
        if (total.notna() & (area.isna() | area.le(0))).any():
            raise ValueError(f"QA measurement without valid area: {name}")
        if (area.gt(0) & total.isna()).any():
            raise ValueError(f"Missing QA measurement on valid area: {name}")
        if not qa[name + "_mean"].dropna().between(-1e-9, 1 + 1e-9).all():
            raise ValueError(f"QA completeness outside [0,1]: {name}")
        if not qa[name + "_valid_fraction"].between(-1e-9, 1 + 1e-9).all():
            raise ValueError(f"QA valid fraction outside [0,1]: {name}")
    return {"climate_equivalent": True, "rows": len(a), "climate_fields": len(names),
            "temporal_qa_fields": len(QA_NAMES), "qa_pixelwise_accuracy_verified": False,
            "qa_null_counts": {name: int(qa[name + "_mean"].isna().sum()) for name in QA_NAMES}}
