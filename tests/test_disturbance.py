"""Local MTBS event selection and non-double-counted geometry checks."""

import unittest
from unittest.mock import MagicMock, patch

import geopandas as gpd 
import pandas as pd
from shapely.geometry import box

from wildfire_recovery_forecasting import disturbance as d


FIRE = {"fire_id": "focal", "ignition_date": pd.Timestamp("2020-07-15T12:00:00Z"),
        "season5_year": 2025}


def events(rows):
    return gpd.GeoDataFrame([
        {"Event_ID": name, "Ig_Date": pd.Timestamp(when).value // 10**6,
         "Incid_Type": kind, "geometry": geometry}
        for name, when, kind, geometry in rows
    ], columns=["Event_ID", "Ig_Date", "Incid_Type", "geometry"],
        geometry="geometry", crs="EPSG:5070")


class HistoryTests(unittest.TestCase):
    def test_each_prior_boundary_and_same_year(self):
        for years in [10]:
            start = FIRE["ignition_date"] - pd.DateOffset(years=years)
            source = events([(name, when, "Wildfire", box(0, 0, 90, 90)) for name, when in [
                ("before", start - pd.Timedelta(milliseconds=1)), ("start", start),
                ("same_year", FIRE["ignition_date"] - pd.Timedelta(milliseconds=1)),
                ("same_time", FIRE["ignition_date"]), ("focal", start),
            ]])
            selected = d.history_from_mtbs(FIRE, source)["windows"][f"prior_fire_fraction_{years}yr"]
            self.assertEqual(set(selected["events"].Event_ID), {"start", "same_year"})

    def test_reburn_strict_start_and_season5_end(self):
        rows = [(name, when, "Wildfire", box(0, 0, 1, 1)) for name, when in [
            ("equal", "2020-07-15T12:00:00Z"), ("later", "2020-07-15T12:00:00.001Z"),
            ("last", "2025-09-30T23:59:59.999Z"), ("outside", "2025-10-01T00:00:00Z"),
            ("focal", "2021-01-01T00:00:00Z"),
        ]]
        rows.append(("prescribed", "2021-01-01", "Prescribed Fire", box(0, 0, 1, 1)))
        result = d.history_from_mtbs(FIRE, events(rows))["windows"]["reburn_fraction"]
        self.assertEqual(set(result["events"].Event_ID), {"later", "last"})

    def test_union_preserves_parts_without_double_counting(self):
        source = events([(name, "2019-01-01", "Wildfire", geom) for name, geom in [
            ("a", box(0, 0, 60, 90)), ("b", box(30, 0, 90, 90)),
        ]])
        result = d.history_from_mtbs(FIRE, source)
        self.assertEqual(set(result["windows"]), {"prior_fire_fraction_10yr", "reburn_fraction"})
        window = result["windows"]["prior_fire_fraction_10yr"]
        self.assertEqual(result["union_crs"], "EPSG:5070")
        self.assertEqual(window["union_geometry"].intersection(box(0, 0, 90, 90)).area / 8100, 1)
        self.assertEqual(list(window["events"].geometry.to_wkb()), list(source.geometry.to_wkb()))
        self.assertGreater(source.geometry.area.sum(), window["union_geometry"].area)

    def test_empty_is_unverified_and_missing_data_rejected(self):
        for window in d.history_from_mtbs(FIRE, events([]))["windows"].values():
            self.assertTrue(window["union_geometry"].is_empty)
            self.assertEqual(window["coverage_status"], "unverified")
        for field in ["Event_ID", "Ig_Date", "geometry", "Incid_Type"]:
            source = events([("a", "2019-01-01", "Wildfire", box(0, 0, 1, 1))])
            source.at[0, field] = None
            with self.subTest(field=field), self.assertRaises(ValueError):
                d.history_from_mtbs(FIRE, source)

    def test_leap_anniversary_and_timezone(self):
        fire = dict(FIRE, ignition_date=pd.Timestamp("2020-02-29T08:00:00-04:00"))
        window = d.wildfire_history_windows(fire)["prior_fire_fraction_10yr"]
        self.assertEqual(window["start"], pd.Timestamp("2010-02-28T12:00:00Z"))

    def test_bounded_retrieval_and_errors_propagate(self):
        records = events([]).set_crs(None, allow_override=True)
        with patch.object(d, "ee") as ee:
            ee.data.computeFeatures.return_value = records
            geometry = MagicMock()
            result = d.build_wildfire_history(FIRE, geometry)
            ee.FeatureCollection.assert_called_once_with(d.MTBS_SOURCE)
            ee.FeatureCollection.return_value.filterBounds.assert_called_once_with(geometry)
            self.assertEqual(set(result["windows"]), {"prior_fire_fraction_10yr", "reburn_fraction"})
            ee.data.computeFeatures.side_effect = RuntimeError("Source unavailable")
            with self.assertRaises(RuntimeError):
                d.build_wildfire_history(FIRE, geometry)


if __name__ == "__main__":
    unittest.main()
