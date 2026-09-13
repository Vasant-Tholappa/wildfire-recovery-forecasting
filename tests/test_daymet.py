"""Offline checks of Daymet temporal windows, formulas and EE graphs."""

import unittest
from unittest.mock import patch

import ee
import pandas as pd
from ee.apitestcase import ApiTestCase

from wildfire_recovery_forecasting import daymet, reductions
from test_landsat import graph_calls


class DaymetTests(ApiTestCase):
    def test_historical_collections_normalize_each_annual_summary(self):
        projection = ee.Projection("EPSG:4326")
        for period in ("pre_s1", "s1"):
            annual = []

            def aggregate(daily, year, requested_period, source_projection):
                # Distinct year-specific constant types, with an existing mask.
                image = (ee.Image.constant([year] * 4).rename(daymet.SUMMARY_BANDS)
                         .updateMask(ee.Image.constant(0)).setDefaultProjection(projection))
                annual.append(image)
                return {"image": image, "valid_days": image}

            with patch.object(daymet, "aggregate_daymet_window", side_effect=aggregate), \
                 patch.object(ee.ImageCollection, "fromImages", wraps=ee.ImageCollection.fromImages) as stack:
                daymet.historical_baseline(ee.ImageCollection([]), 2018, period, projection)
            members = stack.call_args.args[0]
            self.assertEqual(len(members), 10)
            for member, original in zip(members, annual):
                self.assertEqual(member.serialize(), original.toDouble().serialize())

    def test_historical_daily_coverage_normalized_after_yearly_division(self):
        images = daymet.build_fire_climate_images(
            {"fire_year": 2018, "season1_year": 2019}, ee.Geometry.Point([-122, 43]))

        def area_sums(values, patches, projection):
            row = {"patch_id": "p", "fire_id": "f", reductions.RASTER_FULL_AREA: 8100.}
            for name in values:
                row[name + "_sum"] = 8100.
                row[name + "_area"] = 8100.
            return pd.DataFrame([row])

        with patch.object(reductions, "build_area_sums", side_effect=area_sums), \
             patch.object(reductions, "fetch_patch_table", side_effect=lambda fc, columns: fc), \
             patch.object(ee.ImageCollection, "fromImages", wraps=ee.ImageCollection.fromImages) as stack:
            reductions.reduce_daymet(images, ee.FeatureCollection([]))
        self.assertEqual(stack.call_count, 2)
        for call in stack.call_args_list:
            self.assertEqual(len(call.args[0]), 10)
            for member in call.args[0]:
                self.assertEqual(member.func.getSignature()["name"], "Image.toDouble")
                self.assertEqual(member.args["value"].func.getSignature()["name"], "Image.divide")

    def test_calendar_windows(self):
        self.assertEqual(daymet.climate_window(2016, "pre_s1"), ("2016-01-01", "2016-06-01"))
        self.assertEqual(daymet.climate_window(2016, "s1"), ("2016-06-01", "2016-10-01"))
        with self.assertRaises(ValueError):
            daymet.climate_window(2016, "annual")

    def test_vpd_formula(self):
        calls = graph_calls(daymet.add_daymet_vpd(ee.Image("daily")))
        names = [name for name, _ in calls]
        self.assertEqual(names.count("Image.exp"), 2)
        self.assertIn("Image.max", names)
        self.assertNotIn("Image.unmask", names)
        constants = [args["image2"][1].get("value") for name, args in calls
                     if name in ["Image.multiply", "Image.add", "Image.divide", "Image.max"]
                     and args["image2"][0] == "Image.constant"]
        for value in [17.27, 237.3, 0.6108, 2, 1000, 0]:
            self.assertIn(value, constants)

    def test_window_aggregation_missingness_and_projection(self):
        projection = ee.Projection("EPSG:4326")
        result = daymet.aggregate_daymet_window(ee.ImageCollection([]), 2016, "s1", projection)
        calls = graph_calls(result["image"])
        names = [name for name, _ in calls]
        self.assertIn("reduce.sum", names)
        self.assertIn("reduce.mean", names)
        masks = [args["mask"] for name, args in calls if name == "Image.updateMask"]
        self.assertTrue(any(mask[0] == "Image.gt" for mask in masks))
        self.assertIn("reduce.count", result["valid_days"].serialize())
        self.assertEqual(result["image"].args["crs"].serialize(), projection.serialize())
        for forbidden in ["Image.reproject", "Image.resample", "Image.clip"]:
            self.assertNotIn(forbidden, result["image"].serialize())
        # Only count QA can be unmasked; the measurement stack is updateMasked.
        unmasked = [args for name, args in calls if name == "Image.unmask"]
        self.assertEqual(len(unmasked), 1)
        self.assertIn("reduce.count", str(unmasked))

    def test_manifest_windows_matching_and_early_separation(self):
        for year in [2015, 2020]:
            projection = ee.Projection("EPSG:4326")
            windows = []

            def aggregate(daily, requested_year, period, source_projection):
                windows.append((requested_year, period))
                value = requested_year + (100 if period == "pre_s1" else 0)
                image = ee.Image.constant([value] * 4).rename(daymet.SUMMARY_BANDS)
                return {"image": image.setDefaultProjection(projection), "valid_days": image}

            with patch.object(daymet, "aggregate_daymet_window", side_effect=aggregate), \
                 patch.object(ee.data, "computeValue", side_effect=AssertionError("Network")):
                result = daymet.build_fire_climate_images(
                    {"fire_year": year, "season1_year": year + 1}, ee.Geometry.Point([-115, 45]))
            self.assertEqual(windows, [(y, "pre_s1") for y in range(year - 10, year)]
                             + [(y, "s1") for y in range(year - 10, year)]
                             + [(year + 1, "pre_s1"), (year + 1, "s1")])
            self.assertEqual(set(result["historical_s1"]["valid_days_by_year"]), set(range(year - 10, year)))
            subtracts = [args for name, args in graph_calls(result["primary"]) if name == "Image.subtract"]
            expected = [(result["pre_s1"]["image"], result["historical_pre_s1"]["image"]),
                        (result["s1"]["image"], result["historical_s1"]["image"])]
            self.assertEqual(len(subtracts), 2)
            for current, baseline in expected:
                # The assembled anomaly graph must contain each exact matched subtraction.
                args = next(args for name, args in graph_calls(current.subtract(baseline))
                            if name == "Image.subtract")
                self.assertIn(args, subtracts)
            early = result["historical"].serialize()
            self.assertNotIn("pre_s1_", early)
            self.assertNotIn("Image.subtract", early)

    def test_full_graph_cutoff_and_invalid_manifest(self):
        result = daymet.build_fire_climate_images(
            {"fire_year": 2015, "season1_year": 2016}, ee.Geometry.Point([-115, 45]))
        graph = result["primary"].serialize()
        for text in ["NASA/ORNL/DAYMET_V4", "2005-01-01", "2014-10-01", "2016-10-01"]:
            self.assertIn(text, graph)
        for text in ["2017-", "Image.reproject", "Image.resample", "Image.clip"]:
            self.assertNotIn(text, graph)
        for bad in [{}, {"fire_year": 2015, "season1_year": 2017}]:
            with self.assertRaises(ValueError):
                daymet.build_fire_climate_images(bad, ee.Geometry.Point([0, 0]))


if __name__ == "__main__":
    unittest.main()
