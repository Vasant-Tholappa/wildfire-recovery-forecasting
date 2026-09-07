"""Offline LCMS class, union, timing and missing-coverage checks."""

import unittest
from unittest.mock import patch

import ee
from ee.apitestcase import ApiTestCase
import numpy as np

from wildfire_recovery_forecasting import disturbance as d
from test_landfire import Pixels as RemapPixels


class Pixels(RemapPixels):
    def remap(self, codes, values, default=None):
        result = super().remap(codes, values).values
        if default is not None:
            result = np.ma.array(result.filled(default), mask=np.ma.getmaskarray(self.values))
        return Pixels(result)

    def multiply(self, value):
        return Pixels(self.values * value)

    def add(self, value):
        return Pixels(self.values + value)

    def eq(self, value):
        return Pixels(self.values == value)

    def updateMask(self, value):
        return Pixels(super().updateMask(value).values)

    def unmask(self, value, **kwargs):
        return Pixels(self.values.filled(value))

    def setDefaultProjection(self, projection):
        return self

    def toInt16(self):
        return Pixels(self.values.astype(np.int16))


class Collection:
    def __init__(self, images):
        self.values = np.ma.stack([image.values for image in images])

    def max(self):
        return Pixels(self.values.max(axis=0))

    def min(self):
        return Pixels(self.values.min(axis=0))

    def count(self):
        return Pixels(self.values.count(axis=0))


class LcmsNumericTests(unittest.TestCase):
    def test_classes_and_unavailable_codes(self):
        flags = d.lcms_change_flags(Pixels(list(range(1, 18))))
        for name, expected in [("clear_disturbance", {1, 2, 5, 6, 8, 9, 10, 11}),
                               ("other_loss", {13})]:
            self.assertEqual(set(np.where(flags[name].values.filled(0) == 1)[0] + 1), expected)
            np.testing.assert_array_equal(flags[name].values.mask, [False] * 15 + [True] * 2)

    def test_union_timing_and_partial_coverage(self):
        # Repeat clear; repeat Other Loss; stable; partial stable; all missing.
        annual = {2017: Pixels([9, 13, 15, 16, 16]),
                  2018: Pixels([9, 13, 15, 15, 16])}
        with patch.object(d.ee.ImageCollection, "fromImages", side_effect=Collection):
            result = d.summarize_lcms_window(annual, [2017, 2018], None)
        np.testing.assert_array_equal(result["clear_disturbance"].values[:4], [1, 0, 0, 0])
        np.testing.assert_array_equal(result["other_loss"].values[:4], [0, 1, 0, 0])
        self.assertTrue(result["clear_disturbance"].values.mask[4])
        np.testing.assert_array_equal(result["valid_years"].values, [2, 2, 2, 1, 0])
        np.testing.assert_array_equal(result["complete_coverage"].values, [1, 1, 1, 0, 0])
        self.assertEqual(result["clear_disturbance_first_year"].values[0], 2017)
        self.assertEqual(result["other_loss_first_year"].values[1], 2017)
        self.assertTrue(result["clear_disturbance_first_year"].values.mask[2:].all())

    def test_year_bounds(self):
        for year in [2015, 2020]:
            windows = d.lcms_year_windows({"fire_year": year, "season5_year": year + 5})
            for n in [5, 7, 10]:
                self.assertEqual(windows[f"pre_{n}yr"], list(range(year - n, year)))
            self.assertEqual(windows["post_y_to_y5"], list(range(year, year + 6)))


class LcmsGraphTests(ApiTestCase):
    def test_each_annual_timing_image_has_common_int16_type(self):
        years = [2012, 2013]
        annual = {year: ee.Image.constant(9).rename("Change") for year in years}
        with patch.object(ee.data, "computeValue", side_effect=AssertionError("Network")), \
                patch.object(ee.ImageCollection, "fromImages", wraps=ee.ImageCollection.fromImages) as collections:
            d.summarize_lcms_window(annual, years, ee.Projection("EPSG:5070"))
        timing = [call.args[0] for call in collections.call_args_list
                  if "Image.toInt16" in call.args[0][0].serialize()]
        self.assertEqual(len(timing), 2)  # Clear disturbance and Other Loss.
        for images in timing:
            self.assertEqual(len(images), len(years))
            for image, year in zip(images, years):
                graph = image.serialize()
                self.assertIn("Image.toInt16", graph)
                self.assertIn("Image.updateMask", graph)
                self.assertIn(str(year), graph)

    def test_source_missing_year_branch_native_grid_and_qa_role(self):
        with patch.object(ee.data, "computeValue", side_effect=AssertionError("Network")):
            result = d.build_lcms_history({"fire_year": 2020, "season5_year": 2025},
                                         ee.Geometry.Point([-115, 45]))
            self.assertEqual(set(result["annual_image_counts"]), set(range(2010, 2026)))
            self.assertEqual(result["role"], "retrospective_qa_only")
            self.assertIn("uncertain", result["timing_resolution"])
            image = result["windows"]["post_y_to_y5"]["clear_disturbance"]
            graph = image.serialize()
            for text in [d.LCMS_SOURCE, "CONUS", "If", "Collection.size", "reduce.max",
                         "Image.setDefaultProjection", "Image.updateMask"]:
                self.assertIn(text, graph)
            for text in ["Image.reproject", "Image.resample", "Image.clip", "reduce.sum"]:
                self.assertNotIn(text, graph)
            self.assertNotIn("Image.unmask", graph)  # Observed flags are not filled with zero.


if __name__ == "__main__":
    unittest.main()
