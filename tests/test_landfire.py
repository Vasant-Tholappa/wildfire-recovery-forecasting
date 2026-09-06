"""Network-independent LANDFIRE decoding and EE graph checks."""

import unittest
from unittest.mock import patch

import ee
from ee.apitestcase import ApiTestCase
import numpy as np

from wildfire_recovery_forecasting import landfire


class Pixels:
    """Evaluate remap/masking on local pixels, including masked input values."""

    def __init__(self, values):
        self.values = np.ma.array(values, dtype=float)

    def remap(self, codes, values):
        output = np.ma.masked_all(self.values.shape)
        for code, value in zip(codes, values):
            selected = np.asarray((self.values == code).filled(False))
            output[selected] = value
        return Pixels(output)

    def rename(self, name):
        return self

    def updateMask(self, mask):
        return Pixels(np.ma.array(
            self.values.data,
            mask=np.ma.getmaskarray(self.values) | ~mask.values.filled(0).astype(bool),
        ))


class DecodeTests(unittest.TestCase):
    def test_cover_midpoints_and_tree_support(self):
        codes = list(range(101, 110)) + [151, 152, 153]
        expected = [15, 25, 35, 45, 55, 65, 75, 85, 95, 17.5, 42.5, 80]
        np.testing.assert_array_equal(landfire.decode_tree_cover(Pixels(codes)).values, expected)
        np.testing.assert_array_equal(landfire.forest_indicator(Pixels(codes)).values, 1)

    def test_non_tree_unknown_and_missing_are_distinct(self):
        # A masked tree code must stay missing despite its otherwise valid value.
        pixels = Pixels(np.ma.array([11, 78, 100, 111, 121, 0, -9999, 999, 101],
                                   mask=[False] * 8 + [True]))
        forest = landfire.forest_indicator(pixels).values
        np.testing.assert_array_equal(forest.mask, [False] * 5 + [True] * 4)
        np.testing.assert_array_equal(forest.compressed(), [0] * 5)
        self.assertTrue(landfire.decode_tree_cover(pixels).values.mask.all())

    def test_height_and_open_ended_class(self):
        pixels = Pixels(np.ma.array([101, 110, 198, 199, 100, 200, 11, 0, 101],
                                   mask=[False] * 8 + [True]))
        heights = landfire.decode_tree_height(pixels).values
        np.testing.assert_array_equal(heights.compressed(), [1, 10, 98, 99])
        np.testing.assert_array_equal(heights.mask, [False] * 4 + [True] * 5)

    def test_height_requires_valid_forest_support(self):
        forest = landfire.forest_indicator(Pixels([101, 111, -9999, 109]))
        height = landfire.decode_tree_height(Pixels([110, 110, 110, 11])).updateMask(forest)
        np.testing.assert_array_equal(height.values.mask, [False, True, True, True])
        self.assertEqual(height.values[0], 10)

    def test_manifest_baseline_lookup(self):
        self.assertEqual(landfire.landfire_sources({
            "landfire_condition_year": 2014, "landfire_version": "v1.4.0",
        }), {product: f"LANDFIRE/Vegetation/{product}/v1_4_0"
             for product in ("EVT", "EVC", "EVH")})
        for year, version in [(2012, "v1.3.0"), (2016, "v2.0.0"),
                              (2014, "v2.0.0"), (2016, "v1.4.0"), (None, None)]:
            with self.subTest(year=year, version=version), self.assertRaises(ValueError):
                landfire.landfire_sources({"landfire_condition_year": year,
                                           "landfire_version": version})


class ImageGraphTests(ApiTestCase):
    def test_sources_categorical_identity_qa_and_native_projection(self):
        with patch.object(ee.data, "computeValue", side_effect=AssertionError("Network")):
            result = landfire.build_landfire_images(
                {"landfire_condition_year": 2014, "landfire_version": "v1.4.0"},
                ee.Geometry.Point([-115, 45]),
            )
            self.assertEqual(set(result), {
                "forest_indicator", "forest_mask", "evt", "tree_cover_pct", "tree_height_m",
                "evc_valid", "evt_valid", "tree_cover_valid", "tree_height_valid",
            })
            evt_args = result["evt"].args["input"].args
            self.assertEqual(evt_args["from"].serialize(), evt_args["to"].serialize())
            self.assertIn("EVT_class_values", result["evt"].serialize())
            for name, image in result.items():
                graph = image.serialize()
                for forbidden in ["Reducer.mean", "reduce.mean", "Image.reproject",
                                  "Image.resample", "Image.clip", "ImageCollection.mosaic"]:
                    self.assertNotIn(forbidden, graph)
                self.assertEqual("Image.unmask" in graph, name.endswith("_valid"))
            for product, key in [("EVT", "evt"), ("EVC", "forest_indicator"),
                                 ("EVH", "tree_height_m")]:
                self.assertIn(f"LANDFIRE/Vegetation/{product}/v1_4_0", result[key].serialize())
                self.assertIn("Collection.first", result[key].serialize())
            self.assertIn("Image.updateMask", result["tree_height_m"].serialize())


if __name__ == "__main__":
    unittest.main()
