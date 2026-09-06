"""Offline numeric and EE graph checks for native 3DEP terrain."""

import unittest
from unittest.mock import patch

import ee
from ee.apitestcase import ApiTestCase
import numpy as np

from wildfire_recovery_forecasting import terrain


class Pixels:
    def __init__(self, values):
        self.values = np.ma.array(values)

    def multiply(self, value):
        return Pixels(self.values * value)

    def cos(self):
        return Pixels(np.ma.cos(self.values))

    def sin(self):
        return Pixels(np.ma.sin(self.values))

    def rename(self, name):
        return self


class AspectTests(unittest.TestCase):
    def test_cardinal_directions_and_missingness(self):
        aspect = Pixels(np.ma.array([0, 90, 180, 270, 360, 45],
                                    mask=[False] * 5 + [True]))
        result = terrain.aspect_components(aspect)
        np.testing.assert_allclose(result["northness"].values[:5], [1, 0, -1, 0, 1], atol=1e-12)
        np.testing.assert_allclose(result["eastness"].values[:5], [0, 1, 0, -1, 0], atol=1e-12)
        for image in result.values():
            self.assertTrue(image.values.mask[5])


class TerrainGraphTests(ApiTestCase):
    def test_source_derivatives_and_projection(self):
        with patch.object(ee.data, "computeValue", side_effect=AssertionError("Network")):
            result = terrain.build_terrain_images(ee.Geometry.Point([-115.227, 45.490]))
            self.assertEqual(set(result), {"elevation", "slope", "northness", "eastness",
                                           "elevation_valid", "slope_valid", "aspect_valid"})
            elevation = result["elevation"]
            self.assertIn("USGS/3DEP/10m_collection", elevation.serialize())
            self.assertIn("Collection.first", elevation.serialize())
            self.assertIn("Filter.intersects", elevation.serialize())
            products = result["slope"].args["input"]
            self.assertEqual(products.func.getSignature()["name"], "Terrain.products")
            self.assertEqual(products.args["input"].serialize(), elevation.serialize())
            for name, image in result.items():
                graph = image.serialize()
                for forbidden in ["Image.reproject", "Image.resample", "Image.clip",
                                  "Image.setDefaultProjection", "ImageCollection.mosaic"]:
                    self.assertNotIn(forbidden, graph)
                self.assertEqual("Image.unmask" in graph, name.endswith("_valid"))
            self.assertIn("Image.cos", result["northness"].serialize())
            self.assertIn("Image.sin", result["eastness"].serialize())


if __name__ == "__main__":
    unittest.main()
