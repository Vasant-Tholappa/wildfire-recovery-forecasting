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
                for forbidden in ["Image.resample", "Image.clip",
                                  "Image.setDefaultProjection", "ImageCollection.mosaic"]:
                    self.assertNotIn(forbidden, graph)
                self.assertIn("Image.unmask", graph)
                self.assertIn("Image.reproject", graph)
                self.assertIn("Collection.iterate", graph)
                self.assertIn("system:index", graph)
                self.assertIn("Geometry.buffer", graph)
            self.assertEqual(elevation.args["input"].func.getSignature()["name"], "Image.reproject")
            self.assertIn("Image.cos", result["northness"].serialize())
            self.assertIn("Image.sin", result["eastness"].serialize())


class DEM:
    """Small masked-array oracle for priority filling, not an EE emulator."""
    def __init__(self, values, key="primary"):
        self.values = np.ma.masked_invalid(np.array(values, dtype=float))
        self.key = key
        self.grid = "exact-primary-transform"
    def unmask(self, other, sameFootprint):
        assert sameFootprint is False
        result = DEM(np.where(np.ma.getmaskarray(self.values), other.values.filled(np.nan), self.values.filled(np.nan)))
        result.grid = self.grid
        return result
    def projection(self): return self.grid
    def reproject(self, grid):
        self.used_grid = grid
        return self
    def rename(self, name):
        self.name = name
        return self


class DEMCollection:
    def __init__(self, images): self.images = images
    def sort(self, key):
        assert key == "system:index"
        return DEMCollection(sorted(self.images, key=lambda i: i.key))
    def iterate(self, fn, primary):
        for image in self.images: primary = fn(image, primary)
        return primary


class CompositionTests(unittest.TestCase):
    def test_primary_priority_ordered_alternatives_and_uncovered_mask(self):
        primary = DEM([10, np.nan, np.nan, np.nan])
        a = DEM([99, 20, np.nan, np.nan], "a")
        b = DEM([98, 25, 30, np.nan], "b")
        with patch.object(ee, "Image", side_effect=lambda x:x):
            result = terrain.compose_elevation(primary, DEMCollection([b,a]))
            reversed_result = terrain.compose_elevation(primary, DEMCollection([a,b]))
        np.testing.assert_allclose(result.values.filled(np.nan), [10,20,30,np.nan], equal_nan=True)
        np.testing.assert_allclose(result.values.filled(np.nan), reversed_result.values.filled(np.nan), equal_nan=True)
        self.assertEqual(result.used_grid, primary.projection())
        self.assertEqual(result.name, "elevation")
        self.assertTrue(result.values.mask[-1])

    def test_single_image_preserved(self):
        primary=DEM([10,np.nan,30])
        with patch.object(ee, "Image", side_effect=lambda x:x):
            result=terrain.compose_elevation(primary,DEMCollection([primary]))
        np.testing.assert_allclose(result.values.filled(np.nan),primary.values.filled(np.nan),equal_nan=True)


if __name__ == "__main__":
    unittest.main()
