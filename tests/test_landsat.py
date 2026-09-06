"""Offline EE graph checks using the installed API's bundled algorithm fixture."""

import json
import unittest
from unittest.mock import patch

import ee
import numpy as np
from ee.apitestcase import ApiTestCase

from wildfire_recovery_forecasting import landsat


class PixelImage:
    """Small masked pixel arrays for exercising orchestration without EE."""

    def __init__(self, values):
        self.values = np.ma.array(values)

    def rename(self, name):
        return self

    def setDefaultProjection(self, projection):
        return self

    def clip(self, geometry):
        return self

    def _binary(self, other, operation):
        return PixelImage(operation(self.values, other.values if isinstance(other, PixelImage) else other))

    def gte(self, other):
        return self._binary(other, np.greater_equal)

    def lt(self, other):
        return self._binary(other, np.less)

    def eq(self, other):
        return self._binary(other, np.equal)

    def And(self, other):
        return self._binary(other, np.logical_and)

    def add(self, other):
        return self._binary(other, np.add)

    def subtract(self, other):
        return self._binary(other, np.subtract)

    def mask(self):
        return PixelImage(~np.ma.getmaskarray(self.values))

    def unmask(self, value, sameFootprint=False):
        return PixelImage(self.values.filled(value))

    def updateMask(self, mask):
        return PixelImage(np.ma.array(
            self.values.data,
            mask=np.ma.getmaskarray(self.values) | ~np.asarray(mask.values.filled(0), dtype=bool),
        ))


def graph_calls(value):
    """Resolve serialized argument references so graph assertions are readable."""
    graph = json.loads(value.serialize())
    nodes = graph["values"]

    def resolve(argument):
        if isinstance(argument, list):
            return [resolve(item) for item in argument]
        if not isinstance(argument, dict):
            return argument
        if "valueReference" in argument:
            return resolve(nodes[argument["valueReference"]])
        if "constantValue" in argument:
            return argument["constantValue"]
        if "functionInvocationValue" in argument:
            call = argument["functionInvocationValue"]
            return (call["functionName"], {k: resolve(v) for k, v in call["arguments"].items()})
        if "functionDefinitionValue" in argument:
            definition = argument["functionDefinitionValue"]
            return (definition["argumentNames"], resolve(nodes[definition["body"]]))
        return {key: resolve(value) for key, value in argument.items()}

    calls = []

    def visit(node):
        if isinstance(node, dict):
            if "functionInvocationValue" in node:
                calls.append(resolve(node))
            for child in node.values():
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(nodes)
    return calls


class LandsatTests(ApiTestCase):
    def test_sensor_transition(self):
        for year, expected in [(2008, "L7"), (2012, "L7"), (2013, "L8"), (2025, "L8")]:
            with self.subTest(year=year):
                self.assertEqual(landsat.sensor_for_year(year), expected)
        with self.assertRaises(ValueError):
            landsat.preprocess_landsat(ee.Image(0), "L9")

    def test_sensor_bands_qa_and_scaling(self):
        for sensor, nir, pixel_bits, saturation_bits in [
            ("L7", "SR_B4", 59, 584), ("L8", "SR_B5", 63, 2128),
        ]:
            with self.subTest(sensor=sensor):
                image = landsat.preprocess_landsat(ee.Image("scene"), sensor)
                calls = graph_calls(image)
                selectors = [args["bandSelectors"] for name, args in calls if name == "Image.select"]
                self.assertIn([nir, "SR_B7"], selectors)
                self.assertIn(["QA_PIXEL"], selectors)
                self.assertIn(["QA_RADSAT"], selectors)
                self.assertEqual(len(selectors), 3)  # No aerosol or other-band exclusions.
                masks = [args["image2"][1]["value"] for name, args in calls if name == "Image.bitwiseAnd"]
                self.assertCountEqual(masks, [pixel_bits, saturation_bits])
                scales = [args["image2"][1]["value"] for name, args in calls if name == "Image.multiply"]
                offsets = [args["image2"][1]["value"] for name, args in calls if name == "Image.add"]
                self.assertEqual(scales, [0.0000275])
                self.assertEqual(offsets, [-0.2])
                self.assertIn("Image.copyProperties", [name for name, _ in calls])

    def test_nbr_uses_explicit_arithmetic_and_masks_zero_denominator(self):
        calls = graph_calls(landsat.calculate_nbr(ee.Image("reflectance")))
        names = [name for name, _ in calls]
        self.assertNotIn("Image.normalizedDifference", names)  # Would mask negative inputs.
        divide = next(args for name, args in calls if name == "Image.divide")
        self.assertEqual(divide["image1"][0], "Image.subtract")
        self.assertEqual(divide["image2"][0], "Image.add")
        mask = next(args for name, args in calls if name == "Image.updateMask")["mask"]
        self.assertEqual(mask[0], "Image.neq")
        self.assertEqual(mask[1]["image1"], divide["image2"])
        self.assertEqual(mask[1]["image2"][1]["value"], 0)

    def test_season_dates_collections_and_daily_reduction(self):
        geometry = ee.Geometry.Rectangle([-116, 45, -115, 46])
        for year, collection in [(2012, "LE07"), (2013, "LC08")]:
            with self.subTest(year=year), patch.object(ee.data, "computeValue", side_effect=AssertionError("Network evaluation")):
                result = landsat.build_seasonal_nbr(year, geometry, reference_projection=ee.Projection("EPSG:32611"))
                calls = graph_calls(result["nbr"])
                names = [name for name, _ in calls]
                text = result["nbr"].serialize()
                self.assertIn(f"LANDSAT/{collection}/C02/T1_L2", text)
                self.assertIn(f"{year}-06-01", text)
                self.assertIn(f"{year}-10-01", text)
                self.assertIn("List.distinct", names)
                self.assertIn("List.sort", names)
                self.assertIn("Date.format", names)
                self.assertEqual(names.count("reduce.median"), 2)
                count_calls = graph_calls(result["obs_count"])
                count_input = next(args["collection"] for name, args in count_calls if name == "reduce.count")
                median_inputs = [args["collection"] for name, args in calls if name == "reduce.median"]
                self.assertIn(count_input, median_inputs)  # Both reduce the same daily collection.
                self.assertEqual(set(result), {"nbr", "obs_count", "scene_count", "date_count"})

    def test_projection_empty_schema_and_missingness(self):
        result = landsat.build_seasonal_nbr(
            2018, ee.Geometry.Point([-115, 45]), reference_projection=ee.Projection("EPSG:32611"),
        )
        calls = graph_calls(result["nbr"])
        names = [name for name, _ in calls]
        self.assertNotIn("Image.reproject", names)
        self.assertNotIn("Image.unmask", names)  # Missing NBR never becomes zero.
        self.assertEqual(names.count("Image.setDefaultProjection"), 3)  # Daily, seed, seasonal.
        scale = next(args["meters"] for name, args in calls if name == "Projection.atScale")
        self.assertEqual(scale, 30)
        merge = next(args for name, args in calls if name == "ImageCollection.merge")
        self.assertIn("Image.updateMask", str(merge["collection2"]))
        count_calls = graph_calls(result["obs_count"])
        self.assertIn("Image.unmask", [name for name, _ in count_calls])
        self.assertNotIn("ImageCollection.merge", [name for name, _ in graph_calls(result["date_count"])])
        with self.assertRaises(ValueError):
            landsat.build_seasonal_nbr(2018, ee.Geometry.Point([0, 0]), reference_projection=None)


class FireNBRTests(ApiTestCase):
    def setUp(self):
        super().setUp()
        self.fire = dict(prefire_search_start_year=2012, prefire_search_end_year=2016,
                         season1_year=2018, season5_year=2022)

    def test_latest_three_selection_and_missing_pixels_numerically(self):
        # Columns are independent pixels: all usable, recent unusable,
        # only two usable, missing annual NBR despite count, no usable years.
        counts = {
            2016: [3, 2, 3, 3, 0], 2015: [3, 3, 0, 3, 0],
            2014: [3, 3, 2, 3, 0], 2013: [3, 3, 3, 3, 0],
            2012: [3, 3, 0, 3, 0],
        }
        values = {2016: -0.7, 2015: 0.8, 2014: 0.4, 2013: 0.9, 2012: 0.6}

        def season(year, geometry, reference_projection):
            if year in counts:
                nbr = np.ma.array([values[year]] * 5, mask=[False, False, False, year == 2015, False])
                return {"nbr": PixelImage(nbr), "obs_count": PixelImage(counts[year])}
            # A single date remains allowed for Season 1/5 in this substep.
            return {"nbr": PixelImage([0.2 if year == 2018 else 0.9] * 5),
                    "obs_count": PixelImage([1] * 5)}

        with patch.object(landsat, "build_seasonal_nbr", side_effect=season) as seasonal, \
             patch.object(landsat, "ee") as api:
            api.Image.constant.side_effect = PixelImage
            api.ImageCollection.fromImages.return_value.median.side_effect = lambda: PixelImage(
                np.ma.median(np.ma.stack([
                    image.values for image in api.ImageCollection.fromImages.call_args.args[0]
                ]), axis=0)
            )
            result = landsat.build_fire_nbr(self.fire, "geometry", reference_projection=api.Projection())
            self.assertEqual([call.args[0] for call in seasonal.call_args_list],
                             [2016, 2015, 2014, 2013, 2012, 2018, 2022])
            np.testing.assert_allclose(result["nbr_pre"].values.compressed(), [0.4, 0.8, 0.4])
            np.testing.assert_array_equal(np.ma.getmaskarray(result["nbr_pre"].values), [0, 0, 1, 0, 1])
            np.testing.assert_array_equal(result["prefire_selected_year_count"].values, [3, 3, 2, 3, 0])
            np.testing.assert_array_equal(result["nbr_pre_valid"].values, [1, 1, 0, 1, 0])
            np.testing.assert_allclose(result["nbr_drop"].values.compressed(), [0.2, 0.6, 0.2])
            np.testing.assert_array_equal(np.ma.getmaskarray(result["nbr_drop"].values), [0, 0, 1, 0, 1])
            for key in ["nbr_1", "nbr_5"]:
                self.assertFalse(np.ma.getmaskarray(result[key].values).any())
            for key in ["season1_obs_count", "season5_obs_count"]:
                np.testing.assert_array_equal(result[key].values, [1] * 5)

    def test_manifest_years_projection_and_offline_graph(self):
        self.fire.update(prefire_search_start_year=2008, prefire_search_end_year=2012,
                         season1_year=2014, season5_year=2018)
        projection = ee.Projection("EPSG:32611")
        geometry = ee.Geometry.Point([-115, 45])
        with patch.object(landsat, "build_seasonal_nbr") as seasonal, \
             patch.object(ee.data, "computeValue", side_effect=AssertionError("Network evaluation")):
            seasonal.return_value = {"nbr": ee.Image(0.5).rename("NBR"),
                                     "obs_count": ee.Image(3).rename("valid_obs_count")}
            result = landsat.build_fire_nbr(self.fire, geometry, reference_projection=projection)
            self.assertEqual([call.args[0] for call in seasonal.call_args_list],
                             [2012, 2011, 2010, 2009, 2008, 2014, 2018])
            for call in seasonal.call_args_list:
                self.assertEqual(call.args[1], geometry)
                self.assertEqual(call.kwargs, {"reference_projection": projection})
            calls = graph_calls(result["nbr_pre"])
            self.assertNotIn("Image.reproject", [name for name, _ in calls])
            self.assertIn(("Projection.atScale", {"projection": ("Projection", {"crs": "EPSG:32611"}), "meters": 30}), calls)
            for image in result.values():
                self.assertIsInstance(image, ee.Image)
                image.serialize()

    def test_invalid_manifest_pool_is_rejected_before_seasonal_calls(self):
        for change in [dict(prefire_search_start_year=2011), dict(prefire_search_end_year=2015),
                       dict(season1_year=None), dict(season5_year=2022.5)]:
            with self.subTest(change=change), patch.object(landsat, "build_seasonal_nbr") as seasonal:
                with self.assertRaises(ValueError):
                    landsat.build_fire_nbr(self.fire | change, None, reference_projection=ee.Projection("EPSG:32611"))
                seasonal.assert_not_called()


if __name__ == "__main__":
    unittest.main()
