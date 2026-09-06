"""Offline EE graph checks using the installed API's bundled algorithm fixture."""

import json
import unittest
from unittest.mock import patch

import ee
from ee.apitestcase import ApiTestCase

from wildfire_recovery_forecasting import landsat


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


if __name__ == "__main__":
    unittest.main()
