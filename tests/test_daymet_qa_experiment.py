"""Candidate C pixel arithmetic and serialized-graph tests, without networking."""

import json
import unittest
from unittest.mock import Mock, patch

import ee
from ee.apitestcase import ApiTestCase
import numpy as np
import pandas as pd

import manual_daymet_qa_experiment as candidate
from wildfire_recovery_forecasting import daymet, reductions


class Pixels:
    """Four bands by pixels; model EE min's masked-member skipping explicitly."""
    def __init__(self, values):
        self.values = np.ma.array(values, dtype=float)

    def select(self, names):
        return self

    def divide(self, days):
        return Pixels(self.values / days)

    def toDouble(self):
        return self

    def mask(self):
        return Pixels((~np.ma.getmaskarray(self.values)).astype(float))

    def unmask(self, value):
        return Pixels(self.values.filled(value))

    def reduce(self, reducer):
        return Pixels(self.values.min(axis=0, keepdims=True))

    def updateMask(self, mask):
        return Pixels(np.ma.array(self.values.data,
            mask=np.ma.getmaskarray(self.values) | ~mask.values.filled(0).astype(bool)))

    def eq(self, value):
        return Pixels(self.values == value)

    def rename(self, name):
        return self

    def setDefaultProjection(self, projection):
        return self


class Annual:
    def __init__(self, images):
        self.values = np.ma.stack([i.values for i in images])

    def min(self):
        return Pixels(self.values.min(axis=0))

    def count(self):
        return Pixels((~np.ma.getmaskarray(self.values)).sum(axis=0))


class ArithmeticTests(unittest.TestCase):
    def test_worst_variable_is_pixelwise_preserves_zero_and_unknown(self):
        pixels = Pixels(np.ma.array([[100, 50, 0, 100], [50, 100, 100, 100],
                                     [100, 100, 100, 100], [100, 100, 100, 100]],
                                    mask=[[0, 0, 0, 1], [0]*4, [0]*4, [0]*4]))
        with patch.object(candidate.ee, "Reducer"):
            result = candidate.worst_variable_coverage(pixels, 100, "native")
        np.testing.assert_allclose(result.values.compressed(), [.5, .5, 0])
        self.assertTrue(result.values.mask[0, 3])
        # Mean of pixel minima is .5, whereas minimum of band means is .75.
        self.assertEqual(result.values[0, :2].mean(), .5)
        self.assertEqual(pixels.values[:, :2].mean(axis=1).min() / 100, .75)

    def test_worst_year_normalization_and_missing_year_mask(self):
        images = {}
        for period in ("pre_s1", "s1"):
            annual = {}
            for year in range(2009, 2019):
                values = np.ma.array(np.ones((4, 3)) * candidate.expected_days(year, period), mask=False)
                if year == 2012:
                    values[0, 0] *= .25
                    values[1, 1] = 0
                    values.mask[2, 2] = True
                annual[year] = Pixels(values)
            images["historical_" + period] = {"valid_days_by_year": annual, "image": Mock()}
            images[period] = {"valid_days": Pixels(np.ones((4, 3)) * candidate.expected_days(2020, period))}
        with patch.object(candidate.ee, "Reducer"), \
             patch.object(candidate.ee.ImageCollection, "fromImages", side_effect=Annual):
            result = candidate.build_temporal_qa(images, {"fire_year": 2019, "season1_year": 2020})
        self.assertEqual(tuple(result), candidate.QA_NAMES)
        for period in ("pre_s1", "s1"):
            observed = result[f"daymet_hist_{period}_completeness"].values
            np.testing.assert_allclose(observed.compressed(), [.25, 0])
            self.assertTrue(observed.mask[0, 2])
            np.testing.assert_allclose(result[f"daymet_{period}_completeness"].values, 1)
        del images["historical_pre_s1"]["valid_days_by_year"][2009]
        with self.assertRaisesRegex(ValueError, "every year"):
            candidate.build_temporal_qa(images, {"fire_year": 2019, "season1_year": 2020})
        self.assertEqual(candidate.expected_days(2020, "pre_s1"), 152)
        self.assertEqual(candidate.expected_days(2019, "pre_s1"), 151)
        self.assertEqual(candidate.expected_days(2020, "s1"), 122)


class GraphTests(ApiTestCase):
    def test_climate_inputs_areas_and_production_unchanged_one_reduction(self):
        fire = {"fire_year": 2019, "season1_year": 2020}
        images = daymet.build_fire_climate_images(fire, ee.Geometry.Point([-111, 33]))
        fc = ee.FeatureCollection([])
        with patch.object(ee.data, "computeValue", side_effect=AssertionError("Network")), \
             patch.object(ee.data, "computeFeatures", side_effect=AssertionError("Network")), \
             patch.object(ee.batch.Export.table, "toCloudStorage", side_effect=AssertionError("Export")), \
             patch.object(reductions, "build_area_product", wraps=reductions.build_area_product) as build:
            full = candidate.combined.build_combined_product(images, fc)
            old_inputs = build.call_args.args[0]
            product = candidate.build_qa4_product(images, fc, fire)
            new_inputs, _, projection = build.call_args.args
            separate = reductions.build_daymet_products(images, fc)
        self.assertEqual(set(separate), {"combined"})
        self.assertEqual(len(full["columns"]), 65)
        self.assertEqual(len(product["columns"]), 41)
        self.assertEqual(len(set(product["columns"])), 41)
        self.assertEqual(projection.serialize(), ee.Projection("EPSG:5070").atScale(30).serialize())
        primary, _, _ = reductions.daymet_area_inputs(images)
        for name in primary:
            self.assertEqual(new_inputs[name].serialize(), old_inputs[name].serialize())
            self.assertEqual(new_inputs[name].mask().serialize(), old_inputs[name].mask().serialize())
        nodes = json.loads(product["collection"].serialize())["values"].values()
        calls = [n.get("functionInvocationValue", {}).get("functionName") for n in nodes]
        self.assertEqual(calls.count("Image.reduceRegions"), 1)
        for name in candidate.QA_NAMES:
            graph = new_inputs[name].serialize()
            self.assertIn('"functionName": "Image.reduce"', graph)
            self.assertNotIn("Image.reduceRegions", graph)
        self.assertIn("2020-10-01", product["collection"].serialize())
        self.assertNotIn("2021-", product["collection"].serialize())
        key = candidate.experiment_identity(60, "r235_c-259")
        self.assertEqual(key["production_version"], "phase3_daymet_qa4_bench_v1_t60")
        self.assertNotEqual(key["production_version"], candidate.combined.VERSION + "_t60")


class ComparisonTests(unittest.TestCase):
    def test_retained_climate_values_validity_and_nulls_checked(self):
        names = daymet.HISTORICAL_BANDS + daymet.PRE_S1_BANDS + daymet.PRE_S1_ANOMALY_BANDS + daymet.S1_ANOMALY_BANDS
        raw = {"patch_id": ["p", "q"], "fire_id": [candidate.combined.baseline.FIRE_ID]*2,
               reductions.RASTER_FULL_AREA: [8100., 8100.]}
        for name in names:
            raw[name + "_sum"] = [100., np.nan]
            raw[name + "_area"] = [8100., np.nan]
        reference = pd.DataFrame(raw)
        for name in candidate.QA_NAMES:
            raw[name + "_sum"] = [0., np.nan]
            raw[name + "_area"] = [8100., np.nan]
        table = pd.DataFrame(raw)
        result = candidate.compare_climate_tables(table, reference, ["p", "q"])
        self.assertTrue(result["climate_equivalent"])
        self.assertFalse(result["qa_pixelwise_accuracy_verified"])
        self.assertEqual(set(result["qa_null_counts"].values()), {1})
        for field in (names[0] + "_sum", names[0] + "_area"):
            bad = table.copy()
            bad.loc[0, field] += 1
            with self.assertRaises(ValueError):
                candidate.compare_climate_tables(bad, reference, ["p", "q"])


if __name__ == "__main__":
    unittest.main()
