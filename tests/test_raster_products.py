"""Offline Plan B graph inventory and numerical equivalence checks."""

from unittest.mock import patch

import ee
from ee.apitestcase import ApiTestCase
import numpy as np
import pandas as pd
from shapely.geometry import box

from wildfire_recovery_forecasting import pipeline, reductions as r, production
from wildfire_recovery_forecasting.fire_catalog import derive_fire_fields
from wildfire_recovery_forecasting.grid import build_fire_grid


def weighted_raw(values):
    """Two unequal raster contributions per patch, including fractional edge area."""
    weights = np.array([2700., 5402.43])
    data = {"patch_id": ["mixed", "nonforest", "missing"], "fire_id": ["f"] * 3,
            r.RASTER_FULL_AREA: [weights.sum()] * 3}
    for name, pixels in values.items():
        pixels = np.ma.masked_invalid(np.array(pixels, dtype=float))
        area = (~np.ma.getmaskarray(pixels) * weights).sum(axis=1)
        total = (pixels * weights).sum(axis=1).filled(np.nan)
        data[name + "_sum"] = total
        data[name + "_area"] = np.where(area > 0, area, np.nan)
    return pd.DataFrame(data)


class RasterProductTests(ApiTestCase):
    def setUp(self):
        super().setUp()
        self.collection = ee.FeatureCollection([])
        self.projection = ee.Projection("EPSG:5070").atScale(30)

    def test_real_builders_enumerate_12_without_fetch_export_or_grid_rebuild(self):
        ignition = pd.Timestamp("2019-06-08", tz="UTC")
        fire = {"fire_id": "f", "ignition_date": ignition, **derive_fire_fields(ignition)}
        patches = build_fire_grid(box(0, 0, 180, 90), "f", perimeter_crs="EPSG:5070")
        with patch.object(ee.data, "computeFeatures", side_effect=AssertionError("Network")), \
             patch.object(ee.data, "computeValue", side_effect=AssertionError("Network")), \
             patch.object(ee.batch.Export.table, "toCloudStorage", side_effect=AssertionError("Export")), \
             patch.object(pipeline, "build_fire_grid", side_effect=AssertionError("Grid rebuilt")), \
             patch.object(pipeline.disturbance, "build_wildfire_history", side_effect=AssertionError("Vector fetch")):
            products = pipeline.build_raster_products(fire, patches, reference_projection=self.projection)
            expected = {"landsat": {"nbrpre", "season1", "nbr5", "season5_count"},
                        "landfire": {"evc", "height", "evt_support", "evt_categories"},
                        "daymet": {"combined"}, "terrain": {None},
                        "lcms": {None}, "mtbs_severity": {None}}
            self.assertEqual({source: set(parts) for source, parts in products.items()}, expected)
            self.assertEqual(sum(map(len, products.values())), 12)
            paths = set()
            for source, parts in products.items():
                for name, product in parts.items():
                    graph = product["collection"].serialize()
                    self.assertEqual(graph.count('"functionName": "Image.reduceRegions"'), 1)
                    self.assertEqual(len(product["columns"]), len(set(product["columns"])))
                    paths.add(production.raw_export_prefix(source, "f", "r0_c0", product=name))
            self.assertEqual(len(paths), 12)
            self.assertEqual(products["lcms"][None]["expected_years"],
                             {"pre_10yr": 10, "post_y_to_y5": 6})
            self.assertEqual(len(products["lcms"][None]["columns"]), 17)
            self.assertEqual(len(products["daymet"]["combined"]["columns"]), 65)
            for product in products["landsat"].values():
                graph = product["collection"].serialize()
                self.assertIn("Collection.geometry", graph)
                self.assertNotIn("Geometry.bounds", graph)

    def test_evc_combination_preserves_known_nonforest_and_measurement_support(self):
        images = {name: ee.Image(name) for name in
                  ("forest_indicator", "forest_mask", "tree_cover_pct", "tree_height_m", "evt")}
        forest = [[1, 0], [0, 0], [np.nan, np.nan]]
        cover = [[40, np.nan], [np.nan, np.nan], [np.nan, np.nan]]
        captured = []
        def build(values, patches, projection):
            captured.append((values, projection))
            return self.collection
        with patch.object(r, "build_area_sums", side_effect=build):
            products = r.build_landfire_products(images, self.collection)
        values, projection = captured[0]
        self.assertEqual(set(values), {"forest", "tree_cover", "support"})
        self.assertIs(values["forest"], images["forest_indicator"])
        self.assertIs(values["support"], images["forest_indicator"])
        self.assertIs(values["tree_cover"], images["tree_cover_pct"])
        self.assertEqual(projection.serialize(), images["forest_indicator"].projection().serialize())
        self.assertEqual(captured[1][1].serialize(), images["tree_height_m"].projection().serialize())
        self.assertEqual(captured[2][1].serialize(), images["evt"].projection().serialize())
        self.assertEqual(set(products), {"evc", "height", "evt_support", "evt_categories"})
        combined = weighted_raw({"forest": forest, "tree_cover": cover, "support": forest})
        old_forest = r.summarize_area_sums(weighted_raw({"forest": forest}), ["forest"], fractions=["forest"])
        old_cover = r.summarize_area_sums(weighted_raw({"tree_cover": cover, "support": forest}),
                                         ["tree_cover"], support="support")
        pd.testing.assert_frame_equal(old_forest, r.summarize_area_sums(combined, ["forest"], fractions=["forest"]))
        pd.testing.assert_frame_equal(old_cover, r.summarize_area_sums(combined, ["tree_cover"], support="support"))
        self.assertEqual(old_forest.forest_valid_fraction.tolist(), [1., 1., 0.])
        self.assertEqual(old_forest.forest_fraction.iloc[1], 0.)
        self.assertTrue(pd.isna(old_forest.forest_fraction.iloc[2]))
        self.assertEqual(old_cover.tree_cover_mean.iloc[0], 40.)
        self.assertTrue(old_cover.tree_cover_mean.iloc[1:].isna().all())
        self.assertEqual(old_cover.tree_cover_valid_fraction.tolist(), [1., 0., 0.])

    def test_lcms_shared_sum_matches_separate_windows_with_different_masks(self):
        names = ("clear_disturbance", "other_loss", "complete_coverage", "valid_years")
        pixels = {
            "pre_10yr": {"clear_disturbance": [[1, np.nan], [0, 0], [np.nan, np.nan]],
                         "other_loss": [[0, np.nan], [0, 0], [np.nan, np.nan]],
                         "complete_coverage": [[1, 0], [1, 1], [0, 0]],
                         "valid_years": [[10, 0], [10, 10], [0, 0]]},
            "post_y_to_y5": {"clear_disturbance": [[0, 0], [np.nan, 1], [np.nan, np.nan]],
                              "other_loss": [[0, 1], [np.nan, 0], [np.nan, np.nan]],
                              "complete_coverage": [[1, 1], [0, 0], [0, 0]],
                              "valid_years": [[6, 6], [0, 3], [0, 0]]}}
        history = {"windows": {window: {**{name: ee.Image(window + name) for name in names},
                                       "expected_years": expected}
                               for window, expected in (("pre_10yr", 10), ("post_y_to_y5", 6))}}
        with patch.object(r, "build_area_sums", return_value=self.collection) as build:
            product = r.build_lcms_product(history, self.collection)
        self.assertEqual(build.call_count, 1)
        values = build.call_args.args[0]
        for window, data in history["windows"].items():
            for name in names:
                self.assertIs(values[f"lcms_{window}_{name}"], data[name])
        raw = weighted_raw({f"lcms_{window}_{name}": value
                            for window, data in pixels.items() for name, value in data.items()})
        result = r.summarize_lcms_product(raw, product["expected_years"])
        old = []
        for window, data in pixels.items():
            part = r.summarize_area_sums(weighted_raw(data), names,
                                        fractions=names[:3])
            part = part.rename(columns={name: f"lcms_{window}_{name}" for name in part
                                        if name not in ("patch_id", "fire_id")})
            part[f"lcms_{window}_expected_years"] = product["expected_years"][window]
            old.append(part)
        expected = old[0].merge(old[1], on=["patch_id", "fire_id"], validate="one_to_one")
        pd.testing.assert_frame_equal(result, expected)
        self.assertEqual(len(result.columns), 20)
        self.assertFalse(any("first_year" in name or "timing" in name for name in result))
        self.assertTrue(pd.isna(result.lcms_post_y_to_y5_clear_disturbance_fraction.iloc[2]))
        self.assertEqual(result.lcms_pre_10yr_clear_disturbance_fraction.iloc[1], 0)
