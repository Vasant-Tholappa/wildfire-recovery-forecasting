"""Offline checks for exact-union execution, product boundaries and batch joins."""

from unittest.mock import patch

import ee
from ee.apitestcase import ApiTestCase
import numpy as np
import pandas as pd
from shapely.geometry import box

from wildfire_recovery_forecasting import pipeline, reductions
from wildfire_recovery_forecasting.grid import build_fire_grid


class LandsatExecutionTests(ApiTestCase):
    def setUp(self):
        super().setUp()
        self.patches = build_fire_grid(box(0, 0, 270, 90), "fire", perimeter_crs="EPSG:5070")
        self.fire = {"fire_id": "fire", "season1_year": 2018}
        self.projection = ee.Projection("EPSG:32611").atScale(30)
        self.names = ["nbr_pre", "nbr_1", "nbr_5", "nbr_drop",
                      "prefire_selected_year_count", "season1_obs_count", "season5_obs_count"]
        self.images = {name: ee.Image.constant(i).rename(name) for i, name in enumerate(self.names)}

    def raw_sums(self, images, patches, projection):
        result = pd.DataFrame(self.patches[["patch_id", "fire_id"]]).iloc[::-1].reset_index(drop=True)
        result[reductions.RASTER_FULL_AREA] = 8100.
        for name in images:
            # Drop is intentionally not the difference of the patch means:
            # a different common valid support can produce a different result.
            value = {"nbr_pre": .6, "nbr_1": .2, "nbr_drop": .1}.get(name, .5)
            result[name + "_sum"] = [value * 8100, np.nan, 0.]
            result[name + "_area"] = [8100., 0., 8100.]
        return result

    def test_five_independent_requests_schema_missingness_and_pixelwise_drop(self):
        with patch.object(reductions, "area_sums", side_effect=self.raw_sums) as sums:
            result = reductions.reduce_landsat(self.images, ee.FeatureCollection([]),
                expected=self.patches, reference_projection=self.projection)
        self.assertEqual([list(call.args[0]) for call in sums.call_args_list], [
            ["nbr_pre", "prefire_selected_year_count"], ["nbr_1", "season1_obs_count"],
            ["nbr_5"], ["season5_obs_count"], ["nbr_drop"]])
        self.assertIs(sums.call_args_list[-1].args[0]["nbr_drop"], self.images["nbr_drop"])
        self.assertEqual(result.patch_id.tolist(), self.patches.patch_id.tolist())
        self.assertEqual(list(result), ["patch_id", "fire_id"] + [
            name + suffix for name in self.names for suffix in ("_mean", "_valid_fraction")])
        self.assertAlmostEqual(result.nbr_drop_mean.iloc[2], .1)
        self.assertTrue(pd.isna(result.nbr_drop_mean.iloc[1]))
        self.assertEqual(result.nbr_drop_valid_fraction.iloc[1], 0)
        self.assertEqual(result.nbr_drop_mean.iloc[0], 0)

    def test_every_subtable_rejects_missing_extra_duplicate_or_wrong_fire_ids(self):
        for fault in ("missing", "extra", "duplicate", "wrong_fire"):
            def broken(images, patches, projection):
                raw = self.raw_sums(images, patches, projection)
                if list(images) != ["nbr_5"]:
                    return raw
                if fault == "missing": return raw.iloc[:-1]
                if fault == "duplicate": return pd.concat([raw, raw.iloc[:1]])
                if fault == "extra": return raw.assign(patch_id=["extra", *raw.patch_id.iloc[1:]])
                return raw.assign(fire_id="wrong")
            with self.subTest(fault=fault), patch.object(reductions, "area_sums", side_effect=broken):
                with self.assertRaises(ValueError):
                    reductions.reduce_landsat(self.images, ee.FeatureCollection([]),
                        expected=self.patches, reference_projection=self.projection)

    def test_async_products_match_sync_reduction_graphs(self):
        collection = ee.FeatureCollection([])
        graphs = []
        def fetch(raw, columns):
            graphs.append((raw.serialize(), columns))
            names = [name.removesuffix("_sum") for name in columns if name.endswith("_sum")]
            return self.raw_sums({name: self.images[name] for name in names}, collection, self.projection)
        with patch.object(reductions, "fetch_patch_table", side_effect=fetch):
            reductions.reduce_landsat(self.images, collection, expected=self.patches,
                                       reference_projection=self.projection)
        with patch.object(reductions, "fetch_patch_table", side_effect=AssertionError("Fetched")), \
             patch.object(ee.data, "computeFeatures", side_effect=AssertionError("Network")):
            products = reductions.build_landsat_products(self.images, collection,
                                                         reference_projection=self.projection)
        self.assertEqual(len(products), 5)
        self.assertEqual(graphs, [(item["collection"].serialize(), item["columns"])
                                 for item in products.values()])

    def test_exact_union_and_optional_batches_preserve_ids_and_reference(self):
        requested = self.patches.iloc[[2, 0, 1]]
        for size, expected_sizes in ((None, [3]), (2, [2, 1])):
            def reduced(images, collection, *, expected, reference_projection):
                self.assertIs(reference_projection, self.projection)
                result = pd.DataFrame(expected[["patch_id", "fire_id"]]).copy()
                result["nbr_pre_mean"] = np.nan
                return result
            with patch.object(pipeline.landsat, "build_fire_nbr", return_value=self.images) as build, \
                    patch.object(reductions, "reduce_landsat", side_effect=reduced) as reduce:
                result = pipeline.landsat_patch_table(self.fire, requested, batch_size=size,
                                                      reference_projection=self.projection)
            self.assertEqual([len(call.kwargs["expected"]) for call in reduce.call_args_list], expected_sizes)
            self.assertEqual(result.patch_id.tolist(), requested.patch_id.tolist())
            self.assertEqual(result.patch_id.nunique(), 3)
            self.assertTrue(result.nbr_pre_mean.isna().all())
            for call, reduction in zip(build.call_args_list, reduce.call_args_list):
                geometry = call.args[1]
                self.assertIn("Collection.geometry", geometry.serialize())
                self.assertNotIn("Geometry.bounds", geometry.serialize())
                self.assertEqual(geometry.serialize(), reduction.args[1].geometry().serialize())
            if size is not None:
                ids = [value for call in reduce.call_args_list for value in call.kwargs["expected"].patch_id]
                self.assertEqual(ids, sorted(requested.patch_id))

    def test_bad_batches_are_rejected_before_concatenation(self):
        with patch.object(pipeline.landsat, "build_fire_nbr", return_value=self.images), \
                patch.object(reductions, "reduce_landsat", return_value=pd.DataFrame(
                    self.patches[["patch_id", "fire_id"]])):
            with self.assertRaises(ValueError):
                pipeline.landsat_patch_table(self.fire, self.patches, batch_size=2,
                                             reference_projection=self.projection)
        for size in (0, -1, True, 1.5):
            with self.subTest(size=size), self.assertRaises(ValueError):
                pipeline.landsat_patch_table(self.fire, self.patches, batch_size=size)
