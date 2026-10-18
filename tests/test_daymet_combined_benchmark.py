"""Offline schema/graph equivalence and strict benchmark comparison checks."""

import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import ee
from ee.apitestcase import ApiTestCase
import numpy as np
import pandas as pd

import manual_daymet_combined_benchmark as experiment
import manual_planb_benchmark as baseline
from wildfire_recovery_forecasting import daymet, reductions


class CombinedGraphTests(ApiTestCase):
    def test_same_inputs_independent_masks_one_boundary_and_union_schema(self):
        images = daymet.build_fire_climate_images(
            {"fire_year": 2019, "season1_year": 2020}, ee.Geometry.Point([-111, 33]))
        patches = ee.FeatureCollection([])
        with patch.object(reductions, "build_area_product", wraps=reductions.build_area_product) as build:
            values, qa, projection = reductions.daymet_area_inputs(images)
            separate = {"primary": reductions.build_area_product(values, patches, projection),
                        "coverage": reductions.build_area_product(qa, patches, projection)}
            primary, coverage = build.call_args_list
            combined = experiment.build_combined_product(images, patches)
            merged = build.call_args_list[-1]
        production = reductions.build_daymet_products(images, patches)
        self.assertEqual(set(production), {"combined"})
        self.assertEqual(production["combined"]["collection"].serialize(), combined["collection"].serialize())
        self.assertEqual(production["combined"]["columns"], combined["columns"])
        expected_inputs = {**primary.args[0], **coverage.args[0]}
        self.assertEqual(set(merged.args[0]), set(expected_inputs))
        for name, image in expected_inputs.items():
            self.assertEqual(merged.args[0][name].serialize(), image.serialize())
        self.assertEqual(primary.args[2].serialize(), merged.args[2].serialize())
        self.assertEqual(coverage.args[2].serialize(), merged.args[2].serialize())
        expected_columns = set(separate["primary"]["columns"]) | set(separate["coverage"]["columns"])
        self.assertEqual(set(combined["columns"]), expected_columns)
        self.assertEqual(len(combined["columns"]), 65)
        self.assertEqual(len(combined["columns"]), len(expected_columns))
        nodes = json.loads(combined["collection"].serialize())["values"].values()
        self.assertEqual(sum(n.get("functionInvocationValue", {}).get("functionName") == "Image.reduceRegions"
                             for n in nodes), 1)
        # Build_area_sums must still assign each variable its own mask/area.
        with patch.object(ee.Image, "updateMask", autospec=True, wraps=None) as mask:
            mask.side_effect = lambda image, own_mask: image
            reductions.build_area_product(expected_inputs, patches, primary.args[2])
        self.assertEqual(mask.call_count, 32)
        for call, image in zip(mask.call_args_list, expected_inputs.values()):
            self.assertEqual(call.args[1].serialize(), image.mask().serialize())


class ComparisonTests(unittest.TestCase):
    def test_production_fetches_once_preserving_all_climate_and_temporal_qa(self):
        names = daymet.HISTORICAL_BANDS + daymet.PRE_S1_BANDS + daymet.PRE_S1_ANOMALY_BANDS + daymet.S1_ANOMALY_BANDS
        qa = [f"daymet_{prefix}{period}_{band}_{suffix}"
              for period in ("pre_s1", "s1") for band in daymet.SUMMARY_BANDS
              for prefix, suffix in (("hist_", "daily_coverage"), ("", "valid_days"))]
        raw = pd.DataFrame({"patch_id": ["p", "q"], "fire_id": ["f", "f"],
                            reductions.RASTER_FULL_AREA: [8100., 8100.]})
        for name in names:
            raw[name + "_sum"] = [100., np.nan]
            raw[name + "_area"] = [4050., np.nan]
        for name in qa:
            raw[name + "_sum"] = [0., 8100.]
            raw[name + "_area"] = [8100., 8100.]
        descriptor = {"columns": list(raw.columns[2:])}
        with patch.object(reductions, "build_daymet_products", return_value={"combined": descriptor}), \
             patch.object(reductions, "fetch_product", return_value=raw) as fetch:
            result = reductions.reduce_daymet({}, object())
        fetch.assert_called_once_with(descriptor)
        expected = reductions.summarize_area_sums(raw, names).rename(
            columns={name + "_mean": name for name in names})
        old_qa = reductions.summarize_area_sums(raw, qa).drop(
            columns=[name + "_valid_fraction" for name in qa])
        expected = expected.merge(old_qa, on=["patch_id", "fire_id"], validate="one_to_one")
        pd.testing.assert_frame_equal(result, expected)
        self.assertEqual(len(names), 16)
        self.assertEqual(len(qa), 16)
        self.assertTrue(result.loc[1, names].isna().all())
        self.assertEqual(result.loc[0, names[0] + "_valid_fraction"], .5)
        self.assertEqual(result.loc[0, qa[0] + "_mean"], 0.)

    def setUp(self):
        for target in ("ee.Initialize", "google.cloud.storage.Client", "ee.data.computeFeatures",
                       "ee.batch.Export.table.toCloudStorage"):
            guard = patch(target, side_effect=AssertionError("Live call forbidden"))
            guard.start()
            self.addCleanup(guard.stop)
        common = {"patch_id": ["p", "q"], "fire_id": [baseline.FIRE_ID] * 2,
                  reductions.RASTER_FULL_AREA: [8100., 8100.]}
        self.primary = pd.DataFrame({**common, "climate_sum": [1., np.nan], "climate_area": [8100., np.nan]})
        self.coverage = pd.DataFrame({**common, "qa_sum": [0., 122.], "qa_area": [8100., 8100.]})
        self.combined = self.primary.assign(qa_sum=self.coverage.qa_sum, qa_area=self.coverage.qa_area)

    def test_comparison_accepts_reordering_nulls_and_tiny_roundoff(self):
        self.combined.loc[0, "climate_sum"] += 1e-8
        result = experiment.compare_tables(self.combined.iloc[::-1], self.primary, self.coverage, ["p", "q"])
        self.assertTrue(result["equivalent"])
        self.assertTrue(result["null_patterns_identical"])

    def test_comparison_rejects_changed_values_nulls_ids_columns_and_denominator(self):
        bad_tables = [self.combined.assign(qa_sum=[0., np.nan]),
                      self.combined.assign(qa_sum=[0., 123.]),
                      self.combined.assign(patch_id=["p", "p"]),
                      self.combined.assign(patch_id=["p", "extra"]),
                      self.combined.drop(columns="qa_area"),
                      self.combined.assign(raster_full_area_m2=8101.)]
        for bad in bad_tables:
            with self.assertRaises(ValueError):
                experiment.compare_tables(bad, self.primary, self.coverage, ["p", "q"])
        bad_reference = self.coverage.assign(raster_full_area_m2=8000.)
        with self.assertRaises(ValueError):
            experiment.compare_tables(self.combined, self.primary, bad_reference, ["p", "q"])

    def test_isolated_paths_and_unchanged_production_inventory(self):
        self.assertEqual(sum(map(len, baseline.INVENTORY.values())), 14)
        self.assertEqual(baseline.INVENTORY["daymet"], {"primary", "coverage"})
        all_paths = set()
        for tile in (60, 90, 120):
            batch = {"tile_side_cells": tile, "batch_id": "r0_c0"}
            paths = experiment.paths(batch, {"source": "daymet", "product": "combined"})
            for path in paths.values():
                self.assertIn(f"phase3_daymet_combined_bench_v1_t{tile}", path)
                self.assertNotIn("phase3_prod_v1", path)
                self.assertNotIn(baseline.VERSION, path)
                all_paths.add(path)
        self.assertEqual(len(all_paths), 9)

    def test_check_reuses_controller_without_submission_or_loop(self):
        batch = {"tile_side_cells": 120, "batch_id": "r0_c0", "expected_patch_ids": ["p"],
                 "products": [{"source": "daymet", "product": "combined", "expected_columns": ["x"], "task_id": "task"}]}
        client = Mock()
        client.list_blobs.return_value = []
        saved = {"status": "RUNNING", "ee_task_id": "task", "started_at": "2026-10-17T00:00:00Z"}
        with patch.object(baseline.persistence, "read_status", return_value=saved) as read, \
             patch.object(baseline.work_items, "lookup_task_status", return_value={"state": "RUNNING"}) as lookup, \
             patch.object(baseline.work_items, "advance_work_item") as advance, \
             patch.object(baseline, "save"):
            baseline.collect(SimpleNamespace(results=Path("unused.json")), {"batches": {"120": batch}},
                             client, identity_fn=experiment.identity, paths_fn=experiment.paths)
        lookup.assert_called_once_with("task")
        advance.assert_not_called()
        self.assertEqual(read.call_args.kwargs["production_version"], "phase3_daymet_combined_bench_v1_t120")


if __name__ == "__main__":
    unittest.main()
