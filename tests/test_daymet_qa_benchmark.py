"""Single-task Candidate C launcher checks without EE/GCS access."""

import unittest
import json
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

import manual_daymet_qa_benchmark as runner
from wildfire_recovery_forecasting import reductions


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.ids = [f"{runner.baseline.FIRE_ID}_r{row}_c{col}"
                    for row in range(14100, 14160) for col in range(-15540, -15480)]
        self.batch = {"tile_side_cells": 60, "batch_id": runner.BATCH_ID,
                      "patch_count": 3600, "expected_patch_ids": self.ids, "products": []}
        for target in ("ee.Initialize", "google.cloud.storage.Client", "ee.data.computeFeatures",
                       "ee.batch.Export.table.toCloudStorage"):
            guard = patch(target, side_effect=AssertionError("Live call forbidden"))
            guard.start()
            self.addCleanup(guard.stop)

    def test_fixture_is_exact_full_canonical_tile_and_rejects_foreign_cell(self):
        manifest = gpd.GeoDataFrame({"fire_id": [runner.baseline.FIRE_ID],
                                    "fire_year": [2019], "season1_year": [2020]},
            geometry=[box(-15540*90, 14100*90, -15480*90, 14160*90)], crs="EPSG:5070")
        with patch.object(runner.gpd, "read_parquet", return_value=manifest):
            _, patches = runner.load_fixture("unused", self.batch)
            self.assertEqual(patches.patch_id.tolist(), self.ids)
            self.assertTrue(patches.geometry.area.eq(8100).all())
            self.assertTrue(patches.fire_overlap_fraction.eq(1).all())
            bad = {**self.batch, "expected_patch_ids": [self.ids[0].replace("r14100", "r0"), *self.ids[1:]]}
            with self.assertRaises(ValueError):
                runner.load_fixture("unused", bad)

    def test_submit_invokes_one_controller_item_and_reuses_identity(self):
        descriptor = {"collection": object(), "columns": [f"field{i}" for i in range(41)]}
        saved = {"ee_task_id": "task", "started_at": "now", "status": "RUNNING"}
        results = {"batches": {"60": self.batch}}
        with patch.object(runner.pipeline, "patches_to_ee", return_value=Mock()), \
             patch.object(runner.daymet, "build_fire_climate_images", return_value={}), \
             patch.object(runner.experiment, "build_qa4_product", return_value=descriptor), \
             patch.object(runner.baseline, "save"), \
             patch.object(runner.work_items, "advance_work_item",
                          return_value={"action": "SUBMITTED", "status": saved}) as advance:
            runner.submit({}, Mock(), results, Mock())
            advance.assert_called_once()
            self.assertEqual(advance.call_args.kwargs["production_version"], "phase3_daymet_qa4_bench_v1_t60")
            self.assertEqual(advance.call_args.kwargs["product"], "qa4")
            advance.return_value = {"action": "WAITING", "status": saved}
            runner.submit({}, Mock(), results, Mock())
        self.assertEqual(len(self.batch["products"]), 1)
        self.assertEqual(self.batch["products"][0]["task_id"], "task")
        for path in runner.paths(self.batch, self.batch["products"][0]).values():
            self.assertIn("phase3_daymet_qa4_bench_v1_t60", path)
            self.assertNotIn("phase3_daymet_combined", path)

    def test_identity_rejects_other_tiles_or_products(self):
        for batch, product in (({**self.batch, "tile_side_cells": 120}, "qa4"),
                               (self.batch, "combined")):
            with self.assertRaises(ValueError):
                runner.identity(batch, {"source": "daymet", "product": product})


class CollectionComparisonTests(unittest.TestCase):
    """Exercise CLI dispatch, real collector/controller, and real comparison offline."""

    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "results.json"
        self.batch = {"tile_side_cells": 60, "batch_id": runner.BATCH_ID,
                      "patch_count": 2, "expected_patch_ids": ["p", "q"]}
        self.item = {"source": "daymet", "product": "qa4", "task_id": "saved-task"}
        self.item.update(runner.paths(self.batch, self.item))
        self.names = (runner.daymet.HISTORICAL_BANDS + runner.daymet.PRE_S1_BANDS
                      + runner.daymet.PRE_S1_ANOMALY_BANDS + runner.daymet.S1_ANOMALY_BANDS)
        raw = {"patch_id": ["p", "q"], "fire_id": [runner.baseline.FIRE_ID]*2,
               reductions.RASTER_FULL_AREA: [8100., 8100.]}
        for name in self.names:
            raw[name + "_sum"] = [100., None]
            raw[name + "_area"] = [8100., None]
        self.reference_table = pd.DataFrame(raw)
        for name in runner.experiment.QA_NAMES:
            raw[name + "_sum"] = [0., None]
            raw[name + "_area"] = [8100., None]
        self.table = pd.DataFrame(raw)
        self.item["expected_columns"] = list(self.table.columns[2:])
        self.batch["products"] = [self.item]
        self.results = {"benchmark": runner.experiment.VERSION,
                        "fire_id": runner.baseline.FIRE_ID, "batches": {"60": self.batch}}
        self.old = {"source": "daymet", "product": "combined",
                    "status": {"status": "COMPLETE"},
                    "ee_task": {"batch_eecu_usage_seconds": 302.1731262207031},
                    "parquet_path": runner.production.parquet_path(
                        "daymet", runner.baseline.FIRE_ID, runner.BATCH_ID,
                        product="combined", production_version=runner.combined.VERSION + "_t60")}
        self.reference = {**self.batch, "products": [self.old]}
        self.client = Mock()
        self.client.list_blobs.return_value = []
        self.saved = {"status": "RUNNING", "stage": "EE_EXPORT",
                      "ee_task_id": "saved-task", "started_at": "2026-10-17T00:00:00Z"}
        self.task = {"state": "COMPLETED", "creation_timestamp_ms": 1000,
                     "start_timestamp_ms": 2000, "update_timestamp_ms": 5000,
                     "batch_eecu_usage_seconds": 123.5}
        def mock_target(target, **kwargs):
            mocker = patch(target, **kwargs)
            result = mocker.start()
            self.addCleanup(mocker.stop)
            return result
        self.export = mock_target("wildfire_recovery_forecasting.exports.start_table_export",
                                  side_effect=AssertionError("Collection must never submit"))
        mock_target("ee.batch.Export.table.toCloudStorage", side_effect=AssertionError("Export forbidden"))
        mock_target("ee.data.computeFeatures", side_effect=AssertionError("Live evaluation forbidden"))
        self.lookup = mock_target("wildfire_recovery_forecasting.work_items.lookup_task_status",
                                  return_value=self.task)
        self.read_status = mock_target("wildfire_recovery_forecasting.persistence.read_status",
                                       return_value=self.saved)
        self.write_status = mock_target("wildfire_recovery_forecasting.persistence.write_status",
                                        side_effect=lambda **kw: {
                                            k: v.isoformat() if isinstance(v, datetime) else v
                                            for k, v in kw.items() if k != "client"})
        self.convert = mock_target("wildfire_recovery_forecasting.persistence.csv_to_parquet",
                                   return_value={"row_count": 2, "output_uri": "gs://test/result.parquet"})

    def run_mode(self, mode):
        self.path.write_text(json.dumps(self.results))
        with patch.object(runner, "RESULTS", self.path), \
             patch.object(runner, "load_reference", return_value=self.reference), \
             patch.object(runner.baseline, "record_git_commit"), \
             patch.object(runner.ee, "Initialize") as initialize, \
             patch.object(runner.storage, "Client", return_value=self.client), \
             patch("sys.argv", ["manual_daymet_qa_benchmark.py", mode]):
            runner.main()
            if mode == "compare":
                initialize.assert_not_called()
        self.export.assert_not_called()
        return json.loads(self.path.read_text())

    def test_check_completes_using_experimental_identity_and_records_metrics(self):
        blob = Mock(size=100, download_as_bytes=Mock(return_value=self.table.to_parquet(index=False)))
        self.client.bucket.return_value.get_blob.return_value = blob
        raw = Mock(size=50)
        raw.name = self.item["raw_prefix"] + ".csv"
        self.client.list_blobs.return_value = [raw]
        result = self.run_mode("check")["batches"]["60"]["products"][0]
        self.lookup.assert_called_once_with("saved-task")
        self.convert.assert_called_once()
        self.assertEqual(self.convert.call_args.kwargs["production_version"], "phase3_daymet_qa4_bench_v1_t60")
        self.assertEqual(self.convert.call_args.kwargs["product"], "qa4")
        for call in self.read_status.call_args_list:
            self.assertEqual(call.kwargs["product"], "qa4")
        self.client.bucket.return_value.get_blob.assert_called_once_with(self.item["parquet_path"])
        self.assertEqual(result["status"]["status"], "COMPLETE")
        self.assertEqual(result["eecu_seconds"], 123.5)
        self.assertEqual((result["execution_seconds"], result["elapsed_seconds"]), (3, 4))
        self.assertEqual((result["row_count"], result["raw_shard_count"], result["raw_size_bytes"]), (2, 1, 50))
        self.assertGreater(result["parquet_null_value_count"], 0)
        self.assertNotIn("harness_error", result)

    def test_check_waits_without_conversion_or_submission(self):
        self.task["state"] = "RUNNING"
        result = self.run_mode("check")["batches"]["60"]["products"][0]
        self.assertEqual(result["action"], "WAITING")
        self.lookup.assert_called_once_with("saved-task")
        self.convert.assert_not_called()
        self.write_status.assert_not_called()

    def test_check_records_ee_failure_and_does_not_convert(self):
        self.task.update(state="FAILED", error_message="User memory limit exceeded")
        result = self.run_mode("check")["batches"]["60"]["products"][0]
        self.assertEqual(result["status"]["status"], "FAILED")
        self.assertEqual(result["status"]["stage"], "EE_EXPORT")
        self.assertIn("memory", result["status"]["error_message"])
        self.convert.assert_not_called()

    def test_check_missing_status_records_error_without_submission(self):
        self.read_status.return_value = None
        result = self.run_mode("check")["batches"]["60"]["products"][0]
        self.assertIn("Missing persisted task ID", result["harness_error"])
        self.lookup.assert_not_called()
        self.convert.assert_not_called()

    def test_check_records_actual_downstream_failure_stage(self):
        def fail(*args, **kwargs):
            kwargs["on_stage"]("STRUCTURAL_VALIDATION")
            raise ValueError("Wrong patch IDs")
        self.convert.side_effect = fail
        result = self.run_mode("check")["batches"]["60"]["products"][0]
        self.assertEqual(result["status"]["status"], "FAILED")
        self.assertEqual(result["status"]["stage"], "STRUCTURAL_VALIDATION")
        self.assertEqual(result["status"]["error_message"], "Wrong patch IDs")

    def prepare_compare(self):
        self.item.update(status={"status": "COMPLETE"}, ee_task=self.task)
        tables = {self.item["parquet_path"]: self.table, self.old["parquet_path"]: self.reference_table}
        self.client.bucket.return_value.blob.side_effect = lambda path: Mock(
            download_as_bytes=Mock(return_value=tables[path].to_parquet(index=False)))

    def test_compare_reads_combined_reference_and_leaves_pixel_audit_unverified(self):
        self.prepare_compare()
        result = self.run_mode("compare")["comparison"]
        self.assertTrue(result["climate_equivalent"])
        self.assertFalse(result["qa_pixelwise_accuracy_verified"])
        self.assertEqual(result["combined_reference_eecu_seconds"], 302.1731262207031)
        self.assertEqual([c.args[0] for c in self.client.bucket.return_value.blob.call_args_list],
                         [self.item["parquet_path"], self.old["parquet_path"]])
        self.lookup.assert_not_called()
        self.convert.assert_not_called()

    def test_compare_rejects_changed_climate_sums_areas_and_nulls(self):
        for suffix, value in (("_sum", 101.), ("_area", 8000.), ("_sum", None)):
            with self.subTest(suffix=suffix, value=value):
                original = self.table.copy()
                self.table.loc[0, self.names[0] + suffix] = value
                self.prepare_compare()
                with self.assertRaisesRegex(ValueError, "Climate field changed"):
                    self.run_mode("compare")
                self.assertNotIn("comparison", json.loads(self.path.read_text()))
                self.table = original
        self.lookup.assert_not_called()
        self.export.assert_not_called()


if __name__ == "__main__":
    unittest.main()
