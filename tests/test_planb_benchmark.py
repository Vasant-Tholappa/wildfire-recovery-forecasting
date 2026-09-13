"""Targeted offline checks for the manual benchmark harness."""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

import manual_planb_benchmark as b


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        for target in ("manual_planb_benchmark.ee.Initialize", "manual_planb_benchmark.storage.Client",
                       "manual_planb_benchmark.ee.data.computeFeatures",
                       "manual_planb_benchmark.ee.batch.Export.table.toCloudStorage"):
            guard = patch(target, side_effect=AssertionError("Live call forbidden"))
            guard.start()
            self.addCleanup(guard.stop)

    def test_densest_selection_is_order_independent_and_preserves_batch(self):
        rows = [1, 2, 3, 70, -1]
        grid = gpd.GeoDataFrame({"fire_id": [b.FIRE_ID] * 5, "patch_id": list("abcde"),
                                "grid_row": rows, "grid_col": rows},
                               geometry=[box(n * 90, n * 90, (n+1)*90, (n+1)*90) for n in rows],
                               crs="EPSG:5070")
        for tile in (60, 90, 120, 150):
            first = b.select_batch(grid, tile)
            second = b.select_batch(grid.sample(frac=1, random_state=7), tile)
            self.assertEqual(first.patch_id.tolist(), second.patch_id.tolist())
            self.assertTrue(first.geometry.equals(second.geometry))
            self.assertEqual(first.batch_id.nunique(), 1)
            self.assertEqual(len(first), 3 if tile == 60 else 4)

    def test_inventory_and_distinct_benchmark_paths(self):
        products = {source: {name: {"collection": object(), "columns": ["x_sum"]}
                             for name in names} for source, names in b.INVENTORY.items()}
        entries = b.product_entries(products)
        self.assertEqual(len(entries), 14)
        all_paths = set()
        for tile in (60, 90, 120, 150):
            batch = {"tile_side_cells": tile, "batch_id": "r0_c0"}
            for source, name, _ in entries:
                item = {"source": source, "product": name}
                paths = b.paths(batch, item)
                self.assertEqual(len(paths), 3)
                for path in paths.values():
                    self.assertNotIn("phase3_prod_v1", path)
                    self.assertIn(b.benchmark_version(tile), path)
                    self.assertEqual("product=" in path, name is not None)
                    all_paths.add(path)
        self.assertEqual(len(all_paths), 168)
        for tile in (10, 180, 0, True):
            with self.assertRaises(ValueError):
                b.benchmark_version(tile)
        products["landsat"].pop("nbr5")
        with self.assertRaises(ValueError):
            b.product_entries(products)

    def test_check_active_task_looks_up_once_without_submission(self):
        item = {"source": "terrain", "product": None, "expected_columns": ["x_sum"], "task_id": "id"}
        batch = {"tile_side_cells": 60, "batch_id": "r0_c0", "expected_patch_ids": ["p"], "products": [item]}
        client = Mock()
        client.list_blobs.return_value = []
        saved = {"status": "RUNNING", "ee_task_id": "id", "started_at": "2026-09-13T00:00:00+00:00"}
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(b.persistence, "read_status", return_value=saved), \
             patch.object(b.work_items, "lookup_task_status", return_value={"state": "RUNNING"}) as lookup, \
             patch.object(b.work_items, "advance_work_item", side_effect=AssertionError("Unexpected advance")):
            b.collect(SimpleNamespace(results=Path(directory) / "results.json"),
                      {"batches": {"60": batch}}, client)
            lookup.assert_called_once_with("id")
            self.assertEqual(item["action"], "WAITING")
            self.assertNotIn("harness_error", item)

    def test_git_commit_recorded_and_initial_commit_preserved(self):
        results = {}
        with patch.object(b.subprocess, "run", return_value=SimpleNamespace(stdout="a" * 40 + "\n")):
            b.record_git_commit(results)
        self.assertEqual(json.loads(json.dumps(results))["git_commit"], "a" * 40)
        with patch.object(b.subprocess, "run", return_value=SimpleNamespace(stdout="b" * 40)):
            b.record_git_commit(results)
        self.assertEqual(results["git_commit"], "a" * 40)
        self.assertEqual(results["last_invocation_git_commit"], "b" * 40)
        for failure in (FileNotFoundError("git missing"),
                        subprocess.CalledProcessError(128, "git", stderr="not a repository")):
            with patch.object(b.subprocess, "run", side_effect=failure), \
                 self.assertRaisesRegex(RuntimeError, "Cannot determine benchmark Git commit"):
                b.record_git_commit({})
        with patch.object(b.subprocess, "run", return_value=SimpleNamespace(stdout="")), \
             self.assertRaisesRegex(RuntimeError, "invalid git rev-parse"):
            b.record_git_commit({})

    def test_completed_evt_collect_records_groups_diagnostics(self):
        representative = '[{"class":9000,"sum":1800}]'
        table = pd.DataFrame({"patch_id": ["p", "q"], "fire_id": [b.FIRE_ID] * 2,
                              "groups": [representative, None]})
        item = {"source": "landfire", "product": "evt_categories", "expected_columns": ["groups"], "task_id": "id"}
        batch = {"tile_side_cells": 60, "batch_id": "r0_c0", "expected_patch_ids": ["p", "q"], "products": [item]}
        client = Mock()
        client.list_blobs.return_value = []
        blob = client.bucket.return_value.get_blob.return_value
        blob.size = 50
        blob.download_as_bytes.return_value = b"mock parquet"
        saved = {"status": "COMPLETE", "ee_task_id": "id", "started_at": "2026-09-13T00:00:00+00:00", "row_count": 2}
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(b.persistence, "read_status", return_value=saved), \
             patch.object(b.work_items, "lookup_task_status", return_value={"state": "COMPLETED"}), \
             patch.object(b.pd, "read_parquet", return_value=table):
            path = Path(directory) / "results.json"
            b.collect(SimpleNamespace(results=path), {"batches": {"60": batch}}, client)
            diagnostic = json.loads(path.read_text())["batches"]["60"]["products"][0]["groups_diagnostics"]
        self.assertNotIn("harness_error", item)
        self.assertEqual(diagnostic["pandas_dtype"], str(table["groups"].dtype))
        self.assertEqual(diagnostic["python_value_type"], "builtins.str")
        self.assertEqual(diagnostic["representative_value"], representative)
        self.assertTrue(diagnostic["all_expected_rows_have_column"])
        self.assertEqual(diagnostic["null_row_count"], 1)
        self.assertEqual(diagnostic["non_null_row_count"], 1)

    def test_groups_missing_null_and_nested_values_are_diagnostic_only(self):
        self.assertFalse(b.groups_diagnostics(pd.DataFrame({"patch_id": ["p"]}), ["p"])
                         ["all_expected_rows_have_column"])
        empty = b.groups_diagnostics(pd.DataFrame({"patch_id": ["p"], "groups": [None]}), ["p"])
        self.assertTrue(empty["all_expected_rows_have_column"])
        self.assertIsNone(empty["representative_value"])
        nested = [{"class": 9000, "sum": 1800}]
        result = b.groups_diagnostics(pd.DataFrame({"patch_id": ["p"], "groups": [nested]}), ["p", "q"])
        self.assertFalse(result["all_expected_rows_have_column"])
        self.assertEqual(result["python_value_type"], "builtins.list")
        self.assertEqual(result["representative_value"], repr(nested))
        json.dumps(result, allow_nan=False)
