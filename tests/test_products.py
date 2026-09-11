"""Product-scoped paths, durable statuses and controller isolation, offline."""

from datetime import datetime, timezone
from io import BytesIO
import json
import unittest
from unittest.mock import Mock, patch

from google.api_core.exceptions import NotFound
import pandas as pd

from wildfire_recovery_forecasting import persistence, production, reductions, work_items


class ProductTests(unittest.TestCase):
    def test_five_distinct_paths_and_legacy_paths(self):
        names = ["nbrpre", "season1", "nbr5", "season5_count", "nbrdrop"]
        self.assertEqual(list(reductions.LANDSAT_PRODUCTS), names)
        for helper, folder, extension in [(production.raw_export_prefix, "raw_exports", ""),
                (production.parquet_path, "parquet", ".parquet"),
                (production.status_path, "status", ".json")]:
            paths = [helper("landsat", "fire", "r0_c0", product=name) for name in names]
            self.assertEqual(len(set(paths)), 5)
            self.assertEqual(paths[0], f"production/phase3_prod_v1/{folder}/landsat/product=nbrpre/fire_id=fire/batch_r0_c0{extension}")
            self.assertEqual(helper("terrain", "fire", "r0_c0"),
                             f"production/phase3_prod_v1/{folder}/terrain/fire_id=fire/batch_r0_c0{extension}")
            for bad in ("", "a/b", ".."):
                with self.assertRaises(ValueError):
                    helper("landsat", "fire", "r0_c0", product=bad)

    def test_real_status_persistence_isolates_completion_resume_and_retry(self):
        objects = {}
        client = Mock()
        def blob(name):
            item = Mock()
            def download():
                if name not in objects:
                    raise NotFound(name)
                return objects[name]
            item.download_as_bytes.side_effect = download
            item.upload_from_string.side_effect = lambda payload, **kw: objects.update({name: payload.encode()})
            return item
        client.bucket.return_value.blob.side_effect = blob
        client.list_blobs.return_value = []
        lookup = Mock(return_value={"state": "COMPLETED"})
        submit = Mock(side_effect=[Mock(id="pre-task"), Mock(id="s1-task"), Mock(id="s1-retry")])
        def convert(*args, **kw):
            self.assertEqual(kw["product"], "nbrpre")
            kw["on_stage"]("PARQUET_UPLOAD")
            return {"row_count": 1, "output_uri": "gs://bucket/pre.parquet"}
        def run(product, **kw):
            return work_items.advance_work_item(None, ["x"], source="landsat", fire_id="fire",
                batch_id="r0_c0", expected_patch_ids=["p"], product=product, client=client,
                task_status_lookup=lookup, **kw)
        with patch.object(work_items.exports, "start_table_export", submit), \
             patch.object(persistence, "csv_to_parquet", side_effect=convert), \
             patch.object(persistence.storage, "Client", side_effect=AssertionError("Live GCS")):
            self.assertEqual(run("nbrpre")["action"], "SUBMITTED")
            self.assertEqual(run("nbrpre")["action"], "COMPLETE")
            self.assertEqual(run("season1")["action"], "SUBMITTED")
            lookup.return_value = {"state": "RUNNING"}
            self.assertEqual(run("season1")["action"], "WAITING")
            lookup.assert_called_with("s1-task")
            lookup.return_value = {"state": "FAILED", "error_message": "test failure"}
            self.assertEqual(run("season1")["action"], "FAILED")
            self.assertEqual(run("nbrpre")["action"], "SKIPPED_COMPLETE")
            self.assertEqual(run("season1")["action"], "NEEDS_RETRY")
            self.assertEqual(run("season1", retry_failed=True)["action"], "SUBMITTED")
            self.assertEqual([c.kwargs["product"] for c in submit.call_args_list],
                             ["nbrpre", "season1", "season1"])
            prefix = production.raw_export_prefix("landsat", "fire", "r0_c0", product="season1")
            self.assertEqual(client.list_blobs.call_args.kwargs["prefix"], prefix)
        pre = persistence.read_status("landsat", "fire", "r0_c0", product="nbrpre", client=client)
        s1 = persistence.read_status("landsat", "fire", "r0_c0", product="season1", client=client)
        self.assertEqual((pre["status"], pre["ee_task_id"]), ("COMPLETE", "pre-task"))
        self.assertEqual(s1["ee_task_id"], "s1-retry")
        # A corrupted record at the right path cannot impersonate another product.
        objects[production.status_path("landsat", "fire", "r0_c0", product="season1")] = json.dumps(pre).encode()
        with self.assertRaises(ValueError):
            persistence.read_status("landsat", "fire", "r0_c0", product="season1", client=client)

    def test_product_csv_conversion_uses_product_destination(self):
        client = Mock()
        raw = Mock()
        raw.name = production.raw_export_prefix("landsat", "fire", "r0_c0", product="nbr5") + ".csv"
        raw.download_as_bytes.return_value = b"patch_id,fire_id,nbr_5_sum\np,fire,\n"
        client.list_blobs.return_value = [raw]
        result = persistence.csv_to_parquet("landsat", "fire", "r0_c0", product="nbr5",
            expected_patch_ids=["p"], expected_columns=["nbr_5_sum"], client=client)
        destination = production.parquet_path("landsat", "fire", "r0_c0", product="nbr5")
        client.bucket.return_value.blob.assert_called_with(destination)
        self.assertTrue(result["output_uri"].endswith(destination))
        payload = client.bucket.return_value.blob.return_value.upload_from_string.call_args.args[0]
        self.assertTrue(pd.read_parquet(BytesIO(payload)).nbr_5_sum.isna().all())
