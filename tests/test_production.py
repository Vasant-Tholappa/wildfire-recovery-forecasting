"""Offline checks of the locked cloud contract and deterministic object names."""

import unittest

from wildfire_recovery_forecasting import config, production


class ProductionTests(unittest.TestCase):
    def test_locked_constants(self):
        self.assertEqual(config.GCP_PROJECT, "ee-wildfire-recovery-vasant")
        self.assertEqual(config.GCS_BUCKET, "ee-wildfire-recovery-vasant-data")
        self.assertEqual(config.GCS_REGION, "us-central1")
        self.assertEqual(config.PRODUCTION_VERSION, "phase3_prod_v1")
        self.assertEqual(config.PRODUCTION_CONTRACT_VERSION, "v2")
        self.assertEqual(config.GRID_VERSION, "v1")
        self.assertEqual(config.SCHEMA_VERSION, "v1")
        self.assertEqual(production.WORK_ITEM_STATUSES, {"RUNNING", "FAILED", "COMPLETE"})

    def test_object_paths(self):
        args = ("landsat", "OR4293812266120180715", "0007")
        suffix = "landsat/fire_id=OR4293812266120180715/batch_0007"
        root = "production/phase3_prod_v1/"
        for helper, kind, extension in [
            (production.raw_export_prefix, "raw_exports", ""),
            (production.parquet_path, "parquet", ".parquet"),
            (production.status_path, "status", ".json"),
        ]:
            expected = root + kind + "/" + suffix + extension
            self.assertEqual(helper(*args), expected)
            self.assertEqual(helper(*args), helper(*args))
            self.assertEqual(helper(*args, production_version="next"),
                             expected.replace("phase3_prod_v1", "next"))
        self.assertEqual(production.run_metadata_path(), root + "metadata/run.json")
        self.assertEqual(production.run_metadata_path(production_version="next"),
                         "production/next/metadata/run.json")

    def test_invalid_components_rejected(self):
        for bad in ("", "..", ".", "a/b", "a\\b", " x", None, 7):
            for index in range(3):
                args = ["source", "fire", "0001"]
                args[index] = bad
                for helper in (production.raw_export_prefix, production.parquet_path,
                               production.status_path):
                    with self.assertRaises(ValueError):
                        helper(*args)
            with self.assertRaises(ValueError):
                production.run_metadata_path(production_version=bad)


if __name__ == "__main__":
    unittest.main()
