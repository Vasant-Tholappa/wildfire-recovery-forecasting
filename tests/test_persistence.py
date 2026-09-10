"""In-memory CSV/Parquet round trips with fake GCS objects; no cloud access."""

from io import BytesIO
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from wildfire_recovery_forecasting import config, persistence, production


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.prefix = production.raw_export_prefix("terrain", "fire", "r0_c1")
        self.client = Mock()
        self.output = self.client.bucket.return_value.blob.return_value
        self.no_client = patch.object(persistence.storage, "Client",
                                      side_effect=AssertionError("Live client"))
        self.no_client.start()
        self.addCleanup(self.no_client.stop)

    def blob(self, suffix, csv):
        blob = Mock()
        blob.name = self.prefix + suffix
        blob.download_as_bytes.return_value = csv.encode()
        return blob

    def convert(self, expected=("001", "002")):
        return persistence.csv_to_parquet(
            "terrain", "fire", "r0_c1", expected_patch_ids=expected,
            expected_columns=["value"], client=self.client)

    def read_output(self):
        return pd.read_parquet(BytesIO(self.output.upload_from_string.call_args.args[0]))

    def test_single_csv_destination_ids_and_nulls(self):
        raw = self.blob(".csv", "patch_id,fire_id,value\n002,fire,0\n001,fire,\n")
        self.client.list_blobs.return_value = [raw]
        result = self.convert()
        table = self.read_output()
        self.assertEqual(table.patch_id.tolist(), ["001", "002"])
        self.assertTrue(pd.isna(table.value.iloc[0]))
        self.assertEqual(table.value.iloc[1], 0)
        path = production.parquet_path("terrain", "fire", "r0_c1")
        self.client.list_blobs.assert_called_once_with(config.GCS_BUCKET, prefix=self.prefix)
        self.client.bucket.assert_called_once_with(config.GCS_BUCKET)
        self.client.bucket.return_value.blob.assert_called_once_with(path)
        self.output.upload_from_string.assert_called_once()
        self.assertEqual(result, {"row_count": 2, "output_uri": f"gs://{config.GCS_BUCKET}/{path}"})
        raw.delete.assert_not_called()

    def test_shards_combined_neighboring_prefixes_ignored(self):
        first = self.blob("-000000000000.csv", "patch_id,fire_id,value\n001,fire,1\n")
        second = self.blob("-000000000001.csv", "patch_id,fire_id,value\n002,fire,2\n")
        neighbors = [self.blob(suffix, "bad") for suffix in
                     ("0.csv", "_other.csv", "/other.csv", ".csv.tmp", "-notes.csv")]
        self.client.list_blobs.return_value = [second, *neighbors, first]
        self.convert()
        self.assertEqual(self.read_output().value.tolist(), [1, 2])
        for blob in neighbors:
            blob.download_as_bytes.assert_not_called()

    def test_invalid_rows_and_columns_never_upload(self):
        cases = [
            "patch_id,fire_id,value\n001,fire,1\n001,fire,2\n",
            "patch_id,fire_id,value\n,fire,1\n002,fire,2\n",
            "patch_id,fire_id,value\nwrong,fire,1\n002,fire,2\n",
            "patch_id,fire_id,value\n001,fire,1\n",
            "patch_id,fire_id,value\n001,other,1\n002,fire,2\n",
            "patch_id,fire_id,value\n001,,1\n002,fire,2\n",
            "patch_id,fire_id\n001,fire\n002,fire\n",
            "fire_id,value\nfire,1\n",
            "patch_id,value\n001,1\n",
            "patch_id,fire_id,value\n", "",
        ]
        for csv in cases:
            with self.subTest(csv=csv), self.assertRaises(ValueError):
                self.client.list_blobs.return_value = [self.blob(".csv", csv)]
                self.convert()
        self.output.upload_from_string.assert_not_called()

    def test_cross_shard_duplicate_and_missing_column_rejected(self):
        first = self.blob("-0.csv", "patch_id,fire_id,value\n001,fire,1\n")
        for csv in ("patch_id,fire_id,value\n001,fire,2\n", "patch_id,fire_id\n002,fire\n"):
            self.client.list_blobs.return_value = [first, self.blob("-1.csv", csv)]
            with self.assertRaises(ValueError):
                self.convert()
        self.output.upload_from_string.assert_not_called()

    def test_no_raw_output_and_bad_expected_ids_rejected(self):
        self.client.list_blobs.return_value = []
        with self.assertRaisesRegex(ValueError, "No completed CSV"):
            self.convert()
        for expected in ([], ["001", "001"], [None]):
            with self.assertRaises(ValueError):
                self.convert(expected)
        self.output.upload_from_string.assert_not_called()

    def test_stage_callback_tracks_actual_conversion_boundaries(self):
        events = []
        self.client.list_blobs.side_effect = lambda *a, **kw: (
            events.append("list") or [self.blob(".csv", "patch_id,fire_id,value\n001,fire,1\n002,fire,2\n")])
        self.output.upload_from_string.side_effect = lambda *a, **kw: events.append("upload")
        persistence.csv_to_parquet("terrain", "fire", "r0_c1", expected_patch_ids=["001", "002"],
            expected_columns=["value"], client=self.client, on_stage=events.append)
        self.assertEqual(events, ["STRUCTURAL_VALIDATION", "RAW_OUTPUT_CHECK", "list",
            "CSV_TO_PARQUET", "STRUCTURAL_VALIDATION", "CSV_TO_PARQUET", "PARQUET_UPLOAD", "upload"])


if __name__ == "__main__":
    unittest.main()
