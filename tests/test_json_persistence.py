"""Offline JSON persistence with injected storage and real serialization."""

from datetime import datetime, timezone, timedelta
import json
import unittest
from unittest.mock import Mock, patch

from google.api_core.exceptions import Forbidden, NotFound

from wildfire_recovery_forecasting import config, persistence, production


class JsonPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.client = Mock()
        self.blob = self.client.bucket.return_value.blob.return_value
        self.blob.upload_from_string.side_effect = self.store
        self.now = datetime(2026, 9, 10, 9, tzinfo=timezone(timedelta(hours=-4)))
        guard = patch.object(persistence.storage, "Client", side_effect=AssertionError("Live GCS"))
        guard.start()
        self.addCleanup(guard.stop)

    def store(self, payload, **kwargs):
        self.assertEqual(kwargs, {"content_type": "application/json"})
        self.blob.download_as_bytes.return_value = payload.encode()

    def test_status_round_trip_nullable_fields_and_path(self):
        written = persistence.write_status("terrain", "fire", "r-1_c2", status="RUNNING",
                                           stage="EE_EXPORT", started_at=self.now, client=self.client)
        read = persistence.read_status("terrain", "fire", "r-1_c2", client=self.client)
        self.assertEqual(written, read)
        self.assertEqual(read["started_at"], "2026-09-10T13:00:00+00:00")
        for field in ("ee_task_id", "row_count", "output_uri", "completed_at", "error_message"):
            self.assertIsNone(read[field])
        self.assertEqual(set(read), {"production_version", "fire_id", "source", "batch_id",
            "status", "stage", "ee_task_id", "row_count", "output_uri", "started_at", "completed_at", "error_message"})
        self.client.bucket.assert_called_with(config.GCS_BUCKET)
        self.client.bucket.return_value.blob.assert_called_with(production.status_path("terrain", "fire", "r-1_c2"))

    def test_complete_and_failed_are_explicit_and_round_trip(self):
        for status in ("COMPLETE", "FAILED"):
            written = persistence.write_status("terrain", "fire", "r0_c0", status=status,
                stage="COMPLETE" if status == "COMPLETE" else "PARQUET_UPLOAD",
                started_at=self.now, completed_at="2026-09-10T14:00:00Z", ee_task_id="task",
                row_count=4, output_uri="gs://bucket/object", error_message="error" if status == "FAILED" else None,
                client=self.client, production_version="phase3_smoke_v1")
            self.assertEqual(written, persistence.read_status("terrain", "fire", "r0_c0",
                client=self.client, production_version="phase3_smoke_v1"))

    def test_run_metadata_round_trip(self):
        written = persistence.write_run_metadata(git_commit="abc", batching_configuration={
            "tile_side_cells": 10, "sources": ["terrain"]}, created_at=self.now, client=self.client)
        self.assertEqual(written, persistence.read_run_metadata(client=self.client))
        self.assertEqual(set(written), {"production_version", "contract_version", "grid_version",
            "schema_version", "git_commit", "gcp_project", "gcs_bucket", "gcs_region",
            "batching_configuration", "created_at"})
        self.assertEqual(written["gcs_region"], "us-central1")
        self.assertEqual(written["contract_version"], "v2")
        self.client.bucket.return_value.blob.assert_called_with(production.run_metadata_path())

    def test_missing_is_only_not_found(self):
        self.blob.download_as_bytes.side_effect = NotFound("absent")
        self.assertIsNone(persistence.read_status("terrain", "fire", "r0_c0", client=self.client))
        self.assertIsNone(persistence.read_run_metadata(client=self.client))
        self.blob.download_as_bytes.side_effect = Forbidden("denied")
        with self.assertRaises(Forbidden):
            persistence.read_status("terrain", "fire", "r0_c0", client=self.client)

    def test_malformed_status_is_not_missing(self):
        for payload in (b"not json", b"null", b"[]", b"{}", b'{"status":"MISSING"}'):
            self.blob.download_as_bytes.return_value = payload
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                persistence.read_status("terrain", "fire", "r0_c0", client=self.client)

    def test_invalid_writes_fail_before_upload(self):
        for status in ("MISSING", "complete", "", None):
            with self.assertRaises(ValueError):
                persistence.write_status("terrain", "fire", "r0_c0", status=status,
                                         stage="EE_EXPORT", started_at=self.now, client=self.client)
        with self.assertRaises(ValueError):
            persistence.write_status("terrain", "fire", "r0_c0", status="RUNNING",
                                     stage="EE_EXPORT", started_at=datetime(2026, 1, 1), client=self.client)
        for batching in (None, [], {"bad": object()}, {"bad": float("nan")}):
            with self.assertRaises((ValueError, TypeError)):
                persistence.write_run_metadata(git_commit="abc", batching_configuration=batching,
                                               created_at=self.now, client=self.client)
        self.blob.upload_from_string.assert_not_called()

    def test_operation_stages_round_trip(self):
        stages = {"EE_EXPORT", "RAW_OUTPUT_CHECK", "CSV_TO_PARQUET",
                  "STRUCTURAL_VALIDATION", "PARQUET_UPLOAD", "COMPLETE"}
        self.assertEqual(production.WORK_ITEM_STAGES, stages)
        for stage in stages - {"COMPLETE"}:
            for status in ("RUNNING", "FAILED"):
                written = persistence.write_status("terrain", "fire", "r0_c0",
                    status=status, stage=stage, started_at=self.now, client=self.client)
                self.assertEqual(written["stage"], stage)
                self.assertEqual(written, persistence.read_status(
                    "terrain", "fire", "r0_c0", client=self.client))

    def test_invalid_stage_and_complete_mismatch_rejected(self):
        for status, stage in [("RUNNING", value) for value in (None, "", "export", [])] + [
            ("COMPLETE", value) for value in production.WORK_ITEM_STAGES - {"COMPLETE"}
        ] + [("RUNNING", "COMPLETE"), ("FAILED", "COMPLETE")]:
            with self.subTest(status=status, stage=stage), self.assertRaises(ValueError):
                persistence.write_status("terrain", "fire", "r0_c0", status=status,
                    stage=stage, started_at=self.now, client=self.client)
        self.blob.upload_from_string.assert_not_called()

    def test_reader_rejects_missing_invalid_or_inconsistent_stage(self):
        record = persistence.write_status("terrain", "fire", "r0_c0", status="COMPLETE",
            stage="COMPLETE", started_at=self.now, client=self.client)
        for stage in (None, "unknown", "EE_EXPORT"):
            bad = {**record, "stage": stage}
            if stage is None:
                del bad["stage"]
            self.blob.download_as_bytes.return_value = json.dumps(bad).encode()
            with self.assertRaises(ValueError):
                persistence.read_status("terrain", "fire", "r0_c0", client=self.client)
        for status in ("RUNNING", "FAILED"):
            self.blob.download_as_bytes.return_value = json.dumps({**record, "status": status}).encode()
            with self.assertRaises(ValueError):
                persistence.read_status("terrain", "fire", "r0_c0", client=self.client)
