"""Offline one-step control, restart safety, and precise failure stages."""

import unittest
from unittest.mock import Mock, patch

from wildfire_recovery_forecasting import work_items, persistence


class WorkItemTests(unittest.TestCase):
    def setUp(self):
        self.saved = None
        self.writes = []
        self.client = Mock()
        self.client.list_blobs.return_value = []
        self.lookup = Mock(return_value={"state": "READY"})
        self.submit = Mock(return_value=Mock(id="task-1"))
        self.convert = Mock(side_effect=self.finish)
        for target, replacement in [
            ("read_status", lambda **kw: self.saved), ("write_status", self.write),
            ("csv_to_parquet", self.convert),
        ]:
            guard = patch.object(persistence, target, side_effect=replacement)
            guard.start()
            self.addCleanup(guard.stop)
        guard = patch.object(work_items.exports, "start_table_export", self.submit)
        guard.start()
        self.addCleanup(guard.stop)
        guard = patch.object(persistence.storage, "Client", side_effect=AssertionError("Live GCS"))
        guard.start()
        self.addCleanup(guard.stop)

    def write(self, **record):
        record.pop("client")
        self.saved = record
        self.writes.append(record)
        return record

    def run_item(self, **kwargs):
        return work_items.advance_work_item("collection", ["value"], source="terrain",
            fire_id="fire", batch_id="r0_c0", expected_patch_ids=["p"], client=self.client,
            task_status_lookup=self.lookup, **kwargs)

    def finish(self, *args, **kwargs):
        for stage in ("RAW_OUTPUT_CHECK", "CSV_TO_PARQUET", "STRUCTURAL_VALIDATION", "PARQUET_UPLOAD"):
            kwargs["on_stage"](stage)
        return {"row_count": 1, "output_uri": "gs://bucket/out.parquet"}

    def test_submit_once_and_wait_after_restart(self):
        self.assertEqual(self.run_item()["action"], "SUBMITTED")
        self.assertIsNone(self.writes[0]["ee_task_id"])
        self.assertEqual(self.saved["ee_task_id"], "task-1")
        self.assertEqual((self.saved["status"], self.saved["stage"]), ("RUNNING", "EE_EXPORT"))
        for state in ("READY", "RUNNING"):
            self.lookup.return_value = {"state": state}
            self.assertEqual(self.run_item()["action"], "WAITING")
        self.lookup.assert_called_with("task-1")
        self.submit.assert_called_once()
        self.convert.assert_not_called()

    def test_complete_then_skip_without_other_operations(self):
        self.run_item()
        self.lookup.return_value = {"state": "COMPLETED"}
        self.assertEqual(self.run_item()["action"], "COMPLETE")
        self.assertEqual((self.saved["status"], self.saved["stage"]), ("COMPLETE", "COMPLETE"))
        self.assertEqual(self.saved["row_count"], 1)
        self.assertIsNotNone(self.saved["completed_at"])
        before = len(self.writes)
        self.lookup.reset_mock()
        self.convert.reset_mock()
        self.assertEqual(self.run_item()["action"], "SKIPPED_COMPLETE")
        self.assertEqual(len(self.writes), before)
        self.lookup.assert_not_called()
        self.convert.assert_not_called()
        self.client.assert_not_called()
        self.submit.assert_called_once()

    def test_ee_failure_and_explicit_retry(self):
        self.run_item()
        self.lookup.return_value = {"state": "FAILED", "error_message": "EE memory limit"}
        self.assertEqual(self.run_item()["action"], "FAILED")
        self.assertEqual(self.saved["error_message"], "EE memory limit")
        self.assertEqual(self.saved["stage"], "EE_EXPORT")
        self.lookup.reset_mock()
        self.assertEqual(self.run_item()["action"], "NEEDS_RETRY")
        self.lookup.assert_not_called()
        self.assertEqual(self.run_item(retry_failed=True)["action"], "SUBMITTED")
        self.assertEqual(self.submit.call_count, 2)
        self.assertEqual(self.run_item()["action"], "FAILED")

    def test_explicit_retry_never_duplicates_active_task(self):
        self.run_item()
        self.saved = {**self.saved, "status": "FAILED", "error_message": "connection lost"}
        self.assertEqual(self.run_item(retry_failed=True)["action"], "WAITING")
        self.submit.assert_called_once()
        self.lookup.return_value = {"state": "COMPLETED"}
        self.assertEqual(self.run_item(retry_failed=True)["action"], "COMPLETE")
        self.submit.assert_called_once()

    def test_each_downstream_failure_is_recorded_and_retried_without_export(self):
        self.run_item()
        running = self.saved.copy()
        self.lookup.return_value = {"state": "COMPLETED"}
        for stage in ("RAW_OUTPUT_CHECK", "CSV_TO_PARQUET", "STRUCTURAL_VALIDATION", "PARQUET_UPLOAD"):
            self.saved = running.copy()
            def broken(*args, **kwargs):
                kwargs["on_stage"](stage)
                raise ValueError("broken " + stage)
            self.convert.side_effect = broken
            self.assertEqual(self.run_item()["action"], "FAILED")
            self.assertEqual(self.saved["stage"], stage)
            self.assertEqual(self.saved["error_message"], "broken " + stage)
            self.assertIsNotNone(self.saved["completed_at"])
        self.convert.side_effect = self.finish
        self.assertEqual(self.run_item(retry_failed=True)["action"], "COMPLETE")
        self.submit.assert_called_once()

    def test_interrupted_or_unknown_submission_requires_review(self):
        self.run_item()
        self.saved = {**self.saved, "ee_task_id": None}
        self.assertEqual(self.run_item(retry_failed=True)["action"], "NEEDS_REVIEW")
        self.saved["ee_task_id"] = "task-1"
        self.lookup.return_value = {"state": "UNKNOWN"}
        self.assertEqual(self.run_item()["action"], "NEEDS_REVIEW")
        self.submit.assert_called_once()

    def test_failed_export_with_raw_objects_requires_review(self):
        self.run_item()
        self.lookup.return_value = {"state": "CANCELLED"}
        self.assertEqual(self.run_item()["action"], "FAILED")
        self.client.list_blobs.return_value = [Mock()]
        self.assertEqual(self.run_item(retry_failed=True)["action"], "NEEDS_REVIEW")
        self.submit.assert_called_once()

    def test_task_lookup_failure_does_not_submit_or_convert(self):
        self.run_item()
        self.lookup.side_effect = PermissionError("EE denied")
        with self.assertRaises(PermissionError):
            self.run_item()
        self.submit.assert_called_once()
        self.convert.assert_not_called()

    def test_submission_intent_survives_ambiguous_start_failure(self):
        def interrupted(*args, **kwargs):
            self.assertEqual(self.saved["status"], "RUNNING")
            self.assertIsNone(self.saved["ee_task_id"])
            raise ConnectionError("Submission response lost")
        self.submit.side_effect = interrupted
        self.assertEqual(self.run_item()["action"], "FAILED")
        self.assertEqual(self.saved["stage"], "EE_EXPORT")
        self.assertEqual(self.run_item(retry_failed=True)["action"], "NEEDS_REVIEW")
        self.submit.assert_called_once()

    def test_interrupted_downstream_work_resumes_without_export(self):
        self.run_item()
        self.saved = {**self.saved, "stage": "PARQUET_UPLOAD"}
        self.lookup.return_value = {"state": "COMPLETED"}
        self.assertEqual(self.run_item()["action"], "COMPLETE")
        self.submit.assert_called_once()
