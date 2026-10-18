"""Offline bounded-wave checks; never initialize cloud clients."""
import unittest
from unittest.mock import Mock, patch
import manual_population as driver


class PopulationTests(unittest.TestCase):
    def setUp(self):
        self.items = [{"source": "daymet", "product": "combined", "fire_id": "f",
                       "batch_id": f"r0_c{i}", "patch_count": 1} for i in range(5)]
        self.payload = Mock(return_value=({"collection": object(), "columns": ["x"]}, ["p"]))

    def record(self, i, status, task="t"):
        return {driver.production.status_path(**driver.key(self.items[i])):
                {"status": status, "ee_task_id": task}}

    def test_california_namespace_isolates_all_paths_and_metadata_identity(self):
        item = self.items[0]
        k = driver.key(item)
        self.assertEqual(k["production_version"], "phase3_ca175_v1")
        for helper in (driver.production.status_path, driver.production.raw_export_prefix, driver.production.parquet_path):
            self.assertIn("/phase3_ca175_v1/", helper(**k))
            old = {**k, "production_version": "phase3_prod_v1"}
            self.assertNotEqual(helper(**k), helper(**old))
        client = Mock(); client.list_blobs.return_value = []
        driver.read_cloud_state(self.items, client, [])
        self.assertEqual(client.list_blobs.call_args.kwargs["prefix"], "production/phase3_ca175_v1/status/")

    def test_inventory_paths_and_limits(self):
        self.assertEqual(sum(map(len, driver.INVENTORY.values())), 12)
        self.assertEqual(driver.INVENTORY["daymet"], ("combined",))
        self.assertNotIn("nbrdrop", driver.INVENTORY["landsat"])
        records = {**self.record(0, "COMPLETE"), **self.record(1, "RUNNING"), **self.record(2, "FAILED")}
        self.assertEqual(driver.proposed_wave(self.items, records, 3, 20, 4), [self.items[3]])
        self.assertEqual(driver.proposed_wave(self.items, records, 4, 20, 4), [])
        self.assertEqual(driver.proposed_wave(self.items, self.record(0, "RUNNING", None), 0, 2, 4), [])

    def test_submissions_bounded_and_failed_items_not_retried(self):
        records = self.record(0, "FAILED")
        with patch.object(driver.work_items, "advance_work_item", return_value={"action": "SUBMITTED"}) as advance:
            actions = driver.advance(self.items, records, [], Mock(), self.payload, 2, 4, 1)
        self.assertEqual(len(actions), 2)
        self.assertEqual(advance.call_count, 2)
        self.assertEqual(advance.call_args_list[0].kwargs["batch_id"], "r0_c1")

    def test_active_and_complete_skip_without_graph_build(self):
        records = {**self.record(0, "RUNNING"), **self.record(1, "COMPLETE")}
        with patch.object(driver.work_items, "advance_work_item") as advance:
            driver.advance(self.items, records, [{"id": "t", "state": "RUNNING"}], Mock(), self.payload, 4, 1, 12)
        advance.assert_not_called()
        self.payload.assert_not_called()

    def test_completed_ee_progress_is_bounded_and_never_supplies_export_graph(self):
        records = {**self.record(0, "RUNNING", "a"), **self.record(1, "RUNNING", "b")}
        tasks = [{"id": t, "state": "COMPLETED"} for t in ("a", "b")]
        with patch.object(driver.work_items, "advance_work_item", return_value={"action": "COMPLETE"}) as advance:
            driver.advance(self.items, records, tasks, Mock(), self.payload, 0, 4, 1)
        advance.assert_called_once()
        self.assertIsNone(advance.call_args.args[0])
        self.assertEqual(advance.call_args.kwargs["task_status_lookup"]("a")["state"], "COMPLETED")

    def test_failure_stops_new_wave_and_unknown_tasks_reserve_capacity(self):
        with patch.object(driver.work_items, "advance_work_item", return_value={"action": "FAILED"}) as advance:
            driver.advance(self.items, {}, [], Mock(), self.payload, 5, 5, 0)
        advance.assert_called_once()
        with patch.object(driver.work_items, "advance_work_item") as advance:
            driver.advance(self.items, self.record(0, "RUNNING", "unknown"), [], Mock(), self.payload, 5, 1, 0)
        advance.assert_not_called()

    def test_status_scan_uses_validating_reader_and_rejects_foreign_identity(self):
        client = Mock()
        blob = Mock(); blob.name = driver.production.status_path(**driver.key(self.items[0]))
        client.list_blobs.return_value = [blob]
        with patch.object(driver.persistence, "read_status", return_value={"status": "COMPLETE"}) as read:
            records, active = driver.read_cloud_state(self.items, client, [{"state": "READY"}])
        self.assertEqual((len(records), active), (1, 1))
        self.assertEqual(read.call_args.kwargs["product"], "combined")
        blob.name = "foreign/status.json"
        with self.assertRaises(ValueError):
            driver.read_cloud_state(self.items, client, [])

    def test_session_cache_only_reuses_terminal_complete_records(self):
        client = Mock()
        blob = Mock(); blob.name = driver.production.status_path(**driver.key(self.items[0]))
        client.list_blobs.return_value = [blob]
        cache = {}
        with patch.object(driver.persistence, "read_status", return_value={"status": "COMPLETE"}) as read:
            driver.read_cloud_state(self.items, client, [], cache)
            driver.read_cloud_state(self.items, client, [], cache)
        read.assert_called_once()

    def test_session_total_cap_and_wait_interval(self):
        step = Mock(side_effect=[
            {"actions": [{"action": "SUBMITTED"}], "running_status": 1, "visible_active_ee_tasks": 1},
            {"actions": [], "running_status": 1, "visible_active_ee_tasks": 1},
            {"actions": [{"action": "SUBMITTED"}], "running_status": 1, "visible_active_ee_tasks": 1}])
        sleep = Mock()
        with patch.object(driver, "SNAPSHOT"):
            result = driver.run_session(step, submission_cap=2, poll_seconds=60,
                max_minutes=10, sleep=sleep, clock=lambda: 0)
        self.assertEqual([c.args[0] for c in step.call_args_list], [2, 1, 1])
        self.assertEqual(result["session_newly_submitted"], 2)
        self.assertEqual(sleep.call_count, 2)
        sleep.assert_called_with(60)

    def test_session_failure_and_ctrl_c_stop_without_retry(self):
        step = Mock(return_value={"actions": [{"action": "FAILED"}],
            "running_status": 0, "visible_active_ee_tasks": 0})
        with patch.object(driver, "SNAPSHOT"):
            driver.run_session(step, submission_cap=20, poll_seconds=60,
                max_minutes=10, sleep=Mock(side_effect=AssertionError("Must stop")), clock=lambda: 0)
        step.assert_called_once()
        with self.assertRaises(KeyboardInterrupt):
            driver.run_session(Mock(side_effect=KeyboardInterrupt), submission_cap=20,
                poll_seconds=60, max_minutes=10, clock=lambda: 0)

    def test_session_time_limit_stops(self):
        step = Mock(return_value={"actions": [], "running_status": 1, "visible_active_ee_tasks": 1})
        sleep = Mock()
        clock = Mock(side_effect=[0, 0, 60])
        with patch.object(driver, "SNAPSHOT"):
            driver.run_session(step, submission_cap=20, poll_seconds=60,
                max_minutes=1, sleep=sleep, clock=clock)
        step.assert_called_once()
        sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
