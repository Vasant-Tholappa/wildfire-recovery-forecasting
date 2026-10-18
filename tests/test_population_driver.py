"""Offline bounded-wave checks; never initialize cloud clients."""
import unittest
from unittest.mock import Mock, patch
import manual_population as driver


class PopulationTests(unittest.TestCase):
    def test_wall_clock_rate_counts_only_new_persisted_outputs_and_includes_idle(self):
        from datetime import datetime, timezone
        records = {"old":{"status":"COMPLETE","production_version":"phase3_ca175_v1"},
                   "new":{"status":"COMPLETE","production_version":driver.VERSION},
                   "ee_done":{"status":"RUNNING","production_version":driver.VERSION}}
        result=driver.wall_clock_progress([1,2,3,4],records,"2026-10-18T12:00:00+00:00",
                                         datetime(2026,10,18,14,tzinfo=timezone.utc))
        self.assertEqual(result["new_persisted_products"],1)
        self.assertEqual(result["reused_products"],1)
        self.assertEqual(result["persisted_new_products_per_hour"],.5)
        self.assertEqual(result["projected_remaining_hours"],4)
        self.assertIsNone(driver.wall_clock_progress([1],{},None)["projected_remaining_hours"])

    def test_reuse_requires_complete_original_status_and_never_submits(self):
        fire, batch = next(iter(driver.REUSED_BATCHES))
        item = {"fire_id":fire,"batch_id":batch,"patch_count":3602,"source":"daymet","product":"combined"}
        old_key = {**driver.key(item), "production_version":driver.REUSED_BATCHES[(fire,batch)]}
        record = {"status":"COMPLETE","stage":"COMPLETE","row_count":3602,
                  "output_uri":f"gs://{driver.config.GCS_BUCKET}/{driver.production.parquet_path(**old_key)}"}
        client=Mock(); client.list_blobs.return_value=[]
        with patch.object(driver.persistence,"read_status",return_value=record) as read:
            records,_=driver.read_cloud_state([item],client,[])
            self.assertEqual(read.call_args.kwargs["production_version"],"phase3_ca175_v1")
        self.assertEqual(records[driver.production.status_path(**driver.key(item))],record)
        self.assertEqual(driver.proposed_wave([item],{},0,12,12),[])
        with patch.object(driver.persistence,"read_status",return_value={**record,"status":"RUNNING"}):
            with self.assertRaises(ValueError): driver.read_cloud_state([item],client,[])

    def test_upper_median_tile_selection_is_order_independent(self):
        import pandas as pd
        from manual_freeze_ca175_tile1 import select_tiles
        rows=pd.DataFrame({"fire_id":["f"]*4+["g"],"batch_id":["r0_c3","r0_c2","r0_c1","r0_c0","r1_c1"],
                           "patch_count":[1,100,100,1000,20]})
        selected=select_tiles(rows)
        self.assertEqual(selected.batch_id.tolist(),["r0_c2","r1_c1"])
        self.assertTrue(selected.patch_count.gt(0).all())
        pd.testing.assert_frame_equal(selected,select_tiles(rows.sample(frac=1,random_state=7)))

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
        self.assertEqual(k["production_version"], "phase3_ca175_tile1_v1")
        for helper in (driver.production.status_path, driver.production.raw_export_prefix, driver.production.parquet_path):
            self.assertIn("/phase3_ca175_tile1_v1/", helper(**k))
            old = {**k, "production_version": "phase3_prod_v1"}
            self.assertNotEqual(helper(**k), helper(**old))
        client = Mock(); client.list_blobs.return_value = []
        driver.read_cloud_state(self.items, client, [])
        self.assertEqual(client.list_blobs.call_args.kwargs["prefix"], "production/phase3_ca175_tile1_v1/status/")

    def test_batch_only_geometry_matches_canonical_and_reconcile_never_builds_graphs(self):
        from shapely.geometry import box, Polygon, MultiPolygon
        import geopandas as gpd
        import pandas as pd
        shapes = [box(-90,-45,100,90), Polygon([(0,0),(270,0),(270,270),(0,270)],
                  holes=[[(90,90),(180,90),(180,180),(90,180)]]),
                  MultiPolygon([box(0,0,40,40),box(180,0,220,40)])]
        for geometry in shapes:
            canonical = driver.assign_spatial_batches(driver.build_fire_grid(geometry,"f",perimeter_crs="EPSG:5070"),tile_side_cells=120)
            for bid, expected in canonical.groupby("batch_id"):
                actual = driver.batch_patches(geometry,"f",bid)
                pd.testing.assert_frame_equal(actual[expected.columns].reset_index(drop=True),expected.reset_index(drop=True),check_dtype=False)
                manifest = gpd.GeoDataFrame({"fire_id":["f"]},geometry=[geometry],crs="EPSG:5070")
                with patch.object(driver.pipeline,"build_raster_products",side_effect=AssertionError("No source graphs")), patch.object(driver,"build_fire_grid",side_effect=AssertionError("No full grid")):
                    descriptor,ids=driver.BatchProducts(manifest).reconciliation({"fire_id":"f","batch_id":bid,"patch_count":len(expected),"source":"landsat","product":"nbrpre"})
                self.assertEqual(ids,expected.patch_id.tolist())
                self.assertNotIn("collection",descriptor)

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
            reconcile=Mock(return_value=({"columns":["x"]},["p"]))
            driver.advance(self.items, records, tasks, Mock(), self.payload, 0, 4, 1,reconcile=reconcile)
        self.payload.assert_not_called()
        reconcile.assert_called_once()
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
