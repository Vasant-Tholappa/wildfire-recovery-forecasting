"""Offline export submission and shared pre-materialization graph checks."""

from unittest.mock import Mock, patch

import ee
from ee.apitestcase import ApiTestCase

from wildfire_recovery_forecasting import exports, production, reductions


class ExportTests(ApiTestCase):
    def test_product_exports_have_distinct_descriptions_and_prefixes(self):
        task = Mock()
        with patch.object(ee.batch.Export.table, "toCloudStorage", return_value=task) as submit:
            for product in reductions.LANDSAT_PRODUCTS:
                exports.start_table_export(ee.FeatureCollection([]), [], source="landsat",
                    fire_id="fire", batch_id="r0_c0", product=product)
        self.assertEqual(len({c.kwargs["description"] for c in submit.call_args_list}), 4)
        self.assertEqual(len({c.kwargs["fileNamePrefix"] for c in submit.call_args_list}), 4)

    def test_csv_configuration_starts_task_without_fetch_or_poll(self):
        collection = ee.FeatureCollection([ee.Feature(None, {"patch_id": "p", "fire_id": "f"})])
        task = Mock()
        with patch.object(ee.batch.Export.table, "toCloudStorage", return_value=task) as submit, \
             patch.object(ee.data, "computeFeatures", side_effect=AssertionError("Downloaded rows")), \
             patch.object(ee.data, "computeValue", side_effect=AssertionError("Evaluated graph")), \
             patch.object(reductions, "fetch_patch_table", side_effect=AssertionError("Sync fetch")):
            for _ in range(2):
                result = exports.start_table_export(
                    collection, ["value_sum", "value_area"], source="terrain",
                    fire_id="OR4293812266120180715", batch_id="r-2_c3")
                self.assertIs(result, task)
        first, second = [call.kwargs for call in submit.call_args_list]
        self.assertEqual(first["collection"].serialize(), second["collection"].serialize())
        self.assertEqual(first["collection"].serialize(), reductions.without_geometry(collection).serialize())
        for key in ("description", "bucket", "fileNamePrefix", "fileFormat", "selectors"):
            self.assertEqual(first[key], second[key])
        self.assertEqual(first["bucket"], "ee-wildfire-recovery-vasant-data")
        self.assertEqual(first["fileFormat"], "CSV")
        self.assertEqual(first["description"],
                         "phase3_prod_v1_terrain_OR4293812266120180715_batch_r-2_c3")
        self.assertEqual(first["fileNamePrefix"], production.raw_export_prefix(
            "terrain", "OR4293812266120180715", "r-2_c3"))
        self.assertEqual(first["selectors"], ["patch_id", "fire_id", "value_sum", "value_area"])
        self.assertEqual(task.method_calls, [("start", (), {}), ("start", (), {})])

    def test_raw_builder_and_sync_fetch_share_exact_reduction_graph(self):
        images = {"value": ee.Image.constant(1)}
        patches = ee.FeatureCollection([])
        projection = ee.Projection("EPSG:5070").atScale(30)
        with patch.object(ee.data, "computeFeatures", side_effect=AssertionError("Downloaded rows")):
            raw = reductions.build_area_sums(images, patches, projection)
        with patch.object(reductions, "fetch_patch_table", return_value="local table") as fetch:
            self.assertEqual(reductions.area_sums(images, patches, projection), "local table")
        self.assertEqual(fetch.call_args.args[0].serialize(), raw.serialize())
        self.assertEqual(fetch.call_args.args[1],
                         [reductions.RASTER_FULL_AREA, "value_sum", "value_area"])

    def test_invalid_task_description_does_not_submit(self):
        with patch.object(ee.batch.Export.table, "toCloudStorage") as submit:
            with self.assertRaises(ValueError):
                exports.start_table_export(ee.FeatureCollection([]), [],
                                           source="bad source", fire_id="f", batch_id="r0_c0")
            submit.assert_not_called()
