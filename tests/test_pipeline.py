"""Local reduction arithmetic, EE graph contracts, and exact-population joins."""

from datetime import datetime, timezone
import unittest
from unittest.mock import patch

import ee
from ee.apitestcase import ApiTestCase
import numpy as np
import pandas as pd
from shapely.geometry import box
from shapely.ops import unary_union

from wildfire_recovery_forecasting import pipeline, reductions, validation
from wildfire_recovery_forecasting.fire_catalog import derive_fire_fields
from wildfire_recovery_forecasting.grid import build_fire_grid


def local_patches():
    return build_fire_grid(box(0, 0, 135, 90), "fire", perimeter_crs="EPSG:5070")


class ReductionTests(unittest.TestCase):
    def test_same_grid_denominator_for_full_partial_and_missing_raster_support(self):
        full = 8102.43
        raw = pd.DataFrame({
            "patch_id": ["full", "partial", "missing"], "fire_id": ["f"] * 3,
            reductions.RASTER_FULL_AREA: [full, full, 8098.],
            "value_sum": [full, full / 4, np.nan],
            "value_area": [full, full / 4, np.nan],
        })
        fractions = reductions.summarize_area_sums(raw, ["value"], fractions=["value"])
        means = reductions.summarize_area_sums(raw, ["value"])
        np.testing.assert_allclose(fractions.value_fraction.iloc[:2], [1, .25])
        np.testing.assert_allclose(fractions.value_valid_fraction, [1, .25, 0])
        np.testing.assert_allclose(means.value_mean.iloc[:2], [1, 1])
        self.assertTrue(pd.isna(fractions.value_fraction.iloc[2]))
        self.assertTrue(pd.isna(means.value_mean.iloc[2]))
        self.assertNotIn(reductions.RASTER_FULL_AREA, fractions)
        validation.validate_measurements(fractions)

    def test_invalid_raster_reference_is_rejected_not_replaced_with_vector_area(self):
        for full in (0, -1, np.nan, np.inf):
            raw = pd.DataFrame({"patch_id": ["p"], "fire_id": ["f"],
                                reductions.RASTER_FULL_AREA: [full],
                                "value_sum": [1.], "value_area": [1.]})
            with self.subTest(full=full), self.assertRaises(ValueError):
                reductions.summarize_area_sums(raw, ["value"])

    def test_valid_area_not_full_area_for_means_and_missing_not_zero(self):
        raw = pd.DataFrame({
            "patch_id": ["a", "b", "c"], "fire_id": ["f"] * 3,
            reductions.RASTER_FULL_AREA: [8100.] * 3,
            "value_sum": [1800., np.nan, 0.], "value_area": [900., np.nan, 8100.],
        })
        result = reductions.summarize_area_sums(raw, ["value"])
        self.assertEqual(result.value_mean.iloc[0], 2)
        self.assertAlmostEqual(result.value_valid_fraction.iloc[0], 1 / 9)
        self.assertTrue(pd.isna(result.value_mean.iloc[1]))
        self.assertEqual(result.value_valid_fraction.iloc[1], 0)
        self.assertEqual(result.value_mean.iloc[2], 0)
        fractions = reductions.summarize_area_sums(raw, ["value"], fractions=["value"])
        self.assertAlmostEqual(fractions.value_fraction.iloc[0], 1800 / 8100)
        self.assertTrue(pd.isna(fractions.value_fraction.iloc[1]))

    def test_forest_support_denominator_and_unobserved_forest(self):
        raw = pd.DataFrame({"patch_id": ["a", "b"], "fire_id": ["f", "f"],
                            reductions.RASTER_FULL_AREA: [8102.43, 8098.],
                            "support_sum": [1800., 0.], "cover_sum": [45000., np.nan],
                            "cover_area": [900., np.nan]})
        result = reductions.summarize_area_sums(raw, ["cover"], support="support")
        self.assertEqual(result.cover_mean.iloc[0], 50)
        self.assertEqual(result.cover_valid_fraction.iloc[0], .5)
        self.assertEqual(result.cover_valid_fraction.iloc[1], 0)
        self.assertTrue(pd.isna(result.cover_mean.iloc[1]))

    def test_categorical_area_mode_not_code_average(self):
        groups = [{"class": 9000, "sum": 900}, {"class": 100, "sum": 7200}]
        self.assertEqual(reductions.dominant_category(groups), (100, 7200))
        self.assertTrue(np.isnan(reductions.dominant_category([])[0]))
        self.assertEqual(reductions.dominant_category([
            {"class": 2, "sum": 10}, {"class": 1, "sum": 10}]), (1, 10))

    def test_mtbs_union_no_double_count_and_unknown_completeness(self):
        patches = local_patches()
        union = unary_union([box(0, 0, 60, 90), box(30, 0, 90, 90)])
        history = {"union_crs": "EPSG:5070", "windows": {
            "prior_fire_fraction_5yr": {"union_geometry": union, "coverage_status": "unverified"}}}
        result = reductions.reduce_wildfire_history(history, patches)
        self.assertEqual(result.prior_fire_fraction_5yr.tolist(), [1, 0])
        self.assertEqual(result.prior_fire_fraction_5yr_coverage_status.tolist(), ["unverified"] * 2)
        validation.validate_measurements(result)
        history["windows"]["prior_fire_fraction_5yr"]["union_geometry"] = box(0, 0, 135, 90)
        partial = reductions.reduce_wildfire_history(history, patches)
        self.assertEqual(partial.prior_fire_fraction_5yr.tolist(), [1, .5])


class JoinTests(unittest.TestCase):
    def setUp(self):
        self.patches = local_patches()
        self.source = pd.DataFrame(self.patches[["patch_id", "fire_id"]])
        self.source["nbr_pre_mean"] = [0.5, np.nan]
        self.source["nbr_pre_valid_fraction"] = [1., 0.]

    def test_one_to_one_preserves_order_ids_and_missingness(self):
        result = validation.join_source_tables(self.patches, {"landsat": self.source.iloc[::-1]}, fire_id="fire")
        self.assertEqual(result.patch_id.tolist(), self.patches.patch_id.tolist())
        self.assertEqual(len(result), 2)
        self.assertTrue(pd.isna(result.nbr_pre_mean.iloc[1]))

    def test_duplicate_missing_unexpected_and_wrong_fire_rejected(self):
        broken = [pd.concat([self.source, self.source.iloc[:1]]), self.source.iloc[:1],
                  self.source.assign(patch_id=[None, "b"]), self.source.assign(fire_id="other"),
                  self.source.assign(patch_id=["wrong", "ids"])]
        for source in broken:
            with self.subTest(source=source), self.assertRaises(ValueError):
                validation.join_source_tables(self.patches, {"bad": source}, fire_id="fire")

    def test_collision_wrong_area_and_invalid_fraction_rejected(self):
        with self.assertRaises(ValueError):
            validation.join_source_tables(self.patches, {"a": self.source, "b": self.source}, fire_id="fire")
        broken = self.patches.copy()
        broken["patch_area_m2"] = 8000
        with self.assertRaises(ValueError):
            validation.validate_patches(broken, "fire")
        for fraction in [1.01, -.01]:
            with self.assertRaises(ValueError):
                validation.validate_measurements(self.source.assign(nbr_pre_valid_fraction=fraction))
        with self.assertRaises(ValueError):
            validation.validate_measurements(self.source.assign(nbr_pre_mean=0))

    def test_runner_uses_independent_tables_and_declares_roles(self):
        ignition = datetime(2017, 7, 29, tzinfo=timezone.utc)
        fire = {"geometry": box(0, 0, 135, 90), "fire_id": "fire", "fire_size_acres": 3,
                "ignition_date": ignition, **derive_fire_fields(ignition)}
        schema = pipeline.dataset_roles([])
        columns = set(schema["predictors"]) | {schema["target"], "lcms_pre_5yr_clear_disturbance_fraction"}
        table = self.source[["patch_id", "fire_id"]].copy()
        for name in columns - {"fire_size_acres", "ignition_month"}:
            table[name] = .5
        with patch.object(pipeline, "source_patch_tables", return_value=iter([("mock", table)])) as source:
            result = pipeline.run_fire_pipeline(fire, perimeter_crs="EPSG:5070", patch_ids=self.patches.patch_id)
        self.assertEqual(source.call_count, 1)
        self.assertEqual(len(result["table"]), 2)
        self.assertIn("lcms_pre_5yr_clear_disturbance_fraction", result["roles"]["qa"])
        self.assertNotIn("nbr_1_mean", result["roles"]["early_predictors"])
        self.assertTrue(set(pipeline.daymet.HISTORICAL_BANDS) <= set(result["roles"]["early_predictors"]))
        for leaked in ("nbr_5_mean", "lcms_pre_5yr_clear_disturbance_fraction"):
            roles = {**result["roles"], "predictors": result["roles"]["predictors"] + [leaked]}
            with self.assertRaises(ValueError):
                validation.validate_dataset(result["table"], roles, expected_ids=self.patches.patch_id, fire_id="fire")

    def test_sample_is_deterministic_and_contains_full_and_boundary(self):
        grid = build_fire_grid(box(15, 15, 900, 900), "f", perimeter_crs="EPSG:5070")
        ids = pipeline.smoke_patch_ids(grid, 24)
        self.assertEqual(ids, pipeline.smoke_patch_ids(grid, 24))
        selected = grid.loc[grid.patch_id.isin(ids)]
        self.assertEqual(len(selected), 24)
        self.assertEqual(np.isclose(selected.fire_overlap_fraction, 1).sum(), 12)


class ReductionGraphTests(ApiTestCase):
    def test_landfire_reduction_uses_distinct_grids_and_categorical_groups(self):
        images = {name: ee.Image(name) for name in (
            "forest_indicator", "forest_mask", "tree_cover_pct", "tree_height_m", "evt")}
        projections = []
        def fake_area(values, patches, projection):
            projections.append(projection.serialize())
            data = {"patch_id": ["p"], "fire_id": ["f"], reductions.RASTER_FULL_AREA: [8102.43]}
            for name in values:
                total, area = {"forest": (4500, 8100), "support": (4500, 8100),
                               "tree_cover": (45000, 900), "tree_height": (18000, 1800),
                               "evt_coverage": (7200, 8100)}[name]
                data.update({name + "_sum": [total], name + "_area": [area]})
            return pd.DataFrame(data)
        graphs = []
        def fake_fetch(fc, columns):
            graphs.append(fc.serialize())
            return pd.DataFrame({"patch_id": ["p"], "fire_id": ["f"],
                                 "groups": [[{"class": 9000, "sum": 1800}, {"class": 100, "sum": 900}]]})
        with patch.object(reductions, "area_sums", side_effect=fake_area), \
                patch.object(reductions, "fetch_patch_table", side_effect=fake_fetch):
            result = reductions.reduce_landfire(images, ee.FeatureCollection([])).iloc[0]
        self.assertAlmostEqual(result.forest_fraction, 4500 / 8102.43)
        self.assertEqual(result.tree_cover_mean, 50)
        self.assertEqual(result.tree_height_mean, 10)
        self.assertEqual(result.tree_cover_valid_fraction, .2)
        self.assertEqual(result.tree_height_valid_fraction, .4)
        self.assertEqual(result.dominant_evt, 9000)
        self.assertEqual(result.dominant_evt_fraction, .4)
        self.assertAlmostEqual(result.evt_valid_fraction, 7200 / 8102.43)
        self.assertIn("Reducer.group", graphs[0])
        self.assertNotIn("Reducer.mean", graphs[0])
        self.assertNotEqual(projections[0], projections[-1])

    def test_lcms_fraction_coverage_and_annual_timing_are_retained(self):
        categories = ["clear_disturbance", "other_loss"]
        images = {name: ee.Image(name) for name in categories + [
            "valid_years", "complete_coverage", "clear_disturbance_first_year", "other_loss_first_year"]}
        images["expected_years"] = 5
        history = {"windows": {"pre_5yr": images}, "timing_resolution": "annual; uncertain ordering"}
        def fake_area(values, patches, projection):
            data = {"patch_id": ["observed", "unavailable"], "fire_id": ["f", "f"],
                    reductions.RASTER_FULL_AREA: [8100., 8100.]}
            for name in values:
                total = {"clear_disturbance": 900, "other_loss": 0,
                         "complete_coverage": 0, "valid_years": 8100 * 4}[name]
                qa = name in ("valid_years", "complete_coverage")
                data.update({name + "_sum": [total, 0 if qa else np.nan],
                             name + "_area": [8100, 8100 if qa else np.nan]})
            return pd.DataFrame(data)
        def fake_dates(fc, columns):
            self.assertIn("Reducer.min", fc.serialize())
            return pd.DataFrame({"patch_id": ["observed", "unavailable"], "fire_id": ["f", "f"],
                                 "clear_disturbance_first_year": [2014, np.nan],
                                 "other_loss_first_year": [np.nan, np.nan]})
        with patch.object(reductions, "area_sums", side_effect=fake_area), \
                patch.object(reductions, "fetch_patch_table", side_effect=fake_dates):
            result = reductions.reduce_lcms(history, ee.FeatureCollection([]))
        self.assertAlmostEqual(result.lcms_pre_5yr_clear_disturbance_fraction.iloc[0], 1 / 9)
        self.assertEqual(result.lcms_pre_5yr_other_loss_fraction.iloc[0], 0)
        self.assertTrue(pd.isna(result.lcms_pre_5yr_other_loss_fraction.iloc[1]))
        self.assertEqual(result.lcms_pre_5yr_valid_years_mean.tolist(), [4, 0])
        self.assertEqual(result.lcms_pre_5yr_complete_coverage_fraction.tolist(), [0, 0])
        self.assertEqual(result.lcms_timing_resolution.iloc[0], "annual; uncertain ordering")
        validation.validate_measurements(result)

    def test_explicit_grid_area_and_no_measurement_unmask(self):
        captured = []
        with patch.object(reductions, "fetch_patch_table", side_effect=lambda fc, columns: captured.append(fc)):
            reductions.area_sums({"x": ee.Image("source").select("x")},
                                 ee.FeatureCollection([]), ee.Projection("EPSG:5070").atScale(30))
        graph = captured[0].serialize()
        self.assertIn(reductions.RASTER_FULL_AREA, graph)
        self.assertRegex(reductions.RASTER_FULL_AREA, r"^[A-Za-z][A-Za-z0-9_]*$")
        for expected in ("Image.pixelArea", "Image.mask", "Image.multiply", "Reducer.sum", "Projection.atScale"):
            self.assertIn(expected, graph)
        self.assertIsInstance(captured[0].args["crs"], ee.Projection)
        self.assertIsNone(captured[0].args.get("crsTransform"))
        for forbidden in ("Image.unmask", "Image.reproject", "Image.resample", "Image.clip", "Reducer.mean"):
            self.assertNotIn(forbidden, graph)

    def test_vectors_remain_full_planar_squares(self):
        graph = pipeline.patches_to_ee(local_patches()).serialize()
        self.assertIn("EPSG:5070", graph)
        self.assertIn('"geodesic": {"constantValue": false}', graph)
        self.assertIn("patch_id", graph)

    def test_focal_severity_uses_validated_annual_source_and_classes(self):
        captured = {}
        def fake(images, patches, projection):
            captured.update(images)
            values = {"patch_id": ["p"], "fire_id": ["f"], reductions.RASTER_FULL_AREA: [8100.]}
            for name in images:
                values.update({name + "_sum": [0.], name + "_area": [8100.]})
            return pd.DataFrame(values)
        with patch.object(reductions, "area_sums", side_effect=fake):
            result = reductions.reduce_focal_severity({"fire_year": 2017}, ee.FeatureCollection([]))
        graph = captured["mtbs_high"].serialize()
        self.assertIn("mtbs_CONUS_2017", graph)
        self.assertIn(reductions.MTBS_SEVERITY_SOURCE, graph)
        self.assertIn("mtbs_nonmapping_fraction", result)
        self.assertIn("mtbs_source_valid_fraction", result)
        self.assertIn("mtbs_valid_fraction", result)


if __name__ == "__main__":
    unittest.main()
