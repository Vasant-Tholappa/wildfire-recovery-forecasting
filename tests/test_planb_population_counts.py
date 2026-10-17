"""Synthetic, offline checks; never count the real population in tests."""

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon, box

import manual_planb_population_counts as counts


class PopulationCountTests(unittest.TestCase):
    def test_optimized_matches_canonical_geometry_cases(self):
        cases = [
            Polygon(box(-270, -270, 270, 270).exterior.coords,
                    [box(-90, -90, 90, 90).exterior.coords]),
            MultiPolygon([box(-10890, -90, -10800, 0), box(10800, 0, 10801, 1)]),
            Polygon([(0, 0), (180, 0), (0, 180)]),
            box(-91, -1, 1, 91),
            box(0, 0, 10800, 10800),  # fully covered tile shortcut
            box(10800, 0, 10890, 90),  # boundary touch must not add adjacent tiles
        ]
        for i, geometry in enumerate(cases):
            with self.subTest(case=i):
                manifest = gpd.GeoDataFrame({"fire_id": [str(i)]},
                                           geometry=[geometry], crs="EPSG:5070")
                pd.testing.assert_frame_equal(counts.batch_patch_counts(manifest),
                                              counts.batch_patch_counts_reference(manifest))

    def test_batch_patch_counts_are_positive_deterministic_and_match_tile_count(self):
        manifest = gpd.GeoDataFrame({"fire_id": ["a", "b"]}, geometry=[
            MultiPolygon([box(-90, -90, 90, 90), box(10800, 0, 10980, 90)]),
            box(0, 0, 180, 90),
        ], crs="EPSG:5070")
        first = counts.batch_patch_counts(manifest)
        second = counts.batch_patch_counts(manifest.iloc[::-1])
        pd.testing.assert_frame_equal(first, second)
        self.assertTrue((first.patch_count > 0).all())
        self.assertEqual(len(first), counts.count_tiles(manifest, 120)["spatial_batches"])
        self.assertEqual(first.patch_count.sum(), 8)
        self.assertEqual(set(first[first.fire_id == "a"].batch_id),
                         {"r-1_c-1", "r-1_c0", "r0_c-1", "r0_c0", "r0_c1"})

    def test_population_output_requires_exact_6301_batches(self):
        # Synthetic rows test the regression guard, not the real population.
        table = pd.DataFrame({"fire_id": ["f"] * 6301,
                              "batch_id": [f"r0_c{i}" for i in range(6301)],
                              "patch_count": [1] * 6301})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "counts.parquet"
            counts.write_batch_patch_counts(table, path)
            pd.testing.assert_frame_equal(table, pd.read_parquet(path))
            for bad in (table.iloc[:-1], table.assign(patch_count=0)):
                with self.assertRaisesRegex(ValueError, "6,301"):
                    counts.write_batch_patch_counts(bad, path)

    def test_positive_area_negative_coordinates_holes_and_sparse_parts(self):
        # A tile-sized hole contributes no batch; touching boundaries add none.
        ring = Polygon(box(-180, -180, 180, 180).exterior.coords,
                       [box(-90, -90, 90, 90).exterior.coords])
        sparse = MultiPolygon([box(1, 1, 2, 2), box(271, 1, 272, 2)])
        manifest = gpd.GeoDataFrame(geometry=[ring, sparse], crs="EPSG:5070")
        self.assertEqual(counts.count_tiles(manifest, 1), {
            "spatial_batches": 14, "per_fire_median": 7.0, "per_fire_max": 12})

    def test_all_regressions_precede_requested_counts(self):
        manifest = gpd.GeoDataFrame(geometry=[box(0, 0, 90, 90)], crs="EPSG:5070")
        def fake_count(frame, tile):
            self.assertEqual(frame.crs.to_epsg(), 5070)
            return {"spatial_batches": counts.REGRESSION_COUNTS.get(tile, 1),
                    "per_fire_median": 1.0, "per_fire_max": 1}
        with patch.object(counts, "count_tiles", side_effect=fake_count) as counter:
            result = counts.verified_counts(manifest, [90, 120, 150])
        self.assertEqual([call.args[1] for call in counter.call_args_list],
                         [10, 20, 30, 90, 120, 150])
        self.assertEqual(set(result["counts"]), {"90", "120", "150"})
        self.assertEqual(result["fire_count"], 1)
        self.assertEqual(result["method"], counts.METHOD)

    def test_regression_mismatch_stops_before_new_counts(self):
        manifest = gpd.GeoDataFrame(geometry=[box(0, 0, 90, 90)], crs="EPSG:5070")
        with patch.object(counts, "count_tiles", return_value={"spatial_batches": 0}) as counter:
            with self.assertRaisesRegex(ValueError, "Tile 10 regression failed"):
                counts.verified_counts(manifest, [90, 120, 150])
        self.assertEqual(counter.call_count, 1)


if __name__ == "__main__":
    unittest.main()
