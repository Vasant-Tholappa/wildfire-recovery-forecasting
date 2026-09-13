"""Synthetic, offline checks; never count the real population in tests."""

import unittest
from unittest.mock import patch

import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, box

import manual_planb_population_counts as counts


class PopulationCountTests(unittest.TestCase):
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
