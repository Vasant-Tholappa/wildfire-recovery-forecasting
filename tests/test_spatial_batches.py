"""Offline checks of fixed spatial tiles over retained canonical patches."""

import unittest

from shapely.geometry import box

from wildfire_recovery_forecasting.grid import assign_spatial_batches, build_fire_grid


class SpatialBatchTests(unittest.TestCase):
    def setUp(self):
        self.patches = build_fire_grid(box(-270, -270, 270, 270), "fire",
                                      perimeter_crs="EPSG:5070")

    def test_shuffle_preserves_assignment_and_all_input_data(self):
        original = self.patches.copy()
        first = assign_spatial_batches(original, tile_side_cells=2)
        shuffled = assign_spatial_batches(original.sample(frac=1, random_state=7),
                                         tile_side_cells=2)
        self.assertEqual(dict(zip(first.patch_id, first.batch_id)),
                         dict(zip(shuffled.patch_id, shuffled.batch_id)))
        self.assertTrue(first.drop(columns="batch_id").equals(original))
        self.assertTrue(original.equals(self.patches))
        self.assertEqual(first.crs, original.crs)
        grouped_ids = [patch_id for _, group in first.groupby("batch_id")
                       for patch_id in group.patch_id]
        self.assertCountEqual(grouped_ids, original.patch_id)
        self.assertEqual(len(grouped_ids), len(set(grouped_ids)))
        for _, group in first.groupby("batch_id"):
            self.assertLessEqual(len(group), 4)
            self.assertLess(group.grid_row.max() - group.grid_row.min(), 2)
            self.assertLess(group.grid_col.max() - group.grid_col.min(), 2)

    def test_positive_and_negative_tile_boundaries(self):
        result = assign_spatial_batches(self.patches, tile_side_cells=2)
        expected = {-3: -2, -2: -1, -1: -1, 0: 0, 1: 0, 2: 1}
        for row in result.itertuples():
            self.assertEqual(row.batch_id,
                             f"r{expected[row.grid_row]}_c{expected[row.grid_col]}")

    def test_sparse_edges_are_not_filled_from_adjacent_tiles(self):
        sparse = self.patches.loc[(self.patches.grid_row == 0)
                                  & self.patches.grid_col.isin([0, 2])]
        result = assign_spatial_batches(sparse, tile_side_cells=2)
        self.assertEqual(result.batch_id.tolist(), ["r0_c0", "r0_c1"])
        self.assertEqual(result.groupby("batch_id").size().tolist(), [1, 1])

    def test_explicit_tile_size_and_empty_input(self):
        singles = assign_spatial_batches(self.patches, tile_side_cells=1)
        self.assertTrue(singles.batch_id.is_unique)
        empty = assign_spatial_batches(self.patches.iloc[:0], tile_side_cells=10)
        self.assertTrue(empty.empty)
        self.assertEqual(empty.crs, self.patches.crs)
        self.assertIn("batch_id", empty)
        for invalid in (0, -1, 1.5, 2.0, True, False, "10", None):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                assign_spatial_batches(self.patches, tile_side_cells=invalid)

    def test_duplicate_or_missing_patch_ids_rejected(self):
        for value in (None, self.patches.patch_id.iloc[1]):
            bad = self.patches.copy()
            bad.loc[bad.index[0], "patch_id"] = value
            with self.assertRaises(ValueError):
                assign_spatial_batches(bad, tile_side_cells=2)


if __name__ == "__main__":
    unittest.main()
