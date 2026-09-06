"""Local geometric checks for the canonical production lattice."""

import unittest

import geopandas as gpd
from shapely.geometry import MultiPolygon, Polygon, box

from wildfire_recovery_forecasting.grid import build_fire_grid


class FireGridTests(unittest.TestCase):
    def test_full_and_partial_cells_keep_full_squares(self):
        grid = build_fire_grid(box(0, 0, 135, 90), "fire", perimeter_crs="EPSG:5070")
        self.assertEqual(grid.crs.to_epsg(), 5070)
        self.assertEqual(grid.patch_id.tolist(), ["fire_r0_c0", "fire_r0_c1"])
        self.assertEqual(grid.fire_overlap_fraction.tolist(), [1.0, 0.5])
        self.assertTrue(grid.patch_id.is_unique)
        self.assertTrue((grid.fire_id == "fire").all())
        self.assertTrue((grid.patch_area_m2 == 8100).all())
        for cell in grid.geometry:
            min_x, min_y, max_x, max_y = cell.bounds
            self.assertEqual(max_x - min_x, 90)
            self.assertEqual(max_y - min_y, 90)
            self.assertAlmostEqual(cell.area, 8100)
            self.assertTrue(all(value % 90 == 0 for value in cell.bounds))
        self.assertEqual(grid.geometry.iloc[1].bounds, (90, 0, 180, 90))

    def test_determinism_and_alignment_across_fires(self):
        perimeter = box(10, 10, 100, 100)
        first = build_fire_grid(perimeter, "a", perimeter_crs="EPSG:5070")
        again = build_fire_grid(perimeter, "a", perimeter_crs="EPSG:5070")
        self.assertTrue(first.equals(again))
        self.assertEqual(first.geometry.to_wkb().tolist(), again.geometry.to_wkb().tolist())
        other = build_fire_grid(box(20, 30, 110, 120), "b", perimeter_crs="EPSG:5070")
        self.assertEqual(first.grid_row.tolist(), other.grid_row.tolist())
        self.assertEqual(first.grid_col.tolist(), other.grid_col.tolist())
        self.assertEqual(first.geometry.to_wkb().tolist(), other.geometry.to_wkb().tolist())

    def test_negative_coordinates_use_floor_indices(self):
        grid = build_fire_grid(box(-135, -45, 45, 45), "neg", perimeter_crs="EPSG:5070")
        self.assertEqual(
            list(zip(grid.grid_row, grid.grid_col)),
            [(-1, -2), (-1, -1), (-1, 0), (0, -2), (0, -1), (0, 0)],
        )
        self.assertEqual(grid.patch_id.iloc[0], "neg_r-1_c-2")
        self.assertEqual(grid.geometry.iloc[0].bounds, (-180, -90, -90, 0))
        self.assertEqual(grid.fire_overlap_fraction.tolist(), [0.25, 0.5, 0.25] * 2)

    def test_zero_area_line_and_point_touches_are_excluded(self):
        # The upper-right cell touches the diagonal only; it is in the bbox.
        triangle = Polygon([(0, 0), (180, 0), (0, 180)])
        grid = build_fire_grid(triangle, "line", perimeter_crs="EPSG:5070")
        self.assertEqual(grid.patch_id.tolist(), ["line_r0_c0", "line_r0_c1", "line_r1_c0"])
        self.assertEqual(grid.fire_overlap_fraction.tolist(), [1, 0.5, 0.5])
        # The same bbox contains two cells that touch only at (90, 90).
        diagonal = MultiPolygon([box(0, 0, 90, 90), box(90, 90, 180, 180)])
        grid = build_fire_grid(diagonal, "point", perimeter_crs="EPSG:5070")
        self.assertEqual(grid.patch_id.tolist(), ["point_r0_c0", "point_r1_c1"])

    def test_holes_and_disconnected_perimeter_preserve_overlap_area(self):
        ring = Polygon(box(0, 0, 270, 270).exterior.coords, [box(90, 90, 180, 180).exterior.coords])
        perimeter = MultiPolygon([ring, box(360, 0, 405, 45)])
        grid = build_fire_grid(perimeter, "multi", perimeter_crs="EPSG:5070")
        self.assertEqual(len(grid), 9)
        self.assertNotIn("multi_r1_c1", grid.patch_id.tolist())
        self.assertTrue((grid.fire_overlap_fraction > 0).all())
        self.assertTrue((grid.fire_overlap_fraction <= 1).all())
        self.assertAlmostEqual((grid.fire_overlap_fraction * grid.patch_area_m2).sum(), perimeter.area)

    def test_reprojects_geographic_perimeter(self):
        perimeter = box(-115, 45, -114.998, 45.002)
        projected = gpd.GeoSeries([perimeter], crs="EPSG:4326").to_crs("EPSG:5070").iloc[0]
        expected = build_fire_grid(projected, "geo", perimeter_crs="EPSG:5070")
        actual = build_fire_grid(perimeter, "geo", perimeter_crs="EPSG:4326")
        self.assertEqual(actual.crs.to_epsg(), 5070)
        self.assertGreater(len(actual), 0)
        self.assertTrue(actual.equals(expected))
        self.assertAlmostEqual((actual.fire_overlap_fraction * 8100).sum(), projected.area)

    def test_invalid_inputs_are_rejected(self):
        for perimeter in (None, Polygon(), Polygon([(0, 0), (90, 90), (90, 0), (0, 90)])):
            with self.subTest(perimeter=perimeter), self.assertRaises(ValueError):
                build_fire_grid(perimeter, "fire", perimeter_crs="EPSG:5070")
        for fire_id in (None, "", " "):
            with self.subTest(fire_id=fire_id), self.assertRaises(ValueError):
                build_fire_grid(box(0, 0, 90, 90), fire_id, perimeter_crs="EPSG:5070")
        with self.assertRaises(ValueError):
            build_fire_grid(box(0, 0, 90, 90), "fire", perimeter_crs=None)


if __name__ == "__main__":
    unittest.main()
