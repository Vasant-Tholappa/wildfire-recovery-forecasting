"""Local fire-manifest checks; no Earth Engine requests."""

from datetime import date
import unittest
from unittest.mock import MagicMock, patch

import geopandas as gpd
import pandas as pd
from shapely.geometry import GeometryCollection, LineString, MultiPolygon, Point, Polygon, box

from wildfire_recovery_forecasting import fire_catalog as catalog


def mtbs_rows():
    return gpd.GeoDataFrame(
        {
            "Event_ID": ["ID_TEST_2017", "CA_TEST_2020"],
            "Incid_Name": ["Test fire", "Small fire"],
            "Incid_Type": ["Wildfire", "Wildfire"],
            "Ig_Date": [1501311600000, pd.Timestamp("2020-11-01", tz="UTC").value // 10**6],
            "BurnBndAc": [93699, 100],
        },
        geometry=[box(-116, 45, -115, 46), box(-121, 38, -120, 39)],
        crs="EPSG:4326",
    )


class FireDerivationTests(unittest.TestCase):
    def test_prefire_season_boundaries(self):
        for ignition, last in [
            (date(2015, 7, 15), 2014),
            (date(2015, 9, 30), 2014),
            (date(2015, 10, 1), 2015),
            (date(2015, 11, 15), 2015),
        ]:
            with self.subTest(ignition=ignition):
                fields = catalog.derive_fire_fields(ignition)
                self.assertEqual(fields["last_fully_prefire_growing_season_year"], last)
                self.assertEqual(fields["prefire_search_start_year"], last - 4)
                self.assertEqual(fields["prefire_search_end_year"], last)
                self.assertEqual(fields["season1_year"], 2016)
                self.assertEqual(fields["season5_year"], 2020)

    def test_versions_and_splits(self):
        for year, condition, version, split in [
            (2015, 2014, "v1.4.0", "train"),
            (2016, 2014, "v1.4.0", "train"),
            (2017, 2014, "v1.4.0", "train"),
            (2018, 2014, "v1.4.0", "validation"),
            (2019, 2014, "v1.4.0", "validation"),
            (2020, 2014, "v1.4.0", "test"),
        ]:
            with self.subTest(year=year):
                fields = catalog.derive_fire_fields(date(year, 7, 1))
                self.assertEqual(fields["fire_year"], year)
                self.assertEqual(fields["landfire_condition_year"], condition)
                self.assertEqual(fields["landfire_version"], version)
                self.assertEqual(fields["split"], split)

    def test_invalid_dates(self):
        for value in [None, pd.NaT, 1501311600000, date(2012, 12, 31),
                      date(2013, 7, 1), date(2014, 12, 31), date(2021, 1, 1)]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                catalog.derive_fire_fields(value)


class GeometryNormalizationTests(unittest.TestCase):
    def test_polygon_and_multipolygon_unchanged(self):
        for geometry in [box(0, 0, 10, 10), MultiPolygon([box(0, 0, 10, 10), box(20, 0, 21, 1)])]:
            with self.subTest(geometry=geometry):
                self.assertIs(catalog.normalize_fire_geometry(geometry), geometry)

    def test_collection_preserves_holes_and_small_parts(self):
        polygon = Polygon(box(0, 0, 10, 10).exterior.coords, [box(2, 2, 4, 4).exterior.coords])
        tiny = box(20, 0, 20.00001, 0.00001)
        line = LineString([(0, 0), (1, 1)])
        for parts in [[polygon], [polygon, tiny]]:
            with self.subTest(parts=len(parts)):
                collection = GeometryCollection([line, *parts])
                result = catalog.normalize_fire_geometry(collection)
                self.assertEqual(result.geom_type, "Polygon" if len(parts) == 1 else "MultiPolygon")
                actual_parts = [result] if isinstance(result, Polygon) else list(result.geoms)
                self.assertEqual([part.wkb for part in actual_parts], [part.wkb for part in parts])
                self.assertEqual(result.area, collection.area)
                self.assertTrue(result.is_valid)

    def test_nested_polygonal_components_are_retained(self):
        parts = [box(0, 0, 1, 1), box(2, 0, 3, 1)]
        collection = GeometryCollection([GeometryCollection([MultiPolygon(parts), Point(0, 0)])])
        result = catalog.normalize_fire_geometry(collection)
        self.assertEqual([part.wkb for part in result.geoms], [part.wkb for part in parts])
        self.assertEqual(result.area, collection.area)

    def test_no_polygons_invalid_and_unsupported_are_rejected(self):
        line = LineString([(0, 0), (1, 1)])
        invalid = Polygon([(0, 0), (1, 1), (1, 0), (0, 1)])
        with self.assertRaisesRegex(ValueError, "no polygonal components"):
            catalog.normalize_fire_geometry(GeometryCollection([line]))
        for geometry in [None, Polygon(), line, Point(0, 0), invalid, GeometryCollection([invalid, line]),
                         GeometryCollection([box(0, 0, 2, 2), box(1, 1, 3, 3)])]:
            with self.subTest(geometry=geometry), self.assertRaises(ValueError):
                catalog.normalize_fire_geometry(geometry)


class ManifestTests(unittest.TestCase):
    def test_focal_population_date_bounds(self):
        for ignition, accepted in [
            ("2013-07-01", False), ("2014-12-31", False),
            ("2015-01-01", True), ("2020-12-31", True),
            ("2021-01-01", False),
        ]:
            with self.subTest(ignition=ignition):
                source = mtbs_rows()
                source.at[0, "Ig_Date"] = pd.Timestamp(ignition, tz="UTC").value // 10**6
                if accepted:
                    self.assertEqual(len(catalog.manifest_from_mtbs(source)), 2)
                else:
                    with self.assertRaisesRegex(ValueError, "Focal fire year"):
                        catalog.manifest_from_mtbs(source)

    def test_manifest_normalizes_collection_without_changing_source(self):
        source = mtbs_rows()
        polygon = source.geometry.iloc[0]
        collection = GeometryCollection([polygon, LineString([(-116, 45), (-115, 46)])])
        source.at[0, "geometry"] = collection
        manifest = catalog.manifest_from_mtbs(source).set_index("fire_id")
        self.assertEqual(len(manifest), len(source))
        self.assertEqual(manifest.loc["ID_TEST_2017"].geometry.wkb, polygon.wkb)
        self.assertEqual(source.geometry.iloc[0].wkb, collection.wkb)
        self.assertEqual(manifest.crs, source.crs)

    def test_manifest_preserves_dates_geometry_and_metadata(self):
        source = mtbs_rows()
        source.index = [5, 5]
        manifest = catalog.manifest_from_mtbs(source)
        self.assertEqual(len(manifest), 2)
        self.assertTrue(manifest.fire_id.is_unique)
        self.assertTrue(manifest.fire_year.between(2015, 2020).all())
        self.assertTrue(manifest.landfire_condition_year.eq(2014).all())
        self.assertTrue(manifest.landfire_version.eq("v1.4.0").all())
        self.assertFalse(manifest.ignition_date.isna().any())
        self.assertFalse(manifest.geometry.isna().any())
        row = manifest.set_index("fire_id").loc["ID_TEST_2017"]
        self.assertEqual(row.ignition_date, pd.Timestamp("2017-07-29 07:00:00Z"))
        self.assertEqual(row.fire_size_acres, 93699)
        self.assertTrue(row.geometry.equals(source.geometry.iloc[0]))
        self.assertEqual(manifest.crs, source.crs)
        self.assertTrue(manifest.source_dataset.eq(catalog.MTBS_SOURCE).all())

    def test_rejects_incomplete_or_conflicting_records(self):
        cases = [
            ("Event_ID", "CA_TEST_2020"),
            ("Event_ID", None),
            ("Ig_Date", None),
            ("Ig_Date", "2017-07-29"),
            ("Ig_Date", 1501311600),  # Seconds must not be interpreted as milliseconds.
            ("Incid_Type", "Prescribed Fire"),
            ("geometry", None),
            ("geometry", Polygon()),
        ]
        for field, value in cases:
            with self.subTest(field=field, value=value):
                source = mtbs_rows()
                if field == "Ig_Date":
                    source[field] = source[field].astype(object)
                source.at[0, field] = value
                with self.assertRaises(ValueError):
                    catalog.manifest_from_mtbs(source)

    def test_earth_engine_population_query_without_network(self):
        states, fires = MagicMock(), MagicMock()
        states.filter.return_value = states
        fires.filter.return_value = fires
        fires.filterBounds.return_value = fires
        fires.select.return_value = fires
        with patch.object(catalog, "ee") as ee:
            ee.FeatureCollection.side_effect = [states, fires]
            ee.Filter.eq.return_value = "wildfire_filter"
            ee.Filter.gte.return_value = "start_filter"
            ee.Filter.lt.return_value = "end_filter"
            ee.data.computeFeatures.return_value = mtbs_rows().set_crs(None, allow_override=True)
            manifest = catalog.build_fire_manifest()
            ee.Filter.inList.assert_called_once_with("STUSPS", list(catalog.WESTERN_STATE_CODES))
            self.assertEqual(len(catalog.WESTERN_STATE_CODES), 11)
            ee.Filter.eq.assert_called_once_with("Incid_Type", "Wildfire")
            self.assertEqual([c.args[0] for c in ee.Date.call_args_list], ["2015-01-01", "2021-01-01"])
            self.assertEqual([c.args[0] for c in fires.filter.call_args_list], ["wildfire_filter", "start_filter", "end_filter"])
            fires.filterBounds.assert_called_once_with(states.geometry.return_value)
            ee.data.computeFeatures.assert_called_once_with({
                "expression": fires, "fileFormat": "GEOPANDAS_GEODATAFRAME"
            })
            self.assertEqual(len(manifest), 2)
            self.assertEqual(manifest.crs.to_epsg(), 4326)


if __name__ == "__main__":
    unittest.main()
