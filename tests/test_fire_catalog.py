"""Local fire-manifest checks; no Earth Engine requests."""

from datetime import date
import unittest
from unittest.mock import MagicMock, patch

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon, box

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
            (2013, 2012, "v1.3.0", "train"),
            (2014, 2012, "v1.3.0", "train"),
            (2015, 2014, "v1.4.0", "train"),
            (2016, 2014, "v1.4.0", "train"),
            (2017, 2016, "v2.0.0", "train"),
            (2018, 2016, "v2.0.0", "validation"),
            (2019, 2016, "v2.0.0", "validation"),
            (2020, 2016, "v2.0.0", "test"),
        ]:
            with self.subTest(year=year):
                fields = catalog.derive_fire_fields(date(year, 7, 1))
                self.assertEqual(fields["fire_year"], year)
                self.assertEqual(fields["landfire_condition_year"], condition)
                self.assertEqual(fields["landfire_version"], version)
                self.assertEqual(fields["split"], split)

    def test_invalid_dates(self):
        for value in [None, pd.NaT, 1501311600000, date(2012, 12, 31), date(2021, 1, 1)]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                catalog.derive_fire_fields(value)


class ManifestTests(unittest.TestCase):
    def test_manifest_preserves_dates_geometry_and_metadata(self):
        source = mtbs_rows()
        source.index = [5, 5]
        manifest = catalog.manifest_from_mtbs(source)
        self.assertEqual(len(manifest), 2)
        self.assertTrue(manifest.fire_id.is_unique)
        self.assertTrue(manifest.fire_year.between(2013, 2020).all())
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
            self.assertEqual([c.args[0] for c in ee.Date.call_args_list], ["2013-01-01", "2021-01-01"])
            self.assertEqual([c.args[0] for c in fires.filter.call_args_list], ["wildfire_filter", "start_filter", "end_filter"])
            fires.filterBounds.assert_called_once_with(states.geometry.return_value)
            ee.data.computeFeatures.assert_called_once_with({
                "expression": fires, "fileFormat": "GEOPANDAS_GEODATAFRAME"
            })
            self.assertEqual(len(manifest), 2)
            self.assertEqual(manifest.crs.to_epsg(), 4326)


if __name__ == "__main__":
    unittest.main()
