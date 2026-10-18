"""Approved patch-drop contract and retired-work protection, entirely offline."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import manual_planb_eecu_forecast as forecast
from wildfire_recovery_forecasting import daymet, exports, pipeline, validation, work_items


class CandidateATests(unittest.TestCase):
    def table(self):
        data = pd.DataFrame({"patch_id": ["p", "q", "r"], "fire_id": ["f"] * 3,
                             "nbr_pre_mean": [.5, np.nan, .5],
                             "nbr_1_mean": [.1, .1, np.nan], "nbr_5_mean": [.6] * 3,
                             "nbr_pre_valid_fraction": [1., 0., 1.],
                             "nbr_1_valid_fraction": [.5, .5, 0.]})
        for name in daymet.HISTORICAL_BANDS:
            data[name] = 1.
        return data

    def validate(self, table):
        roles = {"target": "nbr_5_mean", "predictors": ["nbr_pre_mean", "nbr_1_mean"],
                 "early_predictors": ["nbr_pre_mean", *daymet.HISTORICAL_BANDS], "qa": []}
        if "nbr_drop_patch" in table:
            roles["predictors"].append("nbr_drop_patch")
        validation.validate_dataset(table, roles, expected_ids=["p", "q", "r"], fire_id="f")

    def test_raw_tables_need_no_drop_and_sql_contract_preserves_nulls(self):
        table = self.table()
        self.validate(table)
        table["nbr_drop_patch"] = table.nbr_pre_mean - table.nbr_1_mean
        self.validate(table)
        self.assertAlmostEqual(table.nbr_drop_patch.iloc[0], .4)
        self.assertTrue(table.nbr_drop_patch.iloc[1:].isna().all())
        # Canonical example: pre pixels .8,.2; S1 .1,missing -> old drop .7, new .4.
        self.assertNotAlmostEqual(table.nbr_drop_patch.iloc[0], .7)
        roles = pipeline.dataset_roles(table)
        self.assertIn("nbr_drop_patch", roles["predictors"])
        self.assertNotIn("nbr_drop_patch", roles["early_predictors"])
        self.assertNotIn("nbr_drop_mean", roles["predictors"])
        for bad in (table.assign(nbr_drop_patch=0), table.assign(nbr_drop_mean=.7),
                    table.assign(nbr_drop_valid_fraction=1)):
            with self.assertRaises(ValueError):
                self.validate(bad)

    def test_retired_product_cannot_submit_or_access_status(self):
        with patch.object(work_items.persistence, "read_status", side_effect=AssertionError("GCS")), \
             patch.object(exports.ee.batch.Export.table, "toCloudStorage", side_effect=AssertionError("EE")):
            with self.assertRaisesRegex(ValueError, "retired"):
                work_items.advance_work_item(None, [], source="landsat", product="nbrdrop",
                    fire_id="f", batch_id="r0_c0", expected_patch_ids=["p"])
            with self.assertRaisesRegex(ValueError, "retired"):
                exports.start_table_export(None, [], source="landsat", product="nbrdrop",
                                           fire_id="f", batch_id="r0_c0")

    def test_forecast_excludes_once_without_changing_history(self):
        old = {"60": {"landsat/nbrdrop": 120., "landsat/nbr5": 18., "daymet/primary": 313.}}
        new = forecast.exclude_retired_products(old)
        self.assertIn("landsat/nbrdrop", old["60"])
        self.assertEqual(new["60"], {"landsat/nbr5": 18., "daymet/primary": 313.})
        self.assertEqual(forecast.exclude_retired_products(new), new)


if __name__ == "__main__":
    unittest.main()
