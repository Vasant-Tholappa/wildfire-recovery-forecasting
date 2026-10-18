"""Local approved selection regression; no cloud access."""
import unittest
from pathlib import Path
import hashlib
import geopandas as gpd
import pandas as pd
from manual_freeze_ca175 import select_ca, IDS_HASH
from wildfire_recovery_forecasting.fire_catalog import derive_fire_fields

class SelectionTests(unittest.TestCase):
    @unittest.skipUnless(Path("data/processed/fire_manifest.parquet").exists(), "Local original manifest required")
    def test_exact_approved_selection_geometry_timing_and_order_independence(self):
        original = gpd.read_parquet("data/processed/fire_manifest.parquet")
        selected = select_ca(original)
        shuffled = select_ca(original.sample(frac=1, random_state=12))
        self.assertEqual(selected.fire_id.tolist(), shuffled.fire_id.tolist())
        self.assertEqual(hashlib.sha256("\n".join(selected.fire_id).encode()).hexdigest(), IDS_HASH)
        self.assertEqual(selected.split.value_counts().to_dict(), {"train":91,"validation":43,"test":41})
        frozen = gpd.read_parquet("data/processed/phase3_ca175_population_v1.parquet")
        expected = original.set_index("fire_id").loc[selected.fire_id].reset_index()
        pd.testing.assert_frame_equal(frozen, expected)
        for row in frozen.itertuples():
            for name,value in derive_fire_fields(row.ignition_date).items():
                self.assertEqual(getattr(row,name),value)
        import manual_population as driver
        manifest, items = driver.load_population()
        self.assertEqual((len(manifest),len(items)),(175,2100))
        self.assertEqual(len({(i["fire_id"],i["batch_id"]) for i in items}),175)
        self.assertEqual(sum(i["patch_count"] for i in items)//12,308048)
