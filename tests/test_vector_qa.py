import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import Mock, patch
import geopandas as gpd
import pandas as pd
from shapely.geometry import box
import manual_vector_qa as driver
from wildfire_recovery_forecasting import disturbance

class VectorQATests(unittest.TestCase):
    def setUp(self):
        self.manifest = gpd.GeoDataFrame({'fire_id': ['CAa', 'CAb'], 'ignition_date': [date(2017,7,1)]*2, 'season5_year': [2022]*2}, geometry=[box(0,0,45,90), box(90,0,180,90)], crs='EPSG:5070')
        self.tiles = [{'fire_id': n, 'batch_id': 'r0_c0', 'patch_count': 1} for n in ('CAa','CAb')]
        self.events = gpd.GeoDataFrame({'Event_ID': ['prior','reburn'], 'Incid_Type': ['Wildfire']*2, 'Ig_Date': [pd.Timestamp(d,tz='UTC').value//10**6 for d in ('2010-01-01','2018-01-01')]}, geometry=[box(0,0,45,90),box(0,0,90,90)],crs='EPSG:5070')
    def history(self, fire, patches):
        return disturbance.history_from_mtbs(fire,self.events)
    def run_driver(self, root, loader, tiles=None):
        return driver.run_tiles(self.manifest,tiles or self.tiles,root,history_loader=loader,provenance={'inputs':'frozen'},filename='supplemental.parquet')
    @patch('ee.Initialize',side_effect=AssertionError('No live EE'))
    def test_fractions_ids_schema_and_resume(self, initialize):
        with tempfile.TemporaryDirectory() as root:
            table = pd.read_parquet(self.run_driver(root,self.history))
            self.assertEqual(list(table),driver.COLUMNS)
            self.assertEqual(table.patch_id.tolist(),['CAa_r0_c0','CAb_r0_c1'])
            self.assertEqual(table.fire_overlap_fraction.tolist(),[.5,1.])
            self.assertEqual(table.prior_fire_fraction_10yr.tolist(),[.5,0.])
            self.assertEqual(table.reburn_fraction.tolist(),[1.,0.])
            self.assertTrue(table[driver.STATUSES].eq('unverified').all().all())
            loader=Mock(side_effect=AssertionError('No resumed retrieval'))
            self.run_driver(root,loader)
            loader.assert_not_called()
            initialize.assert_not_called()
    def test_interruption_and_resume(self):
        with tempfile.TemporaryDirectory() as root:
            calls=[]
            def interrupted(fire,patches):
                calls.append(fire['fire_id'])
                if len(calls)==2: raise KeyboardInterrupt()
                return self.history(fire,patches)
            with self.assertRaises(KeyboardInterrupt): self.run_driver(root,interrupted)
            self.assertFalse((Path(root)/'supplemental.parquet').exists())
            loader=Mock(side_effect=self.history)
            self.run_driver(root,loader)
            self.assertEqual(loader.call_count,1)
            self.assertEqual(loader.call_args.args[0]['fire_id'],'CAb')
    def test_corrupt_resume_and_changed_inputs_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            self.run_driver(root,self.history)
            part=Path(root)/'tiles/CAa_r0_c0.parquet'
            table=pd.read_parquet(part)
            table.loc[0,'patch_id']='wrong'
            table.to_parquet(part,index=False)
            with self.assertRaisesRegex(ValueError,'patch IDs'): self.run_driver(root,Mock())
            with self.assertRaisesRegex(ValueError,'inputs changed'):
                driver.run_tiles(self.manifest,self.tiles,root,history_loader=Mock(),provenance={},filename='supplemental.parquet')
    def test_selection_and_occupancy(self):
        items=list(reversed(self.tiles))*12
        self.assertEqual(driver.selected_tiles(items,1),self.tiles[:1])
        self.assertEqual(driver.selected_tiles(items),self.tiles)
        with self.assertRaises(ValueError): driver.selected_tiles(items,0)
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError,'occupancy mismatch'):
                self.run_driver(root,Mock(),[{**self.tiles[0],'patch_count':2}])

if __name__=='__main__': unittest.main()
