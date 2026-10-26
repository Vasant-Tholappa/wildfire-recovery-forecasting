"""Offline tests for the fixed two-tile terrain recovery workflow."""
import unittest
from unittest.mock import patch
import pandas as pd
import ee
from ee.apitestcase import ApiTestCase
import manual_terrain_recovery as p

class RecoveryTests(unittest.TestCase):
    def test_frozen_inputs_and_isolation(self):
        m, items = p.pilot_inputs()
        self.assertEqual(len(items), 2)
        self.assertEqual({i['fire_id'] for i in items}, set(p.FIRES))
        for i in items:
            patches = p.prepare_patches(m, i)
            self.assertEqual(len(patches), i['patch_count'])
            self.assertTrue(patches.patch_id.is_unique)
            for name, path in p.paths(i).items():
                self.assertIn('/'+p.VERSION+'/', path)
                self.assertNotEqual(path, p.paths(i,p.population.VERSION)[name])

    def test_check_missing_cannot_build_or_submit(self):
        with patch.object(p.persistence,'read_status',return_value=None), patch.object(p,'terrain_product') as build, patch.object(p.work_items,'advance_work_item') as step:
            result=p.advance('check',dict(fire_id='f',batch_id='b'),{},None,object())
        self.assertEqual(result['action'],'NOT_SUBMITTED')
        build.assert_not_called()
        step.assert_not_called()

    def test_check_metadata_no_graph(self):
        task=dict(state='COMPLETED',batch_eecu_usage_seconds=12,start_timestamp_ms=100)
        with patch.object(p.persistence,'read_status',return_value=dict(status='RUNNING',ee_task_id='t')), patch.object(p.work_items,'lookup_task_status',return_value=task), patch.object(p,'terrain_product') as build, patch.object(p.work_items,'advance_work_item',return_value=dict(action='COMPLETE')) as step:
            result=p.advance('check',dict(fire_id='f',batch_id='b'),{},pd.DataFrame(dict(patch_id=['p'])),object())
        build.assert_not_called()
        self.assertIsNone(step.call_args.args[0])
        self.assertEqual(result['ee_task'],task)
        self.assertEqual(step.call_args.kwargs['production_version'],p.VERSION)

    def test_raced_status_cannot_export(self):
        with patch.object(p.persistence,'read_status',side_effect=[dict(status='RUNNING',ee_task_id='t'),None]), patch.object(p.persistence,'write_status',return_value={}), patch.object(p.work_items,'lookup_task_status',return_value=dict(state='COMPLETED')), patch.object(ee.batch.Export.table,'toCloudStorage') as export:
            result=p.advance('check',dict(fire_id='f',batch_id='b'),{},pd.DataFrame(dict(patch_id=['p'])),object())
        export.assert_not_called()
        self.assertEqual(result['action'],'FAILED')

    def test_compare_recovery_and_duplicate_rejection(self):
        old=pd.DataFrame(dict(patch_id=['p','q'],fire_id=['f','f'],raster_full_area_m2=[8100.,8100.],support_sum=[8100.,8100.],support_area=[8100.,8100.]))
        for name in p.NAMES:
            old[name+'_sum']=[None,8100.]
            old[name+'_area']=[None,8100.]
        new=old.copy()
        for name in p.NAMES:
            new.loc[0,name+'_sum']=8100.
            new.loc[0,name+'_area']=8100.
        r=p.compare_tables(old,new,dict(fire_id='f'),['p','q'])
        self.assertEqual(r['variables']['elevation']['recovered'],1)
        self.assertEqual(r['variables']['elevation']['lost'],0)
        with self.assertRaises(ValueError):
            p.compare_tables(old,pd.concat([new,new]),dict(fire_id='f'),['p','q'])

class GraphTests(ApiTestCase):
    def test_existing_terrain_graph_only(self):
        m,items=p.pilot_inputs()
        i=items[0]
        patches=p.prepare_patches(m,i)
        fire=m.set_index('fire_id',drop=False).loc[i['fire_id']].to_dict()
        with patch.object(ee.data,'computeValue',side_effect=AssertionError('Network')), patch.object(p.pipeline,'build_raster_products',side_effect=AssertionError('All sources')):
            product=p.terrain_product(fire,patches)
            self.assertEqual(product['columns'],p.population.raw_columns('terrain',None))
            graph=product['collection'].serialize()
            self.assertIn('Image.reduceRegions',graph)
            self.assertIn('Image.unmask',graph)
            self.assertNotIn('LANDSAT',graph)
            self.assertNotIn('DAYMET',graph)

if __name__=='__main__':
    unittest.main()
