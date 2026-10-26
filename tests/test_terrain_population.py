"""Offline terrain population adapter safety checks."""
import unittest
from unittest.mock import patch
import pandas as pd
import manual_terrain_population as recovery

class RecoveryPopulationTests(unittest.TestCase):
    def test_verify_compares_actual_original_and_recovery(self):
        old=pd.DataFrame(dict(patch_id=['p'],fire_id=['f'],raster_full_area_m2=[8100.],support_sum=[8100.],support_area=[8100.]))
        for name in recovery.pilot.NAMES:
            old[name+'_sum']=[None]
            old[name+'_area']=[None]
        new=old.copy()
        for name in recovery.pilot.NAMES:
            new[name+'_sum']=[8100.]
            new[name+'_area']=[8100.]
        item=dict(fire_id='f',batch_id='r1_c1')
        for original,corrected in ((recovery.ORIGINAL_VERSION,recovery.VERSION),
                                   ('phase3_ca175_v1',recovery.VERSION),
                                   (recovery.ORIGINAL_VERSION,recovery.pilot.VERSION)):
            with self.subTest(original=original,corrected=corrected):
                with patch.dict(recovery.ORIGINAL_REUSED_BATCHES,{('f','r1_c1'):original},clear=True), patch.object(recovery.pop,'VERSION',recovery.VERSION), patch.object(recovery.pop,'REUSED_BATCHES',{}), patch.object(recovery.pilot,'read_parquet',side_effect=[old,new]) as read:
                    result,uri=recovery.compare_original(None,item,corrected,['p'])
                self.assertEqual([c.args[2] for c in read.call_args_list],[original,corrected])
                self.assertIn('/'+original+'/',uri)
                for stats in result['variables'].values():
                    self.assertEqual(stats['old_missing'],1)
                    self.assertEqual(stats['new_missing'],0)
                    self.assertEqual(stats['recovered'],1)

    def test_comparison_rejects_same_original_and_corrected_path(self):
        item=dict(fire_id='f',batch_id='r1_c1')
        with patch.dict(recovery.ORIGINAL_REUSED_BATCHES,{},clear=True), patch.object(recovery.pilot,'read_parquet') as read:
            with self.assertRaises(ValueError):
                recovery.compare_original(None,item,recovery.ORIGINAL_VERSION,['p'])
        read.assert_not_called()

    def test_frozen_counts_and_pilot_reuse(self):
        manifest,items,reuse=recovery.inputs()
        self.assertEqual(len(manifest),175)
        self.assertEqual(len(items),175)
        self.assertEqual(sum(i['patch_count'] for i in items),308048)
        self.assertEqual(len(reuse),2)
        paths=[recovery.pilot.paths(i,reuse.get((i['fire_id'],i['batch_id']),recovery.VERSION))['parquet'] for i in items]
        self.assertEqual(len(set(paths)),175)
        self.assertEqual(sum('/'+recovery.VERSION+'/' in p for p in paths),173)
        first=recovery.initial_five(items,reuse)
        self.assertEqual(len(first),5)
        self.assertTrue(all((i['fire_id'],i['batch_id']) not in reuse for i in first))

    def test_preflight_offline_exact_key_inventory(self):
        import contextlib
        import io
        import json
        from pathlib import Path
        manifest,items,reuse=recovery.inputs()
        with patch.object(recovery.ee,'Initialize',side_effect=AssertionError('EE')), patch.object(recovery.storage,'Client',side_effect=AssertionError('GCS')), patch.object(Path,'write_text') as write, contextlib.redirect_stdout(io.StringIO()):
            recovery.preflight(manifest,items,reuse)
        report=json.loads(write.call_args.args[0])
        self.assertEqual(report['new_exports'],173)
        self.assertEqual(len(report['items']),175)
        self.assertEqual(len({i['status_uri'] for i in report['items']}),175)
        self.assertTrue(all(len(i['patch_ids_sha256'])==64 for i in report['items']))

    def test_preparation_rejects_changed_candidate_ids(self):
        payload=object.__new__(recovery.TerrainPayload)
        payload.patches=pd.DataFrame(dict(fire_id=['f'],patch_id=['wrong']))
        payload.expected_ids={'f':['expected']}
        with patch.object(recovery.pop.BatchProducts,'prepare',return_value=['wrong']):
            with self.assertRaises(ValueError):
                payload.prepare(dict(fire_id='f'))

    def test_reuse_cannot_enter_submission_wave(self):
        pop=recovery.pop
        items=[dict(source='terrain',product=None,fire_id=str(n),batch_id='r1_c1',patch_count=1) for n in range(8)]
        reuse={('0','r1_c1'):recovery.pilot.VERSION,('1','r1_c1'):recovery.pilot.VERSION}
        with patch.object(pop,'VERSION',recovery.VERSION),patch.object(pop,'REUSED_BATCHES',reuse):
            wave=pop.proposed_wave(items,{},0,173,4)
            self.assertEqual([i['fire_id'] for i in wave],['2','3','4','5'])
            self.assertEqual(pop.proposed_wave(items,{},4,173,4),[])
            records={recovery.production.status_path(**pop.key(items[2])):dict(status='COMPLETE')}
            self.assertNotIn(items[2],pop.proposed_wave(items,records,0,173,4))

    def test_payload_only_builds_terrain_once(self):
        payload=object.__new__(recovery.TerrainPayload)
        payload.products=None
        payload.fire={'fire_id':'f'}
        payload.patches=pd.DataFrame(dict(patch_id=['p']))
        product=dict(collection='terrain',columns=recovery.pop.raw_columns('terrain',None))
        with patch.object(payload,'prepare',return_value=['p']),patch.object(recovery.pilot,'terrain_product',return_value=product) as build:
            self.assertEqual(payload({})[0],product)
            payload({})
            self.assertEqual(build.call_count,1)
            desc,ids=payload.reconciliation(dict(source='terrain',product=None))
            self.assertEqual(desc['columns'],product['columns'])
            self.assertEqual(ids,['p'])
            self.assertEqual(build.call_count,1)

    def test_reconciliation_has_zero_submission_budget(self):
        pop=recovery.pop
        item=dict(source='terrain',product=None,fire_id='f',batch_id='r1_c1')
        with patch.object(pop,'VERSION',recovery.VERSION),patch.object(pop,'REUSED_BATCHES',{}):
            key=recovery.production.status_path(**pop.key(item))
            records={key:dict(status='RUNNING',ee_task_id='t')}
            with patch.object(pop.work_items,'advance_work_item',return_value=dict(action='COMPLETE')) as advance:
                actions=pop.advance([item],records,[dict(id='t',state='COMPLETED')],object(),None,0,4,4,reconcile=lambda i:(dict(columns=['x']),['p']))
            self.assertEqual(actions[0]['action'],'COMPLETE')
            self.assertIsNone(advance.call_args.args[0])
            self.assertEqual(advance.call_args.kwargs['production_version'],recovery.VERSION)

if __name__=='__main__':
    unittest.main()
