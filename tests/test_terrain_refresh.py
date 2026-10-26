"""Offline SQL and immutable terrain-refresh checks."""
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import manual_prepare_terrain_refresh as r

class TerrainRefreshTests(unittest.TestCase):
    def test_sql_only_replaces_seven_terrain_fields(self):
        q=r.queries()
        self.assertEqual(set(r.terrain_fields()),set(r.FIELDS))
        self.assertIn('o.* REPLACE(',q['assemble.sql'])
        self.assertIn('USING(fire_id,patch_id)',q['assemble.sql'])
        self.assertIn(r.ORIGINAL,q['assemble.sql'])
        self.assertNotIn('WHERE',q['assemble.sql'])
        self.assertNotIn('OR REPLACE',str(q))
        self.assertIn('nonterrain_unchanged',q['qa.sql'])
        self.assertIn('exact_eligible_keys',q['qa.sql'])
        self.assertIn('support_lost',q['qa.sql'])
        self.assertIn('111257',q['qa.sql'])
        self.assertIn('308048',q['raw_qa.sql'])
        self.assertIn('duplicate_keys',q['raw_qa.sql'])
        self.assertEqual(q['raw_qa.sql'].count(';'),1)
        self.assertEqual(q['qa.sql'].count(';'),1)
        self.assertNotIn('nbr_5_mean',q['summary.sql'])
        self.assertEqual(q['publish.sql'],f'SELECT * FROM `{r.STAGE}`;\n')

    def test_exact_corrected_uri_inventory(self):
        _,items,reuse=r.recovery.inputs()
        uris=[r.recovery.pilot.paths(i,reuse.get((i['fire_id'],i['batch_id']),r.recovery.VERSION))['parquet'] for i in items]
        self.assertEqual(len(set(uris)),175)
        self.assertEqual(sum('/'+r.recovery.pilot.VERSION+'/' in x for x in uris),2)
        self.assertEqual(sum('/'+r.recovery.VERSION+'/' in x for x in uris),173)

    def test_local_integrity_recovery_and_rejections(self):
        n=111257
        old=pd.DataFrame({'fire_id':['synthetic']*n,'patch_id':[str(i) for i in range(n)],'nbr_5_mean':np.zeros(n),'split':['train']*n})
        for c in r.MEANS: old[c]=np.nan
        for c in r.VALID: old[c]=0.
        new=old.copy()
        for c in r.MEANS: new[c]=0.5
        for c in r.VALID: new[c]=1.
        r.validate_local(old,new)
        for field,value in [('nbr_5_mean',1.),('split','test'),('patch_id','duplicate'),('slope_mean',100.)]:
            bad=new.copy()
            bad.loc[0,field]=value
            with self.subTest(field=field),self.assertRaises(ValueError): r.validate_local(old,bad)
        with self.assertRaises(ValueError): r.validate_local(new,old)
        with self.assertRaises(ValueError): r.validate_local(old,new.iloc[:-1])

if __name__=='__main__': unittest.main()
