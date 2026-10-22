"""Offline assembly contract checks; no cloud calls."""
import unittest
import json
import re
import sqlite3
import pandas as pd
import manual_prepare_bigquery as a
from wildfire_recovery_forecasting import reductions

class AssemblyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest,cls.items=a.population.load_population()
        cls.sql,cls.roles=a.build_sql(cls.manifest,cls.items)

    def test_exact_inventory_and_reuse(self):
        defs=a.definitions(self.items)
        uris=[u for d in defs.values() for u in d['sourceUris']]
        self.assertEqual((len(defs),len(uris),len(set(uris))),(13,2101,2101))
        old=[u for u in uris if '/phase3_ca175_v1/' in u]
        self.assertEqual(len(old),12)
        self.assertTrue(all('CA4118512343320200728/batch_r215_c-210' in u for u in old))
        self.assertFalse(any('*' in u for u in uris))

    def test_arithmetic_equivalence(self):
        db=sqlite3.connect(':memory:')
        db.create_function('IF',3,lambda c,a,b:a if c else b)
        db.create_function('SAFE_DIVIDE',2,lambda a,b:None if a is None or b in (None,0) else a/b)
        for support in (False,True):
            for fraction in (False,True):
                for total,area,sup in ((40,20,25),(0,20,25),(None,None,None),(0,0,0)):
                    raw=pd.DataFrame([dict(patch_id='p',fire_id='f',raster_full_area_m2=8102.43,x_sum=total,x_area=area,support_sum=sup)])
                    expected=reductions.summarize_area_sums(raw,['x'],support='support' if support else None,fractions=['x'] if fraction else [])
                    expr=a.area_fields('r','x',support=support,fraction=fraction)
                    db.execute('DROP TABLE IF EXISTS r')
                    db.execute('CREATE TABLE r (raster_full_area_m2 REAL,x_sum REAL,x_area REAL,support_sum REAL)')
                    db.execute('INSERT INTO r VALUES (?,?,?,?)',(8102.43,total,area,sup))
                    values=db.execute('SELECT '+','.join(expr.values())+' FROM r').fetchone()
                    for name,value in zip(expr,values):
                        if pd.isna(expected[name].iloc[0]):self.assertIsNone(value)
                        else:self.assertAlmostEqual(value,expected[name].iloc[0])

    def test_evt_and_leakage_contract(self):
        actual='[{class=3028, sum=29.272803213082106}, {class=3138, sum=0.0}]'
        groups=json.loads(re.sub(r'([A-Za-z_]+)=',lambda m:'"'+m[1]+'":',actual))
        self.assertEqual(reductions.dominant_category(groups),(3028,29.272803213082106))
        sql=self.sql['assemble.sql']
        self.assertNotIn(chr(1),sql)
        self.assertIn('ORDER BY area DESC, code ASC',sql)
        self.assertIn('nbr_pre_mean - nbr_1_mean AS nbr_drop_patch',sql)
        self.assertIn('v.reburn_fraction AS reburn_fraction',sql)
        self.assertNotIn('not_assembled',sql)
        self.assertIn('10 AS lcms_pre_10yr_expected_years',sql)
        self.assertIn('6 AS lcms_post_y_to_y5_expected_years',sql)
        for n in self.roles['predictors']+self.roles['early_predictors']:
            self.assertFalse(n.startswith(('lcms_','nbr_5','season5_','reburn_')))
        self.assertNotIn('nbr_drop_patch',self.roles['early_predictors'])
        self.assertEqual(self.roles['target'],'nbr_5_mean')
        self.assertEqual(len(a.CLIMATE),16)
        for check in ('joined_rows','evt_parse','chronological_split'):
            self.assertIn(check,self.sql['qa.sql'])

    def test_vector_join_and_checks(self):
        self.assertEqual(a.definitions(self.items)['raw_vector_qa']['sourceUris'],[a.VECTOR_URI])
        self.assertIn('v.patch_id=p0.patch_id AND v.fire_id=p0.fire_id',self.sql['assemble.sql'])
        for name in a.VECTOR_STATUSES:
            self.assertIn(name,self.roles['qa'])
            self.assertNotIn(name,self.roles['predictors'])
        for name in ('reburn_fraction_required_bounds','fire_overlap_positive'):
            self.assertIn(name,self.sql['qa.sql'])
        self.assertIn('308048',self.sql['qa.sql'])

    def test_actual_local_supplement_and_rejections(self):
        table=pd.read_parquet(a.VECTOR_FILE)
        self.assertEqual(len(table),308048)
        a.validate_vector(table,self.items)
        with self.assertRaises(ValueError): a.validate_vector(pd.concat([table,table.iloc[:1]]),self.items)
        with self.assertRaises(ValueError): a.validate_vector(table.iloc[1:],self.items)
        broken=table.copy()
        broken.loc[0,'reburn_fraction']=float('nan')
        with self.assertRaises(ValueError): a.validate_vector(broken,self.items)

    def test_materialization_boundaries(self):
        keys=self.sql['source_keys.sql']
        self.assertIn('CREATE TEMP TABLE source_keys',keys)
        self.assertIn('source_duplicate_keys',keys)
        self.assertIn('COUNT(DISTINCT source)!=13',keys)
        self.assertIn('source_fire_counts',keys)
        for name in a.definitions(self.items):
            self.assertEqual(keys.count('FROM `'+name+'`'),1)
        for name in ('qa.sql','nulls.sql','primary.sql','early.sql','publish.sql'):
            sql=self.sql[name]
            self.assertNotIn('raw_',sql.replace('raw_full_areas',''))
            self.assertNotIn('LEFT JOIN',sql)
            self.assertIn(a.STAGE,sql)
        self.assertEqual(self.sql['qa.sql'].count('FROM `'+a.STAGE+'`'),1)
        self.assertEqual(self.sql['nulls.sql'].count('FROM `'+a.STAGE+'`'),1)
        self.assertIn('COUNT(DISTINCT patch_id)',self.sql['qa.sql'])
        for diagnostic in ('_qa_bad_full_areas','_qa_evt_parse','_qa_evt_members'):
            self.assertIn(diagnostic,self.sql['assemble.sql'])
            self.assertIn(diagnostic,self.sql['qa.sql'])
            self.assertIn(diagnostic,self.sql['publish.sql'])
        self.assertNotIn('SELECT DISTINCT',self.sql['assemble.sql'])

    def test_single_scan_aggregates_preserve_issue_counts(self):
        checks=[('nulls','COUNTIF(x IS NULL)'),('rows','ABS(COUNT(*)-2)')]
        sql=a.aggregate_checks(checks,'stage')
        self.assertEqual(sql.count('FROM stage'),1)
        self.assertIn('COUNTIF(x IS NULL) AS c0',sql)
        self.assertIn("STRUCT('nulls' AS check_name,c0 AS issues)",sql)

    def test_high_precision_evt_rounding_regression(self):
        actual='[{"class":3014,"sum":2516.2123965992646},{"class":3129,"sum":742.6311645507812}]'
        groups=json.loads(actual)
        code, area=reductions.dominant_category(groups)
        self.assertEqual(code,3014)
        self.assertEqual(area,2516.2123965992646)
        self.assertAlmostEqual(area/sum(g['sum'] for g in groups),0.7721181914333112,places=6)
        sql=self.sql['assemble.sql']
        self.assertIn("wide_number_mode => 'round') AS decoded",sql)
        self.assertEqual(sql.count('SAFE.PARSE_JSON('),1)
        self.assertIn('evt.winner.code AS dominant_evt',sql)
        self.assertIn('SAFE_DIVIDE(evt.winner.area,IF(p6.support_sum>0,p6.support_sum,NULL)) AS dominant_evt_fraction',sql)
        self.assertIn('ORDER BY area DESC, code ASC',sql)
        self.assertIn("JSON_TYPE(evt.decoded)!='array'",sql)
        self.assertIn("SAFE_CAST(JSON_VALUE(g,'$.class') AS INT64) IS NULL",sql)
        self.assertIn('SUM(_qa_evt_parse)',self.sql['qa.sql'])
        self.assertIn('SUM(_qa_evt_members)',self.sql['qa.sql'])

    def test_frozen_eligibility_predicates_and_boundaries(self):
        from pathlib import Path
        sql=(Path(a.__file__).parent/'sql/phase3_ca175_tile1_v1_eligibility.sql').read_text()
        self.assertIn("SELECT * REPLACE ('eligible_v1' AS eligibility_status)",sql)
        self.assertNotIn('CREATE OR REPLACE',sql)
        self.assertNotIn('CASE WHEN',sql)
        expected={
            'forest_fraction':('>=',.50), 'fire_overlap_fraction':('>=',.50),
            'mtbs_meaningful_burn_fraction':('>=',.25),
            'nbr_pre_valid_fraction':('>=',.80),'nbr_1_valid_fraction':('>=',.80),
            'nbr_5_valid_fraction':('>=',.80),'prefire_selected_year_count_mean':('>=',1),
            'season1_obs_count_mean':('>=',1),'season5_obs_count_mean':('>=',1),
            'prior_fire_fraction_10yr':('=',0),'reburn_fraction':('=',0),
            'lcms_pre_10yr_clear_disturbance_fraction':('<',.05),
            'lcms_post_y_to_y5_clear_disturbance_fraction':('<',.05),
            'lcms_pre_10yr_complete_coverage_fraction':('>=',.80),
            'lcms_post_y_to_y5_complete_coverage_fraction':('>=',.80)}
        where=sql.split('WHERE ',1)[1].rstrip().removesuffix(';')
        parsed={n:(op,float(v)) for n,op,v in re.findall(r'(\w+)\s*(>=|<|=)\s*([0-9.]+)',where)}
        self.assertEqual(parsed,expected)
        db=sqlite3.connect(':memory:')
        db.execute('CREATE TABLE candidates ('+','.join(n+' REAL' for n in expected)+')')
        baseline={n:(v-.01 if op=='<' else v) for n,(op,v) in expected.items()}
        def accepted(values):
            db.execute('DELETE FROM candidates')
            db.execute('INSERT INTO candidates VALUES ('+','.join('?' for _ in expected)+')',list(values.values()))
            return db.execute('SELECT COUNT(*) FROM candidates WHERE '+where).fetchone()[0]
        self.assertEqual(accepted(baseline),1)
        for name,(op,value) in expected.items():
            with self.subTest(field=name):
                self.assertEqual(accepted({**baseline,name:None}),0)
                failing=value if op=='<' else value-.01 if op=='>=' else .00001
                self.assertEqual(accepted({**baseline,name:failing}),0)
