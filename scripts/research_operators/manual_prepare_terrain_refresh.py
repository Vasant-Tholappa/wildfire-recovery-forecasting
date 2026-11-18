"""Local-only terrain refresh SQL preparation and downloaded-Parquet validation."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import numpy as np

import manual_prepare_bigquery as assembly
import manual_terrain_population as recovery
from wildfire_recovery_forecasting import config, validation

DATA = recovery.pop.DATA
OUT = DATA / 'bigquery_terrain_refresh_v2'
BASE = f'{config.GCP_PROJECT}.phase3_ca175_tile1_v1'
RAW = BASE + '.terrain_recovery_raw_v2'
STAGE = BASE + '.patch_model_eligible_terrain_staging_v2'
FINAL = BASE + '.patch_model_eligible_terrain_v2'
ORIGINAL = BASE + '.patch_model_eligible_v1'
LOCAL = DATA / 'patch_model_eligible_terrain_v2.parquet'
ROLES = DATA / 'bigquery_ca175_tile1_v1/roles.json'
MEANS = ('elevation_mean','slope_mean','northness_mean','eastness_mean')
VALID = ('elevation_valid_fraction','slope_valid_fraction','aspect_valid_fraction')
FIELDS = (*MEANS, *VALID)


def terrain_fields():
    result = {}
    for name in recovery.pilot.NAMES:
        result.update(assembly.area_fields('t', name, support=True))
    result['aspect_valid_fraction'] = result.pop('northness_valid_fraction')
    del result['eastness_valid_fraction']
    assert set(result) == set(FIELDS)
    return result


def queries():
    fields = terrain_fields()
    replacements = ',\n'.join(f'{expr} AS {name}' for name,expr in fields.items())
    excluded = ','.join(FIELDS)
    raw_checks = [
        ('rows', 'ABS(COUNT(*)-308048)'),
        ('null_keys', "COUNTIF(fire_id IS NULL OR patch_id IS NULL OR fire_id='' OR patch_id='')"),
        ('duplicate_keys', 'COUNT(*)-COUNT(DISTINCT TO_JSON_STRING(STRUCT(fire_id,patch_id)))'),
        ('full_area', 'COUNTIF(raster_full_area_m2 IS NULL OR raster_full_area_m2<=0 OR IS_NAN(raster_full_area_m2) OR IS_INF(raster_full_area_m2))'),
        ('forest_support', 'COUNTIF(support_sum<0 OR support_sum>raster_full_area_m2+0.01 OR support_sum>COALESCE(support_area,0)+0.01)'),
    ]
    for name in (*recovery.pilot.NAMES,'support'):
        raw_checks += [(name+'_invalid_area',f'COUNTIF({name}_area<0 OR IS_INF({name}_area) OR IS_NAN({name}_area))'),
            (name+'_invalid_sum',f'COUNTIF(IS_INF({name}_sum) OR IS_NAN({name}_sum))')]
    for name in recovery.pilot.NAMES:
        raw_checks.append((name+'_support',f'COUNTIF(({name}_area>0 AND ({name}_sum IS NULL OR COALESCE(support_sum,0)<=0)) OR COALESCE({name}_area,0)>COALESCE(support_sum,0)+0.01)'))
    raw_qa = assembly.aggregate_checks(raw_checks, f'`{RAW}`')
    raw_qa = raw_qa.rstrip().removesuffix(';') + f"""
UNION ALL
SELECT 'exact_candidate_keys' AS check_name, COUNT(*) AS issues
FROM `{RAW}` t FULL OUTER JOIN `{BASE}.patch_candidates_v1` o USING(fire_id,patch_id)
WHERE t.patch_id IS NULL OR o.patch_id IS NULL;
"""
    checks = [('rows','ABS(COUNT(*)-111257)'),
        ('fires','ABS(COUNT(DISTINCT fire_id)-151)'),
        ('duplicate_keys','COUNT(*)-COUNT(DISTINCT TO_JSON_STRING(STRUCT(fire_id,patch_id)))')]
    for split,rows,fires in [('train',61090,78),('validation',19561,36),('test',30606,37)]:
        checks += [(split+'_rows',f"ABS(COUNTIF(split='{split}')-{rows})"),
                   (split+'_fires',f"ABS(COUNT(DISTINCT IF(split='{split}',fire_id,NULL))-{fires})")]
    checks.append(('chronology',"COUNTIF(split IS NULL OR NOT ((fire_year BETWEEN 2015 AND 2017 AND split='train') OR (fire_year BETWEEN 2018 AND 2019 AND split='validation') OR (fire_year=2020 AND split='test')))"))
    for name in FIELDS:
        checks.append((name+'_nonfinite',f'COUNTIF(IS_INF({name}) OR IS_NAN({name}))'))
    for name in VALID:
        checks.append((name+'_range',f'COUNTIF({name} IS NULL OR {name} < -1e-6 OR {name}>1+1e-6)'))
    for name,low,high in [('elevation_mean',-500,5000),('slope_mean',0,90),('northness_mean',-1,1),('eastness_mean',-1,1)]:
        checks.append((name+'_range',f'COUNTIF({name}<{low}-1e-6 OR {name}>{high}+1e-6)'))
    for name,valid in zip(MEANS,('elevation_valid_fraction','slope_valid_fraction','aspect_valid_fraction','aspect_valid_fraction')):
        checks.append((name+'_mask',f'COUNTIF(({valid}=0 AND {name} IS NOT NULL) OR ({valid}>0 AND {name} IS NULL))'))
    qa = assembly.aggregate_checks(checks, f'`{STAGE}`')
    joined = f'`{ORIGINAL}` o FULL OUTER JOIN `{STAGE}` n USING(fire_id,patch_id)'
    equality = f'TO_JSON_STRING((SELECT AS STRUCT o.* EXCEPT({excluded}))) != TO_JSON_STRING((SELECT AS STRUCT n.* EXCEPT({excluded})))'
    comparison_checks = [('exact_eligible_keys','COUNTIF(o.patch_id IS NULL OR n.patch_id IS NULL)'),
                         ('nonterrain_unchanged',f'COUNTIF({equality})')]
    comparison_checks += [(name+'_lost','COUNTIF(o.'+name+' IS NOT NULL AND n.'+name+' IS NULL)') for name in MEANS]
    comparison_checks += [(name+'_support_lost',f'COUNTIF(n.{name}<o.{name}-1e-6)') for name in VALID]
    comparison_qa = assembly.aggregate_checks(comparison_checks, joined)
    qa = ('SELECT * FROM (\n'+qa.rstrip().removesuffix(';')+'\n)\nUNION ALL\n'
          'SELECT * FROM (\n'+comparison_qa.rstrip().removesuffix(';')+'\n);\n')
    # Diagnostics only: target values never appear in output or summaries.
    summaries = []
    for name in MEANS:
        summaries.append(f"SELECT '{name}' AS measurement, COUNTIF(o.{name} IS NULL) AS old_missing, COUNTIF(n.{name} IS NULL) AS new_missing, COUNTIF(o.{name} IS NULL AND n.{name} IS NOT NULL) AS recovered, MIN(n.{name}) AS new_min, MAX(n.{name}) AS new_max FROM {joined}")
    return {
        'load_raw.sql': 'SELECT * FROM `corrected_terrain`;\n',
        'raw_qa.sql': raw_qa,
        'assemble.sql': f'SELECT o.* REPLACE(\n{replacements})\nFROM `{ORIGINAL}` o LEFT JOIN `{RAW}` t USING(fire_id,patch_id);\n',
        'qa.sql': qa,
        'summary.sql': '\nUNION ALL\n'.join(summaries)+';\n',
        'publish.sql': f'SELECT * FROM `{STAGE}`;\n',
    }


def prepare():
    _,items,reuse = recovery.inputs()  # Frozen hashes, no configure()/controller writes.
    definition = assembly.definitions([])['raw_terrain']
    definition['sourceUris'] = [recovery.pilot.paths(i,reuse.get((i['fire_id'],i['batch_id']),recovery.VERSION))['parquet'] for i in items]
    assert len(set(definition['sourceUris'])) == 175
    OUT.mkdir(exist_ok=True)
    (OUT/'corrected_terrain.json').write_text(json.dumps(definition,indent=2))
    for name,sql in queries().items():
        (OUT/name).write_text(sql)
    (OUT/'roles.json').write_bytes(ROLES.read_bytes())
    (OUT/'provenance.json').write_text(json.dumps(dict(original_table=ORIGINAL,
        corrected_table=FINAL, corrected_local=str(LOCAL), changed_columns=FIELDS,
        expected_rows=111257, original_local_sha256=hashlib.sha256((DATA/'patch_model_eligible_v1.parquet').read_bytes()).hexdigest(),
        roles_sha256=hashlib.sha256(ROLES.read_bytes()).hexdigest(),
        corrected_terrain_uris=definition['sourceUris'], status='PREPARED_NOT_EXECUTED'),indent=2))
    print(f'Prepared 175 terrain URIs and SQL in {OUT}; no cloud operations.')


def validate_local(old,new):
    if list(old.columns) != list(new.columns) or len(new)!=111257:
        raise ValueError('Schema/row count changed')
    for frame in (old,new):
        validation.validate_patch_table(frame)
    a=old.set_index(['fire_id','patch_id']).sort_index()
    b=new.set_index(['fire_id','patch_id']).sort_index()
    if not a.index.equals(b.index):
        raise ValueError('Eligible key set changed')
    # Equality only, including target: no test-outcome statistics or display.
    nonterrain=[c for c in a if c not in FIELDS]
    if not a[nonterrain].equals(b[nonterrain]):
        raise ValueError('Nonterrain values or types changed')
    validation.validate_measurements(new[list(FIELDS)])
    if np.isinf(new[list(FIELDS)].to_numpy(dtype=float)).any() or new[list(VALID)].isna().any().any():
        raise ValueError('Invalid terrain numerical values')
    for name,low,high in [('elevation_mean',-500,5000),('slope_mean',0,90),('northness_mean',-1,1),('eastness_mean',-1,1)]:
        if not new[name].dropna().between(low-1e-6,high+1e-6).all():
            raise ValueError('Terrain range requires review')
    for name in MEANS:
        if (a[name].notna() & b[name].isna()).any():
            raise ValueError('Lost terrain measurement')
    for name in VALID:
        if (b[name]<a[name]-1e-6).any():
            raise ValueError('Lost valid terrain support')


def finalize(directory):
    metadata = LOCAL.with_suffix('.metadata.json')
    rolefile = LOCAL.with_suffix('.roles.json')
    if any(p.exists() for p in (LOCAL,metadata,rolefile)):
        raise ValueError('Corrected local output already exists; will not overwrite')
    files=sorted(Path(directory).glob('part-*.parquet'))
    if not files:
        raise ValueError('No downloaded part-*.parquet files')
    provenance=json.loads((OUT/'provenance.json').read_text())
    if hashlib.sha256((DATA/'patch_model_eligible_v1.parquet').read_bytes()).hexdigest()!=provenance['original_local_sha256']:
        raise ValueError('Original local rollback file changed since preparation')
    old=pd.read_parquet(DATA/'patch_model_eligible_v1.parquet')
    new=pd.concat([pd.read_parquet(f) for f in files],ignore_index=True)
    validate_local(old,new)
    new.to_parquet(LOCAL,index=False)
    rolefile.write_bytes((OUT/'roles.json').read_bytes())
    metadata.write_text(json.dumps(dict(source_table=FINAL,rows=len(new),fires=new.fire_id.nunique(),
        splits=new.groupby('split').size().to_dict(), nonterrain_exactly_preserved=True,
        parquet_sha256=hashlib.sha256(LOCAL.read_bytes()).hexdigest(),
        roles=str(rolefile), provenance=str(OUT/'provenance.json')),indent=2))
    print(f'Validated corrected file: {LOCAL}; original preserved. No target statistics printed.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('prepare','finalize'))
    parser.add_argument('--download-dir')
    args=parser.parse_args()
    if args.mode=='prepare': prepare()
    else:
        if not args.download_dir: parser.error('--download-dir required')
        finalize(args.download_dir)
