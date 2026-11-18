"""Render reviewable SQL and temporary external-table definitions; no cloud writes.

prepare is entirely local. dry-run invokes bq with --dry_run (no query execution).
Actual execution/resource creation requires separate operator approval.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import pandas as pd

import manual_population as population
from wildfire_recovery_forecasting import config, daymet, pipeline, production, reductions

OUT = population.DATA / "bigquery_ca175_tile1_v1"
CLIMATE = daymet.HISTORICAL_BANDS + daymet.PRE_S1_BANDS + daymet.PRE_S1_ANOMALY_BANDS + daymet.S1_ANOMALY_BANDS
VECTOR_FILE = population.DATA / (population.VERSION + "_vector_qa") / "supplemental.parquet"
VECTOR_URI = f"gs://{config.GCS_BUCKET}/production/{population.VERSION}/supplemental/vector_qa_v1.parquet"
VECTOR_FRACTIONS = ("fire_overlap_fraction", "prior_fire_fraction_10yr", "reburn_fraction")
VECTOR_STATUSES = ("prior_fire_fraction_10yr_coverage_status", "reburn_fraction_coverage_status")
VECTOR_COLUMNS = ("fire_id", "patch_id", *VECTOR_FRACTIONS, *VECTOR_STATUSES)
STAGE = f"{config.GCP_PROJECT}.{population.VERSION}.patch_candidates_staging_v1"


def aggregate_checks(checks, source):
    """One aggregate scan, then reshape the small result into check rows."""
    expressions = ',\n'.join(f'{expr} AS c{i}' for i, (_, expr) in enumerate(checks))
    rows = ',\n'.join(f"STRUCT('{name}' AS check_name,c{i} AS issues)" for i, (name, _) in enumerate(checks))
    return f'WITH counts AS (SELECT {expressions} FROM {source})\nSELECT result.* FROM counts, UNNEST([{rows}]) result;\n'


def validate_vector(table, items):
    """Local structural gate; SQL additionally checks exact raster-key agreement."""
    from wildfire_recovery_forecasting.validation import validate_patch_table, validate_measurements
    if set(table.columns) != set(VECTOR_COLUMNS):
        raise ValueError("Supplemental vector QA schema mismatch")
    validate_patch_table(table)
    expected = {i["fire_id"]: i["patch_count"] for i in items}
    if table.fire_id.isna().any() or table.groupby("fire_id").size().to_dict() != expected:
        raise ValueError("Supplemental vector QA fire counts mismatch")
    import numpy as np
    if not np.isfinite(table[list(VECTOR_FRACTIONS)].to_numpy(dtype=float)).all():
        raise ValueError("Missing/nonfinite vector QA fractions")
    validate_measurements(table)
    if not table.fire_overlap_fraction.gt(0).all():
        raise ValueError("Fire overlap must be positive")
    if not table[list(VECTOR_STATUSES)].eq("unverified").all().all():
        raise ValueError("Unexpected MTBS coverage status; review completeness evidence")


def table_name(source, product):
    return "raw_" + source + ("_" + product if product else "")


def definitions(items):
    result = {}
    for source, products in population.INVENTORY.items():
        for product in products:
            uris = []
            for item in items:
                if (item["source"], item["product"]) != (source, product):
                    continue
                version = population.REUSED_BATCHES.get((item["fire_id"], item["batch_id"]), population.VERSION)
                path = production.parquet_path(source, item["fire_id"], item["batch_id"], product=product, production_version=version)
                uris.append(f"gs://{config.GCS_BUCKET}/{path}")
            columns = ["patch_id", "fire_id", *population.raw_columns(source, product)]
            result[table_name(source, product)] = {
                "sourceFormat": "PARQUET", "sourceUris": uris,
                "schema": {"fields": [{"name": c, "type": "STRING" if c in ("patch_id", "fire_id", "groups") else "FLOAT64"} for c in columns]},
            }
    result['raw_vector_qa'] = {
        'sourceFormat': 'PARQUET', 'sourceUris': [VECTOR_URI],
        'schema': {'fields': [{'name': c, 'type': 'FLOAT64' if c in VECTOR_FRACTIONS else 'STRING'}
                              for c in VECTOR_COLUMNS]},
    }
    return result


def area_fields(alias, name, *, support=False, fraction=False, output=None, validity=True):
    """SQL translation of summarize_area_sums; zero support never yields a mean."""
    area, total = f"{alias}.{name}_area", f"{alias}.{name}_sum"
    full = f"{alias}.raster_full_area_m2"
    denominator = full if fraction else f"IF({area}>0,{area},NULL)"
    measurement = f"IF({area}>0,SAFE_DIVIDE({total},{denominator}),NULL)"
    output = output or name + ("_fraction" if fraction else "_mean")
    result = {output: measurement}
    if validity:
        valid = f"SAFE_DIVIDE(COALESCE({area},0),{full})"
        if support:
            valid = f"COALESCE(SAFE_DIVIDE(COALESCE({area},0),IF({alias}.support_sum>0,{alias}.support_sum,NULL)),0)"
        result[name + "_valid_fraction"] = valid
    return result


def build_sql(manifest, items):
    metadata = []
    fields = ["fire_id", "fire_name", "ignition_date", "fire_year", "fire_size_acres", "split",
              "last_fully_prefire_growing_season_year", "prefire_search_start_year", "prefire_search_end_year",
              "season1_year", "season5_year", "landfire_condition_year", "landfire_version"]
    batches = {i["fire_id"]: i for i in items}
    for row in manifest.to_dict("records"):
        entry = {k: row[k] for k in fields}
        entry["ignition_date"] = row["ignition_date"].date().isoformat() if hasattr(row["ignition_date"], "date") else row["ignition_date"].isoformat()
        entry.update({k: batches[row["fire_id"]][k] for k in ("batch_id", "patch_count")})
        metadata.append(entry)
    payload = json.dumps(metadata, allow_nan=False)
    if "'''" in payload: raise ValueError("Unsupported metadata literal")
    types = {"ignition_date": "DATE", "fire_size_acres": "FLOAT64", **{k:"INT64" for k in fields if k.endswith("year")}, "patch_count":"INT64"}
    columns = fields + ["batch_id", "patch_count"]
    manifest_sql = "SELECT " + ",\n".join(
        f"CAST(JSON_VALUE(j,'$.{c}') AS {types.get(c, 'STRING')}) AS {c}" for c in columns
    ) + f"\nFROM UNNEST(JSON_QUERY_ARRAY(r'''{payload}''')) AS j"
    ctes = ["manifest AS (" + manifest_sql + ")"]
    aliases = {}
    for index, (source, products) in enumerate(population.INVENTORY.items()):
        for product in products:
            alias = "p" + str(len(aliases))
            aliases[source, product] = alias
            ctes.append(f"{alias} AS (SELECT * FROM `{table_name(source, product)}`)")
    cat = aliases['landfire', 'evt_categories']
    ctes += [fr"""evt_parsed AS (
 SELECT *, SAFE.PARSE_JSON(REGEXP_REPLACE(`groups`, r'([A-Za-z_]+)=', r'"\1":'), wide_number_mode => 'round') AS decoded FROM {cat}
), evt AS (
 SELECT *, ARRAY(SELECT AS STRUCT
   SAFE_CAST(JSON_VALUE(g,'$.class') AS INT64) AS code,
   SAFE_CAST(JSON_VALUE(g,'$.sum') AS FLOAT64) AS area
 FROM UNNEST(JSON_QUERY_ARRAY(decoded)) g
 WHERE SAFE_CAST(JSON_VALUE(g,'$.sum') AS FLOAT64)>0
 ORDER BY area DESC, code ASC LIMIT 1)[SAFE_OFFSET(0)] AS winner
 FROM evt_parsed
)"""]
    out = {}
    for product, names in reductions.LANDSAT_PRODUCTS.items():
        for name in names: out.update(area_fields(aliases['landsat', product], name))
    evc, height, support = [aliases['landfire', p] for p in ('evc', 'height', 'evt_support')]
    out.update(area_fields(evc, 'forest', fraction=True))
    out['evc_valid_fraction'] = out.pop('forest_valid_fraction')
    out.update(area_fields(evc, 'tree_cover', support=True))
    out.update(area_fields(height, 'tree_height', support=True))
    out['dominant_evt'] = 'evt.winner.code'
    out['dominant_evt_fraction'] = f'SAFE_DIVIDE(evt.winner.area,IF({support}.support_sum>0,{support}.support_sum,NULL))'
    out['evt_valid_fraction'] = f'COALESCE(IF({support}.evt_coverage_area>0,SAFE_DIVIDE({support}.evt_coverage_sum,{support}.raster_full_area_m2),NULL),0)'
    for name in ('elevation','slope','northness','eastness'):
        out.update(area_fields(aliases['terrain',None],name,support=True))
    out['aspect_valid_fraction'] = out.pop('northness_valid_fraction')
    del out['eastness_valid_fraction']
    for name in CLIMATE: out.update(area_fields(aliases['daymet','combined'],name,output=name))
    for c in population.raw_columns('daymet','combined'):
        if c.endswith('_sum') and c[:-4] not in CLIMATE:
            out.update(area_fields(aliases['daymet','combined'],c[:-4],validity=False))
    for window, years in (('pre_10yr',10),('post_y_to_y5',6)):
        for name in ('clear_disturbance','other_loss','complete_coverage','valid_years'):
            out.update(area_fields(aliases['lcms',None],f'lcms_{window}_{name}',fraction=name!='valid_years'))
        out[f'lcms_{window}_expected_years'] = str(years)
    severity = aliases['mtbs_severity',None]
    for name in [*reductions.SEVERITY_CLASSES,'valid','meaningful_burn']:
        out.update(area_fields(severity,'mtbs_'+name,fraction=True,validity=False))
    out['mtbs_valid_fraction'] = f"COALESCE({out['mtbs_valid_fraction']},0)"
    out['mtbs_source_valid_fraction'] = f'SAFE_DIVIDE(COALESCE({severity}.mtbs_valid_area,0),{severity}.raster_full_area_m2)'
    base = aliases['landsat','nbrpre']
    joins = [f'FROM {base} JOIN manifest m USING(fire_id)']
    for identity, alias in aliases.items():
        if alias == base: continue
        target = 'evt' if identity == ('landfire','evt_categories') else alias
        joins.append(f'LEFT JOIN {target} ON {target}.patch_id={base}.patch_id AND {target}.fire_id={base}.fire_id')
    joins.append(f'LEFT JOIN `raw_vector_qa` v ON v.patch_id={base}.patch_id AND v.fire_id={base}.fire_id')
    selects = [f'{base}.patch_id', 'm.* EXCEPT(patch_count)',
               fr"CAST(REGEXP_EXTRACT({base}.patch_id,r'_r(-?\d+)_c') AS INT64) AS grid_row",
               fr"CAST(REGEXP_EXTRACT({base}.patch_id,r'_c(-?\d+)$') AS INT64) AS grid_col",
               '8100.0 AS patch_area_m2', 'EXTRACT(MONTH FROM m.ignition_date) AS ignition_month']
    selects += [f'{v} AS {k}' for k,v in out.items()]
    selects += [f'v.{name} AS {name}' for name in (*VECTOR_FRACTIONS, *VECTOR_STATUSES)]
    selects += ["'pending_phase3_5' AS eligibility_status"]
    # Preserve raw-data checks during the one external assembly pass.
    bad_areas = [f'IF({a}.raster_full_area_m2 IS NULL OR {a}.raster_full_area_m2<=0 OR IS_NAN({a}.raster_full_area_m2) OR IS_INF({a}.raster_full_area_m2),1,0)'
                 for identity, a in aliases.items() if identity != ('landfire','evt_categories')]
    selects += [' + '.join(bad_areas)+' AS _qa_bad_full_areas',
                "IF(evt.`groups` IS NOT NULL AND (evt.decoded IS NULL OR JSON_TYPE(evt.decoded)!='array'),1,0) AS _qa_evt_parse",
                "(SELECT COUNTIF(SAFE_CAST(JSON_VALUE(g,'$.class') AS INT64) IS NULL OR SAFE_CAST(JSON_VALUE(g,'$.sum') AS FLOAT64) IS NULL) FROM UNNEST(JSON_QUERY_ARRAY(evt.decoded)) g) AS _qa_evt_members"]
    ctes.append('joined AS (SELECT\n'+',\n'.join(selects)+'\n'+'\n'.join(joins)+')')
    ctes.append('model AS (SELECT *, nbr_pre_mean - nbr_1_mean AS nbr_drop_patch FROM joined)')
    prefix = 'WITH\n'+',\n'.join(ctes)+'\n'
    roles = pipeline.dataset_roles([*columns,'patch_id','grid_row','grid_col','patch_area_m2','ignition_month',*out,
                                    'nbr_drop_patch','fire_overlap_fraction','prior_fire_fraction_10yr','reburn_fraction',
                                    *VECTOR_STATUSES,'eligibility_status'])
    roles['qa'] = sorted(set(roles['qa'])-{'patch_count'})
    assert roles['target'] not in roles['predictors'] + roles['early_predictors']
    assert not any(n.startswith(('lcms_', 'season5_', 'nbr_5', 'reburn_', 'prior_fire_'))
                   for n in roles['predictors'] + roles['early_predictors'])
    assert not any(n.startswith(('nbr_1', 'nbr_drop', 'pre_s1_', 's1_')) for n in roles['early_predictors'])
    checks = []
    def check(name, expr, source):
        checks.append((name, expr, source))
    for identity, alias in aliases.items():
        label = table_name(*identity)
        check(label+'_keys', 'COUNTIF(patch_id IS NULL OR fire_id IS NULL) + COUNT(*)-COUNT(DISTINCT patch_id)',alias)
        check(label+'_batch_counts', 'COUNT(*)', f"(SELECT m.fire_id FROM manifest m LEFT JOIN {alias} r USING(fire_id) GROUP BY m.fire_id,m.patch_count HAVING COUNT(r.patch_id)!=m.patch_count)")
        check(label+'_unexpected_fire', 'COUNTIF(m.fire_id IS NULL)',f'{alias} r LEFT JOIN manifest m USING(fire_id)')
        check(label+'_patch_set', 'COUNT(*)',f"(SELECT patch_id,fire_id FROM {alias} EXCEPT DISTINCT SELECT patch_id,fire_id FROM {base})")
        if identity != ('landfire','evt_categories'):
            check(label+'_full_area', 'COUNTIF(raster_full_area_m2 IS NULL OR raster_full_area_m2<=0 OR IS_NAN(raster_full_area_m2) OR IS_INF(raster_full_area_m2))',alias)
    check('patch_identity_and_tile', r'COUNTIF(NOT REGEXP_CONTAINS(patch_id,CONCAT(r"^",fire_id,r"_r-?\d+_c-?\d+$")) OR CONCAT("r",CAST(CAST(FLOOR(grid_row/120) AS INT64) AS STRING),"_c",CAST(CAST(FLOOR(grid_col/120) AS INT64) AS STRING))!=batch_id)', 'model')
    check('joined_rows', 'ABS(COUNT(*)-308048)+COUNT(*)-COUNT(DISTINCT patch_id)', 'model')
    check('vector_keys', 'COUNTIF(patch_id IS NULL OR fire_id IS NULL) + COUNT(*)-COUNT(DISTINCT patch_id)', '`raw_vector_qa`')
    check('vector_rows', 'ABS(COUNT(*)-308048)', '`raw_vector_qa`')
    for left, right, label in (('`raw_vector_qa`', base, 'extra'), (base, '`raw_vector_qa`', 'missing')):
        check('vector_'+label+'_keys', 'COUNT(*)', f'(SELECT fire_id,patch_id FROM {left} EXCEPT DISTINCT SELECT fire_id,patch_id FROM {right})')
    for name in VECTOR_FRACTIONS:
        check(name+'_required_bounds', f'COUNTIF({name} IS NULL OR IS_NAN({name}) OR IS_INF({name}) OR {name} < -1e-6 OR {name} > 1+1e-6)', 'model')
    check('fire_overlap_positive', 'COUNTIF(fire_overlap_fraction<=0)', 'model')
    for name in VECTOR_STATUSES:
        check(name+'_required', f"COUNTIF({name} IS NULL OR {name}!='unverified')", 'model')
    check('chronological_split', "COUNTIF(split!=CASE WHEN fire_year<=2017 THEN 'train' WHEN fire_year<=2019 THEN 'validation' ELSE 'test' END OR fire_year NOT BETWEEN 2015 AND 2020 OR season1_year!=fire_year+1 OR season5_year!=fire_year+5)",'model')
    check('evt_parse', "COUNTIF(`groups` IS NOT NULL AND (decoded IS NULL OR JSON_TYPE(decoded)!='array'))",'evt_parsed')
    check('evt_members', "COUNTIF(SAFE_CAST(JSON_VALUE(g,'$.class') AS INT64) IS NULL OR SAFE_CAST(JSON_VALUE(g,'$.sum') AS FLOAT64) IS NULL)", 'evt_parsed, UNNEST(JSON_QUERY_ARRAY(decoded)) g')
    for name in out:
        if 'fraction' in name:
            check(name+'_bounds',f'COUNTIF({name} < -1e-6 OR {name} > 1+1e-6 OR IS_NAN({name}) OR IS_INF({name}))','model')
        if name != 'dominant_evt':
            check(name+'_finite',f'COUNTIF(IS_NAN({name}) OR IS_INF({name}))','model')
    nulls = [(n, f'COUNTIF({n} IS NULL)') for n in dict.fromkeys([roles['target'],*roles['predictors'],*VECTOR_FRACTIONS,*VECTOR_STATUSES])]
    stage_checks = [(name, expr) for name, expr, source in checks if source == 'model']
    stage_checks += [('raw_full_areas', 'SUM(_qa_bad_full_areas)'), ('evt_parse', 'SUM(_qa_evt_parse)'),
                     ('evt_members', 'SUM(_qa_evt_members)'),
                     ('joined_null_keys', "COUNTIF(patch_id IS NULL OR fire_id IS NULL OR patch_id='' OR fire_id='')")]
    names = [table_name(*identity) for identity in aliases]+['raw_vector_qa']
    keys = '\nUNION ALL\n'.join(f"SELECT '{name}' AS source,fire_id,patch_id FROM `{name}`" for name in names)
    # Temporary key materialization prevents repeated external reads in this script.
    key_sql = 'CREATE TEMP TABLE source_keys AS\n'+keys+';\n'
    key_sql += 'CREATE TEMP TABLE expected_fires AS '+manifest_sql+';\n'
    key_sql += """SELECT 'source_null_keys' AS check_name, COUNTIF(fire_id IS NULL OR patch_id IS NULL OR fire_id='' OR patch_id='') AS issues FROM source_keys
UNION ALL
SELECT 'source_duplicate_keys', COUNT(*) FROM (SELECT source,fire_id,patch_id FROM source_keys GROUP BY source,fire_id,patch_id HAVING COUNT(*)!=1)
UNION ALL
SELECT 'source_exact_key_sets', COUNT(*) FROM (SELECT fire_id,patch_id FROM source_keys GROUP BY fire_id,patch_id HAVING COUNT(*)!=13 OR COUNT(DISTINCT source)!=13)
UNION ALL
SELECT 'source_unexpected_fire', COUNTIF(m.fire_id IS NULL) FROM source_keys k LEFT JOIN expected_fires m USING(fire_id)
UNION ALL
SELECT 'source_fire_counts', COUNT(*) FROM (
SELECT s.source,m.fire_id,m.patch_count,COUNT(k.patch_id) AS actual
FROM expected_fires m CROSS JOIN UNNEST("""+json.dumps(names)+""") AS source_name
CROSS JOIN UNNEST([STRUCT(source_name AS source)]) s
LEFT JOIN source_keys k ON k.source=s.source AND k.fire_id=m.fire_id
GROUP BY s.source,m.fire_id,m.patch_count HAVING actual!=m.patch_count);
"""
    return {'assemble.sql':prefix+'SELECT * FROM model;\n',
            'source_keys.sql':key_sql,
            'qa.sql':aggregate_checks(stage_checks, f'`{STAGE}`'),
            'nulls.sql':aggregate_checks(nulls, f'`{STAGE}`'),
            'publish.sql':f"SELECT * EXCEPT(_qa_bad_full_areas,_qa_evt_parse,_qa_evt_members) FROM `{STAGE}`;\n",
            'primary.sql':'SELECT patch_id,fire_id,split,'+','.join([*roles['predictors'],roles['target']])+f' FROM `{STAGE}`;\n',
            'early.sql':'SELECT patch_id,fire_id,split,'+','.join([*roles['early_predictors'],roles['target']])+f' FROM `{STAGE}`;\n'}, roles


def prepare():
    manifest, items = population.load_population()
    validate_vector(pd.read_parquet(VECTOR_FILE), items)
    tables = definitions(items)
    queries, roles = build_sql(manifest, items)
    OUT.mkdir(exist_ok=True)
    for name, definition in tables.items(): (OUT/(name+'.json')).write_text(json.dumps(definition,indent=2)+'\n')
    for name, sql in queries.items(): (OUT/name).write_text(sql)
    (OUT/'roles.json').write_text(json.dumps(roles,indent=2)+'\n')
    (OUT/'provenance.json').write_text(json.dumps({'production_version':population.VERSION,
        'inputs_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (population.MANIFEST,population.OCCUPANCY,population.REPORT,VECTOR_FILE)},
        'expected_rows':308048,'products':12,'external_uris':sum(len(d['sourceUris']) for d in tables.values()),
        'reused_uris':12,'vector_uri':VECTOR_URI,'eligibility':'pending Phase 3.5; MTBS history completeness unverified'},indent=2)+'\n')
    return tables


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['prepare','dry-run'])
    parser.add_argument('--query',choices=['source_keys','assemble','qa','nulls','publish','primary','early'],default='assemble')
    args=parser.parse_args()
    tables=prepare()
    print(f'Prepared 12 raster + 1 vector definitions / 2101 exact Parquet URIs in {OUT}',flush=True)
    if args.mode=='dry-run':
        bq=shutil.which('bq.cmd' if os.name=='nt' else 'bq')
        if not bq: raise SystemExit('Google Cloud CLI bq is required; no new Python dependency needed')
        command=[bq,'--project_id='+config.GCP_PROJECT,'--location='+config.GCS_REGION,'query','--use_legacy_sql=false','--dry_run']
        if args.query in ('source_keys','assemble'):
            command += ['--external_table_definition='+n+'::'+str(OUT/(n+'.json')) for n in tables]
        subprocess.run(command,shell=os.name=='nt',input=(OUT/(args.query+'.sql')).read_text(),text=True,check=True)


if __name__=='__main__': main()
