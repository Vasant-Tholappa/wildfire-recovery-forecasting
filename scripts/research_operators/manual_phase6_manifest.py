"""Offline frozen eligible-population imagery manifest; no EE/GCS operations."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import shapely

import manual_population as population
import manual_phase6_imagery as poc
from wildfire_recovery_forecasting import config, imagery
from wildfire_recovery_forecasting.fire_catalog import derive_fire_fields

VERSION='phase6_nbr_context_v1'
OUTPUT=population.DATA / f'{VERSION}_manifest.json'
EXPECTED={'train':(61090,78),'validation':(19561,36),'test':(30606,37)}
KEY_COLUMNS=list(poc.KEY_COLUMNS)


def validate_population(rows, manifest, batches, expected=EXPECTED):
    if rows[KEY_COLUMNS].isna().any().any() or not rows.patch_id.is_unique or rows.duplicated(['fire_id','patch_id']).any():
        raise ValueError('Missing or duplicate eligible keys')
    actual={s:(len(g),g.fire_id.nunique()) for s,g in rows.groupby('split')}
    if actual!=expected: raise ValueError(f'Frozen population/split counts disagree: {actual}')
    if not manifest.fire_id.is_unique or not batches.fire_id.is_unique:
        raise ValueError('Ambiguous fire/batch association')
    joined=rows.merge(batches[['fire_id','batch_id']],on=['fire_id','batch_id'],how='left',indicator=True)
    if not joined['_merge'].eq('both').all() or not set(rows.fire_id)<=set(manifest.fire_id):
        raise ValueError('Missing frozen association')
    fires=manifest.set_index('fire_id')
    for fire_id,group in rows.groupby('fire_id'):
        fire=fires.loc[fire_id]
        timing=derive_fire_fields(pd.Timestamp(fire.ignition_date))
        for key in ('fire_year','split','prefire_search_start_year','prefire_search_end_year','season1_year'):
            if fire[key]!=timing[key]: raise ValueError('Frozen timing disagrees with ignition')
        if not group.fire_year.eq(timing['fire_year']).all() or not group.split.eq(timing['split']).all():
            raise ValueError('Eligible split/year mismatch')


def fire_entry(fire, rows, batch):
    fire_id=fire['fire_id'];batch_id=batch['batch_id']
    region=imagery.prepare_region(rows,fire_id=fire_id,batch_id=batch_id)
    for r in rows.itertuples():
        win=imagery.chip_window(r.grid_row,r.grid_col,region)
        if (region['transform']*(win.col_off+6,win.row_off+6)!=(90*r.grid_col,90*(r.grid_row+1))
                or region['transform']*(win.col_off+9,win.row_off+9)!=(90*(r.grid_col+1),90*r.grid_row)):
            raise ValueError('Central 3x3 geometry mismatch')
    products={}
    for name,spec in imagery.prepare_export_specs(region).items():
        prefix=f'imagery/{VERSION}/{fire_id}/{batch_id}/{name}'
        products[name]=dict(**spec,product_id=f'{VERSION}/{fire_id}/{batch_id}/{name}',
            bucket=config.GCS_BUCKET,fileNamePrefix=prefix,
            planned_uri=f'gs://{config.GCS_BUCKET}/{prefix}.tif',
            action='EXPORT',effective_uri=f'gs://{config.GCS_BUCKET}/{prefix}.tif',
            uncompressed_bytes=region['width']*region['height']*2*4)
    xmin,ymin,xmax,ymax=region['bounds']
    return dict(fire_id=fire_id,batch_id=batch_id,fire_year=int(fire['fire_year']),split=fire['split'],
        eligible_patch_count=len(rows),eligible_keys_sha256=poc.key_digest(rows.patch_id),
        original_batch_patch_count=int(batch['patch_count']),bounds=list(region['bounds']),
        width=region['width'],height=region['height'],transform=list(region['transform'])[:6],
        eligible_rectangle_occupancy=len(rows)*8100/((xmax-xmin-360)*(ymax-ymin-360)),
        ignition_date=pd.Timestamp(fire['ignition_date']).isoformat(),
        prefire_search_years=[int(fire['prefire_search_start_year']),int(fire['prefire_search_end_year'])],
        season1_year=int(fire['season1_year']),reference_selection=dict(
            collection='LANDSAT/LC08/C02/T1_L2',band='SR_B5',
            start=f"{fire['season1_year']}-06-01",end_exclusive=f"{fire['season1_year']}-10-01",
            order='system:time_start ascending; first',
            geometry='Union of ALL positive-area retained patches in original frozen batch; never eligible-only or buffered bounds',
            reconstruction='manual_population.batch_patches with frozen perimeter in EPSG:5070',
            status='UNRESOLVED_OFFLINE'),products=products)


def reuse(entry, record, state, qa, pixel_qa, root):
    """Reuse only when local identities AND saved validation AND file hashes agree."""
    for key in ('fire_id','batch_id','eligible_patch_count','eligible_keys_sha256','width','height','original_batch_patch_count'):
        if entry[key]!=record[key]: raise ValueError('POC identity mismatch: '+key)
    if state['configuration']!=record or state['fire_id']!=entry['fire_id'] or state['version']!=poc.VERSION:
        raise ValueError('POC submission identity mismatch')
    year=str(entry['fire_year'])
    for product,spec in entry['products'].items():
        old=record['exports'][product]
        for key in ('band','crs','crsTransform','region_epsg5070','nodata','validity_band','sampling'):
            if spec[key]!=old[key]: raise ValueError('POC export semantics mismatch')
        result=qa['rasters'][year+'_'+product]
        pixels=pixel_qa['results'][year+'_'+product]
        path=root/f'{year}_{product}.tif'
        uri=f"gs://{old['bucket']}/{old['fileNamePrefix']}.tif"
        if (not all(result['checks'].values()) or result['source']!=uri
                or result['task_id']!=state['products'][product]['task_id']
                or state['products'][product]['destination']!=uri[:-4]
                or pixels['mask_pass']!=pixels['sample_count'] or pixels['value_pass']!=pixels['sample_count']
                or pixels['max_absolute_difference']>1e-6
                or hashlib.sha256(path.read_bytes()).hexdigest()!=result['sha256']):
            raise ValueError('POC verification/hash mismatch')
        spec.update(action='REUSE',effective_uri=uri,reuse_sha256=result['sha256'],
                    reuse_generation=result['generation'],reuse_task_id=result['task_id'])
    entry['reference_selection']['verified_poc_reference']=state['verified_references'][year]
    entry['reference_selection']['status']='VERIFIED_POC_RECORD'


def generate():
    manifest,_=population.load_population()
    batches=pd.read_parquet(population.OCCUPANCY)
    # Column projection is mandatory: never decode target or other measurement columns.
    rows=pd.read_parquet(poc.ELIGIBLE,columns=KEY_COLUMNS).sort_values(['fire_id','patch_id'])
    validate_population(rows,manifest,batches)
    perimeters=manifest.to_crs(config.GRID_CRS).set_index('fire_id').geometry
    fires=manifest.set_index('fire_id',drop=False);batch_lookup=batches.set_index('fire_id')
    entries=[]
    for fire_id,group in rows.groupby('fire_id',sort=True):
        # Prove membership using canonical square positive-area intersections,
        # without rebuilding millions of irrelevant batch cells.
        boxes=shapely.box(group.grid_col.to_numpy()*90,group.grid_row.to_numpy()*90,
                          (group.grid_col.to_numpy()+1)*90,(group.grid_row.to_numpy()+1)*90)
        if not (shapely.area(shapely.intersection(boxes,perimeters.loc[fire_id]))>0).all():
            raise ValueError('Eligible patch outside canonical retained grid')
        entries.append(fire_entry(fires.loc[fire_id],group,batch_lookup.loc[fire_id]))
    root=population.DATA/'phase6_nbr_context_poc_v1_inspection'
    qa=json.loads((root/'qa_summary.json').read_text())
    pixel_qa=json.loads((root/'source_pixel_verification.json').read_text())
    records=json.loads(poc.OUTPUT.read_text())['candidates']
    for record in records:
        entry=next(e for e in entries if e['fire_id']==record['fire_id'])
        patches=population.batch_patches(perimeters.loc[entry['fire_id']],entry['fire_id'],entry['batch_id'])
        if poc.key_digest(patches.patch_id)!=record['original_batch_keys_sha256']:
            raise ValueError('Original POC batch keys changed')
        state=json.loads((population.DATA/f'{poc.VERSION}_{entry["fire_year"]}_submission.json').read_text())
        reuse(entry,record,state,qa,pixel_qa,root)
    products=[p for e in entries for p in e['products'].values()]
    if len(entries)!=151 or len(products)!=302 or len({p['planned_uri'] for p in products})!=302 or len({p['effective_uri'] for p in products})!=302:
        raise ValueError('Population/export identity mismatch')
    sources=[population.MANIFEST,population.OCCUPANCY,population.REPORT,
             poc.ELIGIBLE.with_suffix('.metadata.json'),poc.ELIGIBLE.with_suffix('.roles.json'),poc.OUTPUT,
             root/'qa_summary.json',root/'source_pixel_verification.json']
    provenance={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    return dict(version=VERSION,project=config.GCP_PROJECT,fire_count=len(entries),patch_count=len(rows),
        eligible_file=poc.ELIGIBLE.name,eligible_keys_sha256=poc.key_digest(rows.patch_id),
        provenance_sha256=provenance,split_counts={s:dict(patches=len(g),fires=g.fire_id.nunique()) for s,g in rows.groupby('split')},
        export_products=len(products),reused_products=sum(p['action']=='REUSE' for p in products),
        new_exports=sum(p['action']=='EXPORT' for p in products),
        uncompressed_bytes=sum(p['uncompressed_bytes'] for p in products),fires=entries)


def audit(result):
    entries=result['fires'];ordered=sorted(entries,key=lambda e:(e['width']*e['height'],e['fire_id']))
    candidates=[e for e in ordered if e['split']!='test' and all(p['action']=='EXPORT' for p in e['products'].values())]
    median=ordered[len(ordered)//2]['width']*ordered[len(ordered)//2]['height']
    representative=min(candidates,key=lambda e:(abs(e['width']*e['height']-median),e['fire_id']))
    large=max(candidates,key=lambda e:(e['width']*e['height'],e['fire_id']))
    return dict(**{k:v for k,v in result.items() if k!='fires'},
        dimensions={k:dict(min=min(e[k] for e in entries),median=float(np.median([e[k] for e in entries])),max=max(e[k] for e in entries)) for k in ('width','height')},
        largest=[{k:e[k] for k in ('fire_id','split','width','height','eligible_patch_count','eligible_rectangle_occupancy')} for e in ordered[-5:][::-1]],
        sparse_below_10_percent=[e['fire_id'] for e in entries if e['eligible_rectangle_occupancy']<.1],
        initial_wave=[{k:e[k] for k in ('fire_id','split','batch_id','width','height','eligible_patch_count')} for e in (representative,large)])


if __name__=='__main__':
    result=generate()
    text=json.dumps(result,indent=2,sort_keys=True)+'\n'
    if OUTPUT.exists() and OUTPUT.read_text()!=text: raise FileExistsError('Existing manifest differs; review before replacing')
    OUTPUT.write_text(text)
    print(json.dumps(audit(result),indent=2))
