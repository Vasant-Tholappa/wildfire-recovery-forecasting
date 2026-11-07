"""Bounded imagery exports. Default offline; pilot commands remain compatible."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import re
import time

import pandas as pd
import manual_phase6_manifest as manifest_code
import manual_phase6_imagery as poc
import manual_phase6_submit as safety
from wildfire_recovery_forecasting import config

VERSION=manifest_code.VERSION
MANIFEST=manifest_code.OUTPUT
MANIFEST_SHA256='5e27951fe1bde2dcc22f60af3df44de62034676cc0421a016e23ce5960100634'
PILOT={'CA4083912297820170808':('r212_c-207',714,219,75),
       'CA3646112190120160722':('r166_c-210',3794,363,372)}
STATE=poc.population.DATA/f'{VERSION}_pilot_submission.json'


def load_entries():
    raw=MANIFEST.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=MANIFEST_SHA256:
        raise ValueError('Frozen production manifest hash mismatch')
    document=json.loads(raw)
    if (document['version']!=VERSION or document['fire_count']!=151 or document['patch_count']!=111257
            or document['export_products']!=302 or document['reused_products']!=4 or document['new_exports']!=298):
        raise ValueError('Frozen population totals changed')
    entries=document['fires']
    products=[p for e in entries for p in e['products'].values()]
    if (len(entries)!=151 or len({e['fire_id'] for e in entries})!=151
            or len(products)!=302 or len({p['effective_uri'] for p in products})!=302
            or sum(e['eligible_patch_count'] for e in entries)!=111257):
        raise ValueError('Frozen identities/counts disagree')
    return entries


def load_pilot():
    entries={e['fire_id']:e for e in load_entries()}
    selected=[]
    for fire_id,expected in PILOT.items():
        e=entries[fire_id]
        if (e['batch_id'],e['eligible_patch_count'],e['width'],e['height'])!=expected or e['split']!='train':
            raise ValueError('Pilot membership/configuration mismatch')
        if set(e['products'])!=set(safety.PRODUCTS): raise ValueError('Wrong product inventory')
        for product,p in e['products'].items():
            prefix=f'imagery/{VERSION}/{fire_id}/{e["batch_id"]}/{product}'
            if (p['action']!='EXPORT' or p['bucket']!=config.GCS_BUCKET or p['fileNamePrefix']!=prefix
                    or p['planned_uri']!=f'gs://{config.GCS_BUCKET}/{prefix}.tif' or p['effective_uri']!=p['planned_uri']):
                raise ValueError('Reused or unexpected destination cannot be submitted')
        selected.append(e)
    return selected


def inputs_for(entry, manifest, *, allow_test_inputs=False):
    fire_id=entry['fire_id']
    fire=manifest.set_index('fire_id',drop=False).loc[fire_id].to_dict()
    perimeter=manifest.to_crs(config.GRID_CRS).set_index('fire_id').loc[fire_id].geometry
    patches=poc.population.batch_patches(perimeter,fire_id,entry['batch_id'])
    rows=pd.read_parquet(poc.ELIGIBLE,columns=poc.KEY_COLUMNS,filters=[('fire_id','==',fire_id)])
    region=poc.validate_candidate(fire,patches,rows,entry['original_batch_patch_count'],allow_test_inputs=allow_test_inputs)
    rebuilt=manifest_code.fire_entry(fire,rows,dict(batch_id=entry['batch_id'],patch_count=len(patches)))
    if rebuilt!=entry: raise ValueError('Pilot geometry, dates, keys or destinations disagree with frozen manifest')
    record={**entry,'original_batch_keys_sha256':poc.key_digest(patches.patch_id)}
    return fire,patches,rows,record


def verify_reference(fire,patches):
    """Resolve the original exact-batch first scene; no composite evaluation."""
    import ee
    info=poc.reference_band(fire,patches).getInfo()
    if not isinstance(info,dict): raise ValueError('Unresolved reference')
    match=re.fullmatch(r'LANDSAT/LC08/C02/T1_L2/LC08_\d{6}_(\d{8})',info.get('id',''))
    bands=info.get('bands',[])
    if not match or not (f"{fire['season1_year']}0601"<=match[1]<f"{fire['season1_year']}1001"):
        raise ValueError('Reference scene outside Season 1 or wrong collection')
    if len(bands)!=1 or bands[0].get('id')!='SR_B5': raise ValueError('Reference band missing')
    band=bands[0];transform=band.get('crs_transform',[])
    if not re.fullmatch(r'EPSG:326(?:0[1-9]|[1-5][0-9]|60)',band.get('crs','')):
        raise ValueError('Unexpected native UTM projection; inspect before proceeding')
    if len(transform)!=6 or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in transform):
        raise ValueError('Invalid reference transform')
    a,b,c,d,e,f=transform
    if not (abs(a-30)<1e-9 and abs(e+30)<1e-9 and abs(b)<1e-9 and abs(d)<1e-9):
        raise ValueError('Expected native north-up 30 m grid')
    meta=dict(scene_id=info['id'],crs=band['crs'],transform=transform)
    return ee.Projection(meta['crs'],transform=transform),meta


def description(fire_id,product):
    return f'{VERSION}_{fire_id}_{product}'


def initial_state(entries):
    return dict(version=VERSION,manifest_sha256=MANIFEST_SHA256,products={
        description(e['fire_id'],p):dict(fire_id=e['fire_id'],batch_id=e['batch_id'],product=p,
            destination=spec['effective_uri'],status='PENDING',task_id=None)
        for e in entries for p,spec in e['products'].items()})


def read_state(path,entries):
    expected=initial_state(entries)
    if not path.exists(): return expected
    saved=json.loads(path.read_text())
    if saved.get('manifest_sha256')!=MANIFEST_SHA256 or saved.get('version')!=VERSION or set(saved['products'])!=set(expected['products']):
        raise ValueError('Local record identity mismatch')
    for key,item in expected['products'].items():
        if any(saved['products'][key][k]!=item[k] for k in ('fire_id','batch_id','product','destination')):
            raise ValueError('Local product identity mismatch')
    ids=[p['task_id'] for p in saved['products'].values() if p.get('task_id')]
    if len(ids)!=len(set(ids)) or any(p['status']=='PENDING' and p.get('task_id') for p in saved['products'].values()):
        raise ValueError('Ambiguous local task IDs')
    return saved


@contextmanager
def locked(path):
    path.parent.mkdir(parents=True,exist_ok=True)
    lock=path.with_suffix('.lock')
    with lock.open('x') as handle:
        handle.write(str(os.getpid()));handle.flush();os.fsync(handle.fileno())
    try: yield
    finally: lock.unlink()


def inventory(ee):
    tasks=ee.data.getTaskList()
    if any(not t.get('description') or t.get('state') not in safety.KNOWN for t in tasks):
        raise RuntimeError('Ambiguous task inventory')
    return tasks


def matches(tasks,key,item):
    return [t for t in tasks if t['description']==key or any(item['destination'][:-4] in u for u in t.get('destination_uris',[]))]


def report(state):
    from collections import Counter
    print('Pilot summary (4 products): '+str(dict(Counter(p['status'] for p in state['products'].values()))),flush=True)
    for key,p in state['products'].items():
        print(f"{key}: {p['status']} ID={p['task_id']} -> {p['destination']}",flush=True)
    print('Command finished without polling. Persisted IDs protect continuation; remote COMPLETED is not raster QA acceptance.',flush=True)


def prepare_arguments(entries,inputs,references, *, allow_test_inputs=False):
    arguments={}
    for entry in entries:
        fid=entry['fire_id'];fire,patches,rows,record=inputs[fid]
        args=poc.image_export_arguments(fire,patches,rows,frozen_record=record,reference_projection=references[fid][0],allow_test_inputs=allow_test_inputs)
        if set(args)!=set(safety.PRODUCTS): raise ValueError('Wrong adapter inventory')
        for product,arg in args.items():
            spec=entry['products'][product];key=description(fid,product)
            # Adapter only constructs images; route to the frozen production destination.
            arg.update(description=key,bucket=spec['bucket'],fileNamePrefix=spec['fileNamePrefix'])
            if (arg['crs']!=spec['crs'] or arg['crsTransform']!=spec['crsTransform'] or 'scale' in arg
                    or arg['formatOptions']!={'noData':-9999} or arg['fileFormat']!='GeoTIFF'
                    or arg['maxPixels']!=entry['width']*entry['height']):
                raise ValueError('Adapter output grid/schema mismatch')
            arguments[key]=arg
    return arguments


def submit_one(ee,key,p,arguments,state,path):
    task=ee.batch.Export.image.toCloudStorage(**arguments)
    p.update(status='INTENT',task_id=task.id)
    safety.save(path,state)
    try:
        task.start()
        if not task.id: raise RuntimeError('No returned task ID')
        p.update(status='SUBMITTED',task_id=task.id,submitted_at=datetime.now(timezone.utc).isoformat())
        safety.save(path,state)
        print(f"SUBMITTED {key} ID={task.id} -> {p['destination']}; record={path}",flush=True)
    except safety.AtomicSaveError:
        # Do not overwrite the fsynced SUBMITTED evidence or restart the task.
        raise
    except BaseException as error:
        p.update(status='AMBIGUOUS',task_id=task.id,error=str(error))
        safety.save(path,state)
        raise


def run(mode='dry-run',path=STATE):
    entries=load_pilot()
    if mode=='dry-run':
        manifest,_=poc.population.load_population()
        for e in entries: inputs_for(e,manifest)
        state=read_state(path,entries);report(state)
        print('Offline pilot: 2 fires, 4508 eligible patches, 4 products. POC reuse products excluded.')
        return state
    if mode not in ('submit','status','reconcile'): raise ValueError('Unsupported mode')
    if mode=='status':
        state=read_state(path,entries)
        ids=[p['task_id'] for p in state['products'].values() if p['task_id']]
        if ids:
            import ee
            ee.Initialize(project=config.GCP_PROJECT);ee.data.setMaxRetries(0)
            records={t['id']:t for t in ee.data.getTaskStatus(ids)}
            for p in state['products'].values():
                if p['task_id']:
                    t=records.get(p['task_id'],{})
                    p.update(status=t.get('state','UNKNOWN'),error=t.get('error_message'))
                    if p['error']: print(p['error'])
        report(state)
        return state  # Never save status-mode changes.
    with locked(path):
        state=read_state(path,entries)
        import ee
        ee.Initialize(project=config.GCP_PROJECT);ee.data.setMaxRetries(0)
        tasks=inventory(ee)
        if mode=='reconcile':
            for key,p in state['products'].items():
                found=matches(tasks,key,p)
                if len(found)>1: raise RuntimeError('Multiple matching tasks; manual review required')
                if found:
                    t=found[0]
                    if t['description']!=key or not t.get('id') or (p['task_id'] and p['task_id']!=t['id']):
                        raise RuntimeError('Remote identity mismatch')
                    if p['status']=='PENDING':
                        raise RuntimeError('Unexpected task without local intent; manual review required')
                    p.update(task_id=t['id'],status=t['state'],error=t.get('error_message'))
                elif p['status']!='PENDING':
                    raise RuntimeError('Recorded/ambiguous submission absent from inventory; do not retry')
            safety.save(path,state);report(state);return state
        pending={k:p for k,p in state['products'].items() if p['status']=='PENDING'}
        for key,p in state['products'].items():
            found=matches(tasks,key,p)
            if key in pending:
                if found: raise RuntimeError('Existing matching task; refuse duplicate')
            elif (p['status'] not in ('SUBMITTED','READY','RUNNING','COMPLETED') or not p['task_id']
                  or len(found)!=1 or found[0].get('id')!=p['task_id'] or found[0]['state'] not in ('READY','RUNNING','COMPLETED')):
                raise RuntimeError('Reconcile or review saved task before continuing')
        if not pending: report(state);return state
        if sum(t['state'] in safety.ACTIVE for t in tasks)+len(pending)>4:
            raise RuntimeError('Four-active-task capacity exceeded')
        from google.cloud import storage
        client=storage.Client(project=config.GCP_PROJECT)
        for p in pending.values():
            prefix=p['destination'].split('/',3)[3][:-4]
            if next(iter(client.list_blobs(config.GCS_BUCKET,prefix=prefix,max_results=1)),None) is not None:
                raise RuntimeError('Destination exists; refuse overwrite')
        manifest,_=poc.population.load_population()
        inputs={e['fire_id']:inputs_for(e,manifest) for e in entries}
        references={fid:verify_reference(values[0],values[1]) for fid,values in inputs.items()}
        state['references']={fid:value[1] for fid,value in references.items()}
        arguments=prepare_arguments(entries,inputs,references)
        safety.save(path,state)
        try:
            for key,p in pending.items():
                submit_one(ee,key,p,arguments[key],state,path)
        finally: report(state)
        return state


POPULATION_STATE=STATE.with_name(f'{VERSION}_population_submission.json')
PILOT_QA=STATE.parent/'phase6_nbr_context_v1_pilot_qa'/'qa_summary.json'


def population_state(path,entries):
    """Import accepted products without changing historical records; no cloud reads."""
    baseline=initial_state(entries)
    pilot=read_state(STATE,load_pilot())
    qa=json.loads(PILOT_QA.read_text())
    for e in entries:
        for product,spec in e['products'].items():
            key=description(e['fire_id'],product);item=baseline['products'][key]
            if spec['action']=='REUSE':
                item.update(status='COMPLETED',task_id=spec['reuse_task_id'],
                            generation=str(spec['reuse_generation']),accepted_qa='original POC')
            elif e['fire_id'] in PILOT:
                old=pilot['products'][key]
                evidence=qa['rasters'][e['fire_id']+'_'+product]
                pixels=qa['source_pixels'][e['fire_id']+'_'+product]
                if (evidence['source']!=item['destination'] or evidence['task_id']!=old['task_id']
                        or not all(evidence['checks'].values()) or pixels['sample_count']<=0
                        or pixels['mask_pass']!=pixels['sample_count'] or pixels['value_pass']!=pixels['sample_count']
                        or pixels['max_absolute_difference']>1e-6):
                    raise ValueError('Pilot acceptance evidence mismatch')
                item.update(old)
                item.update(status='COMPLETED',generation=str(evidence['generation']),accepted_qa='production pilot')
    if not path.exists(): return baseline
    saved=read_state(path,entries)
    for key,item in baseline['products'].items():
        if item.get('accepted_qa'):
            if any(saved['products'][key].get(k)!=item[k] for k in ('task_id','generation','accepted_qa','status')):
                raise ValueError('Accepted product provenance changed')
    return saved


def wave_selection(entries,state,cap,include_test=False):
    """Round-robin years; within each year median raster, largest, then smallest."""
    groups={}
    for e in entries:
        if e['split']=='test' and not include_test: continue
        if any(state['products'][description(e['fire_id'],p)]['status']=='PENDING' for p in safety.PRODUCTS):
            groups.setdefault(e['fire_year'],[]).append(e)
    for year,group in groups.items():
        ordered=sorted(group,key=lambda e:(e['width']*e['height'],e['fire_id']))
        middle=ordered.pop((len(ordered)-1)//2)
        groups[year]=[middle]+( [ordered.pop()] if ordered else [] )+ordered
    ordered=[]
    while any(groups.values()):
        for year in sorted(groups):
            if groups[year]: ordered.append(groups[year].pop(0))
    selected=[];keys=[]
    for e in ordered:
        pending=[description(e['fire_id'],p) for p in safety.PRODUCTS
                 if state['products'][description(e['fire_id'],p)]['status']=='PENDING']
        if len(keys)+len(pending)>cap: break
        selected.append(e);keys.extend(pending)
    return selected,keys


ORIGINAL_WAVE_FIRES=(
    'CA4049212319820150803','CA3622311855820161021','CA3334911719120171207',
    'CA3664611822120180708','CA3703411834520190904','CA3687411890520150801',
    'CA3453012005420160615','CA3442911910020171205')


def wave_path(path):
    return path.with_suffix('.waves.json')


def wave_record(keys,state):
    return dict(product_ids=list(keys),fire_ids=[state['products'][k]['fire_id'] for k in keys],
                manifest_sha256=MANIFEST_SHA256,expected_product_count=len(keys),
                membership={})


def refresh_wave(wave,state):
    keys=wave['product_ids']
    wave['membership']={name:[k for k in keys if predicate(state['products'][k])]
        for name,predicate in dict(completed=lambda p:p['status']=='COMPLETED',
            pending=lambda p:p['status']=='PENDING',
            submitted=lambda p:p['status'] not in ('PENDING','COMPLETED')).items()}
    wave['task_ids']={k:state['products'][k]['task_id'] for k in keys if state['products'][k].get('task_id')}


def read_waves(path,state):
    document=json.loads(wave_path(path).read_text()) if wave_path(path).exists() else dict(
        version=VERSION,manifest_sha256=MANIFEST_SHA256,waves=[])
    if document['version']!=VERSION or document['manifest_sha256']!=MANIFEST_SHA256:
        raise ValueError('Wave manifest identity mismatch')
    seen=set()
    for wave in document['waves']:
        keys=wave['product_ids']
        if (wave['manifest_sha256']!=MANIFEST_SHA256 or len(keys)!=wave['expected_product_count']
                or not keys or len(set(keys))!=len(keys) or seen.intersection(keys)
                or any(k not in state['products'] for k in keys)
                or wave['fire_ids']!=[state['products'][k]['fire_id'] for k in keys]):
            raise ValueError('Invalid persisted wave membership')
        for k,task_id in wave.get('task_ids',{}).items():
            if k not in keys or state['products'][k].get('task_id')!=task_id:
                raise ValueError('Recorded wave task ID changed')
        seen.update(keys)
        refresh_wave(wave,state)
    return document


def planned_wave(entries,state,cap,include_test,path,persist=False):
    document=read_waves(path,state)
    by_fire={e['fire_id']:e for e in entries}
    if not include_test and any(by_fire[state['products'][k]['fire_id']]['split']=='test'
            for w in document['waves'] for k in w['product_ids']
            if state['products'][k]['status']!='COMPLETED'):
        raise ValueError('Resuming 2020 inputs requires --include-test-inputs')
    outstanding=[w for w in document['waves'] if w['membership']['pending']]
    if len(outstanding)>1: raise ValueError('Multiple waves with pending submissions')
    if outstanding:
        wave=outstanding[0]
        by_fire={e['fire_id']:e for e in entries}
        if not include_test and any(by_fire[state['products'][k]['fire_id']]['split']=='test'
                                   for k in wave['product_ids'] if state['products'][k]['status']!='COMPLETED'):
            raise ValueError('Resuming 2020 inputs requires --include-test-inputs')
        keys=wave['membership']['pending'][:cap]
        selected=[e for e in entries if e['fire_id'] in {state['products'][k]['fire_id'] for k in keys}]
    else:
        selected,keys=wave_selection(entries,state,cap,include_test)
        if keys:
            wave=wave_record(keys,state);refresh_wave(wave,state);document['waves'].append(wave)
    if persist: safety.save(wave_path(path),document)
    return selected,keys


def save_wave_progress(path,state):
    if wave_path(path).exists(): safety.save(wave_path(path),read_waves(path,state))


def recover_original_wave(path=POPULATION_STATE):
    """One-time offline recovery; never edit existing submission records."""
    with locked(path),locked(STATE):
        entries=load_entries();state=population_state(path,entries)
        if wave_path(path).exists(): raise FileExistsError('Wave ledger already exists; preserve it')
        by_fire={e['fire_id']:e for e in entries}
        keys=[]
        for fid in ORIGINAL_WAVE_FIRES:
            e=by_fire[fid]
            if e['split']=='test' or set(e['products'])!=set(safety.PRODUCTS):
                raise ValueError('Original wave manifest mismatch')
            for product in ('nbrpre','season1'):
                if e['products'][product]['action']!='EXPORT': raise ValueError('Unexpected reused product')
                keys.append(description(fid,product))
        completed=keys[:11];pending=keys[11:]
        ids=[state['products'][k].get('task_id') for k in completed]
        if len(set(ids))!=11 or not all(ids): raise ValueError('Expected 11 unique recorded task IDs')
        if (any(state['products'][k]['status']!='COMPLETED' for k in completed)
                or any(state['products'][k]['status']!='PENDING' or state['products'][k].get('task_id') for k in pending)
                or any(p.get('issue') for p in state['products'].values())
                or sum(p['status']=='COMPLETED' for p in state['products'].values())!=19
                or sum(p['status']=='PENDING' for p in state['products'].values())!=283
                or any(p.get('task_id') and not p.get('accepted_qa') and k not in completed
                       for k,p in state['products'].items())):
            raise ValueError('Original-wave recovery disagrees with existing records')
        wave=wave_record(keys,state);refresh_wave(wave,state)
        wave['recovered_from_submission_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        safety.save(wave_path(path),dict(version=VERSION,manifest_sha256=MANIFEST_SHA256,waves=[wave]))
        return wave


def population_report(state,new=0):
    from collections import Counter
    c=Counter(p['status'] for p in state['products'].values())
    result=dict(total=len(state['products']),completed=c['COMPLETED'],running=c['RUNNING'],
                queued=c['READY']+c['SUBMITTED'],failed=c['FAILED']+c['CANCELLED'],pending=c['PENDING'],
                remaining=len(state['products'])-c['COMPLETED'],newly_submitted=new,
                issues=sum(bool(p.get('issue')) for p in state['products'].values()),
                synchronizing=sum(bool(p.get('output_sync')) for p in state['products'].values()))
    print(json.dumps(result),flush=True)
    for key,item in state['products'].items():
        if item.get('issue'): print(f"STOP {key}: {item['issue']}",flush=True)
    return result


def reconcile_population(ee,client,state,sync_grace_seconds=600):
    """Read task states and object metadata only. Caller decides whether to save."""
    tasks=inventory(ee)
    ids=[p['task_id'] for p in state['products'].values() if p.get('task_id') and not p.get('accepted_qa')]
    remote=ee.data.getTaskStatus(ids) if ids else []
    if len({t.get('id') for t in remote})!=len(remote): raise RuntimeError('Duplicate remote IDs')
    by_id={t.get('id'):t for t in remote}
    objects=[]
    for version in (VERSION,poc.VERSION):
        found=list(client.list_blobs(config.GCS_BUCKET,prefix=f'imagery/{version}/',max_results=1000,retry=None,timeout=30))
        if len(found)>=1000: raise RuntimeError('Object inventory truncated; stop')
        objects.extend(found)
    for key,item in state['products'].items():
        item.pop('issue',None)
        if item.get('unrecorded_conflict'):
            item['issue']='Unexpected task/object without local intent; manual investigation required'
            continue
        found=matches(tasks,key,item)
        object_name=item['destination'].split('/',3)[3]
        blobs=[b for b in objects if b.name.startswith(object_name[:-4])]
        exact=len(blobs)==1 and blobs[0].name==object_name and int(blobs[0].size or 0)>0
        if len(found)>1 or (found and item.get('task_id') and found[0].get('id')!=item['task_id']):
            item['issue']='Conflicting EE tasks';continue
        if item.get('accepted_qa'):
            if not exact or str(blobs[0].generation)!=item['generation']:
                item['issue']='Accepted object missing, changed or conflicting'
            elif found and found[0]['state']!='COMPLETED': item['issue']='Accepted task no longer COMPLETED'
            continue
        if found and found[0]['description']!=key:
            item['issue']='Task destination matches but description differs';continue
        if item['status']=='PENDING':
            if found or blobs:
                item.update(status='AMBIGUOUS',unrecorded_conflict=True,
                            issue='Unexpected task/object without local intent')
            continue
        recorded_id=item.get('task_id')
        task=by_id.get(item.get('task_id')) if item.get('task_id') else (found[0] if found else None)
        if (not task or task.get('state') not in safety.KNOWN or not task.get('id')
                or (found and task['id']!=found[0].get('id'))):
            item['issue']='Unresolved recorded submission; never retry';continue
        if not item.get('task_id') and (not found or found[0]['description']!=key):
            item['issue']='Cannot recover ambiguous identity';continue
        item.update(task_id=task['id'],status=task['state'],error=task.get('error_message'))
        for field in ('creation_timestamp_ms','start_timestamp_ms','update_timestamp_ms','batch_eecu_usage_seconds'):
            if field in task: item[field]=task[field]
        if item['status']=='COMPLETED':
            if not exact: item['issue']='Completed task has missing/conflicting output; reconcile later'
            elif item.get('generation') and item['generation']!=str(blobs[0].generation):
                item['issue']='Completed output generation changed'
            elif item.get('output_sync') and item['output_sync']['generation']!=str(blobs[0].generation):
                item['issue']='Provisional output generation changed'
            else:
                item.update(generation=str(blobs[0].generation),bytes=int(blobs[0].size))
                item.pop('output_sync',None)
        elif item['status'] in ('FAILED','CANCELLED','CANCEL_REQUESTED'):
            item['issue']=item.get('error') or item['status']
        elif blobs or item.get('output_sync'):
            if (not exact or not recorded_id or len(found)!=1 or found[0].get('id')!=recorded_id
                    or item['status'] not in ('READY','RUNNING')):
                item['issue']='Unexpected/conflicting output or unverified active identity'
            else:
                sync=item.setdefault('output_sync',dict(first_seen_epoch=time.time(),
                                                       generation=str(blobs[0].generation)))
                if sync['generation']!=str(blobs[0].generation):
                    item['issue']='Provisional output generation changed'
                elif time.time()-sync['first_seen_epoch']>=sync_grace_seconds:
                    item['issue']='Output/status synchronization grace period exceeded'
                # Otherwise remain READY/RUNNING, occupying a slot; never count as complete.

    # Inventory can contain unrelated work; union avoids double-counting recorded tasks.
    active={t['id'] for t in tasks if t['state'] in safety.ACTIVE}
    active.update(p['task_id'] for p in state['products'].values()
                  if p['status'] in safety.ACTIVE|{'SUBMITTED'} and p.get('task_id'))
    return len(active)


def population_run(mode='dry-run',max_new=16,max_minutes=60,poll_seconds=60,
                   include_test=False,path=POPULATION_STATE,sync_grace_seconds=600):
    if mode not in ('dry-run','submit','status','reconcile'): raise ValueError('Unsupported mode')
    if not 1<=max_new<=302 or not math.isfinite(max_minutes) or not 0<max_minutes<=1440:
        raise ValueError('Require finite wave cap (1..302) and duration (0..1440 minutes)')
    if not math.isfinite(poll_seconds) or not 30<=poll_seconds<=300:
        raise ValueError('Poll interval must be 30..300 seconds')
    if not math.isfinite(sync_grace_seconds) or not 30<=sync_grace_seconds<=3600:
        raise ValueError('Synchronization grace must be 30..3600 seconds')
    entries=load_entries()
    if mode=='dry-run':
        state=population_state(path,entries)
        selected,keys=planned_wave(entries,state,max_new,include_test,path)
        manifest,_=poc.population.load_population()
        for e in selected:
            inputs_for(e,manifest,**({'allow_test_inputs':True} if include_test else {}))
        population_report(state)
        for e in selected:
            print(f"{e['fire_id']} {e['batch_id']} {e['fire_year']} {e['split']} patches={e['eligible_patch_count']} raster={e['width']}x{e['height']}")
        for key in keys: print(f'PENDING {key}')
        print(f'OFFLINE: {len(selected)} fires, {len(keys)} proposed new products; no state writes or cloud calls.')
        return state
    # Status can run beside a controller: atomic snapshots, no writes/lock acquisition.
    from contextlib import nullcontext
    with (nullcontext() if mode=='status' else locked(path)), (nullcontext() if mode=='status' else locked(STATE)):
        state=population_state(path,entries)
        if mode=='submit' and any(p['status'] in ('INTENT','AMBIGUOUS','FAILED','CANCELLED') or p.get('issue')
                                  for p in state['products'].values()):
            raise RuntimeError('Explicit reconciliation/manual review required before submission')
        import ee
        import httplib2
        from google.cloud import storage
        ee.Initialize(project=config.GCP_PROJECT,http_transport=httplib2.Http(timeout=90))
        ee.data.setMaxRetries(0);ee.data.setDeadline(90000)
        client=storage.Client(project=config.GCP_PROJECT)
        deadline=time.monotonic()+max_minutes*60
        active=reconcile_population(ee,client,state,sync_grace_seconds)
        if mode!='status':
            safety.save(path,state)
            save_wave_progress(path,state)
        population_report(state)
        if mode in ('status','reconcile'): return state
        if any(p.get('issue') for p in state['products'].values()): raise RuntimeError('Reconciliation requires intervention')
        selected,keys=planned_wave(entries,state,max_new,include_test,path,persist=True)
        synchronizing=lambda:any(p.get('output_sync') for p in state['products'].values())
        if not keys and not synchronizing(): print('No pending products within requested scope/cap.');return state
        manifest=poc.population.load_population()[0] if keys else None
        new=0;position=0;cache={}
        try:
            while time.monotonic()<deadline:
                if position==len(keys) and new<max_new:
                    selected,keys=planned_wave(entries,state,max_new-new,include_test,path,persist=True)
                    position=0
                if position==len(keys) and not synchronizing(): break
                if position<len(keys) and new<max_new and active<4:
                    key=keys[position];item=state['products'][key]
                    fid=item['fire_id']
                    if fid not in cache:
                        e=next(e for e in selected if e['fire_id']==fid)
                        values=inputs_for(e,manifest,**({'allow_test_inputs':True} if include_test else {}))
                        reference=verify_reference(values[0],values[1])
                        state.setdefault('references',{})[fid]=reference[1]
                        cache[fid]=prepare_arguments([e],{fid:values},{fid:reference},**({'allow_test_inputs':True} if include_test else {}))
                        safety.save(path,state)
                    # Recheck immediately before every start, after potentially slow graph preparation.
                    active=reconcile_population(ee,client,state,sync_grace_seconds)
                    safety.save(path,state)
                    if any(p.get('issue') for p in state['products'].values()): raise RuntimeError('Conflict/failure; stop')
                    if time.monotonic()>=deadline: break
                    if active<4:
                        submit_one(ee,key,item,cache[fid][key],state,path)
                        new+=1;position+=1;active+=1;population_report(state,new)
                        continue
                remaining=deadline-time.monotonic()
                if remaining<=0: break
                time.sleep(min(poll_seconds,remaining))
                active=reconcile_population(ee,client,state,sync_grace_seconds)
                safety.save(path,state)
                population_report(state,new)
                if any(p.get('issue') for p in state['products'].values()): raise RuntimeError('Conflict/failure; stop')
        finally:
            save_wave_progress(path,state)
            population_report(state,new)
            print(f'Session ended: {new}/{max_new} new submissions. Recorded EE tasks continue remotely; safe to close terminal. State: {path}',flush=True)
        return state


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--submit',action='store_true')
    modes.add_argument('--status',action='store_true')
    modes.add_argument('--reconcile',action='store_true',help='Remote reads and local status updates only')
    parser.add_argument('--population',action='store_true',help='Use full frozen population state instead of legacy pilot')
    parser.add_argument('--max-new',type=int,default=16)
    parser.add_argument('--max-minutes',type=float,default=60)
    parser.add_argument('--poll-seconds',type=float,default=60)
    parser.add_argument('--sync-grace-seconds',type=float,default=600)
    parser.add_argument('--include-test-inputs',action='store_true',help='Allow 2020 imagery inputs; never read targets')
    args=parser.parse_args()
    mode='submit' if args.submit else 'status' if args.status else 'reconcile' if args.reconcile else 'dry-run'
    if args.population:
        population_run(mode,args.max_new,args.max_minutes,args.poll_seconds,args.include_test_inputs,
                       sync_grace_seconds=args.sync_grace_seconds)
    else:
        run(mode)
