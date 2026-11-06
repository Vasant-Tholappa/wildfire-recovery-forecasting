"""One-shot allowlisted imagery POC. Default is entirely local dry-run; no retries."""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path

import pandas as pd
import manual_phase6_imagery as poc
from wildfire_recovery_forecasting import config

FIRE = 'CA3839112224120171009'
BATCH = 'r187_c-209'
PRODUCTS = ('nbrpre', 'season1')
STATE = poc.population.DATA / 'phase6_nbr_context_poc_v1_2017_submission.json'


APPROVED = {'2017': (FIRE, BATCH, 'train', 104, 84, 204),
            '2018': ('CA3839611945120180904', 'r180_c-186', 'validation', 215, 162, 138)}
ACTIVE = {'READY', 'RUNNING', 'CANCEL_REQUESTED'}
KNOWN = ACTIVE | {'COMPLETED', 'FAILED', 'CANCELLED'}


def load_inputs(year='2017'):
    fire_id, batch_id, split, count, width, height = APPROVED[year]
    saved = json.loads(poc.OUTPUT.read_text())
    if saved['version'] != poc.VERSION or saved['project'] != config.GCP_PROJECT:
        raise ValueError('Wrong POC version/project')
    record, = [r for r in saved['candidates'] if r['fire_id'] == fire_id]
    if (record['batch_id'] != batch_id or record['fire_year'] != int(year)
            or record['split'] != split or (record['width'], record['height']) != (width,height)
            or record['eligible_patch_count'] != count or set(record['exports']) != set(PRODUCTS)):
        raise ValueError('Only an approved two-export POC is allowed')
    manifest, _ = poc.population.load_population()
    fire = manifest.set_index('fire_id',drop=False).loc[fire_id].to_dict()
    perimeter = manifest.to_crs(config.GRID_CRS).set_index('fire_id').loc[fire_id].geometry
    patches = poc.population.batch_patches(perimeter, fire_id, batch_id)
    rows = pd.read_parquet(poc.ELIGIBLE, columns=poc.KEY_COLUMNS, filters=[('fire_id','==',fire_id)])
    region = poc.validate_candidate(fire, patches, rows, record['original_batch_patch_count'])
    if (poc.key_digest(patches.patch_id) != record['original_batch_keys_sha256']
            or poc.key_digest(rows.patch_id) != record['eligible_keys_sha256']):
        raise ValueError('Frozen POC keys changed')
    specs = poc.imagery.prepare_export_specs(region)
    for product in PRODUCTS:
        specs[product].update(bucket=config.GCS_BUCKET,
            fileNamePrefix=f'imagery/{poc.VERSION}/{fire_id}/{batch_id}/{product}')
    if specs != record['exports']:
        raise ValueError('Saved export configuration disagrees with reconstructed geometry')
    return fire, patches, rows, record


def reject_duplicates(tasks, client, record):
    descriptions = {f"{poc.VERSION}_{record['fire_id']}_{p}" for p in PRODUCTS}
    prefixes = [v['fileNamePrefix'] for v in record['exports'].values()]
    for task in tasks:
        if not task.get('description'):
            raise RuntimeError('Task inventory has ambiguous identity; stop for review')
        if (task['description'] in descriptions or any(
                prefix in uri for prefix in prefixes for uri in task.get('destination_uris', []))):
            raise RuntimeError('Matching EE task exists; no automatic retry, regardless of state')
    for spec in record['exports'].values():
        if next(iter(client.list_blobs(spec['bucket'], prefix=spec['fileNamePrefix'], max_results=1)), None) is not None:
            raise RuntimeError('Destination objects already exist; no submission')


def save(path, state):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(state, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


EXPECTED_REFERENCES = {'2017': ('LC08_044033_20180601', 'EPSG:32610'),
                       '2018': ('LC08_042033_20190606', 'EPSG:32611')}


def verify_reference(year, fire, patches):
    """One selected-band metadata request; never evaluates composite pixels."""
    import ee
    info = poc.reference_band(fire, patches).getInfo()
    scene, crs = EXPECTED_REFERENCES[year]
    if not isinstance(info, dict) or info.get('id') != f'LANDSAT/LC08/C02/T1_L2/{scene}':
        raise ValueError('Missing or mismatched reference scene')
    bands = info.get('bands', [])
    if len(bands) != 1 or bands[0].get('id') != 'SR_B5' or bands[0].get('crs') != crs:
        raise ValueError('Missing or mismatched reference projection')
    transform = bands[0].get('crs_transform')
    if (not isinstance(transform, list) or len(transform) != 6
            or any(isinstance(v, bool) or not isinstance(v, (int, float))
                   or not math.isfinite(v) for v in transform)):
        raise ValueError('Invalid native transform')
    a,b,c,d,e,f = transform
    if not (math.isclose(a,30,abs_tol=1e-9) and math.isclose(e,-30,abs_tol=1e-9)
            and abs(b)<1e-9 and abs(d)<1e-9):
        raise ValueError('Reference is not the expected native north-up 30 m grid')
    metadata = dict(scene_id=info['id'], crs=bands[0]['crs'], transform=transform)
    # Pin the verified server-returned grid, not a hardcoded scientific projection
    # or a second lazy scene selection that could differ from the checked scene.
    return ee.Projection(metadata['crs'], transform=transform), metadata


def record_path(year, directory):
    return directory / f'{poc.VERSION}_{year}_submission.json'


def years_for(selection):
    if selection not in ('2017', '2018', 'both'):
        raise ValueError('Select --fires 2017, 2018 or both')
    return ('2017', '2018') if selection == 'both' else (selection,)


def summary(states, intended):
    products = [p for s in states.values() for p in s['products'].values()]
    successes = sum(p['status']=='SUBMITTED' for p in products)
    failed = sum(p['status'] in ('AMBIGUOUS','FAILED') for p in products)
    print(f'Summary: {successes}/{intended} submitted; {failed} failed/ambiguous; '
          f'{intended-successes-failed} not submitted.', flush=True)
    print('Recorded IDs: '+str([p['task_id'] for p in products if p.get('task_id')]), flush=True)
    print('Safe to close terminal after this command exits; submitted EE tasks run remotely. '
          'No polling/background process. Review any stopped/ambiguous record before further action.', flush=True)


def run(*, selection=None, submit=False, directory=STATE.parent):
    if submit and selection is None:
        raise ValueError('Live submission requires explicit --fires selection')
    years = years_for(selection or 'both')
    inputs = {y: load_inputs(y) for y in years}
    states = {}
    for year, (_, _, _, record) in inputs.items():
        states[year] = dict(version=poc.VERSION, fire_id=record['fire_id'], batch_id=record['batch_id'],
            configuration=record, status='DRY_RUN', products={p:dict(status='NOT_SUBMITTED',task_id=None,
            destination=f"gs://{record['exports'][p]['bucket']}/{record['exports'][p]['fileNamePrefix']}") for p in PRODUCTS})
    if not submit:
        for y, state in states.items():
            state['existing_local_record'] = record_path(y,directory).exists()
        return states
    directory.mkdir(parents=True,exist_ok=True)
    lock = directory / f'{poc.VERSION}_submission.lock'
    # One controller across both selections. An abandoned lock requires review.
    with lock.open('x') as handle:
        handle.write(str(os.getpid()))
        handle.flush()
        os.fsync(handle.fileno())
    reserved = []
    try:
        for y in years:
            if record_path(y,directory).exists():
                raise FileExistsError(f'Existing record: {record_path(y,directory)}; review, never retry blindly')
        for y, state in states.items():
            state.update(status='RESERVED',created_at=datetime.now(timezone.utc).isoformat())
            path=record_path(y,directory)
            with path.open('x',encoding='utf-8') as handle:
                json.dump(state,handle,indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            reserved.append(y)
        import ee
        from google.cloud import storage
        ee.Initialize(project=config.GCP_PROJECT)
        ee.data.setMaxRetries(0)
        tasks=ee.data.getTaskList()
        if any(t.get('state') not in KNOWN for t in tasks):
            raise RuntimeError('Unknown task state; stop for review')
        if sum(t['state'] in ACTIVE for t in tasks)+2*len(years)>4:
            raise RuntimeError('Insufficient capacity under four-active-task limit')
        client=storage.Client(project=config.GCP_PROJECT)
        references={}
        for y in APPROVED:
            fire, patches, _, _ = inputs[y] if y in inputs else load_inputs(y)
            references[y] = verify_reference(y, fire, patches)
        for y in years:
            states[y]['verified_references'] = {k:v[1] for k,v in references.items()}
            save(record_path(y,directory),states[y])
        arguments={}
        for y, (fire,patches,rows,record) in inputs.items():
            reject_duplicates(tasks,client,record)
            args=poc.image_export_arguments(fire,patches,rows,frozen_record=record,
                reference_projection=references[y][0])
            if set(args)!=set(PRODUCTS):
                raise ValueError('Exactly two pre-cutoff products required per fire')
            for p in PRODUCTS:
                arg,spec=args[p],record['exports'][p]
                if (any(arg[k]!=spec[k] for k in ('bucket','fileNamePrefix','crs','crsTransform','fileFormat'))
                    or arg['formatOptions']!={'noData':-9999} or 'scale' in arg
                    or arg['maxPixels']!=record['width']*record['height']
                    or arg['description']!=f"{poc.VERSION}_{record['fire_id']}_{p}"):
                    raise ValueError('Adapter configuration changed')
            arguments[y]=args
        for y in years:
            state,path=states[y],record_path(y,directory)
            for product in PRODUCTS:
                entry=state['products'][product]
                task=None
                try:
                    task=ee.batch.Export.image.toCloudStorage(**arguments[y][product])
                    entry.update(status='SUBMISSION_INTENT',task_id=task.id)
                    save(path,state)
                    task.start()
                    if not task.id:
                        raise RuntimeError('Submission returned no ID')
                    entry.update(status='SUBMITTED',task_id=task.id,
                        submitted_at=datetime.now(timezone.utc).isoformat(),initial_state='NOT_QUERIED')
                    save(path,state)
                    print(f"{y} {state['fire_id']} {product}: SUBMITTED; ID={task.id}; "
                          f'initial EE state=not queried; record={path}',flush=True)
                except BaseException:
                    entry.update(status='AMBIGUOUS' if entry['status'] in ('SUBMISSION_INTENT','SUBMITTED') else 'FAILED')
                    if task is not None and task.id:
                        entry['task_id']=task.id
                    save(path,state)
                    print(f"{y} {product}: {entry['status']}; ID={entry['task_id']}; record={path}",flush=True)
                    raise
            state['status']='SUBMITTED'
            save(path,state)
        return states
    except BaseException as error:
        for y in reserved:
            if states[y]['status']!='SUBMITTED':
                states[y].update(status='STOPPED_REVIEW_REQUIRED',error=f'{type(error).__name__}: {error}')
                save(record_path(y,directory),states[y])
        raise
    finally:
        summary(states,2*len(years))
        lock.unlink()


def check_status(selection='both', directory=STATE.parent):
    entries=[]
    for y in years_for(selection):
        path=record_path(y,directory)
        if not path.exists():
            print(f'{y}: no local submission record')
            continue
        state=json.loads(path.read_text())
        if (state['version']!=poc.VERSION or state['fire_id']!=APPROVED[y][0]
                or state['batch_id']!=APPROVED[y][1] or set(state['products'])!=set(PRODUCTS)):
            raise ValueError('Unrecognized submission record')
        for p,entry in state['products'].items():
            if entry.get('task_id'): entries.append((y,p,entry['task_id']))
            else: print(f"{y} {p}: {entry['status']}; no recorded task ID")
    if not entries: return []
    import ee
    ee.Initialize(project=config.GCP_PROJECT)
    ee.data.setMaxRetries(0)
    results=ee.data.getTaskStatus([entry[2] for entry in entries])
    indexed={r['id']:r for r in results}
    for y,p,task_id in entries:
        result=indexed.get(task_id,{})
        print(f"{y} {p}: {result.get('state','UNKNOWN')} ID={task_id} "
              f"error={result.get('error_message','none reported')}")
    return results


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fires',choices=['2017','2018','both'])
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--submit',action='store_true')
    mode.add_argument('--status',action='store_true',help='Read recorded EE states once; no writes or submissions')
    args=parser.parse_args(argv)
    if args.submit and args.fires is None: parser.error('--submit requires --fires')
    if args.status: return check_status(args.fires or 'both')
    result=run(selection=args.fires,submit=args.submit)
    if not args.submit: print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    main()
