"""Two-fire production pilot QA only: four GCS reads and bounded EE point queries."""
import json
import numpy as np
import pandas as pd
import ee
import httplib2
from google.cloud import storage

import manual_phase6_production as launch
import manual_phase6_inspect as local
import manual_phase6_verify_pixels as pixels
from wildfire_recovery_forecasting import imagery, landsat

ROOT=launch.STATE.parent/'phase6_nbr_context_v1_pilot_qa'


def run(first_wave=False):
    root=launch.STATE.parent/'phase6_nbr_context_v1_wave1_qa' if first_wave else ROOT
    root.mkdir(exist_ok=False)
    if first_wave:
        all_entries=launch.load_entries()
        state=launch.population_state(launch.POPULATION_STATE,all_entries)
        wave=launch.read_waves(launch.POPULATION_STATE,state)['waves'][0]
        expected=[launch.description(fid,p) for fid in launch.ORIGINAL_WAVE_FIRES for p in ('nbrpre','season1')]
        if wave['product_ids']!=expected: raise ValueError('Original wave identity mismatch')
        entries=[next(e for e in all_entries if e['fire_id']==fid) for fid in launch.ORIGINAL_WAVE_FIRES]
    else:
        entries=launch.load_pilot();state=launch.read_state(launch.STATE,entries)
    manifest,_=launch.poc.population.load_population()
    report=dict(rasters={},chips={},source_pixels={},tolerance=pixels.TOLERANCE,
                note='No filtering; sampled export consistency, not independent scientific algorithm validation.')
    coverage_tables=[]
    client=storage.Client(project=launch.config.GCP_PROJECT)
    ee.Initialize(project=launch.config.GCP_PROJECT,http_transport=httplib2.Http(timeout=90))
    ee.data.setDeadline(90000);ee.data.setMaxRetries(0)
    try:
        # Verify the entire requested set before downloading or evaluating composites.
        keys=[launch.description(e['fire_id'],p) for e in entries for p in e['products']]
        ids=[state['products'][k]['task_id'] for k in keys]
        if not all(ids) or len(set(ids))!=len(ids): raise ValueError('Missing/duplicate task IDs')
        statuses=ee.data.getTaskStatus(ids)
        if len(statuses)!=len(ids) or {t['id'] for t in statuses}!=set(ids) or any(t['state']!='COMPLETED' for t in statuses):
            raise ValueError('Incomplete/failed/ambiguous remote task')
        tasks=launch.inventory(ee)
        report['remote_tasks']=statuses
        verified_objects={}
        for entry in entries:
            for product,spec in entry['products'].items():
                key=launch.description(entry['fire_id'],product);saved=state['products'][key]
                if saved['destination']!=spec['effective_uri']: raise ValueError('Destination identity mismatch')
                matches=launch.matches(tasks,key,saved)
                if len(matches)>1 or (matches and (matches[0]['id']!=saved['task_id'] or matches[0]['description']!=key)):
                    raise ValueError('Conflicting task inventory')
                objects=list(client.list_blobs(spec['bucket'],prefix=spec['fileNamePrefix'],max_results=3,retry=None,timeout=30))
                if (len(objects)!=1 or f'gs://{spec["bucket"]}/{objects[0].name}'!=spec['effective_uri']
                        or not objects[0].size or (saved.get('generation') and str(objects[0].generation)!=str(saved['generation']))):
                    raise ValueError('Unexpected/missing/sharded/changed output')
                verified_objects[key]=objects[0]
        print(f'Preflight PASS: {len(ids)} unique COMPLETED tasks and exact GCS objects',flush=True)
        for entry in entries:
            fid=entry['fire_id']
            fire,patches,rows,record=launch.inputs_for(entry,manifest)
            region=imagery.prepare_region(rows,fire_id=fid,batch_id=entry['batch_id'])
            arrays={}
            for product,spec in entry['products'].items():
                saved=state['products'][launch.description(fid,product)]
                if saved['destination']!=spec['effective_uri'] or not saved['task_id']:
                    raise ValueError('Submission identity mismatch')
                blob=verified_objects[launch.description(fid,product)];path=root/f'{fid}_{product}.tif'
                with path.open('xb') as handle:
                    blob.download_to_file(handle,if_generation_match=int(blob.generation),retry=None,timeout=90)
                result,nbr,valid=local.inspect_raster(path,{**entry,'exports':entry['products']},product)
                result.update(source=spec['effective_uri'],generation=blob.generation,task_id=saved['task_id'])
                report['rasters'][fid+'_'+product]=result
                arrays[product]=np.stack([nbr,valid.astype('float32')])
                if not all(result['checks'].values()) or result['outside_minus1_plus1']:
                    raise ValueError('Raster QA discrepancy; stop for review')
            coverage=local.inspect_chips(rows,region,{p:(a[0],a[1]==1) for p,a in arrays.items()})
            if len(coverage)!=entry['eligible_patch_count']: raise ValueError('Patch count mismatch')
            coverage_tables.append(coverage)
            fields=[p+'_'+part for p in arrays for part in ('full','central')]
            report['chips'][fid]=dict(total=len(coverage),joint_ge80=int(coverage.joint_ge80.sum()),
                coverage={f:dict(minimum=float(coverage[f].min()),p25=float(coverage[f].quantile(.25)),
                median=float(coverage[f].median()),mean=float(coverage[f].mean()),maximum=float(coverage[f].max()),
                ge80=int((coverage[f]>=.8).sum()),zero=int((coverage[f]==0).sum())) for f in fields})
            local.figures(root,fid,{p:(a[0],a[1]==1) for p,a in arrays.items()},coverage,region)
            reference,metadata=launch.verify_reference(fire,patches)
            if metadata!=state['references'][fid]: raise ValueError('Source reference changed')
            report.setdefault('references',{})[fid]=metadata
            samples=pixels.choose_pixels(rows,region,arrays)
            # Include the first observed missing pixel of each channel, if not already sampled.
            for product,array in arrays.items():
                missing=np.argwhere(array[1]==0)
                if len(missing):
                    r,c=map(int,missing[0]);pixel_id=f'r{r}_c{c}'
                    existing=next((p for p in samples if p['pixel_id']==pixel_id),None)
                    if existing: existing['kinds'].append('missing_'+product)
                    else:
                        x,y=region['transform']*(c+.5,r+.5)
                        samples.append(dict(pixel_id=pixel_id,row=r,col=c,x=x,y=y,kinds=['missing_'+product]))
            if len(samples)>14: raise ValueError('Point budget exceeded')
            points=ee.FeatureCollection([ee.Feature(ee.Geometry.Point([p['x'],p['y']],proj='EPSG:5070'),
                                        {'pixel_id':p['pixel_id']}) for p in samples])
            output_projection=ee.Projection('EPSG:5070',transform=list(region['transform'])[:6])
            context=ee.Geometry.Rectangle(list(region['bounds']),proj='EPSG:5070',geodesic=False)
            inputs=landsat.build_fire_inputs(fire,context,reference_projection=reference)
            for product,key in [('nbrpre','nbr_pre'),('season1','nbr_1')]:
                raw=inputs[key]
                source=raw.rename('source_nbr').unmask(-9999,sameFootprint=False).addBands(
                    raw.mask().rename('source_mask').unmask(0,sameFootprint=False))
                print(f'{fid} {product}: querying {len(samples)} points',flush=True)
                result=source.sampleRegions(collection=points,properties=['pixel_id'],projection=output_projection,geometries=False).getInfo()
                checked=pixels.compare(samples,result['features'],arrays[product])
                report['source_pixels'][fid+'_'+product]=checked
                (root/'qa_summary.json').write_text(json.dumps(report,indent=2)+'\n')
                print(json.dumps({k:v for k,v in checked.items() if k!='samples'}),flush=True)
                if checked['mask_pass']!=len(samples) or checked['value_pass']!=len(samples):
                    raise ValueError('Source mismatch; tolerance unchanged')
        report['verdict']='PASS: local and sampled export-consistency QA'
    except BaseException as exc:
        report['verdict']='STOPPED: '+str(exc)
        raise
    finally:
        if coverage_tables: pd.concat(coverage_tables).to_csv(root/'chip_coverage.csv',index=False)
        (root/'qa_summary.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-qa',action='store_true')
    p.add_argument('--first-wave',action='store_true')
    args=p.parse_args()
    if not args.run_qa: p.error('Explicit --run-qa required for GCS reads and bounded EE queries')
    run(first_wave=args.first_wave)
