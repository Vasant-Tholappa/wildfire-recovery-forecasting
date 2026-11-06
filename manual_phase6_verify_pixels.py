"""Small output-grid point queries only; no exports, GCS or target reads."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import rasterio

import manual_phase6_submit as launch
from manual_phase6_inspect import ROOT
from wildfire_recovery_forecasting import imagery, landsat

TOLERANCE = 1e-6  # Absolute NBR units, fixed before requesting results; rtol=0.
OUTPUT = ROOT / 'source_pixel_verification.json'


def choose_pixels(rows, region, arrays):
    h,w=region['height'],region['width']
    chosen={}
    def add(r,c,kind):
        if 0<=r<h and 0<=c<w:
            chosen.setdefault((int(r),int(c)),[]).append(kind)
    for r,c in [(0,0),(0,w-1),(h-1,0),(h-1,w-1)]: add(r,c,'boundary')
    ordered=rows.sort_values(['grid_row','grid_col'])
    central=np.zeros((h,w),dtype=bool)
    for row in ordered.itertuples():
        win=imagery.chip_window(row.grid_row,row.grid_col,region)
        r,c=int(win.row_off),int(win.col_off)
        central[r+6:r+9,c+6:c+9]=True
    for index in [0,len(ordered)//2,len(ordered)-1]:
        row=ordered.iloc[index]
        win=imagery.chip_window(int(row.grid_row),int(row.grid_col),region)
        add(win.row_off+7,win.col_off+7,'eligible_center')
        # Select context actually outside ALL eligible central patch footprints.
        for r,c in [(int(win.row_off),int(win.col_off)),(int(win.row_off)+14,int(win.col_off)+14)]:
            if not central[r,c]:
                add(r,c,'surrounding_context');break
    for p,array in arrays.items():
        zero=np.argwhere((array[0]==0)&(array[1]==1))
        if len(zero): add(*zero[0],f'observed_zero_{p}')
    return [dict(pixel_id=f'r{r}_c{c}',row=r,col=c,kinds=kinds,
                 x=(region['transform']*(c+.5,r+.5))[0],
                 y=(region['transform']*(c+.5,r+.5))[1]) for (r,c),kinds in sorted(chosen.items())]


def compare(samples, features, array):
    by_id={f['properties']['pixel_id']:f['properties'] for f in features}
    if len(by_id)!=len(features) or set(by_id)!={p['pixel_id'] for p in samples}:
        raise ValueError('Missing/duplicate/unexpected sampled pixels')
    records=[]
    for p in samples:
        source=by_id[p['pixel_id']]
        local=float(array[0,p['row'],p['col']]);local_mask=float(array[1,p['row'],p['col']])
        mask=source.get('source_mask');value=source.get('source_nbr')
        mask_pass=mask in (0,1) and mask==local_mask
        delta=abs(value-local) if mask==1 and value is not None else None
        value_pass=(delta is not None and np.isfinite(delta) and delta<=TOLERANCE) if mask==1 else (mask==0 and local==-9999)
        records.append(dict(**p,local_nbr=local,source_nbr=value,local_valid=local_mask,source_mask=mask,
                            absolute_difference=delta,mask_pass=bool(mask_pass),value_pass=bool(value_pass)))
    differences=[r['absolute_difference'] for r in records if r['absolute_difference'] is not None]
    return dict(sample_count=len(records),mask_pass=sum(r['mask_pass'] for r in records),
        value_pass=sum(r['value_pass'] for r in records),
        max_absolute_difference=max(differences) if differences else None,
        mean_absolute_difference=float(np.mean(differences)) if differences else None, samples=records)


def run():
    import ee
    import httplib2
    if OUTPUT.exists(): raise FileExistsError('Preserve existing verification results')
    report=dict(tolerance=TOLERANCE,rtol=0,results={},
        independence='Raw composites and their masks sampled without export adapter or GeoTIFF serialization; same scientific seasonal builder reused. Not independent algorithm validation.')
    ee.Initialize(project=launch.config.GCP_PROJECT,http_transport=httplib2.Http(timeout=90))
    ee.data.setDeadline(90000);ee.data.setMaxRetries(0)
    try:
        previous=json.loads((ROOT/'qa_summary.json').read_text())
        for year in ('2017','2018'):
            fire,patches,rows,record=launch.load_inputs(year)
            state=json.loads(launch.record_path(year,launch.STATE.parent).read_text())
            reference,metadata=launch.verify_reference(year,fire,patches)  # one metadata request
            if metadata!=state['verified_references'][year]: raise ValueError('Reference changed since export')
            region=imagery.prepare_region(rows,fire_id=fire['fire_id'],batch_id=record['batch_id'])
            arrays={}
            for p in launch.PRODUCTS:
                path=ROOT/f'{year}_{p}.tif'
                if hashlib.sha256(path.read_bytes()).hexdigest()!=previous['rasters'][year+'_'+p]['sha256']:
                    raise ValueError('Local raster changed')
                with rasterio.open(path) as ds:
                    if ds.transform!=region['transform']: raise ValueError('Output grid changed')
                    arrays[p]=ds.read()
            samples=choose_pixels(rows,region,arrays)
            if len(samples)>12: raise ValueError('Point budget exceeded')
            points=ee.FeatureCollection([ee.Feature(ee.Geometry.Point([p['x'],p['y']],proj='EPSG:5070'),
                                        {'pixel_id':p['pixel_id']}) for p in samples])
            output_projection=ee.Projection('EPSG:5070',transform=list(region['transform'])[:6])
            context=ee.Geometry.Rectangle(list(region['bounds']),proj='EPSG:5070',geodesic=False)
            inputs=landsat.build_fire_inputs(fire,context,reference_projection=reference)
            for product,key in [('nbrpre','nbr_pre'),('season1','nbr_1')]:
                # Inspect the raw scientific NBR mask, not the exported validity band.
                raw=inputs[key]
                source=raw.rename('source_nbr').unmask(-9999,sameFootprint=False).addBands(
                    raw.mask().rename('source_mask').unmask(0,sameFootprint=False))
                print(f'Querying {year} {product}: {len(samples)} pixel centers',flush=True)
                result=source.sampleRegions(collection=points,properties=['pixel_id'],
                    projection=output_projection,geometries=False).getInfo()
                checked=compare(samples,result['features'],arrays[product])
                report['results'][year+'_'+product]=checked
                OUTPUT.write_text(json.dumps(report,indent=2)+'\n')
                print(json.dumps({k:v for k,v in checked.items() if k!='samples'}),flush=True)
                if checked['value_pass']!=len(samples) or checked['mask_pass']!=len(samples):
                    raise ValueError('Pixel mismatch: stop without changing tolerance/science')
        report['verdict']='PASS: sampled export consistency only'
    except BaseException as exc:
        report['verdict']='STOPPED: '+str(exc)
        raise
    finally:
        OUTPUT.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--query',action='store_true',help='Authorize at most four bounded pixel queries and two scene metadata checks')
    if not parser.parse_args().query: parser.error('Explicit --query required')
    run()
