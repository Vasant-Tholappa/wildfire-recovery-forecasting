"""Bounded four-file GCS read/download and local POC raster/chip QA; no EE calls."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from rasterio.windows import Window, bounds

import manual_phase6_submit as launcher
from wildfire_recovery_forecasting import imagery

ROOT = launcher.STATE.parent / 'phase6_nbr_context_poc_v1_inspection'


def inspect_raster(path, record, product):
    spec=record['exports'][product]
    with rasterio.open(path) as ds:
        values=ds.read()
        checks=dict(dimensions=(ds.width,ds.height)==(record['width'],record['height']),
            crs=ds.crs==rasterio.crs.CRS.from_epsg(5070),
            transform=ds.transform==Affine(*spec['crsTransform']),
            bounds=list(ds.bounds)==spec['region_epsg5070'],
            resolution=ds.res==(30,30), north_up=ds.transform.a==30 and ds.transform.e==-30 and ds.transform.b==ds.transform.d==0,
            bands=ds.count==2, float32=ds.dtypes==('float32','float32'), nodata=ds.nodatavals==(-9999.,-9999.),
            descriptions=ds.descriptions in ((spec['band'],'valid'),(None,None)))
        if ds.count!=2: raise ValueError(f'Unexpected band count: {path}')
        nbr,valid=values
        checks['binary_validity']=bool(np.isin(valid,[0,1]).all())
        good=valid==1
        checks['finite_valid_nbr']=bool(np.isfinite(nbr[good]).all())
        checks['nodata_consistency']=bool(np.array_equal(nbr==-9999,~good))
        checks['rasterio_mask_consistency']=bool(np.array_equal(ds.read_masks(1)>0,good))
        checks['validity_band_unmasked']=bool((ds.read_masks(2)>0).all())
        valid_values=nbr[good]
        outliers=int(((valid_values < -1)|(valid_values > 1)).sum())
        result=dict(path=str(path),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            width=ds.width,height=ds.height,crs=str(ds.crs),transform=list(ds.transform)[:6],bounds=list(ds.bounds),
            dtypes=ds.dtypes,descriptions=ds.descriptions,nodata=ds.nodatavals,checks=checks,
            valid_pixels=int(good.sum()),invalid_pixels=int((~good).sum()),valid_fraction=float(good.mean()),
            valid_min=float(valid_values.min()) if len(valid_values) else None,
            valid_max=float(valid_values.max()) if len(valid_values) else None,
            outside_minus1_plus1=outliers,valid_zero_pixels=int(((nbr==0)&good).sum()))
        return result,nbr,good


def inspect_chips(rows, region, arrays):
    records=[]
    for r in rows.itertuples():
        window=imagery.chip_window(r.grid_row,r.grid_col,region)
        central=Window(window.col_off+6,window.row_off+6,3,3)
        if bounds(central,region['transform'])!=(90*r.grid_col,90*r.grid_row,90*(r.grid_col+1),90*(r.grid_row+1)):
            raise ValueError('Central patch geometry mismatch')
        slices=window.toslices()
        entry=dict(fire_id=r.fire_id,patch_id=r.patch_id,grid_row=r.grid_row,grid_col=r.grid_col,
                   raster_row=int(window.row_off),raster_col=int(window.col_off))
        for p,(_,valid) in arrays.items():
            mask=valid[slices]
            if mask.shape!=(15,15): raise ValueError('Chip containment mismatch')
            entry[p+'_full']=float(mask.mean())
            entry[p+'_central']=float(mask[6:9,6:9].mean())
        entry['joint_ge80']=all(entry[p+'_'+part]>=.8 for p in arrays for part in ('full','central'))
        records.append(entry)
    return pd.DataFrame(records)


def figures(output,year,arrays,coverage,region):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    cmap=plt.get_cmap('RdYlGn').copy();cmap.set_bad('#999999')
    fig,axes=plt.subplots(1,2,figsize=(8,7),layout='constrained')
    xmin,ymin,xmax,ymax=region['bounds']
    for ax,(p,(nbr,valid)) in zip(axes,arrays.items()):
        im=ax.imshow(np.ma.array(nbr,mask=~valid),cmap=cmap,vmin=-1,vmax=1,
                     extent=(xmin,xmax,ymin,ymax),origin='upper',interpolation='nearest')
        ax.set_title(f'{year} {p}');ax.set_xlabel('EPSG:5070 easting (m)');ax.set_ylabel('Northing (m)')
    fig.colorbar(im,ax=axes,label='NBR; gray = invalid/nodata')
    fig.savefig(output/f'{year}_rasters.png',dpi=140);plt.close(fig)
    # Input-only choices: worst joint coverage, median coverage and best coverage.
    score=coverage[[p+'_'+part for p in arrays for part in ('full','central')]].min(axis=1)
    ordered=coverage.assign(score=score).sort_values(['score','patch_id'])
    picks=ordered.iloc[sorted(set([0,len(ordered)//2,len(ordered)-1]))]
    fig,axes=plt.subplots(len(picks),2,figsize=(8,3*len(picks)),squeeze=False,layout='constrained')
    for row,(_,r) in enumerate(picks.iterrows()):
        slices=(slice(int(r.raster_row),int(r.raster_row)+15),slice(int(r.raster_col),int(r.raster_col)+15))
        for col,(p,(nbr,valid)) in enumerate(arrays.items()):
            ax=axes[row,col]
            im=ax.imshow(np.ma.array(nbr[slices],mask=~valid[slices]),cmap=cmap,vmin=-1,vmax=1,interpolation='nearest')
            ax.add_patch(Rectangle((5.5,5.5),3,3,fill=False,edgecolor='blue',linewidth=2))
            ax.set_title(f'{r.patch_id}\n{p}: full {r[p+"_full"]:.1%}; center {r[p+"_central"]:.1%}',fontsize=8)
    fig.colorbar(im,ax=axes.ravel().tolist(),label='NBR; gray = nodata; blue = 90 m target footprint')
    fig.savefig(output/f'{year}_chips.png',dpi=140);plt.close(fig)


def run(output,download=False):
    # Exclusive directory: never overwrite an earlier download/report.
    output.mkdir(parents=True,exist_ok=False)
    report={'rasters':{},'chips':{},'note':'Coverage >=80% is diagnostic only; no patches filtered. Scientific equivalence remains unverified.'}
    client=None
    if download:
        from google.cloud import storage
        client=storage.Client(project=launcher.config.GCP_PROJECT)
    all_coverage=[]
    for year in ('2017','2018'):
        fire,patches,rows,record=launcher.load_inputs(year)  # identifiers only
        state=json.loads(launcher.record_path(year,launcher.STATE.parent).read_text())
        if state['configuration']!=record or state['fire_id']!=fire['fire_id']:
            raise ValueError('Submission/configuration mismatch')
        region=imagery.prepare_region(rows,fire_id=fire['fire_id'],batch_id=record['batch_id'])
        arrays={}
        for product in launcher.PRODUCTS:
            spec=record['exports'][product]
            expected=f"gs://{spec['bucket']}/{spec['fileNamePrefix']}"
            if state['products'][product]['destination']!=expected: raise ValueError('Destination mismatch')
            if client is None: raise ValueError('First retrieval requires --download (GCS reads only)')
            blobs=list(client.list_blobs(spec['bucket'],prefix=spec['fileNamePrefix'],max_results=3))
            if len(blobs)!=1 or not blobs[0].name.endswith('.tif'):
                raise ValueError('Expected one GeoTIFF per product; stop to inspect shards/objects')
            blob=blobs[0]; path=output/f'{year}_{product}.tif'
            with path.open('xb') as handle: blob.download_to_file(handle,if_generation_match=int(blob.generation))
            result,nbr,valid=inspect_raster(path,record,product)
            result.update(source=f'gs://{spec["bucket"]}/{blob.name}',generation=blob.generation,task_id=state['products'][product]['task_id'])
            report['rasters'][year+'_'+product]=result
            arrays[product]=(nbr,valid)
        if not all(all(report['rasters'][year+'_'+p]['checks'].values()) for p in arrays):
            report['chips'][year]={'blocked':'Raster checks failed'}
            continue
        coverage=inspect_chips(rows,region,arrays)
        if len(coverage)!=record['eligible_patch_count']: raise ValueError('Eligible count mismatch')
        all_coverage.append(coverage)
        fields=[p+'_'+part for p in arrays for part in ('full','central')]
        report['chips'][year]=dict(total=len(coverage),joint_ge80=int(coverage.joint_ge80.sum()),
            joint_below80=int((~coverage.joint_ge80).sum()),coverage={field:dict(
                minimum=float(coverage[field].min()),p25=float(coverage[field].quantile(.25)),
                median=float(coverage[field].median()),mean=float(coverage[field].mean()),maximum=float(coverage[field].max()),
                ge80=int((coverage[field]>=.8).sum()),zero=int((coverage[field]==0).sum())) for field in fields})
        figures(output,year,arrays,coverage,region)
    if all_coverage: pd.concat(all_coverage).to_csv(output/'chip_coverage.csv',index=False)
    (output/'qa_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download',action='store_true')
    parser.add_argument('--output',type=Path,default=ROOT)
    args=parser.parse_args()
    run(args.output,args.download)
