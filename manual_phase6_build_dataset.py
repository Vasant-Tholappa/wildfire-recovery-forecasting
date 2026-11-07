"""Local, frozen CNN dataset construction. No cloud or model training."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap
from scipy.spatial import cKDTree
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import manual_phase6_local_qa as qa
from wildfire_recovery_forecasting import imagery
from wildfire_recovery_forecasting.modeling import CATEGORICAL_FEATURES

DATA = Path('data/processed')
COUNTS = {'train': 61073, 'validation': 19561, 'test': 30606}
FEATURES = ['nbr_pre_mean','dominant_evt','dominant_evt_fraction','tree_cover_mean',
    'tree_height_mean','elevation_mean','slope_mean','northness_mean','eastness_mean',
    'fire_size_acres','ignition_month','mtbs_unburned_low_fraction','mtbs_low_fraction',
    'mtbs_moderate_fraction','mtbs_high_fraction','mtbs_increased_greenness_fraction',
    'nbr_1_mean','nbr_drop_patch']
KEYS = ['fire_id', 'patch_id', 'split']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def align(left, right, columns):
    if left.duplicated(KEYS).any() or right.duplicated(KEYS).any():
        raise ValueError('Duplicate alignment keys')
    result = left[KEYS].merge(right[KEYS + columns], on=KEYS, how='left',
                              validate='one_to_one', indicator=True, sort=False)
    if not result['_merge'].eq('both').all() or not result[KEYS].equals(left[KEYS].reset_index(drop=True)):
        raise ValueError('Missing or reordered join keys')
    return result[columns]


def load_targets(path, split):
    if split not in ('train', 'validation'):
        raise ValueError('Test target access forbidden')
    return pd.read_parquet(path, columns=KEYS + ['nbr_5_mean'], filters=[('split','==',split)])


def fill_images(values, masks, medians):
    result = np.array(values, dtype=np.float32, copy=True)
    if not np.isfinite(result[masks]).all():
        raise ValueError('Nonfinite valid image values')
    for channel, median in enumerate(medians):
        v = result[:, channel]; valid = masks[:, channel]
        v[valid] = np.clip(v[valid], -1, 1)
        v[~valid] = median
    if not np.isfinite(result).all() or (np.abs(result)>1).any():
        raise ValueError('Invalid filled image')
    return result


def training_medians(images, masks):
    # Caller passes retained TRAIN chips only. Shared pixels count once per chip occurrence.
    values = []
    for channel in range(2):
        good = np.asarray(images[:,channel])[np.asarray(masks[:,channel])]
        if not len(good) or not np.isfinite(good).all():
            raise ValueError('Invalid training pixels')
        values.append(float(np.median(np.clip(good, -1, 1), overwrite_input=True)))
    return values


def context_overlap(rows):
    groups = [(f,g,g[['grid_row','grid_col']].to_numpy()) for f,g in rows.groupby('fire_id')]
    records = []; validation_ids = set()
    for i,(fa,ga,a) in enumerate(groups):
        for fb,gb,b in groups[i+1:]:
            if ga.split.iloc[0] == gb.split.iloc[0]: continue
            if np.any(a.min(0)-b.max(0)>4) or np.any(b.min(0)-a.max(0)>4): continue
            neighbors = cKDTree(b).query_ball_point(a, r=4, p=np.inf)
            ai = [j for j,ns in enumerate(neighbors) if ns]; bi = sorted({k for ns in neighbors for k in ns})
            if not ai: continue
            central = sum(np.array_equal(a[j],b[k]) for j,ns in enumerate(neighbors) for k in ns)
            records.append(dict(fire_a=fa,split_a=ga.split.iloc[0],fire_b=fb,split_b=gb.split.iloc[0],
                                patches_a=len(ai),patches_b=len(bi),pairs=sum(map(len,neighbors)),central_overlap_pairs=central))
            for g, indices in ((ga,ai),(gb,bi)):
                if g.split.iloc[0]=='validation': validation_ids.update(g.iloc[indices].patch_id)
    return records, sorted(validation_ids)


def build(output):
    start = time.monotonic()
    source = DATA/'patch_model_eligible_terrain_v2.parquet'
    roles_path = source.with_suffix('.roles.json')
    manifest_path = DATA/'phase6_nbr_context_v1_manifest.json'
    qa_path = DATA/'phase6_nbr_context_v1_population_local_qa_v1/qa_summary.json'
    coverage_path = qa_path.parent/'patch_coverage.csv'
    manifest = json.loads(manifest_path.read_text()); report = json.loads(qa_path.read_text())
    roles = json.loads(roles_path.read_text())
    features = [f for f in roles['predictors'] if not f.startswith(('hist_','pre_s1_','s1_'))]
    if features != FEATURES or roles['target'] != 'nbr_5_mean': raise ValueError('Phase 5 feature order changed')
    if digest(manifest_path) != qa.MANIFEST_SHA256 or report['verdict'] != 'PASS' or report['rasters_passed'] != 302:
        raise ValueError('Frozen raster QA prerequisite failed')
    rows = pd.read_parquet(source, columns=qa.KEY_COLUMNS + FEATURES)
    qa.validate_keys(rows, manifest)
    if not np.isfinite(rows[FEATURES].to_numpy(dtype=float)).all(): raise ValueError('Invalid tabular inputs')
    coverage = pd.read_csv(coverage_path)
    if len(coverage)!=len(rows) or not coverage.qa_status.eq('PASS').all(): raise ValueError('Coverage incomplete')
    fields = qa.FIELDS + [f+'_valid_count' for f in qa.FIELDS] + ['joint_ge80']
    cov = align(rows, coverage, fields)
    keep = cov[qa.FIELDS].ge(.8).all(axis=1)
    if not keep.equals(cov.joint_ge80): raise ValueError('Coverage flag mismatch')
    rows = pd.concat([rows.reset_index(drop=True),cov.reset_index(drop=True)],axis=1)
    excluded = rows.loc[~keep, qa.KEY_COLUMNS + fields].copy()
    if len(excluded)!=17 or not excluded.split.eq('train').all(): raise ValueError('Exclusions changed')
    retained = rows.loc[keep].sort_values(['fire_id','patch_id']).reset_index(drop=True)
    if retained.groupby('split').size().to_dict()!=COUNTS: raise ValueError('Retained counts changed')
    if retained.groupby('fire_id').split.nunique().max()!=1: raise ValueError('Fire split leakage')
    overlaps, sensitivity = context_overlap(retained)
    expected_pair = {'CA3779911892220160804','CA3782611887220190726'}
    if (len(overlaps)!=1 or {overlaps[0]['fire_a'],overlaps[0]['fire_b']}!=expected_pair
            or overlaps[0]['pairs']!=233 or overlaps[0]['central_overlap_pairs']!=0 or len(sensitivity)!=53):
        raise ValueError('Spatial overlap differs from approved finding')
    output = Path(output); output.mkdir(parents=True,exist_ok=False)
    excluded.to_parquet(output/'exclusions.parquet',index=False)
    pd.DataFrame({'patch_id':sensitivity}).to_csv(output/'shared_context_validation_ids.csv',index=False)
    arrays = {}; frames = {}; positions = {}
    for split,n in COUNTS.items():
        folder=output/split;folder.mkdir()
        frame=retained[retained.split==split].reset_index(drop=True);frames[split]=frame
        frame[qa.KEY_COLUMNS+fields].assign(array_index=np.arange(n)).to_parquet(folder/'index.parquet',index=False)
        frame[KEYS+FEATURES].to_parquet(folder/'tabular.parquet',index=False)
        np.save(folder/'tabular_raw.npy',frame[FEATURES].to_numpy(dtype=np.float64),allow_pickle=False)
        arrays[split]=(open_memmap(folder/'images.npy',mode='w+',dtype='float32',shape=(n,2,15,15)),
                       open_memmap(folder/'validity.npy',mode='w+',dtype='bool',shape=(n,2,15,15)))
        positions.update({p:(split,i) for i,p in enumerate(frame.patch_id)})
    clipped={s:[0,0] for s in COUNTS};missing={s:[0,0] for s in COUNTS};seen=set()
    for entry in manifest['fires']:
        fid=entry['fire_id']; full=rows[rows.fire_id==fid]
        region=imagery.prepare_region(full,fire_id=fid,batch_id=entry['batch_id'])
        group=retained[retained.fire_id==fid]
        for channel,product in enumerate(qa.CHANNELS):
            evidence=report['rasters'][fid+'_'+product]
            path=qa.local_path(DATA/'phase6_downloads',entry['products'][product]['effective_uri'])
            check,nbr,valid=qa.inspect_raster(path,{**entry,'exports':entry['products']},product)
            if check['sha256']!=evidence['sha256'] or not all(check['checks'].values()): raise ValueError('Raster changed: '+fid)
            for row in group.itertuples():
                win=imagery.chip_window(row.grid_row,row.grid_col,region)
                if (region['transform']*(win.col_off+6,win.row_off+6)!=(row.grid_col*90,(row.grid_row+1)*90)
                        or region['transform']*(win.col_off+9,win.row_off+9)!=((row.grid_col+1)*90,row.grid_row*90)):
                    raise ValueError('Central footprint mismatch')
                v=nbr[win.toslices()];mask=valid[win.toslices()]
                if v.shape!=(15,15): raise ValueError('Chip size mismatch')
                if (int(mask.sum())!=getattr(row,product+'_full_valid_count')
                        or int(mask[6:9,6:9].sum())!=getattr(row,product+'_central_valid_count')):
                    raise ValueError('Coverage changed')
                split,i=positions[row.patch_id];images,masks=arrays[split]
                images[i,channel]=v;masks[i,channel]=mask
                clipped[split][channel]+=int(((np.abs(v)>1)&mask).sum())
                missing[split][channel]+=int((~mask).sum())
                seen.add((row.patch_id,channel))
        print('Extracted',fid,len(group),flush=True)
    if len(seen)!=2*len(retained): raise ValueError('Incomplete extraction')
    medians=training_medians(*arrays['train'])
    for split,(images,masks) in arrays.items():
        for first in range(0,len(images),1024):
            last=first+1024
            images[first:last]=fill_images(images[first:last],masks[first:last],medians)
        images.flush();masks.flush()
    # Same categorical conventions as Phase 5; CNN numerical scaling is train-fitted.
    numeric=[f for f in FEATURES if f not in CATEGORICAL_FEATURES]
    prep=ColumnTransformer([('numeric',StandardScaler(),numeric),
          ('categorical',OneHotEncoder(handle_unknown='ignore',sparse_output=False),list(CATEGORICAL_FEATURES))])
    prep.fit(frames['train'][FEATURES])
    scaler=prep.named_transformers_['numeric'];encoder=prep.named_transformers_['categorical']
    tabmeta=dict(input_order=FEATURES,output_order=prep.get_feature_names_out().tolist(),numeric=numeric,
        mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),variance=scaler.var_.tolist(),
        categorical=list(CATEGORICAL_FEATURES),categories=[v.tolist() for v in encoder.categories_],
        fit_split='train',fit_rows=COUNTS['train'],unknown_category_policy='all-zero one-hot')
    unknown={}
    for split,frame in frames.items():
        x=prep.transform(frame[FEATURES]).astype('float32')
        if not np.isfinite(x).all():raise ValueError('Nonfinite fusion inputs')
        np.save(output/split/'tabular_fusion.npy',x,allow_pickle=False)
        unknown[split]={f:int((~frame[f].isin(cat)).sum()) for f,cat in zip(CATEGORICAL_FEATURES,encoder.categories_)}
        if split!='test':
            targets=load_targets(source,split)
            y=align(frame,targets,['nbr_5_mean']).to_numpy(dtype=np.float64).ravel()
            if not np.isfinite(y).all():raise ValueError('Invalid development target')
            np.save(output/split/'targets.npy',y,allow_pickle=False)
    (output/'tabular_preprocessing.json').write_text(json.dumps(tabmeta,indent=2))
    # Reopen saved arrays/indexes in bounded chunks and verify ordering against frozen inputs.
    for split,frame in frames.items():
        saved=pd.read_parquet(output/split/'index.parquet')
        if not saved[qa.KEY_COLUMNS].equals(frame[qa.KEY_COLUMNS]) or not np.array_equal(saved.array_index,np.arange(len(frame))):
            raise ValueError('Saved ordering mismatch')
        raw=np.load(output/split/'tabular_raw.npy',mmap_mode='r')
        if not np.array_equal(raw,frame[FEATURES].to_numpy(dtype=np.float64)):raise ValueError('Tabular values changed')
        im=np.load(output/split/'images.npy',mmap_mode='r');ms=np.load(output/split/'validity.npy',mmap_mode='r')
        for first in range(0,len(im),1024):
            a=im[first:first+1024];v=ms[first:first+1024]
            if not np.isfinite(a).all() or (abs(a)>1).any():raise ValueError('Saved image integrity')
            for ch in range(2):
                if not np.all(a[:,ch][~v[:,ch]]==np.float32(medians[ch])):raise ValueError('Saved fill mismatch')
        if split=='test' and (output/split/'targets.npy').exists():raise ValueError('Test labels forbidden')
    files={str(f.relative_to(output)):dict(bytes=f.stat().st_size,sha256=digest(f)) for f in sorted(output.rglob('*')) if f.is_file()}
    provenance={str(f):digest(f) for f in [manifest_path,qa_path,coverage_path,roles_path,Path(__file__)]}
    # Hashing parquet bytes is not decoding any target column.
    provenance[str(source)]=digest(source)
    result=dict(version=output.name,status='VALIDATED',counts=COUNTS,excluded=17,
        fires={s:int(f.fire_id.nunique()) for s,f in frames.items()},channel_order=list(qa.CHANNELS),features=FEATURES,
        image_shape_per_row=[2,15,15],image_dtype='float32',mask_dtype='bool',target_dtype='float64',
        policy='>=80% independently in both full chips and central patches; clip valid NBR only; preserve masks; no image scaling',
        fill_medians=medians,median_fit='Retained training chips, valid clipped pixel occurrences; overlapping pixels counted per chip',
        missing_pixel_occurrences=missing,clipped_pixel_occurrences=clipped,unknown_categories=unknown,
        context_overlap=overlaps,spatial_limitation='Chronological and fire-disjoint, not fully spatially independent; limited shared context is not demonstrated target leakage. No test context overlap.',
        target_policy='Only train/validation target columns loaded using split filters; no test labels exported',
        provenance_sha256=provenance,raster_provenance='All 302 raster hashes rechecked against hashed QA report',
        files=files,runtime_seconds=round(time.monotonic()-start,3))
    (output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['status','counts','fill_medians','missing_pixel_occurrences','clipped_pixel_occurrences','runtime_seconds']},indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DATA/'phase6_cnn_dataset_v1')
    args=parser.parse_args()
    build(args.output)
