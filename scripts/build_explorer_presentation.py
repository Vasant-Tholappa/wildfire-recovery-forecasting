"""Offline, explicit export of curated presentation records; never runs a model."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'explorer/presentation'
KEYS = ['fire_id', 'patch_id']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build():
    base = ROOT / 'data/processed'
    test = base / 'phase7_test_v1'
    sources = {
        'predictions': test/'predictions.parquet', 'metrics': test/'comparison.csv',
        'per_fire': test/'per_fire_metrics.csv', 'audit': test/'evaluation_audit.json',
        'baselines': test/'supplemental_baselines_v1/baseline_metrics.csv',
        'reductions': test/'supplemental_baselines_v1/model_mae_reductions.csv',
        'baseline_audit': test/'supplemental_baselines_v1/supplemental_audit.json',
        'index': base/'phase6_cnn_dataset_v1/test/index.parquet',
        'fires': base/'fire_manifest.parquet', 'population': base/'phase6_nbr_context_v1_manifest.json',
    }
    original = {k:digest(p) for k,p in sources.items()}
    audit = json.loads(sources['audit'].read_text())
    assert audit['status']=='PASS' and audit['rows']==30606 and audit['fires']==37
    for name in ('predictions.parquet','comparison.csv','per_fire_metrics.csv'):
        assert digest(test/name)==audit['outputs'][name]
    ba = json.loads(sources['baseline_audit'].read_text())
    assert ba['status']=='PASS' and ba['ordered_keys_sha256']==audit['ordered_keys_sha256']
    assert ba['aligned_target_sha256']==audit['aligned_target_sha256']
    for name in ('baseline_metrics.csv','model_mae_reductions.csv'):
        assert digest(test/'supplemental_baselines_v1'/name)==ba['outputs_sha256'][name]
    models = ['option_a_'+m for m in ('xgb','rf','cnn_only','fusion')]
    pred = pd.read_parquet(sources['predictions'],columns=KEYS+['nbr_5_mean']+models)
    ix = pd.read_parquet(sources['index'],columns=KEYS+['grid_row','grid_col','split','array_index'])
    assert len(ix)==len(pred)==30606 and ix.fire_id.nunique()==37 and ix.split.eq('test').all()
    assert not ix.duplicated(KEYS).any() and not pred.duplicated(KEYS).any()
    assert ix[KEYS].equals(pred[KEYS]) and np.array_equal(ix.array_index,np.arange(len(ix)))
    keyhash = hashlib.sha256(''.join(f'{f}\t{p}\n' for f,p in ix[KEYS].itertuples(index=False,name=None)).encode()).hexdigest()
    assert keyhash==audit['ordered_keys_sha256']
    y=pred.nbr_5_mean.to_numpy(dtype='<f8')
    assert hashlib.sha256(y.tobytes()).hexdigest()==audit['aligned_target_sha256']
    assert np.isfinite(pred[['nbr_5_mean']+models].to_numpy()).all()
    assert all(p==f'{f}_r{r}_c{c}' for f,p,r,c in ix[KEYS+['grid_row','grid_col']].itertuples(index=False,name=None))
    # Existing canonical lattice only: no fire-grid regeneration or extraction.
    transform = Transformer.from_crs(5070,4326,always_xy=True)
    patches = pred.copy()
    patches['lon'],patches['lat']=transform.transform((ix.grid_col+.5)*90,(ix.grid_row+.5)*90)
    for i,(dx,dy) in enumerate(((0,0),(1,0),(1,1),(0,1))):
        patches[f'x{i}'],patches[f'y{i}']=transform.transform((ix.grid_col+dx)*90,(ix.grid_row+dy)*90)
    assert patches.lon.between(-125,-113).all() and patches.lat.between(32,43).all()
    fires=gpd.read_parquet(sources['fires'])
    fires=fires[fires.fire_id.isin(ix.fire_id)].copy()
    assert len(fires)==37 and fires.fire_id.is_unique and fires.fire_year.eq(2020).all()
    points=fires.to_crs(5070).geometry.representative_point().to_crs(4326)
    fires['lon']=points.x;fires['lat']=points.y
    fires['patches']=fires.fire_id.map(ix.groupby('fire_id').size())
    records=fires[['fire_id','fire_name','fire_size_acres','lon','lat','patches']].sort_values(['fire_name','fire_id'])
    population=json.loads(sources['population'].read_text())
    assert population['fire_count']==151 and population['patch_count']==111257
    summary={'studied_fires':151,'test_patches':30606,'test_fires':37,
             'models':json.loads(pd.read_csv(sources['metrics']).to_json(orient='records',double_precision=15)),
             'baselines':pd.read_csv(sources['baselines']).to_dict('records'),
             'reductions':pd.read_csv(sources['reductions']).to_dict('records')}
    OUT.mkdir(parents=True,exist_ok=False)  # Never overwrite reviewed presentation assets.
    records.to_csv(OUT/'fires.csv',index=False)
    patches.to_csv(OUT/'patches.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    pd.read_csv(sources['per_fire']).to_csv(OUT/'per_fire.csv',index=False)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
    provenance={'version':'explorer-v1','decision':'C','rows':30606,'fires':37,
        'ordered_keys_sha256':keyhash,'source_hashes':original,
        'geometry':'90m EPSG:5070 zero-anchored grid; corners/centers transformed to EPSG:4326. Fire markers are MTBS interior representative points, not ignition locations.',
        'sampling':'All 30606 evaluated patches; no spatial display subsampling.',
        'values':'Unchanged saved outcomes and four Option A predictions. No inference or performance rescore.',
        'files':{p.name:digest(p) for p in OUT.iterdir() if p.is_file()}}
    (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    assert all(digest(sources[k])==v for k,v in original.items())
    print('PASS: Option C, 37 fires, 30606 exact keys; source files unchanged')
    print('Presentation bytes:',sum(p.stat().st_size for p in OUT.iterdir()))


if __name__=='__main__':
    build()
