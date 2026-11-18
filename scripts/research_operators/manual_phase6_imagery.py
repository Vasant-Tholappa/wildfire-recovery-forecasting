"""Offline-only two-fire imagery POC configuration; never creates EE tasks."""
import hashlib
import json
from pathlib import Path

import pandas as pd
from shapely.geometry import box

import manual_population as population
from wildfire_recovery_forecasting import config, imagery
from wildfire_recovery_forecasting.fire_catalog import derive_fire_fields

VERSION = 'phase6_nbr_context_poc_v1'
CANDIDATES = ('CA3839112224120171009', 'CA3839611945120180904')
ELIGIBLE = population.DATA / 'patch_model_eligible_terrain_v2.parquet'
OUTPUT = Path('notes/phase6_imagery_poc.json')
KEY_COLUMNS = ['fire_id', 'patch_id', 'batch_id', 'grid_row', 'grid_col', 'fire_year', 'split']


def key_digest(ids):
    return hashlib.sha256('\n'.join(sorted(ids)).encode()).hexdigest()


def validate_candidate(fire, patches, rows, expected_count, *, allow_test_inputs=False):
    expected = derive_fire_fields(pd.Timestamp(fire['ignition_date']))
    for name in ('fire_year', 'split', 'prefire_search_start_year', 'prefire_search_end_year', 'season1_year'):
        if fire[name] != expected[name]:
            raise ValueError('Frozen temporal fields disagree with ignition date')
    if expected['split'] == 'test' and not allow_test_inputs:
        raise ValueError('Test input imagery requires explicit authorization')
    if len(patches) != expected_count:
        raise ValueError('Wrong canonical batch occupancy')
    if (rows.empty or not rows.patch_id.is_unique
            or not set(rows.patch_id).issubset(set(patches.patch_id))
            or not rows.fire_year.eq(expected['fire_year']).all()
            or not rows.split.eq(expected['split']).all()):
        raise ValueError('Eligible key/split membership mismatch')
    region = imagery.prepare_region(rows, fire_id=fire['fire_id'], batch_id=patches.batch_id.iloc[0])
    for row in rows.itertuples():
        imagery.chip_window(row.grid_row, row.grid_col, region)
    return region


def prepare():
    manifest, items = population.load_population()  # Local hashes + frozen identity checks.
    perimeters = manifest.to_crs(config.GRID_CRS).set_index('fire_id').geometry
    eligible = pd.read_parquet(ELIGIBLE, columns=KEY_COLUMNS,
                               filters=[('fire_id', 'in', list(CANDIDATES))])
    records = []
    for fire_id in CANDIDATES:
        fire = manifest.set_index('fire_id', drop=False).loc[fire_id].to_dict()
        item = next(i for i in items if i['fire_id'] == fire_id)
        patches = population.batch_patches(perimeters.loc[fire_id], fire_id, item['batch_id'])
        rows = eligible[eligible.fire_id == fire_id]
        region = validate_candidate(fire, patches, rows, item['patch_count'])
        specs = imagery.prepare_export_specs(region)
        for product, spec in specs.items():
            spec['bucket'] = config.GCS_BUCKET
            spec['fileNamePrefix'] = f'imagery/{VERSION}/{fire_id}/{item["batch_id"]}/{product}'
        records.append(dict(fire_id=fire_id, batch_id=item['batch_id'],
            fire_year=int(fire['fire_year']), split=fire['split'],
            eligible_patch_count=len(rows), original_batch_patch_count=len(patches),
            original_batch_keys_sha256=key_digest(patches.patch_id),
            eligible_keys_sha256=key_digest(rows.patch_id),
            context_extends_outside_fire=not perimeters.loc[fire_id].covers(box(*region['bounds'])),
            width=region['width'], height=region['height'], exports=specs))
    return dict(version=VERSION, project=config.GCP_PROJECT, proposed_tasks=4,
        selection_reason='Input-only small pair: 2017 late-ignition training fire and 2018 validation fire; 104 versus 215 eligible patches. No targets or model results used.',
        source_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (population.MANIFEST, population.OCCUPANCY, population.REPORT)},
        eligible_file=str(ELIGIBLE), candidates=records,
        live_gate='Reference scene/projection, masks, output alignment, sampling, quota and cost unresolved; no tasks created.')


def reference_band(fire, original_patches):
    """Same lazy exact-batch scene selection as Phase 3; no evaluation."""
    import ee
    from wildfire_recovery_forecasting import pipeline
    exact = pipeline.patches_to_ee(original_patches).geometry()
    return ee.Image(ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
        .filterBounds(exact)
        .filterDate(f"{fire['season1_year']}-06-01", f"{fire['season1_year']}-10-01")
        .sort('system:time_start').first()).select('SR_B5')


def image_export_arguments(fire, original_patches, eligible_rows, *, frozen_record, reference_projection=None, allow_test_inputs=False):
    """Build lazy EE images/arguments only, after separate live authorization.

    No initialization, metadata evaluation, task creation or submission. The
    original batch key digest is checked against the offline frozen POC record.
    Reference projection remains a lazy expression, never an invented CRS.
    Default EE nearest sampling is the proposal to verify on actual rasters.
    """
    import ee
    from wildfire_recovery_forecasting import landsat, pipeline

    region = validate_candidate(fire, original_patches, eligible_rows,
                                frozen_record['original_batch_patch_count'], allow_test_inputs=allow_test_inputs)
    if (fire['fire_id'] != frozen_record['fire_id']
            or original_patches.batch_id.nunique() != 1
            or original_patches.batch_id.iloc[0] != frozen_record['batch_id']
            or key_digest(original_patches.patch_id) != frozen_record['original_batch_keys_sha256']
            or key_digest(eligible_rows.patch_id) != frozen_record['eligible_keys_sha256']):
        raise ValueError('Frozen POC identity changed')
    reference = reference_projection
    if reference is None:
        reference = reference_band(fire, original_patches).projection()
    context = ee.Geometry.Rectangle(list(region['bounds']), proj=config.GRID_CRS, geodesic=False)
    inputs = landsat.build_fire_inputs(fire, context, reference_projection=reference)
    from wildfire_recovery_forecasting.config import GCS_BUCKET
    arguments = {}
    for product, key in (('nbrpre', 'nbr_pre'), ('season1', 'nbr_1')):
        nbr = inputs[key].toFloat()
        valid = nbr.mask().gt(0).unmask(0, sameFootprint=False).rename('valid').toFloat()
        image = nbr.unmask(-9999, sameFootprint=False).addBands(valid).clip(context)
        arguments[product] = dict(image=image, bucket=GCS_BUCKET,
            description=f"{VERSION}_{fire['fire_id']}_{product}",
            fileNamePrefix=f"imagery/{VERSION}/{fire['fire_id']}/{frozen_record['batch_id']}/{product}",
            region=context, crs=config.GRID_CRS, crsTransform=list(region['transform'])[:6],
            fileFormat='GeoTIFF', formatOptions={'noData': -9999},
            maxPixels=region['width'] * region['height'])
    return arguments


if __name__ == '__main__':
    result = prepare()
    OUTPUT.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
