"""Tiny synthetic local fixtures only; never invoke population QA."""
import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import rasterio

import manual_phase6_local_qa as qa
from wildfire_recovery_forecasting import imagery


def fixture(tmp_path, row=-1, col=-1):
    rows = pd.DataFrame([dict(fire_id='synthetic', patch_id=f'synthetic_r{row}_c{col}',
        batch_id=f'r{row//120}_c{col//120}', grid_row=row, grid_col=col, fire_year=2020, split='test')])
    region = imagery.prepare_region(rows, fire_id='synthetic', batch_id=rows.batch_id.iloc[0])
    specs = imagery.prepare_export_specs(region)
    values = np.zeros((2, 15, 15), dtype='float32')
    values[1] = 1
    values[:, 6, 6] = [-9999, 0]
    values[0, 0, 0] = 1.2  # Finite diagnostic, never silently clipped.
    path = tmp_path / 'synthetic.tif'
    with rasterio.open(path, 'w', driver='GTiff', width=15, height=15, count=2,
                       dtype='float32', crs='EPSG:5070', transform=region['transform'], nodata=-9999) as ds:
        ds.write(values)
    return rows, region, path, dict(width=15, height=15, exports=specs)


@pytest.mark.parametrize('row,col', [(-1,-1), (0,0), (120,120)])
def test_masks_counts_and_central_index(tmp_path, row, col):
    rows, region, path, record = fixture(tmp_path, row, col)
    result, nbr, valid = qa.inspect_raster(path, record, 'nbrpre')
    assert all(result['checks'].values())
    assert result['invalid_pixels'] == 1
    assert result['valid_zero_pixels'] == 223
    assert result['outside_minus1_plus1'] == 1 and nbr[0,0] > 1
    table = qa.coverage_table(rows, region, {'nbrpre': (nbr, valid), 'season1': (nbr, np.ones_like(valid))})
    assert table.nbrpre_full_valid_count.iloc[0] == 224
    assert table.nbrpre_central_valid_count.iloc[0] == 8
    assert table.season1_full_valid_count.iloc[0] == 225
    assert table.season1_central_valid_count.iloc[0] == 9
    assert table.joint_ge80.iloc[0]
    table['qa_status'] = 'PASS'
    failed = table.copy()
    failed['qa_status'] = 'FAIL'
    summary = qa.summarize(pd.concat([table, failed]))
    assert summary['checked_patches'] == 1 and summary['unassessed_patches'] == 1
    assert summary['coverage']['nbrpre_central']['any_missing'] == 1
    json.dumps(summary, allow_nan=False)


def test_nonfinite_and_mask_mismatch_fail(tmp_path):
    _, _, path, record = fixture(tmp_path)
    with rasterio.open(path, 'r+') as ds:
        values = ds.read()
        values[0, 0, 1] = np.inf
        values[1, 6, 6] = 1
        ds.write(values)
    result, _, _ = qa.inspect_raster(path, record, 'nbrpre')
    assert not result['checks']['finite_valid_nbr']
    assert not result['checks']['nodata_consistency']


def test_uri_routing_and_target_free_column_projection(tmp_path):
    uri = 'gs://bucket/imagery/poc/fire/batch/nbrpre.tif'
    assert qa.local_path(tmp_path, uri) == tmp_path / 'bucket/imagery/poc/fire/batch/nbrpre.tif'
    with pytest.raises(ValueError):
        qa.local_path(tmp_path, 'gs://bucket/../outside.tif')
    manifest = tmp_path/'manifest.json'
    manifest.write_text('{}')
    digest = qa.hashlib.sha256(manifest.read_bytes()).hexdigest()
    with patch.object(qa, 'MANIFEST_SHA256', digest), patch.object(qa.pd, 'read_parquet', return_value=pd.DataFrame()) as read, \
         patch.object(qa, 'validate_keys', side_effect=ValueError('stop before any population loop')):
        with pytest.raises(ValueError, match='stop before'):
            qa.run(manifest, tmp_path/'eligible.parquet', tmp_path, tmp_path/'output')
    assert read.call_args.kwargs['columns'] == ['fire_id','patch_id','batch_id','grid_row','grid_col','fire_year','split']
    assert not (tmp_path/'output').exists()
