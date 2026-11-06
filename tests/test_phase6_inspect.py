"""Synthetic local GeoTIFF checks; no cloud clients or EE initialization."""
import numpy as np
import pandas as pd
import pytest
import rasterio
from affine import Affine
from manual_phase6_inspect import inspect_raster, inspect_chips
from wildfire_recovery_forecasting import imagery


def sample(tmp_path, mismatch=False):
    path=tmp_path/'sample.tif'
    values=np.zeros((2,15,15),dtype='float32');values[1]=1
    values[0,0,0]=-9999;values[1,0,0]=0
    if mismatch: values[1,0,0]=1
    transform=Affine(30,0,-180,0,-30,270)
    with rasterio.open(path,'w',driver='GTiff',width=15,height=15,count=2,
                       dtype='float32',crs='EPSG:5070',transform=transform,nodata=-9999) as ds:
        ds.write(values);ds.set_band_description(1,'NBR_pre');ds.set_band_description(2,'valid')
    record={'width':15,'height':15,'exports':{'nbrpre':{'band':'NBR_pre',
             'crsTransform':list(transform)[:6],'region_epsg5070':[-180,-180,270,270]}}}
    return path,record


def test_valid_zero_and_invalid_mask(tmp_path):
    path,record=sample(tmp_path)
    result,nbr,valid=inspect_raster(path,record,'nbrpre')
    assert all(result['checks'].values())
    assert result['valid_zero_pixels']==224 and result['invalid_pixels']==1
    rows=pd.DataFrame([dict(fire_id='f',patch_id='f_r0_c0',batch_id='r0_c0',grid_row=0,grid_col=0)])
    region=imagery.prepare_region(rows,fire_id='f',batch_id='r0_c0')
    coverage=inspect_chips(rows,region,{'nbrpre':(nbr,valid),'season1':(nbr,valid)})
    assert coverage.nbrpre_full.iloc[0]==pytest.approx(224/225)
    assert coverage.nbrpre_central.iloc[0]==1 and coverage.joint_ge80.iloc[0]


def test_inconsistent_validity_is_reported(tmp_path):
    path,record=sample(tmp_path,True)
    result,_,_=inspect_raster(path,record,'nbrpre')
    assert not result['checks']['nodata_consistency']
    assert not result['checks']['rasterio_mask_consistency']


def test_misalignment_reported(tmp_path):
    path,record=sample(tmp_path)
    record['exports']['nbrpre']['crsTransform'][2]+=30
    result,_,_=inspect_raster(path,record,'nbrpre')
    assert not result['checks']['transform']
