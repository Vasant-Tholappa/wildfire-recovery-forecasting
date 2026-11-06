"""Offline-only imagery geometry and pre-cutoff orchestration checks."""
from datetime import date
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from rasterio.windows import bounds, transform

from wildfire_recovery_forecasting import imagery, landsat
from wildfire_recovery_forecasting.fire_catalog import derive_fire_fields
from test_landsat import PixelImage


def rows(indices):
    return pd.DataFrame([dict(fire_id='fire', patch_id=f'fire_r{r}_c{c}',
        grid_row=r, grid_col=c, batch_id=f'r{r//120}_c{c//120}') for r,c in indices])


@pytest.mark.parametrize('r', [-121,-120,-1,0,1,119,120])
@pytest.mark.parametrize('c', [-121,-120,-1,0,1,119,120])
def test_exact_chip(r,c):
    data=rows([(r,c)])
    region=imagery.prepare_region(data,fire_id='fire',batch_id=data.batch_id.iloc[0])
    window=imagery.chip_window(r,c,region)
    assert (window.width,window.height)==(15,15)
    assert (region['width'],region['height'])==(15,15)
    assert region['transform'].e == -30
    x,y=imagery.patch_center(r,c)
    assert bounds(window,region['transform'])==(x-225,y-225,x+225,y+225)
    local=transform(window,region['transform'])
    assert local*(7.5,7.5)==(x,y)
    assert local*(6,6)==(90*c,90*(r+1))
    assert local*(9,9)==(90*(c+1),90*r)
    assert all(v%30==0 for v in region['bounds'])


def test_full_tile_buffer_and_distinct_specs():
    data=rows([(-120,-240),(-1,-121)])
    region=imagery.prepare_region(data,fire_id='fire',batch_id='r-1_c-2')
    assert (region['width'],region['height'])==(372,372)
    for r,c in zip(data.grid_row,data.grid_col): imagery.chip_window(r,c,region)
    specs=imagery.prepare_export_specs(region)
    assert set(specs)=={'nbrpre','season1'}
    for spec in specs.values():
        assert spec['crs']=='EPSG:5070' and len(spec['crsTransform'])==6
        assert 'scale' not in spec and spec['nodata']==-9999
        assert spec['source_reference'].startswith('UNRESOLVED')
    with pytest.raises(ValueError): imagery.chip_window(0,0,region)


def test_bad_identifiers_rejected():
    data=rows([(0,0)])
    for bad in [data.iloc[:0], pd.concat([data,data]), data.assign(patch_id='wrong'),
                data.assign(grid_row=0.5), data.assign(fire_id='other')]:
        with pytest.raises(ValueError): imagery.prepare_region(bad,fire_id='fire',batch_id='r0_c0')


@pytest.mark.parametrize('ignition,last',[(date(2017,9,30),2016),(date(2017,10,1),2017)])
def test_inputs_preserve_late_ignition_without_season5(ignition,last):
    fire=derive_fire_fields(ignition)
    fire.pop('season5_year')  # The input interface neither needs nor reads it.
    def season(year, geometry, reference_projection):
        return {'nbr':PixelImage([year/10000]),'obs_count':PixelImage([3])}
    with patch.object(landsat,'ee') as api, patch.object(landsat,'build_seasonal_nbr',side_effect=season) as build:
        api.Image.constant.side_effect=PixelImage
        api.ImageCollection.fromImages.return_value.median.side_effect=lambda:PixelImage(
            np.ma.median(np.ma.stack([x.values for x in api.ImageCollection.fromImages.call_args.args[0]]),axis=0))
        result=landsat.build_fire_inputs(fire,'buffered context',reference_projection=api.Projection())
        assert [c.args[0] for c in build.call_args_list]==list(range(last,last-5,-1))+[2018]
        assert set(result)=={'nbr_pre','nbr_1','nbr_pre_valid','prefire_selected_year_count','season1_obs_count'}
        np.testing.assert_allclose(result['nbr_pre'].values,[(last-1)/10000])
        assert all(c.args[1]=='buffered context' for c in build.call_args_list)
        api.Initialize.assert_not_called()
