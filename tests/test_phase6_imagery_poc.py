"""POC configuration and export argument checks; no live EE API is used."""
from datetime import date
from unittest.mock import MagicMock, patch
import json

import pandas as pd
import pytest

import manual_phase6_imagery as poc
from wildfire_recovery_forecasting import imagery, landsat, pipeline
from wildfire_recovery_forecasting.fire_catalog import derive_fire_fields


def fixture():
    fire = dict(fire_id='fire', ignition_date=date(2017,10,9), **derive_fire_fields(date(2017,10,9)))
    rows = pd.DataFrame([dict(fire_id='fire', patch_id='fire_r0_c0', grid_row=0,
        grid_col=0, batch_id='r0_c0', fire_year=2017, split='train')])
    record = dict(fire_id='fire', batch_id='r0_c0', original_batch_patch_count=1,
        original_batch_keys_sha256=poc.key_digest(rows.patch_id),
        eligible_keys_sha256=poc.key_digest(rows.patch_id))
    return fire, rows, record


@pytest.mark.parametrize('year', [2017, 2018])
def test_adapter_uses_exact_reference_and_buffered_inputs_without_tasks(year):
    fire, rows, record = fixture()
    fire.update(ignition_date=date(year,10,9), **derive_fire_fields(date(year,10,9)))
    rows['fire_year']=year
    rows['split']=fire['split']
    api = MagicMock()
    with patch.dict('sys.modules', {'ee': api}), patch.object(pipeline,'patches_to_ee') as convert, \
            patch.object(landsat,'build_fire_inputs') as build:
        build.return_value = {'nbr_pre':MagicMock(), 'nbr_1':MagicMock()}
        args = poc.image_export_arguments(fire, rows, rows, frozen_record=record)
        exact = convert.return_value.geometry.return_value
        api.ImageCollection.return_value.filterBounds.assert_called_once_with(exact)
        api.ImageCollection.return_value.filterBounds.return_value.filterDate.assert_called_once_with(f'{year+1}-06-01',f'{year+1}-10-01')
        api.Geometry.Rectangle.assert_called_once_with([-180,-180,270,270],proj='EPSG:5070',geodesic=False)
        assert build.call_args.args[1] is api.Geometry.Rectangle.return_value
        assert set(args) == {'nbrpre','season1'}
        for product, arg in args.items():
            assert arg['crsTransform'] == [30,0,-180,0,-30,270]
            assert 'scale' not in arg and arg['maxPixels'] == 225
            assert arg['formatOptions'] == {'noData':-9999}
            assert arg['fileNamePrefix'].startswith('imagery/phase6_nbr_context_poc_v1/')
            image = build.return_value[{'nbrpre':'nbr_pre','season1':'nbr_1'}[product]].toFloat.return_value
            build.return_value[{'nbrpre':'nbr_pre','season1':'nbr_1'}[product]].toFloat.assert_called_once()
            image.mask.return_value.gt.return_value.unmask.return_value.rename.return_value.toFloat.assert_called_once()
            image.unmask.assert_called_once_with(-9999,sameFootprint=False)
            image.mask.return_value.gt.assert_called_once_with(0)
            image.mask.return_value.gt.return_value.unmask.assert_called_once_with(0,sameFootprint=False)
        api.Initialize.assert_not_called()
        api.batch.assert_not_called()
        assert not api.batch.mock_calls
        assert not api.data.mock_calls


def test_rejects_changed_membership_and_temporal_fields_before_graph():
    fire, rows, record = fixture()
    with patch.dict('sys.modules', {'ee':MagicMock()}), patch.object(pipeline,'patches_to_ee') as convert:
        with pytest.raises(ValueError):
            poc.image_export_arguments(fire,rows,rows,frozen_record={**record,'original_batch_keys_sha256':'bad'})
        with pytest.raises(ValueError):
            poc.validate_candidate({**fire,'season1_year':2022},rows,rows,1)
        with pytest.raises(ValueError):
            poc.validate_candidate(fire,rows,rows.assign(patch_id='not-frozen'),1)
        convert.assert_not_called()


def test_saved_poc_contract():
    saved=json.loads(poc.OUTPUT.read_text())
    assert [r['fire_id'] for r in saved['candidates']] == list(poc.CANDIDATES)
    assert [r['eligible_patch_count'] for r in saved['candidates']] == [104,215]
    assert saved['proposed_tasks']==4
    prefixes=[]
    for row in saved['candidates']:
        assert row['fire_year']<2020 and row['context_extends_outside_fire']
        assert set(row['exports'])=={'nbrpre','season1'}
        for spec in row['exports'].values():
            assert spec['crs']=='EPSG:5070'
            prefixes.append(spec['fileNamePrefix'])
    assert len(set(prefixes))==4
