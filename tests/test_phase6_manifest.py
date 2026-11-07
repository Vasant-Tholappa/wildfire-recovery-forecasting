"""Offline manifest geometry/integrity tests using synthetic keys only."""
from datetime import date
import copy
import hashlib
import json
from unittest.mock import patch

import pandas as pd
import pytest
import manual_phase6_manifest as m
from wildfire_recovery_forecasting.fire_catalog import derive_fire_fields


def fixture():
    fire=dict(fire_id='f',ignition_date=date(2017,10,9),**derive_fire_fields(date(2017,10,9)))
    rows=pd.DataFrame([dict(fire_id='f',patch_id=f'f_r{r}_c{c}',batch_id='r-1_c-1',
        grid_row=r,grid_col=c,fire_year=2017,split='train') for r,c in [(-120,-120),(-1,-1)]])
    batch=dict(fire_id='f',batch_id='r-1_c-1',patch_count=2)
    return fire,rows,batch


def test_deterministic_geometry_and_destinations():
    fire,rows,batch=fixture()
    a=m.fire_entry(fire,rows,batch);b=m.fire_entry(fire,rows.iloc[::-1],batch)
    assert json.dumps(a,sort_keys=True)==json.dumps(b,sort_keys=True)
    assert (a['width'],a['height'])==(372,372)
    assert a['transform']==[30,0,-10980,0,-30,180]
    assert a['bounds']==[-10980,-10980,180,180]
    assert len({p['planned_uri'] for p in a['products'].values()})==2
    assert all('imagery/phase6_nbr_context_v1/' in p['planned_uri'] for p in a['products'].values())
    assert set(a['products'])=={'nbrpre','season1'}
    assert a['reference_selection']['geometry'].startswith('Union of ALL')


@pytest.mark.parametrize('bad',['duplicate','count','association','split','key'])
def test_bad_population_rejected(bad):
    fire,rows,batch=fixture();expected={'train':(2,1)}
    if bad=='duplicate': rows=pd.concat([rows.iloc[:1],rows.iloc[:1]])
    elif bad=='count': expected={'train':(3,1)}
    elif bad=='association': rows['batch_id']='wrong'
    elif bad=='split': rows['fire_year']=2020
    else: rows.loc[0,'patch_id']='wrong'
    with pytest.raises(ValueError):
        m.validate_population(rows,pd.DataFrame([fire]),pd.DataFrame([batch]),expected)
        m.fire_entry(fire,rows,batch)


def test_population_valid_and_identifier_columns_only():
    fire,rows,batch=fixture()
    m.validate_population(rows,pd.DataFrame([fire]),pd.DataFrame([batch]),{'train':(2,1)})
    assert set(m.KEY_COLUMNS)=={'fire_id','patch_id','batch_id','grid_row','grid_col','fire_year','split'}
    # Check actual entrypoint's projected read without letting it process datasets.
    with patch.object(m.population,'load_population',return_value=(None,None)), \
         patch.object(m.pd,'read_parquet',side_effect=[None,RuntimeError('stop after projected read')]) as read:
        with pytest.raises(RuntimeError): m.generate()
        assert read.call_args.kwargs=={'columns':m.KEY_COLUMNS}


def test_reuse_requires_identity_and_verified_file(tmp_path):
    fire,rows,batch=fixture();entry=m.fire_entry(fire,rows,batch)
    record={k:entry[k] for k in ('fire_id','batch_id','eligible_patch_count','eligible_keys_sha256','width','height','original_batch_patch_count')}
    record['exports']=copy.deepcopy(entry['products'])
    state=dict(configuration=record,fire_id='f',version=m.poc.VERSION,products={},verified_references={'2017':{'scene_id':'verified'}})
    qa={'rasters':{}};pixel={'results':{}}
    for product,spec in record['exports'].items():
        spec['fileNamePrefix']=spec['fileNamePrefix'].replace(m.VERSION,m.poc.VERSION)
        (tmp_path/f'2017_{product}.tif').write_bytes(b'synthetic')
        uri=f"gs://{spec['bucket']}/{spec['fileNamePrefix']}.tif"
        state['products'][product]=dict(task_id=product,destination=uri[:-4])
        qa['rasters']['2017_'+product]=dict(checks={'geometry':True},source=uri,task_id=product,
            sha256=hashlib.sha256(b'synthetic').hexdigest(),generation=1)
        pixel['results']['2017_'+product]=dict(sample_count=1,mask_pass=1,value_pass=1,max_absolute_difference=0)
    m.reuse(entry,record,state,qa,pixel,tmp_path)
    assert all(p['action']=='REUSE' and m.poc.VERSION in p['effective_uri'] for p in entry['products'].values())
    changed=copy.deepcopy(entry);changed['eligible_keys_sha256']='wrong'
    with pytest.raises(ValueError): m.reuse(changed,record,state,qa,pixel,tmp_path)
    (tmp_path/'2017_nbrpre.tif').write_bytes(b'changed')
    with pytest.raises(ValueError): m.reuse(entry,record,state,qa,pixel,tmp_path)
