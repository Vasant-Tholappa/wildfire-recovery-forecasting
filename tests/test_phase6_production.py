"""Pilot production orchestration tests; all cloud calls mocked."""
import copy
import json
from unittest.mock import MagicMock, patch
import pytest
import manual_phase6_production as launch


@pytest.fixture
def env(tmp_path):
    entries=launch.load_pilot()
    def inputs(e,manifest): return {},None,None,e
    def args(*a,frozen_record,**kw):
        e=frozen_record
        return {p:dict(image=object(),region=object(),crs=s['crs'],crsTransform=s['crsTransform'],
            formatOptions={'noData':-9999},fileFormat='GeoTIFF',maxPixels=e['width']*e['height']) for p,s in e['products'].items()}
    with patch.object(launch,'inputs_for',side_effect=inputs), \
         patch.object(launch.poc.population,'load_population',return_value=(None,None)), \
         patch.object(launch,'verify_reference',return_value=('projection',{'crs':'verified'})) as ref, \
         patch.object(launch.poc,'image_export_arguments',side_effect=args), \
         patch('ee.Initialize') as init, patch('ee.data.setMaxRetries'), \
         patch('ee.data.getTaskList',return_value=[]) as listing, \
         patch('ee.data.getTaskStatus') as status, \
         patch('google.cloud.storage.Client') as storage, \
         patch('ee.batch.Export.image.toCloudStorage') as export:
        storage.return_value.list_blobs.return_value=[]
        tasks=[MagicMock(id=f'id{i}') for i in range(4)]
        export.side_effect=tasks
        yield tmp_path/'state.json',entries,ref,init,listing,status,storage,export,tasks


def remote_tasks(entries,number=4,state='COMPLETED'):
    return [dict(id=f'id{i}',description=key,state=state) for i,key in enumerate(launch.initial_state(entries)['products'])][:number]


def test_exact_pilot_and_reuse_protection():
    entries=launch.load_pilot()
    assert set(e['fire_id'] for e in entries)==set(launch.PILOT)
    assert sum(e['eligible_patch_count'] for e in entries)==4508
    assert len(launch.initial_state(entries)['products'])==4
    assert all(p['action']=='EXPORT' and launch.VERSION in p['effective_uri'] and 'poc' not in p['effective_uri'] for e in entries for p in e['products'].values())
    with patch.object(launch,'MANIFEST_SHA256','wrong'):
        with pytest.raises(ValueError):launch.load_pilot()


def test_dry_run_no_remote(env):
    path,_,ref,init,listing,status,storage,export,_=env
    launch.run(path=path)
    assert not path.exists()
    for mock in (ref,init,listing,status,storage,export):mock.assert_not_called()


def test_success_ids_before_next_start_and_no_resubmission(env):
    path,entries,ref,_,listing,_,_,export,tasks=env
    def before_second():
        state=json.loads(path.read_text())
        first=list(state['products'].values())[0]
        assert first['status']=='SUBMITTED' and first['task_id']=='id0'
        assert ref.call_count==2
    tasks[1].start.side_effect=before_second
    launch.run('submit',path)
    assert export.call_count==4
    listing.return_value=remote_tasks(entries)
    launch.run('submit',path)
    assert export.call_count==4
    assert len({c.kwargs['fileNamePrefix'] for c in export.call_args_list})==4


def test_reference_failure_zero_submissions(env):
    path,_,ref,_,_,_,_,export,_=env
    ref.side_effect=[('projection',{}),ValueError('bad reference')]
    with pytest.raises(ValueError):launch.run('submit',path)
    export.assert_not_called()


@pytest.mark.parametrize('case',['task','object','capacity','unknown','permission','lock'])
def test_gates(env,case):
    path,entries,_,_,listing,_,storage,export,_=env
    if case=='task':listing.return_value=remote_tasks(entries,1)
    elif case=='object':storage.return_value.list_blobs.return_value=[object()]
    elif case=='capacity':listing.return_value=[dict(id='other',description='other',state='RUNNING')]
    elif case=='unknown':listing.return_value=[dict(id='other',description='other',state='UNKNOWN')]
    elif case=='permission':storage.return_value.list_blobs.side_effect=PermissionError()
    else:path.with_suffix('.lock').write_text('existing controller')
    with pytest.raises((RuntimeError,PermissionError,FileExistsError)):launch.run('submit',path)
    export.assert_not_called()


@pytest.mark.parametrize('error',[RuntimeError('response lost'),KeyboardInterrupt()])
def test_partial_reconciliation_and_continuation(env,error):
    path,entries,_,_,listing,_,_,export,tasks=env
    tasks[1].start.side_effect=error
    with pytest.raises(type(error)):launch.run('submit',path)
    saved=json.loads(path.read_text());items=list(saved['products'].values())
    assert items[0]['task_id']=='id0' and items[1]['status']=='AMBIGUOUS'
    assert export.call_count==2
    listing.return_value=remote_tasks(entries,2)
    with pytest.raises(RuntimeError):launch.run('submit',path)
    launch.run('reconcile',path)
    assert export.call_count==2
    launch.run('submit',path)
    assert export.call_count==4
    assert list(json.loads(path.read_text())['products'].values())[0]['task_id']=='id0'


def test_unresolved_intent_never_retried(env):
    path,entries,_,_,_,_,_,export,_=env
    state=launch.initial_state(entries)
    next(iter(state['products'].values()))['status']='INTENT'
    launch.safety.save(path,state)
    with pytest.raises(RuntimeError):launch.run('reconcile',path)
    with pytest.raises(RuntimeError):launch.run('submit',path)
    export.assert_not_called()


def test_status_readonly(env):
    path,entries,_,_,_,status,storage,export,_=env
    state=launch.initial_state(entries)
    for i,p in enumerate(state['products'].values()):p.update(status='SUBMITTED',task_id=f'id{i}')
    launch.safety.save(path,state);before=path.read_bytes()
    status.return_value=remote_tasks(entries)
    launch.run('status',path)
    assert path.read_bytes()==before
    export.assert_not_called();storage.assert_not_called()


def test_reference_metadata_validation():
    info=dict(id='LANDSAT/LC08/C02/T1_L2/LC08_044033_20180601',bands=[dict(id='SR_B5',crs='EPSG:32610',crs_transform=[30,0,1,0,-30,2])])
    with patch.object(launch.poc,'reference_band') as band,patch('ee.Projection'):
        band.return_value.getInfo.return_value=info
        launch.verify_reference({'season1_year':2018},None)
        info['bands'][0]['crs_transform'][0]=90
        with pytest.raises(ValueError):launch.verify_reference({'season1_year':2018},None)


def test_reject_duplicate_local_ids(env):
    path,entries,*_=env
    state=launch.initial_state(entries)
    for p in state['products'].values():p.update(status='SUBMITTED',task_id='same')
    launch.safety.save(path,state)
    with pytest.raises(ValueError):launch.read_state(path,entries)
