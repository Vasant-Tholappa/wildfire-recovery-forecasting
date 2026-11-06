"""All cloud APIs are mocked; no actual submissions or status requests."""
import json
from unittest.mock import MagicMock, patch
import pytest
import manual_phase6_submit as launch


@pytest.fixture
def setup(tmp_path):
    records={str(r['fire_year']):r for r in json.loads(launch.poc.OUTPUT.read_text())['candidates']}
    def arguments(*args,frozen_record,reference_projection):
        r=frozen_record
        return {p:dict(image=object(),region=object(),bucket=s['bucket'],fileNamePrefix=s['fileNamePrefix'],
            crs=s['crs'],crsTransform=s['crsTransform'],fileFormat='GeoTIFF',formatOptions={'noData':-9999},
            maxPixels=r['width']*r['height'],description=f"{launch.poc.VERSION}_{r['fire_id']}_{p}") for p,s in r['exports'].items()}
    with patch.object(launch,'verify_reference',return_value=('verified-projection',{'scene_id':'mock'})), \
            patch.object(launch,'load_inputs',side_effect=lambda y:({},None,None,records[y])), \
            patch.object(launch.poc,'image_export_arguments',side_effect=arguments) as adapter, \
            patch('ee.Initialize') as initialize, patch('ee.data.setMaxRetries'), \
            patch('ee.data.getTaskList',return_value=[]) as listing, \
            patch('ee.data.getTaskStatus') as status, \
            patch('google.cloud.storage.Client') as storage, \
            patch('ee.batch.Export.image.toCloudStorage') as export:
        storage.return_value.list_blobs.return_value=[]
        tasks=[MagicMock(id=f'task-{i}') for i in range(4)]
        export.side_effect=tasks
        yield tmp_path,records,adapter,initialize,listing,status,storage,export,tasks


def test_default_dry_run(setup):
    path,_,adapter,initialize,listing,status,storage,export,_=setup
    result=launch.run(directory=path)
    assert set(result)=={'2017','2018'} and sum(len(s['products']) for s in result.values())==4
    assert not list(path.iterdir())
    for mock in (adapter,initialize,listing,status,storage,export): mock.assert_not_called()


@pytest.mark.parametrize('selection',['2017','2018','both'])
def test_success_and_durable_ids(setup,selection):
    path,records,_,_,_,status,_,export,tasks=setup
    years=launch.years_for(selection)
    def check_before_next():
        # The previous successful ID is already durable before the next start.
        result=[]
        for y in years:
            result.extend(json.loads(launch.record_path(y,path).read_text())['products'].values())
        assert sum(p['status']=='SUBMITTED' for p in result)>=1
    tasks[1].start.side_effect=check_before_next
    result=launch.run(selection=selection,submit=True,directory=path)
    assert export.call_count==2*len(years)
    for y in years:
        saved=json.loads(launch.record_path(y,path).read_text())
        assert saved['status']=='SUBMITTED'
        assert all(p['task_id'] for p in saved['products'].values())
    for call in export.call_args_list:
        arg=call.kwargs
        assert arg['maxPixels'] in (84*204,162*138)
        assert arg['crsTransform'] in [r['exports']['nbrpre']['crsTransform'] for r in records.values()]
    status.assert_not_called()
    with pytest.raises(FileExistsError): launch.run(selection=selection,submit=True,directory=path)
    assert export.call_count==2*len(years)


@pytest.mark.parametrize('state',['READY','RUNNING','COMPLETED','FAILED','CANCELLED'])
def test_duplicate_task(setup,state):
    path,_,_,_,listing,_,_,export,_=setup
    listing.return_value=[dict(description=f'{launch.poc.VERSION}_{launch.FIRE}_nbrpre',state=state)]
    with pytest.raises(RuntimeError): launch.run(selection='2017',submit=True,directory=path)
    export.assert_not_called()


@pytest.mark.parametrize('case',['object','unknown','permission','capacity'])
def test_fail_closed(setup,case):
    path,_,_,_,listing,_,storage,export,_=setup
    if case=='object': storage.return_value.list_blobs.return_value=[object()]
    elif case=='unknown': listing.return_value=[{'description':'other','state':'UNKNOWN'}]
    elif case=='capacity': listing.return_value=[{'description':'other','state':'RUNNING'}]
    else: storage.return_value.list_blobs.side_effect=PermissionError('denied')
    with pytest.raises((RuntimeError,PermissionError)): launch.run(selection='both',submit=True,directory=path)
    export.assert_not_called()


@pytest.mark.parametrize('error',[RuntimeError('lost response'),KeyboardInterrupt()])
def test_partial_submission_stops_and_preserves(setup,error):
    path,_,_,_,_,_,_,export,tasks=setup
    tasks[2].start.side_effect=error
    with pytest.raises(type(error)): launch.run(selection='both',submit=True,directory=path)
    assert export.call_count==3
    first=json.loads(launch.record_path('2017',path).read_text())
    second=json.loads(launch.record_path('2018',path).read_text())
    assert [p['task_id'] for p in first['products'].values()]==['task-0','task-1']
    assert second['products']['nbrpre']['status']=='AMBIGUOUS'
    assert second['products']['season1']['status']=='NOT_SUBMITTED'
    tasks[3].start.assert_not_called()


def test_selection_and_explicit_submit(setup):
    path,*_=setup
    for selection in (None,'2020','CA-arbitrary'):
        with pytest.raises(ValueError): launch.run(selection=selection,submit=True,directory=path)
    with pytest.raises(SystemExit): launch.main(['--submit'])


def test_status_read_only(setup,capsys):
    path,_,_,_,_,status,storage,export,_=setup
    launch.run(selection='both',submit=True,directory=path)
    before={p:p.read_bytes() for p in path.iterdir()}
    export.reset_mock();storage.reset_mock()
    status.return_value=[dict(id=f'task-{i}',state=s,error_message='failure detail' if s=='FAILED' else '')
        for i,s in enumerate(['READY','RUNNING','COMPLETED','FAILED'])]
    launch.check_status('both',directory=path)
    status.assert_called_once_with(['task-0','task-1','task-2','task-3'])
    assert 'failure detail' in capsys.readouterr().out
    assert before=={p:p.read_bytes() for p in path.iterdir()}
    export.assert_not_called();storage.assert_not_called()


def test_extra_product_rejected(setup):
    path,_,adapter,_,_,_,_,export,_=setup
    adapter.side_effect=lambda *a,**k:{'nbrpre':{},'season1':{},'nbr5':{}}
    with pytest.raises(ValueError): launch.run(selection='both',submit=True,directory=path)
    export.assert_not_called()

@pytest.mark.parametrize('failed_year',['2017','2018'])
def test_reference_failure_prevents_every_submission(setup,failed_year):
    path,_,adapter,_,_,_,_,export,_=setup
    def verify(year,*args):
        if year==failed_year: raise RuntimeError('Reference inaccessible or unresolved')
        return 'projection', {'scene_id':'verified'}
    with patch.object(launch,'verify_reference',side_effect=verify):
        with pytest.raises(RuntimeError): launch.run(selection='both',submit=True,directory=path)
    export.assert_not_called()
    adapter.assert_not_called()


def test_references_once_before_any_start(setup):
    path,_,adapter,_,_,_,_,export,tasks=setup
    with patch.object(launch,'verify_reference',side_effect=lambda y,*a:(f'projection-{y}',{'year':y})) as verify:
        def check_gate():
            assert verify.call_count==2
        tasks[0].start.side_effect=check_gate
        launch.run(selection='both',submit=True,directory=path)
        assert [c.args[0] for c in verify.call_args_list]==['2017','2018']
    assert [c.kwargs['reference_projection'] for c in adapter.call_args_list]==['projection-2017','projection-2018']
    assert json.loads(launch.record_path('2017',path).read_text())['verified_references']['2018']=={'year':'2018'}


@pytest.mark.parametrize('year',['2017','2018'])
@pytest.mark.parametrize('bad',['none','scene','crs','transform','resolution','nan',None])
def test_verify_reference_metadata(year,bad):
    scene,crs=launch.EXPECTED_REFERENCES[year]
    transform=[30,0,501285,0,-30,4423215]
    info={'id':f'LANDSAT/LC08/C02/T1_L2/{scene}',
          'bands':[{'id':'SR_B5','crs':crs,'crs_transform':transform}]}
    if bad=='none': info=None
    elif bad=='scene': info['id']='wrong'
    elif bad=='crs': info['bands'][0]['crs']='EPSG:4326'
    elif bad=='transform': info['bands'][0]['crs_transform']=[]
    elif bad=='resolution': transform[0]=90
    elif bad=='nan': transform[2]=float('nan')
    with patch.object(launch.poc,'reference_band') as band, patch('ee.Projection') as projection:
        band.return_value.getInfo.return_value=info
        if bad:
            with pytest.raises(ValueError): launch.verify_reference(year,{},None)
            projection.assert_not_called()
        else:
            result,metadata=launch.verify_reference(year,{},None)
            projection.assert_called_once_with(crs,transform=transform)
            assert result is projection.return_value and metadata['transform']==transform
        band.return_value.getInfo.assert_called_once()
