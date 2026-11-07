"""Population waves: synthetic cloud inventories only; no EE/GCS calls."""
import copy
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import pytest
import manual_phase6_production as l


@pytest.fixture
def world(tmp_path):
    entries=l.load_entries()
    state=l.population_state(tmp_path/'absent.json',entries)
    path=tmp_path/'population.json'
    objects=[SimpleNamespace(name=p['destination'].split('/',3)[3],generation=p['generation'],size=100)
             for p in state['products'].values() if p.get('accepted_qa')]
    tasks=[];starts=[];clock=[0.]
    client=MagicMock()
    client.list_blobs.side_effect=lambda bucket,prefix,**kw:[o for o in objects if o.name.startswith(prefix)]
    def task_status(ids): return [dict(t) for t in tasks if t['id'] in ids]
    def build(**kwargs):
        task=MagicMock(id=f'new{len(starts)}')
        def start():
            saved=json.loads(path.read_text())['products'][kwargs['description']]
            assert saved['status']=='INTENT' and saved['task_id']==task.id
            assert sum(t['state'] in l.safety.ACTIVE for t in tasks)<4
            tasks.append(dict(id=task.id,description=kwargs['description'],state='READY'))
            starts.append(kwargs)
        task.start.side_effect=start
        return task
    def sleep(seconds):
        clock[0]+=seconds
        for t in tasks:
            if t['state'] in l.safety.ACTIVE:
                t['state']='COMPLETED'
                p=state['products'].get(t['description'])
                if p: objects.append(SimpleNamespace(name=p['destination'].split('/',3)[3],generation='newgen',size=100))
    def arguments(entries,inputs,references):
        return {l.description(e['fire_id'],p):dict(description=l.description(e['fire_id'],p))
                for e in entries for p in l.safety.PRODUCTS}
    with patch.object(l,'population_state',side_effect=lambda path,es:l.read_state(path,es) if path.exists() else copy.deepcopy(state)), \
         patch.object(l,'STATE',tmp_path/'pilot.json'), \
         patch.object(l.poc.population,'load_population',return_value=(None,None)), \
         patch.object(l,'inputs_for',return_value=({},None,None,{})), \
         patch.object(l,'verify_reference',return_value=('projection',{'scene':'verified'})) as reference, \
         patch.object(l,'prepare_arguments',side_effect=arguments), \
         patch('ee.Initialize') as init,patch('ee.data.setMaxRetries'),patch('ee.data.setDeadline'), \
         patch('ee.data.getTaskList',side_effect=lambda:list(tasks)), \
         patch('ee.data.getTaskStatus',side_effect=task_status), \
         patch('google.cloud.storage.Client',return_value=client), \
         patch('ee.batch.Export.image.toCloudStorage',side_effect=build) as export, \
         patch.object(l.time,'monotonic',side_effect=lambda:clock[0]), \
         patch.object(l.time,'sleep',side_effect=sleep) as sleeping:
        yield SimpleNamespace(entries=entries,state=state,path=path,objects=objects,tasks=tasks,
                              starts=starts,clock=clock,client=client,reference=reference,init=init,
                              export=export,sleep=sleeping)


def test_frozen_population_and_accepted_provenance(tmp_path):
    entries=l.load_entries();s=l.population_state(tmp_path/'state.json',entries)
    assert len(s['products'])==302
    assert sum(p['status']=='COMPLETED' for p in s['products'].values())==8
    assert sum(p['status']=='PENDING' for p in s['products'].values())==294
    chosen,keys=l.wave_selection(entries,s,16)
    assert len(chosen)==8 and len(keys)==16
    assert {e['fire_year'] for e in chosen}==set(range(2015,2020))
    assert all(e['split']!='test' and e['fire_id'] not in l.PILOT for e in chosen)
    assert all(s['products'][k]['status']=='PENDING' for k in keys)
    assert l.wave_selection(list(reversed(entries)),s,16)==(chosen,keys)
    accepted=next(p for p in s['products'].values() if p.get('accepted_qa'))
    accepted['status']='PENDING';l.safety.save(tmp_path/'state.json',s)
    with pytest.raises(ValueError):l.population_state(tmp_path/'state.json',entries)
    with patch.object(l,'MANIFEST_SHA256','changed'):
        with pytest.raises(ValueError):l.load_entries()


def test_dry_run_never_cloud_or_state(world):
    l.population_run(path=world.path)
    world.init.assert_not_called();world.export.assert_not_called()
    assert not world.path.exists()


def test_bounded_wave_capacity_and_durable_records(world):
    s=l.population_run('submit',max_new=16,path=world.path)
    assert len(world.starts)==16 and world.sleep.call_count==3
    assert world.reference.call_count==8
    assert len({x['description'] for x in world.starts})==16
    assert sum(t['state']=='READY' for t in world.tasks)==4
    counts=l.population_report(s,16)
    assert counts['completed']==20 and counts['queued']==4 and counts['remaining']==282
    # Same cap is per invocation; max-new=1 cannot start a fresh two-channel fire.
    before=len(world.starts)
    l.population_run('submit',max_new=1,path=world.path)
    assert len(world.starts)==before


def test_time_limit_with_unrelated_active_tasks(world):
    world.tasks.extend(dict(id=f'other{i}',description=f'other{i}',state='RUNNING') for i in range(4))
    world.sleep.side_effect=lambda seconds:world.clock.__setitem__(0,world.clock[0]+seconds)
    l.population_run('submit',max_minutes=1,path=world.path)
    assert world.clock[0]==60 and not world.starts


def test_status_readonly_and_reconcile_persists(world):
    l.population_run('submit',max_new=2,path=world.path)
    world.sleep.side_effect(1)
    before=world.path.read_bytes()
    s=l.population_run('status',path=world.path)
    assert world.path.read_bytes()==before and len(world.starts)==2
    assert l.population_report(s)['completed']==10
    l.population_run('reconcile',path=world.path)
    assert world.path.read_bytes()!=before and len(world.starts)==2


@pytest.mark.parametrize('case',['object','task','missing_accepted','generation','shard','permission','failed','unknown'])
def test_conflicts_prevent_all_starts(world,case):
    _,keys=l.wave_selection(world.entries,world.state,2);k=keys[0];p=world.state['products'][k]
    name=p['destination'].split('/',3)[3]
    if case=='object':world.objects.append(SimpleNamespace(name=name,generation='1',size=10))
    elif case=='task':world.tasks.append(dict(id='unexpected',description=k,state='READY'))
    elif case=='missing_accepted':world.objects.pop()
    elif case=='generation':world.objects[0].generation='changed'
    elif case=='shard':world.objects.append(SimpleNamespace(name=world.objects[0].name[:-4]+'-000.tif',generation='1',size=10))
    elif case=='permission':world.client.list_blobs.side_effect=PermissionError('denied')
    else:
        p.update(status='SUBMITTED',task_id='known')
        world.tasks.append(dict(id='known',description=k,state='FAILED' if case=='failed' else 'UNKNOWN'))
    with pytest.raises((RuntimeError,PermissionError)):l.population_run('submit',path=world.path)
    assert not world.starts


@pytest.mark.parametrize('error',[KeyboardInterrupt(),RuntimeError('lost response')])
def test_partial_failure_reconcile_and_resume_without_duplicate(world,error):
    original=world.export.side_effect
    def build(**kwargs):
        task=original(**kwargs)
        start=task.start.side_effect
        if len(world.starts)==1:
            def ambiguous():start();raise error
            task.start.side_effect=ambiguous
        return task
    world.export.side_effect=build
    with pytest.raises(type(error)):l.population_run('submit',max_new=4,path=world.path)
    saved=json.loads(world.path.read_text())
    assert len(world.starts)==2 and any(p['status']=='AMBIGUOUS' for p in saved['products'].values())
    with pytest.raises(RuntimeError):l.population_run('submit',max_new=4,path=world.path)
    world.export.side_effect=original
    l.population_run('reconcile',path=world.path)
    l.population_run('submit',max_new=2,path=world.path)
    assert len(world.starts)==4 and len({s['description'] for s in world.starts})==4


def test_reference_failure_and_lock(world):
    world.reference.side_effect=ValueError('reference unavailable')
    with pytest.raises(ValueError):l.population_run('submit',path=world.path)
    assert not world.starts
    world.path.with_suffix('.lock').write_text('existing')
    with pytest.raises(FileExistsError):l.population_run('submit',path=world.path)


def test_deadline_after_reference_prevents_start(world):
    def reference(*args):world.clock[0]=100;return ('projection',{})
    world.reference.side_effect=reference
    l.population_run('submit',max_minutes=1,path=world.path)
    assert not world.starts


def test_completed_without_object_stops(world):
    _,keys=l.wave_selection(world.entries,world.state,2)
    world.state['products'][keys[0]].update(status='SUBMITTED',task_id='done')
    world.tasks.append(dict(id='done',description=keys[0],state='COMPLETED'))
    with pytest.raises(RuntimeError):l.population_run('submit',path=world.path)
    assert not world.starts


def test_explicit_test_inputs_and_key_only_loading(world):
    for e in world.entries:
        if e['split']!='test':
            for p in e['products']:world.state['products'][l.description(e['fire_id'],p)]['status']='COMPLETED'
    assert not l.wave_selection(world.entries,world.state,16)[1]
    assert len(l.wave_selection(world.entries,world.state,16,True)[1])==16
    assert 'nbr_5_mean' not in l.poc.KEY_COLUMNS
    assert set(l.poc.KEY_COLUMNS)<= {'fire_id','patch_id','fire_year','split','grid_row','grid_col','batch_id'}

def test_recorded_active_absent_from_task_list_still_counts(world):
    _,keys=l.wave_selection(world.entries,world.state,4)
    for i,k in enumerate(keys):world.state['products'][k].update(status='SUBMITTED',task_id=f'hidden{i}')
    hidden=[dict(id=f'hidden{i}',state='RUNNING') for i in range(4)]
    with patch('ee.data.getTaskStatus',return_value=hidden):
        world.sleep.side_effect=lambda seconds:world.clock.__setitem__(0,world.clock[0]+seconds)
        l.population_run('submit',max_minutes=1,path=world.path)
    assert not world.starts


def test_status_does_not_resolve_ambiguous_by_retry(world):
    _,keys=l.wave_selection(world.entries,world.state,2)
    world.state['products'][keys[0]].update(status='AMBIGUOUS',task_id=None)
    state=l.population_run('status',path=world.path)
    assert state['products'][keys[0]]['issue']
    assert not world.path.exists() and not world.starts


def test_atomic_save_preserves_previous_on_replace_failure(tmp_path):
    path=tmp_path/'state.json'
    l.safety.save(path,{'value':'old'})
    with patch.object(l.safety.os,'replace',side_effect=OSError('disk error')):
        with pytest.raises(OSError):l.safety.save(path,{'value':'new'})
    assert json.loads(path.read_text())=={'value':'old'}

def test_unrecorded_conflict_cannot_disappear_into_pending(world):
    _,keys=l.wave_selection(world.entries,world.state,2)
    p=world.state['products'][keys[0]]
    blob=SimpleNamespace(name=p['destination'].split('/',3)[3],generation='1',size=10)
    world.objects.append(blob)
    with pytest.raises(RuntimeError):l.population_run('submit',path=world.path)
    world.objects.remove(blob)
    state=l.population_run('reconcile',path=world.path)
    assert state['products'][keys[0]]['status']=='AMBIGUOUS'
    assert state['products'][keys[0]]['issue']
    with pytest.raises(RuntimeError):l.population_run('submit',path=world.path)
    assert not world.starts

def seed_original_wave(world):
    keys=[l.description(fid,p) for fid in l.ORIGINAL_WAVE_FIRES for p in ('nbrpre','season1')]
    for i,k in enumerate(keys[:11]):world.state['products'][k].update(status='COMPLETED',task_id=f'original{i}')
    l.safety.save(world.path,world.state)
    return keys


def test_recover_original_wave_and_exact_five(world):
    keys=seed_original_wave(world);before=world.path.read_bytes()
    wave=l.recover_original_wave(world.path)
    assert world.path.read_bytes()==before
    assert wave['product_ids']==keys and len(wave['task_ids'])==11
    state=l.read_state(world.path,world.entries)
    selected,pending=l.planned_wave(world.entries,state,5,False,world.path)
    assert pending==keys[11:] and len(selected)==3
    assert l.planned_wave(world.entries,state,3,False,world.path)[1]==keys[11:14]
    with pytest.raises(FileExistsError):l.recover_original_wave(world.path)


@pytest.mark.parametrize('problem',['id','status','issue'])
def test_recovery_rejects_discrepancy(world,problem):
    keys=seed_original_wave(world)
    state=l.read_state(world.path,world.entries)
    if problem=='id':state['products'][keys[0]]['task_id']=None
    elif problem=='status':state['products'][keys[11]]['status']='SUBMITTED'
    else:state['products'][keys[0]]['issue']='unresolved'
    l.safety.save(world.path,state)
    with pytest.raises(ValueError):l.recover_original_wave(world.path)
    assert not l.wave_path(world.path).exists()


def test_finished_wave_allows_next_and_dry_run_does_not_write(world):
    keys=seed_original_wave(world);l.recover_original_wave(world.path)
    state=l.read_state(world.path,world.entries)
    for i,k in enumerate(keys[11:]):state['products'][k].update(status='COMPLETED',task_id=f'last{i}')
    before=l.wave_path(world.path).read_bytes()
    _,next_keys=l.planned_wave(world.entries,state,16,False,world.path)
    assert len(next_keys)==16 and not set(keys).intersection(next_keys)
    assert l.wave_path(world.path).read_bytes()==before
    l.planned_wave(world.entries,state,16,False,world.path,persist=True)
    assert len(l.read_waves(world.path,state)['waves'])==2


def test_wave_saved_before_remote_start_and_resume_subset(world):
    original=world.export.side_effect
    def build(**kwargs):
        doc=json.loads(l.wave_path(world.path).read_text())
        assert kwargs['description'] in doc['waves'][0]['product_ids']
        assert doc['waves'][0]['expected_product_count']==4
        task=original(**kwargs)
        if len(world.starts)==1:task.start.side_effect=KeyboardInterrupt()
        return task
    world.export.side_effect=build
    with pytest.raises(KeyboardInterrupt):l.population_run('submit',max_new=4,path=world.path)
    state=l.read_state(world.path,world.entries)
    doc=l.read_waves(world.path,state)
    assert len(doc['waves'])==1
    assert l.planned_wave(world.entries,state,16,False,world.path)[1]==doc['waves'][0]['product_ids'][2:]


def test_persisted_test_wave_requires_explicit_flag(world):
    for e in world.entries:
        if e['split']!='test':
            for p in e['products']:world.state['products'][l.description(e['fire_id'],p)]['status']='COMPLETED'
    _,keys=l.planned_wave(world.entries,world.state,2,True,world.path,persist=True)
    assert len(keys)==2
    with pytest.raises(ValueError):l.planned_wave(world.entries,world.state,2,False,world.path)


def test_changed_wave_manifest_or_ids_rejected(world):
    l.planned_wave(world.entries,world.state,2,False,world.path,persist=True)
    doc=json.loads(l.wave_path(world.path).read_text());doc['waves'][0]['fire_ids'][0]='wrong'
    l.safety.save(l.wave_path(world.path),doc)
    with pytest.raises(ValueError):l.read_waves(world.path,world.state)

def sync_fixture(world):
    _,keys=l.wave_selection(world.entries,world.state,2)
    key=keys[0];item=world.state['products'][key]
    item.update(status='RUNNING',task_id='sync-task')
    task=dict(id='sync-task',description=key,state='RUNNING')
    world.tasks.append(task)
    blob=SimpleNamespace(name=item['destination'].split('/',3)[3],generation='sync-gen',size=100)
    world.objects.append(blob)
    return key,item,task,blob


def test_provisional_then_completed(world):
    import ee
    key,item,task,blob=sync_fixture(world)
    with patch.object(l.time,'time',return_value=100):
        assert l.reconcile_population(ee,world.client,world.state)==1
    assert item['status']=='RUNNING' and 'issue' not in item
    assert item['output_sync']['first_seen_epoch']==100
    task['state']='COMPLETED'
    l.reconcile_population(ee,world.client,world.state)
    assert item['status']=='COMPLETED' and 'output_sync' not in item and 'issue' not in item
    assert not world.starts


def test_grace_survives_resume_and_expires(world):
    import ee
    _,item,_,_=sync_fixture(world)
    with patch.object(l.time,'time',return_value=100):l.reconcile_population(ee,world.client,world.state,60)
    l.safety.save(world.path,world.state)
    state=l.read_state(world.path,world.entries)
    with patch.object(l.time,'time',return_value=161):l.reconcile_population(ee,world.client,state,60)
    assert any(p.get('issue')=='Output/status synchronization grace period exceeded' for p in state['products'].values())
    assert not world.starts


@pytest.mark.parametrize('case',['failed','cancelled','duplicate','changed','missing','identity'])
def test_provisional_does_not_hide_conflicts(world,case):
    import ee
    _,item,task,blob=sync_fixture(world)
    l.reconcile_population(ee,world.client,world.state)
    if case=='failed':task['state']='FAILED'
    elif case=='cancelled':task['state']='CANCELLED'
    elif case=='duplicate':world.objects.append(SimpleNamespace(name=blob.name+'extra',generation='2',size=1))
    elif case=='changed':blob.generation='changed'
    elif case=='missing':world.objects.remove(blob)
    else:task['id']='different'
    l.reconcile_population(ee,world.client,world.state)
    assert item.get('issue')
    assert not world.starts


def test_wave_continues_after_transient_sync_without_duplicates(world):
    key,item,task,blob=sync_fixture(world)
    # Keep a fixed two-product wave, one active and its pending partner.
    keys=[l.description(item['fire_id'],p) for p in ('nbrpre','season1')]
    wave=l.wave_record(keys,world.state);l.refresh_wave(wave,world.state)
    l.safety.save(l.wave_path(world.path),dict(version=l.VERSION,manifest_sha256=l.MANIFEST_SHA256,waves=[wave]))
    def sleep(seconds):
        world.clock[0]+=seconds
        task['state']='COMPLETED'
    world.sleep.side_effect=sleep
    state=l.population_run('submit',max_new=1,path=world.path)
    assert len(world.starts)==1 and world.starts[0]['description']==keys[1]
    assert state['products'][key]['status']=='COMPLETED'
    assert world.sleep.call_count==1


def test_persistent_sync_stops_bounded_monitoring(world):
    _,item,_,_=sync_fixture(world)
    world.sleep.side_effect=lambda seconds:world.clock.__setitem__(0,world.clock[0]+seconds)
    # No pending capacity needed: monitor an existing fully submitted wave.
    key=l.description(item['fire_id'],item['product'])
    wave=l.wave_record([key],world.state);l.refresh_wave(wave,world.state)
    l.safety.save(l.wave_path(world.path),dict(version=l.VERSION,manifest_sha256=l.MANIFEST_SHA256,waves=[wave]))
    with patch.object(l.time,'time',side_effect=lambda:world.clock[0]):
        with pytest.raises(RuntimeError):l.population_run('submit',max_minutes=5,sync_grace_seconds=60,path=world.path)
    assert world.clock[0]==60 and len(world.starts)==3
    assert all(s['description']!=key for s in world.starts)


def test_authorized_dryrun_uses_live_local_input_validation(world):
    # Select a real test fire without reading any target values.
    entry=next(e for e in world.entries if e['split']=='test')
    keys=[l.description(entry['fire_id'],p) for p in ('nbrpre','season1')]
    wave=l.wave_record(keys,world.state);l.refresh_wave(wave,world.state)
    l.safety.save(l.wave_path(world.path),dict(version=l.VERSION,manifest_sha256=l.MANIFEST_SHA256,waves=[wave]))
    with patch.object(l,'inputs_for',return_value=({},None,None,{})) as inputs:
        l.population_run(include_test=True,path=world.path)
        inputs.assert_called_once_with(entry,None,allow_test_inputs=True)
    world.init.assert_not_called();world.export.assert_not_called()
    with pytest.raises(ValueError,match='include-test-inputs'):l.population_run(path=world.path)


def test_authorized_live_path_propagates_to_adapter(world):
    entry=next(e for e in world.entries if e['split']=='test')
    keys=[l.description(entry['fire_id'],p) for p in ('nbrpre','season1')]
    wave=l.wave_record(keys,world.state);l.refresh_wave(wave,world.state)
    l.safety.save(l.wave_path(world.path),dict(version=l.VERSION,manifest_sha256=l.MANIFEST_SHA256,waves=[wave]))
    with patch.object(l,'inputs_for',return_value=({},None,None,{})) as inputs, \
         patch.object(l,'prepare_arguments',return_value={k:dict(description=k) for k in keys}) as adapter:
        l.population_run('submit',max_new=2,include_test=True,path=world.path)
        inputs.assert_called_once_with(entry,None,allow_test_inputs=True)
        assert adapter.call_args.kwargs=={'allow_test_inputs':True}
    assert len(world.starts)==2

def test_multiwave_global_cap_and_capacity(world):
    # Freeze four products, but submit only the first two before interruption.
    l.planned_wave(world.entries,world.state,4,False,world.path,persist=True)
    l.population_run('submit',max_new=2,path=world.path)
    first_ids={t['id'] for t in world.tasks}
    state=l.population_run('submit',max_new=6,path=world.path)
    assert len(world.starts)==8  # 2 earlier + exactly 6 NEW across two waves
    assert len({s['description'] for s in world.starts})==8
    assert first_ids <= {p['task_id'] for p in state['products'].values()}
    waves=l.read_waves(world.path,state)['waves']
    assert [w['expected_product_count'] for w in waves]==[4,4]
    assert world.sleep.call_count==1  # Counts the earlier tasks across the boundary.
    assert sum(t['state'] in l.safety.ACTIVE for t in world.tasks)==4


def test_multiwave_deadline_not_reset(world):
    l.planned_wave(world.entries,world.state,4,False,world.path,persist=True)
    l.population_run('submit',max_new=2,path=world.path)
    l.population_run('submit',max_new=6,max_minutes=1,path=world.path)
    assert world.clock[0]==60
    assert len(world.starts)==4  # Two new tasks; next wave cannot extend deadline.
    assert len(json.loads(l.wave_path(world.path).read_text())['waves'])==2


def test_multiwave_partial_fire_resume(world):
    l.planned_wave(world.entries,world.state,2,False,world.path,persist=True)
    original=world.export.side_effect
    def build(**kwargs):
        task=original(**kwargs)
        start=task.start.side_effect
        if not world.starts:
            def interrupted():start();raise KeyboardInterrupt()
            task.start.side_effect=interrupted
        return task
    world.export.side_effect=build
    with pytest.raises(KeyboardInterrupt):l.population_run('submit',max_new=4,path=world.path)
    old_id=world.tasks[0]['id']
    world.export.side_effect=original
    l.population_run('reconcile',path=world.path)
    state=l.population_run('submit',max_new=5,path=world.path)
    assert len(world.starts)==6 and len({s['description'] for s in world.starts})==6
    assert old_id in {p['task_id'] for p in state['products'].values()}
    assert [w['expected_product_count'] for w in l.read_waves(world.path,state)['waves']]==[2,4]


def test_atomic_save_transient_permission(tmp_path):
    path=tmp_path/'state.json'
    original=l.safety.os.replace
    calls=[]
    def replace(source,destination):
        calls.append(source)
        if len(calls)==1: raise PermissionError('locked')
        original(source,destination)
    with patch.object(l.safety.os,'replace',side_effect=replace), patch.object(l.safety.time,'sleep') as sleep:
        l.safety.save(path,{'task_id':'known'})
    assert len(calls)==2
    sleep.assert_called_once_with(0.1)
    assert json.loads(path.read_text())=={'task_id':'known'}


def test_atomic_save_persistent_permission(tmp_path):
    path=tmp_path/'state.json'
    l.safety.save(path,{'value':'old'})
    with patch.object(l.safety.os,'replace',side_effect=PermissionError('locked')) as replace, patch.object(l.safety.time,'sleep') as sleep:
        with pytest.raises(l.safety.AtomicSaveError,match='Recovery evidence retained'):
            l.safety.save(path,{'task_id':'returned-id'})
    assert replace.call_count==5 and sleep.call_count==4
    assert json.loads(path.read_text())=={'value':'old'}
    assert json.loads(path.with_suffix('.tmp').read_text())=={'task_id':'returned-id'}


def test_post_start_save_failure_recovery_without_duplicate(world):
    import ee
    _,keys=l.wave_selection(world.entries,world.state,2)
    key=keys[0]
    original=l.safety.os.replace
    def replace(source,destination):
        if world.starts: raise PermissionError('locked')
        original(source,destination)
    with patch.object(l.safety.os,'replace',side_effect=replace), patch.object(l.safety.time,'sleep'):
        with pytest.raises(l.safety.AtomicSaveError):
            l.submit_one(ee,key,world.state['products'][key],{'description':key},world.state,world.path)
    saved=json.loads(world.path.read_text())['products'][key]
    evidence=json.loads(world.path.with_suffix('.tmp').read_text())['products'][key]
    assert saved['status']=='INTENT' and evidence['status']=='SUBMITTED'
    assert saved['task_id']==evidence['task_id']=='new0'
    assert len(world.starts)==1
    l.population_run('reconcile',path=world.path)
    l.population_run('submit',max_new=1,path=world.path)
    assert len(world.starts)==2
    assert [s['description'] for s in world.starts].count(key)==1
