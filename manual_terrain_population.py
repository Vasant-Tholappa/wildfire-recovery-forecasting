"""Terrain-only recovery adapter for the existing bounded population controller."""
import argparse
import hashlib
import json
from datetime import datetime, timezone

import ee
import pandas as pd
from google.cloud import storage

import manual_population as pop
import manual_terrain_recovery as pilot
from wildfire_recovery_forecasting import config, production, validation

VERSION = "phase3_ca175_tile1_terrain_v2"
SNAPSHOT = pop.DATA / (VERSION + "_progress.json")
# Preserve original locations before configure() redirects the shared runner.
ORIGINAL_VERSION = pop.VERSION
ORIGINAL_REUSED_BATCHES = dict(pop.REUSED_BATCHES)


def compare_original(client, item, corrected_version, expected_ids):
    original_version = ORIGINAL_REUSED_BATCHES.get(
        (item['fire_id'], item['batch_id']), ORIGINAL_VERSION)
    if original_version == corrected_version:
        raise ValueError('Original and corrected terrain paths must differ')
    old = pilot.read_parquet(client, item, original_version)
    new = pilot.read_parquet(client, item, corrected_version)
    checks = pilot.compare_tables(old, new, item, expected_ids)
    return checks, pilot.paths(item, original_version)['parquet']


def inputs():
    manifest, original = pop.load_population()
    items = [i for i in original if i['source'] == 'terrain']
    reuse = {(i['fire_id'], i['batch_id']): pilot.VERSION
             for i in items if i['fire_id'] in pilot.FIRES}
    if len(items) != 175 or len(reuse) != 2 or sum(i['patch_count'] for i in items) != 308048:
        raise ValueError('Frozen terrain population mismatch')
    return manifest, items, reuse


def initial_five(items, reuse):
    return [i for i in items if (i['fire_id'], i['batch_id']) not in reuse][:5]


class TerrainPayload(pop.BatchProducts):
    def __init__(self, manifest):
        super().__init__(manifest)
        keys = pd.read_parquet(pop.DATA / 'phase3_ca175_tile1_v1_vector_qa/supplemental.parquet',
                               columns=['fire_id','patch_id'])
        validation.validate_patch_table(keys)
        self.expected_ids = keys.groupby('fire_id').patch_id.agg(list).to_dict()

    def prepare(self, item):
        ids = super().prepare(item)
        validation.validate_patch_table(self.patches, fire_id=item['fire_id'],
            expected_ids=self.expected_ids[item['fire_id']])
        return ids

    def __call__(self, item):
        ids = self.prepare(item)
        if self.products is None:
            self.products = pilot.terrain_product(self.fire, self.patches)
            if self.products['columns'] != pop.raw_columns('terrain', None):
                raise ValueError('Terrain schema changed')
        return self.products, ids


def configure(reuse):
    # This dedicated process reuses runner functions, not its original CLI/main.
    pop.VERSION = VERSION
    pop.REUSED_BATCHES = reuse
    pop.SNAPSHOT = SNAPSHOT


def preflight(manifest, items, reuse):
    # Reuse the already validated candidate key set; do not rebuild 175 geometries.
    source = pop.DATA / 'phase3_ca175_tile1_v1_vector_qa/supplemental.parquet'
    keys = pd.read_parquet(source, columns=['fire_id','patch_id'])
    validation.validate_patch_table(keys)
    if len(keys) != 308048 or set(keys.fire_id) != set(manifest.fire_id):
        raise ValueError('Supplemental candidate key population mismatch')
    entries = []
    for item in items:
        part = keys[keys.fire_id.eq(item['fire_id'])]
        if len(part) != item['patch_count']:
            raise ValueError('Candidate IDs disagree with frozen tile occupancy')
        coords = part.patch_id.str.extract(r'^'+item['fire_id']+r'_r(-?\d+)_c(-?\d+)$')
        if coords.isna().any().any():
            raise ValueError('Malformed canonical patch ID')
        coords = coords.astype(int)
        tile_ids = 'r'+(coords[0]//120).astype(str)+'_c'+(coords[1]//120).astype(str)
        if not tile_ids.eq(item['batch_id']).all():
            raise ValueError('Candidate patch outside selected tile')
        ids = part.patch_id.tolist()
        version = reuse.get((item['fire_id'], item['batch_id']), VERSION)
        entries.append({**item, **pilot.paths(item, version),
            'patch_ids_sha256': hashlib.sha256('\n'.join(sorted(ids)).encode()).hexdigest(),
            'version': version})
    if len({e['status_uri'] for e in entries}) != 175:
        raise ValueError('Duplicate status identities')
    report = dict(version=VERSION, project=config.GCP_PROJECT, bucket=config.GCS_BUCKET,
        fires=175, batches=175, products=175, reused=2, new_exports=173,
        candidate_patches=308048, initial_five=initial_five(items,reuse), items=entries,
        key_source=str(source), key_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        cloud_status='Unknown offline; status validates both reused COMPLETE records',
        budget='Current quota/billing headroom must be checked manually; no dollar cap enforcement')
    path = pop.DATA / (VERSION + '_preflight.json')
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='items'}, indent=2))
    print(f'Full paths and exact patch-ID hashes: {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('preflight','status','check','session','verify'))
    parser.add_argument('--initial-five', action='store_true', help='Fixed first five non-pilot tiles, not next five')
    parser.add_argument('--session-new-cap', type=int, default=0)
    parser.add_argument('--max-minutes', type=int, default=60)
    parser.add_argument('--poll-seconds', type=int, default=60)
    args = parser.parse_args()
    if not 0 <= args.session_new_cap <= 173 or args.max_minutes <= 0 or args.poll_seconds < 30:
        parser.error('Require cap 0..173, positive duration, polling >=30s')
    manifest, items, reuse = inputs()
    if args.mode == 'preflight':
        preflight(manifest,items,reuse)
        return
    payload = TerrainPayload(manifest)
    configure(reuse)
    selected = initial_five(items,reuse) if args.initial_five else items
    lock = pop.DATA / (VERSION + '_driver.lock')
    owns_lock = args.mode in ('check','session')
    if owns_lock:
        with lock.open('x'):
            pass
    try:
        client = storage.Client(project=config.GCP_PROJECT)
        ee.Initialize(project=config.GCP_PROJECT)
        cache = {}

        def step(limit):
            tasks = ee.data.getTaskList()
            records, active = pop.read_cloud_state(items,client,tasks,cache)
            # Validate every COMPLETE record, not just reused pilot records.
            for i in items:
                r = records.get(production.status_path(**pop.key(i)))
                if r and r['status'] == 'COMPLETE':
                    version = reuse.get((i['fire_id'],i['batch_id']),VERSION)
                    if (r['stage'] != 'COMPLETE' or r['row_count'] != i['patch_count']
                            or r['output_uri'] != pilot.paths(i,version)['parquet']):
                        raise ValueError('Completed terrain identity/row-count mismatch')
            report = pop.summarize(items,records,active,limit,4)
            report['proposed_next_wave'] = pop.proposed_wave(selected,records,active,limit,4)
            report.update(version=VERSION, checked_at=datetime.now(timezone.utc).isoformat(),
                          reused=2, initial_five=args.initial_five)
            ids = {r.get('ee_task_id') for r in records.values()}
            report['visible_eecu_seconds'] = sum(t.get('batch_eecu_usage_seconds',0) or 0
                for t in tasks if t.get('id') in ids)
            report['usage_note'] = 'Visible task usage only; not current monthly quota or dollar headroom'
            if args.mode in ('check','session'):
                if report['failed']:
                    raise RuntimeError('Failed terrain work requires review; no automatic retry')
                report['actions'] = pop.advance(selected,records,tasks,client,payload,
                    limit,4,4,reconcile=payload.reconciliation)
                report['counts_note'] = 'Pre-action counts; run status for final persisted counts'
            if args.mode == 'verify':
                verified = []
                for i in selected:
                    r = records.get(production.status_path(**pop.key(i)))
                    if not r or r['status'] != 'COMPLETE':
                        raise ValueError('Every selected terrain item must be COMPLETE before verify')
                    version = reuse.get((i['fire_id'],i['batch_id']),VERSION)
                    # Compare genuine original/recovery inputs with unchanged checks.
                    checks, original_uri = compare_original(client,i,version,payload.prepare(i))
                    verified.append(dict(fire_id=i['fire_id'],batch_id=i['batch_id'],
                        original_uri=original_uri, output_uri=r['output_uri'],
                        rows=checks['rows'],variables=checks['variables']))
                report['verified_parquets'] = verified
            return report

        if args.mode == 'session':
            pop.run_session(step,submission_cap=args.session_new_cap,
                poll_seconds=args.poll_seconds,max_minutes=args.max_minutes)
        else:
            report = step(0)
            out = SNAPSHOT if args.mode != 'verify' else pop.DATA/(VERSION+'_verified.json')
            out.write_text(json.dumps(report,indent=2))
            print(json.dumps(report,indent=2))
    finally:
        if owns_lock:
            lock.unlink()


if __name__ == '__main__':
    main()
