"""Explicit local-only population imagery QA. Never downloads or contacts EE/GCS.

Local files must mirror effective_uri beneath --raster-root:
  ROOT/<bucket>/imagery/<version>/<fire>/<batch>/<channel>.tif
Includes reused POC paths. Run only after downloads finish; --output must be new.
"""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pandas as pd

from manual_phase6_inspect import inspect_raster, inspect_chips
from manual_phase6_imagery import KEY_COLUMNS, key_digest
from wildfire_recovery_forecasting import imagery

CHANNELS = ('nbrpre', 'season1')
FIELDS = [f'{p}_{part}' for p in CHANNELS for part in ('full', 'central')]
MANIFEST_SHA256 = '5e27951fe1bde2dcc22f60af3df44de62034676cc0421a016e23ce5960100634'
DATA = Path('data/processed')


def local_path(root, uri):
    parsed = urlsplit(uri)
    if (parsed.scheme != 'gs' or not parsed.netloc or parsed.query or parsed.fragment
            or any(p in ('.', '..') for p in parsed.path.split('/'))):
        raise ValueError('Invalid effective_uri')
    root = Path(root).resolve()
    path = (root / parsed.netloc / parsed.path.lstrip('/')).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Raster path escapes root')
    return path


def validate_keys(rows, manifest):
    if rows[KEY_COLUMNS].isna().any().any() or not rows.patch_id.is_unique:
        raise ValueError('Missing/duplicate frozen keys')
    entries = manifest['fires']
    if (len(rows) != 111257 or len(entries) != 151 or rows.fire_id.nunique() != 151
            or manifest['export_products'] != 302
            or set(rows.fire_id) != {e['fire_id'] for e in entries}
            or key_digest(rows.patch_id) != manifest['eligible_keys_sha256']):
        raise ValueError('Frozen population identity/count mismatch')
    actual = {s: {'patches': len(g), 'fires': g.fire_id.nunique()} for s, g in rows.groupby('split')}
    if actual != manifest['split_counts']:
        raise ValueError('Frozen split counts mismatch')
    for e in entries:
        group = rows[rows.fire_id == e['fire_id']]
        year = e['fire_year']
        split = 'train' if 2015 <= year <= 2017 else 'validation' if year in (2018, 2019) else 'test' if year == 2020 else None
        if (e['split'] != split or not group.split.eq(split).all()
                or not group.fire_year.eq(year).all() or not group.batch_id.eq(e['batch_id']).all()
                or len(group) != e['eligible_patch_count']
                or key_digest(group.patch_id) != e['eligible_keys_sha256']
                or set(e['products']) != set(CHANNELS)):
            raise ValueError('Frozen per-fire identity mismatch: ' + e['fire_id'])


def coverage_table(rows, region, arrays):
    table = inspect_chips(rows, region, arrays)
    for field in FIELDS:
        size = 225 if field.endswith('_full') else 9
        table[field + '_valid_count'] = np.rint(table[field] * size).astype(int)
        table[field + '_ge80'] = table[field] >= .8
    return table.merge(rows[['fire_id', 'patch_id', 'fire_year', 'split']],
                       on=['fire_id', 'patch_id'], validate='one_to_one')


def summarize(table):
    checked = table[table.qa_status == 'PASS']
    result = dict(expected_patches=len(table), checked_patches=len(checked),
                  unassessed_patches=len(table)-len(checked), fires=table.fire_id.nunique(),
                  joint_ge80=int(checked.joint_ge80.sum()),
                  joint_below80=int((~checked.joint_ge80.astype(bool)).sum()), coverage={})
    for field in FIELDS:
        values = checked[field].dropna()
        result['coverage'][field] = dict(
            ge80=int((values >= .8).sum()), below80=int((values < .8).sum()),
            zero=int((values == 0).sum()), any_missing=int((values < 1).sum()),
            minimum=float(values.min()) if len(values) else None,
            p25=float(values.quantile(.25)) if len(values) else None,
            median=float(values.median()) if len(values) else None,
            mean=float(values.mean()) if len(values) else None,
            maximum=float(values.max()) if len(values) else None)
    return result


def run(manifest_path, eligible, raster_root, output):
    # No submission records, controller imports, cloud clients, or target columns.
    raw = Path(manifest_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != MANIFEST_SHA256:
        raise ValueError('Frozen manifest checksum changed; stop for review')
    manifest = json.loads(raw)
    rows = pd.read_parquet(eligible, columns=KEY_COLUMNS)
    validate_keys(rows, manifest)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = dict(manifest_sha256=MANIFEST_SHA256, rasters={}, failures={},
                  note='Local QA only, not remote completion or source equivalence. '
                       '80% is diagnostic only; raw values are unchanged. '
                       'Coverage summaries exclude unassessed rows explicitly; no patches are filtered.')
    tables = []
    for entry in manifest['fires']:
        fid = entry['fire_id']
        group = rows[rows.fire_id == fid]
        try:
            region = imagery.prepare_region(group, fire_id=fid, batch_id=entry['batch_id'])
            if (list(region['bounds']) != entry['bounds']
                    or list(region['transform'])[:6] != entry['transform']
                    or (region['width'], region['height']) != (entry['width'], entry['height'])):
                raise ValueError('Reconstructed region differs from frozen manifest')
            arrays = {}
            errors = []
            for channel in CHANNELS:
                spec = entry['products'][channel]
                try:
                    path = local_path(raster_root, spec['effective_uri'])
                    result, nbr, valid = inspect_raster(path, {**entry, 'exports': entry['products']}, channel)
                    result['nonfinite_valid_pixels'] = int((~np.isfinite(nbr[valid])).sum())
                    for field in ('valid_min', 'valid_max'):
                        if result[field] is not None and not np.isfinite(result[field]):
                            result[field] = None
                    result['source'] = spec['effective_uri']
                    result['checks']['reuse_hash'] = not spec.get('reuse_sha256') or result['sha256'] == spec['reuse_sha256']
                    report['rasters'][fid + '_' + channel] = result
                    failed = [k for k, ok in result['checks'].items() if not ok]
                    if failed:
                        raise ValueError('Failed checks: ' + ', '.join(failed))
                    arrays[channel] = (nbr, valid)
                except (OSError, ValueError) as error:
                    errors.append(channel + ': ' + str(error))
            if errors:
                raise ValueError('; '.join(errors))
            table = coverage_table(group, region, arrays)
            table['qa_status'] = 'PASS'
        except (OSError, ValueError) as error:
            report['failures'][fid] = str(error)
            table = group.copy()
            table['qa_status'] = 'FAIL'
            table['qa_error'] = str(error)
            for field in FIELDS:
                table[field] = np.nan
            table['joint_ge80'] = False
        tables.append(table)
    combined = pd.concat(tables, ignore_index=True)
    assert len(combined) == len(rows) and combined.patch_id.is_unique
    report['population'] = summarize(combined)
    report['groups'] = {field: {str(key): summarize(group) for key, group in combined.groupby(field)}
                        for field in ('fire_id', 'fire_year', 'split')}
    report['raster_totals'] = {field: sum(r[field] for r in report['rasters'].values())
                               for field in ('valid_pixels', 'invalid_pixels', 'valid_zero_pixels',
                                             'outside_minus1_plus1', 'nonfinite_valid_pixels')}
    report['rasters_inspected'] = len(report['rasters'])
    report['rasters_passed'] = sum(all(r['checks'].values()) for r in report['rasters'].values())
    report['verdict'] = 'FAIL' if report['failures'] else 'PASS'
    combined.to_csv(output / 'patch_coverage.csv', index=False)
    (output / 'qa_summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: report[k] for k in ('verdict', 'rasters_inspected', 'rasters_passed', 'population')}, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-local-qa', action='store_true', required=True)
    parser.add_argument('--manifest', type=Path, default=DATA / 'phase6_nbr_context_v1_manifest.json')
    parser.add_argument('--eligible', type=Path, default=DATA / 'patch_model_eligible_terrain_v2.parquet')
    parser.add_argument('--raster-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=DATA / 'phase6_nbr_context_v1_population_local_qa_v1')
    args = parser.parse_args()
    result = run(args.manifest, args.eligible, args.raster_root, args.output)
    raise SystemExit(0 if result['verdict'] == 'PASS' else 1)
