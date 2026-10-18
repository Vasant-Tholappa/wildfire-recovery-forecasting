"""Freeze a budget-sized whole-fire sample using local, existing forecast artifacts."""
import hashlib
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from manual_planb_eecu_forecast import scenarios

DATA = Path(__file__).resolve().parent / 'data/processed'
SEED = 'phase3_fire_sample_v1'
BUDGET_HOURS = 700
RETRY_MULTIPLIER = 1.20


def main():
    manifest_path = DATA / 'fire_manifest.parquet'
    occupancy_path = DATA / 'phase3_planb_tile120_batch_patch_counts.parquet'
    forecast_path = DATA / 'phase3_planb_eecu_forecast_combined_daymet_candidate_a.json'
    manifest = gpd.read_parquet(manifest_path)
    batches = pd.read_parquet(occupancy_path)
    forecast = json.loads(forecast_path.read_text())
    assert len(manifest) == 1904 and manifest.fire_id.is_unique and manifest.crs
    assert manifest.fire_year.between(2015, 2020).all()
    assert manifest.fire_size_acres.notna().all() and manifest.fire_size_acres.gt(0).all()
    assert len(batches) == 6301 and batches.patch_count.sum() == 15275928
    assert not batches.duplicated(['fire_id', 'batch_id']).any()
    assert set(batches.fire_id) == set(manifest.fire_id)
    assert forecast['candidate_a'] and forecast['combined_daymet']
    assert len(forecast['products']) == 12
    central = np.zeros(len(batches))
    planning = np.zeros(len(batches))
    for product in forecast['products']:
        prediction = scenarios(batches.patch_count.to_numpy(float), np.array(product['measured_seconds']))
        central += prediction['piecewise_power'] / 3600
        planning += prediction['small_batch_floor'] / 3600
    np.testing.assert_allclose(central.sum(), forecast['central_hours'])
    np.testing.assert_allclose(planning.sum(), forecast['planning_hours'])
    batches['central_hours'] = central
    batches['planning_hours'] = planning
    # The existing planning interpolant need not dominate a concave central segment.
    batches['budget_hours'] = np.maximum(central, planning)
    costs = batches.groupby('fire_id').agg(spatial_batches=('batch_id', 'size'),
        patches=('patch_count', 'sum'), central_hours=('central_hours', 'sum'),
        planning_hours=('planning_hours', 'sum'), budget_hours=('budget_hours', 'sum'))
    fires = manifest.drop(columns='geometry').merge(costs, on='fire_id', validate='one_to_one')
    fires['state_code'] = fires.fire_id.str[:2]  # Recorded event code, not a new geographic filter.
    fires['size_band'] = pd.cut(fires.fire_size_acres, [0, 5000, 20000, 100000, np.inf],
        right=False, labels=['under_5k', '5k_to_20k', '20k_to_100k', '100k_plus']).astype(str)
    fires['selection_hash'] = fires.fire_id.map(lambda fid: hashlib.sha256(f'{SEED}:{fid}'.encode()).hexdigest())
    groups = list(fires.groupby(['state_code', 'fire_year', 'size_band'], sort=True))
    ranked = [group.sort_values(['selection_hash', 'fire_id']) for _, group in groups]

    def sample(rate):
        return pd.concat([group.head(max(1, int(np.floor(rate * len(group))))) for group in ranked])

    selected = None
    rate = None
    for step in range(1001):
        candidate = sample(step / 1000)
        if candidate.budget_hours.sum() * RETRY_MULTIPLIER > BUDGET_HOURS:
            break
        selected, rate = candidate, step / 1000
    if selected is None:
        raise ValueError('Even one fire per stratum exceeds budget; no artifacts written')
    selected = selected.sort_values('fire_id')
    assert selected.fire_id.is_unique
    assert len(selected.groupby(['state_code', 'fire_year', 'size_band'])) == len(groups)
    assert set(selected.fire_year) == set(range(2015, 2021))
    assert set(selected.state_code) == set(fires.state_code)
    assert selected.budget_hours.sum() * RETRY_MULTIPLIER <= BUDGET_HOURS
    frozen = manifest.set_index('fire_id').loc[selected.fire_id].reset_index()
    assert len(frozen) == len(selected) and frozen.geometry.notna().all()
    prefix = DATA / 'phase3_selected_fire_population_v1'
    frozen.to_parquet(prefix.with_suffix('.parquet'), index=False)
    fires['selected'] = fires.fire_id.isin(selected.fire_id)
    fires.sort_values('fire_id').to_csv(str(prefix) + '_selection.csv', index=False)
    distribution = fires.groupby(['state_code', 'fire_year', 'size_band']).agg(
        available=('fire_id', 'size'), selected=('selected', 'sum')).reset_index()
    distribution.to_csv(str(prefix) + '_strata.csv', index=False)
    report = {
        'seed': SEED, 'sampling_fraction_parameter': rate, 'strata': len(groups),
        'fire_count': len(selected), 'spatial_batches': int(selected.spatial_batches.sum()),
        'patches': int(selected.patches.sum()), 'raster_tasks_after_combined_daymet': int(selected.spatial_batches.sum()) * 12,
        'central_hours': float(selected.central_hours.sum()),
        'planning_hours': float(selected.planning_hours.sum()),
        'budget_hours_before_retries': float(selected.budget_hours.sum()),
        'retry_multiplier': RETRY_MULTIPLIER,
        'planning_hours_including_retries': float(selected.budget_hours.sum()) * RETRY_MULTIPLIER,
        'year_counts': selected.fire_year.value_counts().sort_index().to_dict(),
        'state_counts': selected.state_code.value_counts().sort_index().to_dict(),
        'size_counts': selected.size_band.value_counts().sort_index().to_dict(),
        'split_counts': selected.split.value_counts().to_dict(),
        'largest_selected_acres': float(selected.fire_size_acres.max()),
        'inputs_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (manifest_path, occupancy_path, forecast_path)},
        'selected_ids_sha256': hashlib.sha256('\n'.join(selected.fire_id).encode()).hexdigest(),
        'assumptions': forecast['assumptions'] + [
            'Candidate A plus combined Daymet; combined Daymet promotion is a launch prerequisite.',
            'Budget uses max(central, planning) per batch plus 20% retry allowance; not a guaranteed cap.',
            'Fixed SHA256 rank within state-code/year/acreage strata; common fraction, minimum one per stratum.',
            'No per-fire cost ranking or removal. Sparse strata are oversampled; do not claim a self-weighted sample.',
            'State is the event-ID prefix; the original intersecting TX-coded event remains eligible.',
            'All canonical patches retained for each selected fire before later eligibility screening.',
        ]}
    prefix.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
