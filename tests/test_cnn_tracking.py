"""Tracking checks mock fitting: no training or frozen dataset access."""
import json
import sys
from unittest.mock import MagicMock

import pytest

import manual_phase6_train as cli


def test_tracking_completeness_and_distinct_trials(tmp_path, monkeypatch):
    tracker = MagicMock()
    tracker.active_run.return_value.info.run_id = "mock-run-id"
    monkeypatch.setitem(sys.modules, 'mlflow', tracker)
    monkeypatch.setattr(cli, 'ROOT', tmp_path)
    monkeypatch.setattr(cli, 'seed_everything', lambda *a: None)
    monkeypatch.setattr(cli, 'make_loaders', lambda *a: (None, None))
    dataset = MagicMock(side_effect=AssertionError('Frozen data must not be accessed'))
    monkeypatch.setattr(cli, 'PatchDataset', dataset)
    record = dict(epoch=1, seconds=2.5, train_loss=.03, train_mae=.12,
                  val_loss=.02, val_mae=.1, val_rmse=.14, val_r2=.6, val_equal_fire_mae=.11)

    def fake_fit(model, train, val, device, output, **kwargs):
        output.mkdir(parents=True)
        (output/'best_state.pt').write_bytes(b'synthetic checkpoint fixture, not a model')
        kwargs['log_epoch'](record)
        return dict(best_epoch=1, epochs_completed=1, stopped_early=False,
                    best_validation={k.removeprefix('val_'): v for k, v in record.items() if k.startswith('val_')})

    monkeypatch.setattr(cli, 'fit', fake_fit)
    for trial in (0, 1):
        monkeypatch.setattr(sys, 'argv', ['manual_phase6_train.py', '--model', 'fusion',
                          '--smoke-test', '--study-name', 'bounded-study', '--trial-number', str(trial)])
        cli.main()
    assert tracker.start_run.call_count == 2
    assert all('run_id' not in call.kwargs for call in tracker.start_run.call_args_list)
    dataset.assert_not_called()
    for i in (0, 2):
        call = tracker.log_metrics.call_args_list[i]
        assert call.kwargs['step'] == 1
        assert call.args[0] == {k:v for k,v in record.items() if k != 'epoch'}
        summary = tracker.log_metrics.call_args_list[i+1].args[0]
        assert summary['best_val_mae'] == .1
        assert summary['epoch_seconds_total'] == 2.5
        assert summary['epochs_completed'] == summary['best_epoch'] == 1
        assert summary['fit_seconds'] >= 0
    configs = list((tmp_path/'data/processed/phase6_training').glob('*/configuration.json'))
    assert len(configs) == 2
    for path in configs:
        cfg = json.loads(path.read_text())
        assert cfg['training_loss'] == 'MSE' and cfg['optimizer'] == 'AdamW'
        assert cfg['selection_metric'] == 'val_mae' and cfg['seed'] == 42
        assert (path.parent/'architecture.txt').is_file()
    tags = tracker.set_tag.call_args_list
    assert any(c.args == ('optuna.trial_number', '0') for c in tags)
    assert any(c.args == ('optuna.trial_number', '1') for c in tags)
    assert sum(c.args[0] == 'best_checkpoint_sha256' for c in tags) == 2
    assert tracker.log_artifacts.call_count == 2


@pytest.mark.parametrize('extra', [ ['--study-name','s'], ['--trial-number','0'],
                                  ['--study-name','s','--trial-number','-1'] ])
def test_incomplete_trial_identity_rejected_before_data_access(monkeypatch, extra):
    monkeypatch.setattr(sys, 'argv', ['manual_phase6_train.py','--model','cnn_only'] + extra)
    with pytest.raises(SystemExit):
        cli.main()


def test_run_resume_environment_rejected(monkeypatch):
    monkeypatch.setenv('MLFLOW_RUN_ID', 'historical-run')
    monkeypatch.setattr(sys, 'argv', ['manual_phase6_train.py', '--model', 'cnn_only'])
    with pytest.raises(SystemExit):
        cli.main()
