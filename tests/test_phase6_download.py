"""Small synthetic downloader tests; no cloud clients or population execution."""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import manual_phase6_download as d


@pytest.fixture
def example(tmp_path):
    source = tmp_path / 'bytes'
    source.write_bytes(b'synthetic raster bytes')
    payload = source.read_bytes()
    blob = SimpleNamespace(name='imagery/poc/fire/batch/nbrpre.tif', generation='123',
                           size=len(payload), crc32c=d.crc32c(source))
    blob.download_to_file = MagicMock(side_effect=lambda handle, **kw: handle.write(payload))
    client = MagicMock()
    client.list_blobs.return_value = [blob]
    uri = 'gs://bucket/' + blob.name
    item = dict(uri=uri, bucket='bucket', name=blob.name,
                path=d.local_path(tmp_path/'downloads', uri), spec={},
                entry={'products': {}}, channel='nbrpre')
    with patch.object(d, 'inspect_raster', return_value=({'checks': {'synthetic': True}, 'sha256': 'hash'}, None, None)):
        yield client, blob, item, payload


def test_mapping_reuses_qa_and_rejects_escape(tmp_path):
    assert d.local_path(tmp_path, 'gs://bucket/imagery/poc/f/b/nbrpre.tif') == tmp_path/'bucket/imagery/poc/f/b/nbrpre.tif'
    with pytest.raises(ValueError):
        d.local_path(tmp_path, 'gs://bucket/../escape.tif')


def test_partial_restart_then_skip_valid_completed(example):
    client, blob, item, payload = example
    part = item['path'].with_suffix('.tif.download.part')
    part.parent.mkdir(parents=True)
    part.write_bytes(b'partial')
    assert d.download_one(client, item)[0] == 'downloaded'
    assert item['path'].read_bytes() == payload and not part.exists()
    assert d.download_one(client, item)[0] == 'already_present'
    blob.download_to_file.assert_called_once()
    assert blob.download_to_file.call_args.kwargs['if_generation_match'] == 123


def test_complete_temp_finalizes_without_download_and_bad_final_preserved(example):
    client, blob, item, payload = example
    part = item['path'].with_suffix('.tif.download.part')
    part.parent.mkdir(parents=True)
    part.write_bytes(payload)
    assert d.download_one(client, item)[0] == 'downloaded'
    blob.download_to_file.assert_not_called()
    item['path'].write_bytes(b'conflict')
    with pytest.raises(ValueError, match='CRC32C'):
        d.download_one(client, item)
    assert item['path'].read_bytes() == b'conflict'
    blob.download_to_file.assert_not_called()


def test_missing_ambiguous_and_failed_download(example):
    client, blob, item, _ = example
    client.list_blobs.return_value = []
    assert d.download_one(client, item)[0] == 'missing'
    client.list_blobs.return_value = [blob, blob]
    with pytest.raises(ValueError, match='Ambiguous'):
        d.download_one(client, item)
    client.list_blobs.return_value = [blob]
    blob.download_to_file.side_effect = RuntimeError('interrupted')
    with pytest.raises(RuntimeError):
        d.download_one(client, item)
    assert not item['path'].exists()
    assert item['path'].with_suffix('.tif.download.part').exists()


def test_default_is_offline_no_writes(tmp_path, capsys):
    item = {'uri': 'gs://bucket/nbrpre.tif', 'path': tmp_path/'nbrpre.tif'}
    with patch.object(d, 'plan', return_value=({}, [item])), \
         patch('google.cloud.storage.Client') as client, patch.object(d, 'download_one') as download:
        assert d.run(tmp_path/'manifest.json', tmp_path/'absent') == 0
    client.assert_not_called()
    download.assert_not_called()
    assert not (tmp_path/'absent').exists()
    assert 'offline only' in capsys.readouterr().out
