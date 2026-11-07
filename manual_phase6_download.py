"""Offline plan by default; --download explicitly enables GCS reads only.

Files mirror manifest effective_uri beneath raster-root. Completed local files
are checked against remote CRC32C/size and raster metadata before being skipped.
Interrupted .download.part files restart individually, not at byte offsets.
No controller/state access, EE operations, patch/target reads, or cloud writes.
"""
import argparse
import base64
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import urlsplit

from manual_phase6_local_qa import DATA, MANIFEST_SHA256, local_path, inspect_raster


def plan(manifest_path, root):
    raw = Path(manifest_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != MANIFEST_SHA256:
        raise ValueError('Frozen manifest checksum mismatch')
    manifest = json.loads(raw)
    items = []
    for entry in manifest['fires']:
        if set(entry['products']) != {'nbrpre', 'season1'}:
            raise ValueError('Unexpected imagery products')
        for channel, spec in entry['products'].items():
            uri = spec['effective_uri']
            parsed = urlsplit(uri)
            path = local_path(root, uri)
            if not parsed.path.endswith('.tif') or parsed.netloc != spec['bucket']:
                raise ValueError('Invalid object identity')
            items.append(dict(entry=entry, channel=channel, spec=spec, uri=uri,
                              bucket=parsed.netloc, name=parsed.path.lstrip('/'), path=path))
    if (len(manifest['fires']) != 151 or len(items) != 302
            or len({i['uri'] for i in items}) != 302
            or len({i['path'] for i in items}) != 302):
        raise ValueError('Population count or destination collision')
    return manifest, items


def crc32c(path):
    import google_crc32c
    checksum = google_crc32c.Checksum()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            checksum.update(chunk)
    return base64.b64encode(checksum.digest()).decode('ascii')


def validate_file(path, blob, item):
    if not blob.crc32c or path.stat().st_size != int(blob.size) or crc32c(path) != blob.crc32c:
        raise ValueError('Local size/CRC32C differs from remote object')
    result, _, _ = inspect_raster(path, {**item['entry'], 'exports': item['entry']['products']}, item['channel'])
    if not all(result['checks'].values()):
        raise ValueError('Raster checks failed: ' + ', '.join(k for k, ok in result['checks'].items() if not ok))
    expected = item['spec'].get('reuse_sha256')
    if expected and result['sha256'] != expected:
        raise ValueError('Reused POC content changed')
    # Finite NBR outliers are diagnostics; never clip or reject them here.


def download_one(client, item):
    # Inspect the product prefix, refusing shards/extra matching objects.
    blobs = list(client.list_blobs(item['bucket'], prefix=item['name'][:-4],
                                  max_results=3, retry=None, timeout=30))
    if not blobs:
        return 'missing', 'Expected GCS object absent'
    if len(blobs) != 1 or blobs[0].name != item['name']:
        raise ValueError('Ambiguous/sharded/conflicting GCS object prefix')
    blob = blobs[0]
    if not blob.generation or not blob.size:
        raise ValueError('Missing object generation or empty object')
    expected = item['spec'].get('reuse_generation')
    if expected and str(blob.generation) != str(expected):
        raise ValueError('Reused POC generation changed')
    path = item['path']
    if path.exists():
        validate_file(path, blob, item)  # Invalid final files require manual review.
        return 'already_present', 'Verified; unchanged'
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tif.download.part')
    complete = False
    if temporary.exists():
        try:
            validate_file(temporary, blob, item)
            complete = True
        except (OSError, ValueError):
            pass  # Owned incomplete scratch file; restart this object only.
    if not complete:
        with temporary.open('wb') as handle:
            blob.download_to_file(handle, if_generation_match=int(blob.generation),
                                  checksum='crc32c', retry=None, timeout=120)
            handle.flush()
            os.fsync(handle.fileno())
        validate_file(temporary, blob, item)
    # Atomic no-clobber publication, on the same filesystem. Never replace final.
    os.link(temporary, path)
    temporary.unlink()
    return 'downloaded', 'Validated and finalized'


def run(manifest_path, root, download=False):
    manifest, items = plan(manifest_path, root)
    if not download:
        for item in items:
            print(f"PLAN {item['uri']} -> {item['path']}")
        print(f'{len(items)} objects; offline only, no local writes or cloud access.')
        return 0
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    lock = root / '.phase6_download.lock'
    # Separate downloader lock; never touch the production controller lock.
    with lock.open('x') as handle:
        handle.write(str(os.getpid()))
    counts = Counter(downloaded=0, already_present=0, missing=0, failed=0)
    try:
        from google.cloud import storage
        client = storage.Client(project=manifest['project'])
        for item in items:
            try:
                status, message = download_one(client, item)
            except Exception as error:
                status, message = 'failed', f'{type(error).__name__}: {error}'
            counts[status] += 1
            print(f"{status.upper()} {item['uri']} -> {item['path']}: {message}", flush=True)
            if status in ('missing', 'failed'):
                break  # Fail closed; don't continue on permissions or conflicts.
    finally:
        counts['not_attempted'] = len(items) - sum(counts.values())
        print(json.dumps(dict(counts)), flush=True)
        lock.unlink()
    return 1 if counts['missing'] or counts['failed'] else 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true', help='Explicitly enable GCS read/download operations')
    parser.add_argument('--manifest', type=Path, default=DATA / 'phase6_nbr_context_v1_manifest.json')
    parser.add_argument('--raster-root', type=Path, default=DATA / 'phase6_downloads')
    args = parser.parse_args()
    raise SystemExit(run(args.manifest, args.raster_root, args.download))
