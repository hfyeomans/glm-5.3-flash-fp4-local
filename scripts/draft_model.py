#!/usr/bin/env python3
"""Resolve the pinned offline DFlash2 draft; never fetch weights or use GPUs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

LOCK_PATH = Path(__file__).resolve().parents[1] / 'config/dflash2-lock.json'
CONTAINER_HUB = Path('/root/.cache/huggingface/hub')


def plan_draft(hub, model_dir=None, lock=None):
    lock = lock or json.loads(LOCK_PATH.read_text())
    hub = Path(hub).resolve()
    path = (Path(model_dir) if model_dir else hub / ('models--' + lock['model_id'].replace('/', '--')) / 'snapshots' / lock['revision']).resolve()
    source, target = (hub, CONTAINER_HUB) if path.is_relative_to(hub) else (path, Path('/draft'))
    return {'model_id': lock['model_id'], 'revision': lock['revision'], 'license': lock['license'],
            'path': str(path), 'mount_source': str(source), 'mount_target': str(target),
            'container_path': str(target / path.relative_to(source))}


def inspect_draft(hub, model_dir=None, *, verify=False, lock=None):
    lock = lock or json.loads(LOCK_PATH.read_text())
    result = plan_draft(hub, model_dir, lock)
    path, source = Path(result['path']), Path(result['mount_source'])
    for name in ('config.json', lock['weights_file']):
        file = path / name
        if not file.is_file():
            raise ValueError(f'DFlash2 file missing: {file}. Select the pinned local draft with DFLASH2_MODEL_DIR; no automatic download.')
        if not file.resolve().is_relative_to(source):
            raise ValueError(f'{file}: symlink escapes draft mount; retain the complete Hub cache.')
    config = (path / 'config.json').read_bytes()
    if hashlib.sha256(config).hexdigest() != lock['config_sha256']:
        raise ValueError('DFlash2 metadata differs from the pinned revision; original DFlash/other drafts cannot substitute.')
    weights = path / lock['weights_file']
    if weights.stat().st_size != lock['weights_bytes']:
        raise ValueError('DFlash2 weights are truncated or differ from the pinned size.')
    with weights.open('rb') as stream:
        prefix = stream.read(8)
        n = int.from_bytes(prefix, 'little')
        if len(prefix) != 8 or not 0 < n <= 1 << 20:
            raise ValueError('Invalid DFlash2 safetensors header size.')
        header = stream.read(n)
    if hashlib.sha256(header).hexdigest() != lock['safetensors_header_sha256']:
        raise ValueError('DFlash2 tensor layout differs from the pinned BF16 checkpoint.')
    if verify:
        digest = hashlib.sha256()
        with weights.open('rb') as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
        if digest.hexdigest() != lock['weights_sha256']:
            raise ValueError('DFlash2 weight SHA256 mismatch.')
    result.update(config_sha256=lock['config_sha256'], weights_bytes=lock['weights_bytes'],
                  expected_weights_sha256=lock['weights_sha256'], weights_sha256_verified=verify,
                  header_sha256=lock['safetensors_header_sha256'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', action='store_true')
    parser.add_argument('--verify', action='store_true', help='Read the full draft and verify its pinned SHA256')
    args = parser.parse_args()
    hub = Path(os.environ.get('HF_CACHE_ROOT', '/models_engrams/huggingface')) / 'hub'
    directory = os.environ.get('DFLASH2_MODEL_DIR') or None
    try:
        result = plan_draft(hub, directory) if args.plan else inspect_draft(hub, directory, verify=args.verify)
        print(json.dumps(result, indent=2))
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f'ERROR: {exc}\n')


if __name__ == '__main__':
    main()
