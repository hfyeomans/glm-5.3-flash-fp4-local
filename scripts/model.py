#!/usr/bin/env python3
"""Locate and validate an offline NVIDIA FP4 snapshot without loading tensors."""
import hashlib
import json
import os
import sys
from pathlib import Path

MODEL = 'nvidia/GLM-5.3-Flash-FP4-Local'
REVISION = '76348ddb2be2e35e39ecfc00e75dc07703f4f3d5'
CONTAINER_HUB = Path('/root/.cache/huggingface/hub')


def inspect_model(hub, revision=REVISION, model_dir=None):
    hub = Path(hub).resolve()
    home = Path(os.environ.get('HF_HOME', str(hub.parent)))
    if model_dir:
        candidates = [Path(model_dir)]
    else:
        candidates = [hub / 'models--nvidia--GLM-5.3-Flash-FP4-Local' / 'snapshots' / revision,
                      home / 'nvidia/glm-5.3-flash-fp4-local',
                      home / 'nvidia/GLM-5.3-Flash-FP4-Local',
                      home / 'glm-5.3-flash-fp4-local',
                      hub / 'glm-5.3-flash-fp4-local']
    existing = [p for p in candidates if (p / 'config.json').is_file()]
    if not existing:
        raise ValueError('NVIDIA FP4 checkpoint not found. Checked: ' + ', '.join(map(str, candidates))
                         + '. Set MODEL_DIR to the complete local checkpoint. No weights will be downloaded.')
    if not model_dir and len(existing) > 1:
        raise ValueError('Multiple local copies found; select one explicitly with MODEL_DIR: ' + ', '.join(map(str, existing)))
    path = existing[0].resolve()
    config = json.loads((path / 'config.json').read_text())
    quant = config.get('quantization_config', {})
    if config.get('model_type') != 'glm5_next' or quant.get('quant_method') != 'modelopt':
        raise ValueError(f'{path}: expected NVIDIA glm5_next ModelOpt weights; another GLM format cannot substitute.')
    if 'csf_32x2_v1' not in json.dumps(quant):
        raise ValueError(f'{path}: missing NVIDIA compressed FP4 scale encoding csf_32x2_v1.')
    required = ['config.json', 'model.safetensors.index.json', 'tokenizer.json', 'tokenizer_config.json']
    for name in required:
        if not (path / name).is_file() or (path / name).stat().st_size == 0:
            raise ValueError(f'Missing or empty model file: {path / name}')
    index = json.loads((path / 'model.safetensors.index.json').read_text())
    shards = sorted(set(index.get('weight_map', {}).values()))
    if not shards:
        raise ValueError('The safetensors index contains no weights.')
    mount_source, mount_target = (hub, CONTAINER_HUB) if path.is_relative_to(hub) else (path, Path('/model'))
    total = 0
    for name in required + shards:
        file = path / name
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError(f'Unsafe shard path in index: {name}')
        if not file.is_file() or file.stat().st_size == 0:
            raise ValueError(f'Missing, empty or broken shard: {file}')
        if not file.resolve().is_relative_to(mount_source):
            raise ValueError(f'{file}: symlink escapes the container mount; use the original HF_HUB_CACHE instead of a detached snapshot.')
        if name in shards:
            total += file.stat().st_size
    expected = index.get('metadata', {}).get('total_size', 0)
    if total < expected:
        raise ValueError(f'Truncated checkpoint: {total} bytes on disk; index expects at least {expected}.')
    hashes = {name: hashlib.sha256((path / name).read_bytes()).hexdigest()
              for name in ['config.json', 'model.safetensors.index.json']}
    lock = json.loads((Path(__file__).resolve().parent.parent / 'config/model-lock.json').read_text())
    pinned = hashes == lock['metadata_sha256']
    if revision == REVISION and not pinned:
        raise ValueError('Model metadata differs from the pinned NVIDIA revision; inspect the checkpoint before changing MODEL_REVISION.')
    reported_revision = REVISION if pinned else revision if path.name == revision else 'local-directory-unverified-revision'
    return {'model_id': MODEL, 'revision': reported_revision,
            'path': str(path), 'mount_source': str(mount_source), 'mount_target': str(mount_target),
            'container_path': str(mount_target / path.relative_to(mount_source)),
            'shards': len(shards), 'bytes': total, 'metadata_sha256': hashes}


def main():
    try:
        result = inspect_model(os.environ.get('HF_HUB_CACHE', '/models_engrams/huggingface/hub'),
                               os.environ.get('MODEL_REVISION', REVISION), os.environ.get('MODEL_DIR') or None)
        print(json.dumps(result, indent=2))
    except (ValueError, OSError, KeyError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
