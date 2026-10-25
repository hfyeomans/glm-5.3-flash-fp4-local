#!/usr/bin/env python3
"""Locate and validate an offline NVIDIA FP4 snapshot without loading tensors."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

MODEL = 'nvidia/GLM-5.3-Flash-FP4-Local'
REVISION = '76348ddb2be2e35e39ecfc00e75dc07703f4f3d5'
CONTAINER_HUB = Path('/root/.cache/huggingface/hub')


def mount_paths(hub, path, external_target='/model'):
    hub, path = Path(hub).resolve(), Path(path).resolve()
    source, target = (hub, CONTAINER_HUB) if path.is_relative_to(hub) else (path, Path(external_target))
    return {'path': str(path), 'mount_source': str(source), 'mount_target': str(target),
            'container_path': str(target / path.relative_to(source))}


def validate_mounted_file(file, source):
    """Require every link component to survive relocation of the bind mount."""
    if not file.resolve().is_relative_to(source):
        raise ValueError(f'{file}: symlink escapes the container mount; retain the complete Hub cache.')
    pending = list(file.relative_to(source).parts)
    current = source
    links = 0
    while pending:
        part = pending.pop(0)
        if part == '..':
            current = current.parent
            if not current.is_relative_to(source):
                raise ValueError(f'{file}: symlink escapes the container mount.')
            continue
        candidate = current / part
        if candidate.is_symlink():
            links += 1
            if links > 40:
                raise ValueError(f'{file}: too many symlink levels.')
            target = candidate.readlink()
            if target.is_absolute():
                raise ValueError(f'{file}: absolute symlink cannot survive the relocated container mount; use relative links.')
            pending = list(target.parts) + pending
        else:
            current = candidate


def shard_tensor_bytes(file, expected_names):
    """Check the shard header and exact data extent without reading tensor payloads."""
    size = file.stat().st_size
    with file.open('rb') as stream:
        prefix = stream.read(8)
        header_size = int.from_bytes(prefix, 'little')
        if len(prefix) != 8 or not 0 < header_size <= min(100_000_000, size - 8):
            raise ValueError(f'{file}: invalid safetensors header size.')
        header = json.loads(stream.read(header_size))
    if not isinstance(header, dict) or set(header) - {'__metadata__'} != expected_names:
        raise ValueError(f'{file}: shard tensors differ from the index.')
    offsets = []
    for name in expected_names:
        tensor = header[name]
        extent = tensor.get('data_offsets') if isinstance(tensor, dict) else None
        if (not isinstance(extent, list) or len(extent) != 2
                or any(type(n) is not int or n < 0 for n in extent) or extent[1] < extent[0]):
            raise ValueError(f'{file}: invalid tensor offsets for {name}.')
        offsets.append(extent)
    end = 0
    for start, stop in sorted(offsets):
        if start != end:
            raise ValueError(f'{file}: tensor data contains a gap or overlap.')
        end = stop
    if 8 + header_size + end != size:
        raise ValueError(f'Truncated or oversized shard: {file}; expected {8 + header_size + end} bytes, got {size}.')
    return end


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
    mounted = mount_paths(hub, path)
    mount_source = Path(mounted['mount_source'])
    total = tensor_bytes = 0
    for name in required + shards:
        file = path / name
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError(f'Unsafe shard path in index: {name}')
        if not file.is_file() or file.stat().st_size == 0:
            raise ValueError(f'Missing, empty or broken shard: {file}')
        validate_mounted_file(file, mount_source)
        if name in shards:
            total += file.stat().st_size
            tensor_bytes += shard_tensor_bytes(file, {key for key, shard in index['weight_map'].items() if shard == name})
    expected = index.get('metadata', {}).get('total_size', 0)
    if tensor_bytes != expected:
        raise ValueError(f'Tensor data size differs from index: {tensor_bytes} bytes; index expects {expected}.')
    hashes = {name: hashlib.sha256((path / name).read_bytes()).hexdigest()
              for name in ['config.json', 'model.safetensors.index.json']}
    lock = json.loads((Path(__file__).resolve().parent.parent / 'config/model-lock.json').read_text())
    pinned = hashes == lock['metadata_sha256']
    if revision == REVISION and not pinned:
        raise ValueError('Model metadata differs from the pinned NVIDIA revision; inspect the checkpoint before changing MODEL_REVISION.')
    # Metadata hashes identify the format, not the revision of the tensor payloads.
    reported_revision = revision if path.name == revision else 'local-directory-unverified-revision'
    return {'model_id': MODEL, 'revision': reported_revision,
            'requested_revision': revision, **mounted,
            'shards': len(shards), 'bytes': total, 'metadata_sha256': hashes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', action='store_true')
    args = parser.parse_args()
    try:
        hub = Path(os.environ.get('HF_HUB_CACHE', '/models_engrams/huggingface/hub'))
        revision = os.environ.get('MODEL_REVISION', REVISION)
        directory = os.environ.get('MODEL_DIR') or None
        result = (mount_paths(hub, directory or hub / 'models--nvidia--GLM-5.3-Flash-FP4-Local' / 'snapshots' / revision)
                  if args.plan else inspect_model(hub, revision, directory))
        print(json.dumps(result, indent=2))
    except (ValueError, OSError, KeyError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
