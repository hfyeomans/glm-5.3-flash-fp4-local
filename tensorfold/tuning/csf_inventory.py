#!/usr/bin/env python3
"""Read-only TP2 CSF32 storage accounting for the pinned NVIDIA checkpoint; no GPU work."""
import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import mmap
import os
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[2]
PARTS = ('bases', 'codes', 'page_bases', 'offsets', 'exceptions')


def exception_shares(kind, rows, cols, pages, offsets, total_bytes):
    """Count bytes in aligned TP2 rectangles without decoding scales; only these geometries are admitted."""
    if total_bytes % 2 or rows % 128 or cols not in (128, 256):
        raise ValueError('Unsupported CSF32 geometry/record extent')
    page_count = (rows // 128) * (cols // 4) // 32
    if len(pages) != 4 * page_count or len(offsets) != 2 * (rows // 32 * (cols // 2) + page_count):
        raise ValueError('Invalid index extents')
    # Validate page origins against terminal prefix counts, in their different units.
    consumed = 0
    for page in range(page_count):
        origin = struct.unpack_from('<I', pages, 4 * page)[0]
        start, end = [struct.unpack_from('<H', offsets, 2 * (page * 257 + i))[0] for i in (0, 256)]
        if origin != consumed or start != 0:
            raise ValueError('Invalid page origin or initial prefix')
        consumed += 2 * end
    if consumed != total_bytes:
        raise ValueError('Exception extent disagrees with page totals')
    if kind in ('gate_proj', 'up_proj'):
        if rows % 256 or page_count % 2:
            raise ValueError('Row cut must end at a complete page')
        first = struct.unpack_from('<I', pages, 4 * (page_count // 2))[0]
    elif kind == 'down_proj' and cols == 128:
        # Each 128-row tile spans exactly one 256-block page. The TP cut is after 128 blocks.
        first = 0
        counted = 0
        for page in range(page_count):
            origin = page * 257
            lo, middle, end = [struct.unpack_from('<H', offsets, 2 * (origin + i))[0] for i in (0, 128, 256)]
            if lo != 0 or not lo <= middle <= end:
                raise ValueError('Invalid page prefixes')
            first += 2 * middle
            counted += 2 * end
        if counted != total_bytes:
            raise ValueError('Exception extent disagrees with page totals')
    else:
        raise ValueError('Unsupported TP cut')
    if first % 2 or not 0 <= first <= total_bytes:
        raise ValueError('Invalid exception boundary')
    return [first, total_bytes - first]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('locked_model', ROOT / 'scripts/model.py')
    model = importlib.util.module_from_spec(spec); spec.loader.exec_module(model)
    cache = Path(os.environ.get('HF_CACHE_ROOT', '/models_engrams/huggingface'))
    identity = model.inspect_model(cache / 'hub', model.REVISION, os.environ.get('MODEL_DIR') or None)
    snapshot = Path(identity['path'])
    index = json.loads((snapshot / 'model.safetensors.index.json').read_text())['weight_map']
    files, maps, headers, starts = {}, {}, {}, {}
    try:
        for shard in sorted(set(index.values())):
            f = (snapshot / shard).open('rb'); files[shard] = f
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ); maps[shard] = mm
            length = struct.unpack_from('<Q', mm, 0)[0]
            if not 0 < length < 128 * 1024 * 1024 or 8 + length > len(mm):
                raise ValueError('Invalid safetensors header')
            headers[shard] = json.loads(mm[8:8 + length]); starts[shard] = 8 + length

        def metadata(name):
            shard = index[name]; h = headers[shard][name]
            lo, hi = h['data_offsets']
            if not 0 <= lo <= hi <= len(maps[shard]) - starts[shard]:
                raise ValueError('Invalid tensor extent')
            return h, hi - lo

        def read(name):
            shard = index[name]; h, _ = metadata(name); lo, hi = h['data_offsets']
            return maps[shard][starts[shard] + lo:starts[shard] + hi]

        totals = Counter(); ranks = [Counter(), Counter()]; shapes = Counter(); layers = defaultdict(Counter)
        prefixes = sorted(n[:-len('.weight_scale.csf32_bases')] for n in index if n.endswith('.weight_scale.csf32_bases'))
        for prefix in prefixes:
            w, weight_bytes = metadata(prefix + '.weight')
            rows, packed_k = w['shape']; cols = packed_k * 2 // 16; kind = prefix.rsplit('.', 1)[1]
            if w['dtype'] != 'U8' or (rows, cols, kind) not in ((2048, 256, 'gate_proj'), (2048, 256, 'up_proj'), (4096, 128, 'down_proj')):
                raise ValueError('Unexpected checkpoint projection shape')
            if weight_bytes != rows * packed_k:
                raise ValueError('Weight extent disagrees with shape')
            shapes[f'{kind}:{rows}x{cols}'] += 1
            sizes = {}
            expected = {'bases': rows, 'codes': rows * cols // 2,
                        'page_bases': 4 * (rows // 128) * (cols // 4) // 32,
                        'offsets': 2 * (rows // 32 * (cols // 2) + (rows // 128) * (cols // 4) // 32)}
            for part in PARTS:
                h, size = metadata(prefix + '.weight_scale.csf32_' + part)
                if h['dtype'] != {'page_bases': 'U32', 'offsets': 'U16', 'exceptions': 'U16'}.get(part, 'U8'):
                    raise ValueError('Unexpected component dtype')
                if part in expected and size != expected[part]:
                    raise ValueError('Unexpected component size')
                sizes[part] = size; totals[part] += size
            split = exception_shares(kind, rows, cols, read(prefix + '.weight_scale.csf32_page_bases'),
                                     read(prefix + '.weight_scale.csf32_offsets'), sizes['exceptions'])
            totals['decoded_scales'] += rows * cols
            totals['weight_codes'] += weight_bytes
            layer = int(prefix.split('.layers.')[1].split('.')[0])
            layers[layer]['gate_up' if kind != 'down_proj' else 'down'] += rows * cols // 2
            for rank in (0, 1):
                ranks[rank]['decoded_scales'] += rows * cols // 2
                ranks[rank]['weight_codes'] += weight_bytes // 2
                for part, size in sizes.items():
                    ranks[rank][part] += split[rank] if part == 'exceptions' else size if part == 'bases' and kind == 'down_proj' else size // 2
        if len(prefixes) != 36288 or set(layers) != set(range(3, 45)):
            raise ValueError('Checkpoint coverage differs from the pinned inventory')
        if totals['decoded_scales'] != 19025362944 or sum(totals[k] for k in PARTS) != 10321055136:
            raise ValueError('Scale inventory differs from the pinned checkpoint')
        scratch_up = max(v['gate_up'] for v in layers.values())
        scratch_down = max(v['down'] for v in layers.values())
        scenarios = {'separate_up_down': scratch_up + scratch_down, 'aliased_serial': max(scratch_up, scratch_down),
                     'two_full_layer_sets': 2 * (scratch_up + scratch_down),
                     'full_interleaved_layer_copy': 9 * (scratch_up + scratch_down)}
        for rank, r in enumerate(ranks):
            r['compressed_scales'] = sum(r[k] for k in PARTS)
            r['gross_scale_saving'] = r['decoded_scales'] - r['compressed_scales']
        output = {'model_id': identity['model_id'], 'revision': identity['revision'],
                  'metadata_sha256': identity['metadata_sha256'],
                  'audit_client_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'matrices': len(prefixes), 'layers': len(layers), 'shapes': dict(shapes),
                  'checkpoint_bytes': dict(totals), 'tp2_rank_bytes': [dict(r) for r in ranks],
                  'scratch_bytes_per_rank': scenarios, 'projected_scenarios': {},
                  'projection_inputs': {'baseline_context': 524288, 'target_context': 1048576,
                                        'baseline_free_MiB_by_rank': [1547, 1700],
                                        'incremental_bytes_per_capacity_token_per_rank': 7500,
                                        'desired_free_GiB': 1.5,
                                        'evidence': ['tensorfold/results/host-embedding-capacity-v1.json',
                                                     'tensorfold/TUNING-RESULTS.md']},
                  'assumptions': ['Preserve compact u16 page prefixes and u32 page origins after aligned TP2 slicing.',
                                  'Down-projection row bases are duplicated across ranks; exceptions partition by actual page prefixes.',
                                  'New packed weight codes remove the old persistent scale slots; no persistent expanded scale copy remains.',
                                  'Scratch reuse requires proven stream ownership; these are estimates, not implemented allocations.',
                                  'Excludes new metadata, allocator padding, launch/graph state and retained loading staging.',
                                  'Reads headers and scale indexes only; not a complete codec-integrity or GPU test.']}
        for name, scratch in scenarios.items():
            output['projected_scenarios'][name] = []
            for rank, r in enumerate(ranks):
                saving = r['gross_scale_saving'] - scratch
                after = [1547, 1700][rank] * 2**20 + saving - 524288 * 7500
                output['projected_scenarios'][name].append({'rank': rank, 'net_saving_GiB': saving / 2**30,
                                                           'projected_free_at_1m_GiB': after / 2**30})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as f: json.dump(output, f, indent=2); f.write('\n')
        print(json.dumps({'matrices': len(prefixes), 'scenarios': output['projected_scenarios']}, indent=2))
    finally:
        for mm in maps.values(): mm.close()
        for f in files.values(): f.close()


if __name__ == '__main__':
    main()
