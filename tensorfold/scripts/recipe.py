#!/usr/bin/env python3
"""Isolated native TensorFold lifecycle; never controls the vLLM container."""
import contextlib
import fcntl
import json
import os
from pathlib import Path
import re
import socket
import struct
import subprocess
import sys
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
COMMIT = '36415df4f76e4458e82d0fa729ecb28e287ebf38'
PR_COMMIT = '9b23b6a2e7ff226ca61934189c4cd57bf2933980'
MERGED_TREE = '1d52effd182d9319d775d5787904c36783e532f1'
LABEL = 'lab.glm53.tensorfold.root'
BINARY = '/opt/tensorfold/bin/tensorfold-native'
DEFAULTS = {
    'HF_CACHE_ROOT': '/models_engrams/huggingface', 'MODEL_DIR': '',
    'TF_IMAGE': 'glm53-fp4-tensorfold:1.1.3-pr1050-sm120',
    'TF_CONTAINER': 'glm53-fp4-tensorfold', 'TF_VOLUME': 'glm53-fp4-tensorfold-pr1050',
    'TF_GPUS': '0,1', 'TF_HOST': '0.0.0.0', 'TF_PORT': '8001', 'TF_COMM_PORT': '29600',
    'TF_CONTEXT': '8192', 'TF_PARALLEL': '1',
    'TF_MODEL_NAME': 'glm-5.3-flash-fp4-local', 'BUILD_JOBS': '4',
}


def config(path=None, environ=None):
    env = os.environ if environ is None else environ
    result = DEFAULTS.copy()
    path = path or HERE / '.env'
    if path.exists():
        for num, line in enumerate(path.read_text().splitlines(), 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            key, sep, value = line.partition('=')
            if not sep or key not in DEFAULTS:
                raise ValueError(f'{path}:{num}: expected an allowed KEY=VALUE (literal, unquoted)')
            result[key] = value
    result.update({k: env[k] for k in result if k in env})
    for key, low, high in [('TF_PORT', 1, 65535), ('TF_COMM_PORT', 1, 65534),
                           ('TF_CONTEXT', 256, 1048576), ('TF_PARALLEL', 1, 8),
                           ('BUILD_JOBS', 1, 32)]:
        if not result[key].isdigit() or not low <= int(result[key]) <= high:
            raise ValueError(f'{key} must be an integer from {low} to {high}')
    if int(result['TF_PORT']) in (int(result['TF_COMM_PORT']), int(result['TF_COMM_PORT']) + 1):
        raise ValueError('HTTP, NCCL and NCCL watchdog ports must be distinct')
    if not re.fullmatch(r'\d+,\d+', result['TF_GPUS']) or len(set(result['TF_GPUS'].split(','))) != 2:
        raise ValueError('TF_GPUS must name two distinct numeric GPU indices, such as 0,1')
    for k in ('TF_CONTAINER', 'TF_VOLUME'):
        if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.-]+', result[k]):
            raise ValueError(f'Invalid {k}')
    if result['TF_CONTAINER'] == 'glm53-fp4-tp2':
        raise ValueError('TensorFold must use a separate container name')
    for k, v in result.items():
        if '\x00' in v or '\n' in v or '\r' in v:
            raise ValueError(f'Invalid control character in {k}')
    return result


def run(args, capture=False, **kwargs):
    return subprocess.run(args, check=True, text=True, capture_output=capture, **kwargs)


def model(c):
    # Same locked metadata, shard and safe-symlink checks as the measured baseline.
    env = dict(os.environ, HF_HOME=c['HF_CACHE_ROOT'],
               HF_HUB_CACHE=str(Path(c['HF_CACHE_ROOT']) / 'hub'),
               MODEL_DIR=c['MODEL_DIR'], MODEL_REVISION='76348ddb2be2e35e39ecfc00e75dc07703f4f3d5')
    return json.loads(run([sys.executable, str(ROOT / 'scripts/model.py')], capture=True, env=env).stdout)


def image(c):
    data = json.loads(run(['docker', 'image', 'inspect', c['TF_IMAGE']], capture=True).stdout)[0]
    labels = data.get('Config', {}).get('Labels', {}) or {}
    if (labels.get('org.opencontainers.image.revision') != COMMIT or
            labels.get('lab.glm53.tensorfold.recipe') != 'native-fp4-tp2' or
            labels.get('lab.glm53.tensorfold.pr1050') != PR_COMMIT or
            labels.get('lab.glm53.tensorfold.tree') != MERGED_TREE):
        raise ValueError('Image is not this pinned native recipe; run ./tensorfold/build.sh')
    return data['Id']


def owned(data):
    labels = data.get('Config', {}).get('Labels', {}) or {}
    if labels.get(LABEL) != str(HERE):
        raise ValueError('Refusing to control a container not owned by this TensorFold recipe')


def container(c):
    result = subprocess.run(['docker', 'container', 'inspect', c['TF_CONTAINER']],
                            text=True, capture_output=True, check=False)
    if result.returncode:
        # Do not mistake an inaccessible daemon for an absent container.
        if 'No such' in result.stderr:
            return None
        raise RuntimeError(result.stderr.strip())
    data = json.loads(result.stdout)[0]
    owned(data)
    return data


def ports(c):
    for port in (int(c['TF_PORT']), int(c['TF_COMM_PORT']), int(c['TF_COMM_PORT']) + 1):
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(('0.0.0.0', port))
            except OSError as exc:
                raise ValueError(f'Port {port} is unavailable: {exc}') from exc


def gpu_preflight(c):
    rows = run(['nvidia-smi', '--query-gpu=index,uuid,name,memory.free',
                '--format=csv,noheader,nounits'], capture=True).stdout.splitlines()
    found = {r.split(',')[0].strip(): [x.strip() for x in r.split(',')] for r in rows}
    selected = []
    for idx in c['TF_GPUS'].split(','):
        if idx not in found:
            raise ValueError(f'GPU {idx} not found')
        row = found[idx]
        if 'RTX PRO 6000' not in row[2] or 'Blackwell' not in row[2]:
            raise ValueError(f'GPU {idx}: expected RTX PRO 6000 Blackwell')
        if int(row[3]) < 90000:
            raise ValueError(f'GPU {idx} has only {row[3]} MiB free. Free the intended test GPUs first; this recipe never stops other services.')
        selected.append(row[1])
    jobs = run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                '--format=csv,noheader,nounits'], capture=True).stdout
    if any(uuid in jobs for uuid in selected):
        raise ValueError('A compute process occupies a selected GPU; no services were stopped')
    return [found[i] for i in c['TF_GPUS'].split(',')]



def compatibility(m):
    """Necessary checkpoint header contracts for the pinned PR; full GPU gate is separate."""
    directory = Path(m['path'])
    weights = json.loads((directory / 'model.safetensors.index.json').read_text())['weight_map']
    headers = {}
    issues = []
    module = 'model.language_model.layers.3.mlp.experts.0.gate_proj'
    expected = {module + '.weight': 'U8'}
    for part, dtype in [('bases', 'U8'), ('codes', 'U8'), ('page_bases', 'U32'),
                        ('offsets', 'U16'), ('exceptions', 'U16')]:
        expected[module + '.weight_scale.csf32_' + part] = dtype
    expected['model.language_model.layers.0.self_attn.q_proj.weight'] = 'F8_E4M3'
    expected['lm_head.weight'] = 'F8_E4M3'
    for name, dtype in expected.items():
        if name not in weights:
            issues.append({'code': 'missing_tensor', 'tensor': name})
            continue
        shard = weights[name]
        if shard not in headers:
            with (directory / shard).open('rb') as stream:
                length = struct.unpack('<Q', stream.read(8))[0]
                if length > 128 * 1024 * 1024:
                    raise ValueError('Unexpectedly large safetensors header')
                headers[shard] = json.loads(stream.read(length))
        actual = headers[shard][name]['dtype']
        if actual != dtype:
            issues.append({'code': 'unexpected_tensor_dtype', 'tensor': name,
                           'dtype': actual, 'expected': dtype})
    return {'base_commit': COMMIT, 'pr1050_commit': PR_COMMIT, 'merged_tree': MERGED_TREE,
            'model_revision': m.get('revision'), 'loader_contract_checks_passed': not issues,
            'issues': issues, 'note': 'Header checks and the CPU codec probe do not qualify GPU inference.'}


def require_compatible(m):
    report = compatibility(m)
    if report['issues']:
        raise ValueError('Checkpoint headers do not match the pinned PR #1050 recipe: '
                         + '; '.join(x['code'] for x in report['issues']))


def probe(m, ident):
    run(['docker', 'run', '--rm', '--network', 'none', *mount(m),
         '--entrypoint', '/opt/tensorfold/bin/glm-loader-probe', ident, m['container_path']])


def mount(m):
    if ',' in m['mount_source']:
        raise ValueError('Docker mount paths containing commas are unsupported')
    return ['--mount', f"type=bind,src={m['mount_source']},dst={m['mount_target']},readonly"]


def launch(c, m, image_id):
    args = ['docker', 'run', '-d', '--name', c['TF_CONTAINER'], '--init',
            '--gpus', f'"device={c["TF_GPUS"]}"', '--network', 'host', '--ipc', 'host',
            '--shm-size', '32g', '--stop-timeout', '30', '--restart', 'no',
            '--ulimit', 'memlock=-1', '--ulimit', 'stack=67108864:67108864',
            '--label', f'{LABEL}={HERE}', '--log-opt', 'max-size=100m', '--log-opt', 'max-file=3',
            '--mount', f'type=volume,src={c["TF_VOLUME"]},dst=/cache', *mount(m)]
    for k in ('TF_HOST', 'TF_PORT', 'TF_COMM_PORT', 'TF_CONTEXT', 'TF_PARALLEL', 'TF_MODEL_NAME'):
        args += ['-e', f'{k}={c[k]}']
    return args + ['-e', f'MODEL_PATH={m["container_path"]}', image_id]


def gpu_checks(c, m, ident):
    """Bounded upstream arithmetic and real-loader checks, one selected device at a time."""
    folder = ROOT / 'logs/tensorfold' / datetime.now(timezone.utc).strftime('gpu-check-%Y%m%dT%H%M%SZ')
    folder.mkdir(parents=True, exist_ok=False)
    results = []
    for rank, device in enumerate(c['TF_GPUS'].split(',')):
        for case in ('info', 'glm-ref', 'mxfp8', 'projection-f32', 'glm-loader-gate'):
            name = c['TF_CONTAINER'] + '-check'
            args = ['docker', 'run', '--rm', '--name', name, '--init', '--network', 'none',
                    '--gpus', 'device=' + device, '--ipc', 'host', '--shm-size', '32g',
                    '--ulimit', 'memlock=-1', '--ulimit', 'stack=67108864:67108864',
                    '--label', f'{LABEL}={HERE}', *mount(m),
                    '-e', 'TF_GATE_WORLDS=2', '-e', f'TF_GATE_RANKS={rank}',
                    '-e', 'TF_GATE_LAYERS=0,3,4,43,44',
                    '--entrypoint', '/opt/tensorfold/bin/tf-cuda-test', ident, case]
            if case == 'glm-loader-gate':
                args.append(m['container_path'])
            logfile = folder / f'gpu{device}-{case}.log'
            print(f'GPU {device}, {case}: {logfile}', flush=True)
            with logfile.open('w') as out:
                try:
                    result = subprocess.run(args, text=True, stdout=out, stderr=subprocess.STDOUT,
                                            timeout=1200, check=False)
                except (subprocess.TimeoutExpired, KeyboardInterrupt):
                    target = dict(c, TF_CONTAINER=name)
                    if container(target):
                        run(['docker', 'stop', '--time', '10', name])
                    raise
            results.append({'device': device, 'rank_geometry': rank, 'case': case,
                            'exit_code': result.returncode, 'log': str(logfile)})
            (folder / 'checks.json').write_text(json.dumps(results, indent=2) + '\n')
            if result.returncode:
                raise RuntimeError(f'GPU check failed: {case}; inspect {logfile}')
    print('PASS both GPUs: arithmetic and five decoder layers plus MTP loader checks', flush=True)


@contextlib.contextmanager
def lock():
    folder = ROOT / '.state/tensorfold'
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'lifecycle.lock').open('w') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def main(argv):
    c = config()
    if not argv:
        raise ValueError('Choose build, inspect, compatibility, probe-loader, doctor, gpu-check, start, status, logs, stop, smoke or bench')
    command, *extra = argv
    if command == 'build':
        if extra:
            raise ValueError('build accepts no arguments; set BUILD_JOBS in the environment')
        folder = ROOT / 'logs/tensorfold'
        folder.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        logfile = folder / f'build-{stamp}.log'
        print(f'Build log: {logfile}', flush=True)
        with logfile.open('w') as out:
            run(['docker', 'build', '--progress=plain', '--build-arg', f'BUILD_JOBS={c["BUILD_JOBS"]}',
                 '--tag', c['TF_IMAGE'], str(HERE)], stdout=out, stderr=subprocess.STDOUT)
        print('Built:', image(c))
        return
    if command in ('smoke', 'bench'):
        env = dict(os.environ, BASE_URL=f'http://127.0.0.1:{c["TF_PORT"]}',
                   SERVED_MODEL_NAME=c['TF_MODEL_NAME'], CONTAINER_NAME=c['TF_CONTAINER'])
        env.pop('VLLM_API_KEY', None)
        run([sys.executable, str(HERE / 'scripts/api_check.py'), command, *extra], env=env)
        return
    if extra and command != 'logs':
        raise ValueError(f'{command} accepts no arguments')
    if command == 'compatibility':
        m = model(c)
        print(json.dumps(compatibility(m), indent=2), flush=True)
        require_compatible(m)
        probe(m, image(c))
        return
    if command == 'probe-loader':
        m, ident = model(c), image(c)
        probe(m, ident)
        return
    if command == 'inspect':
        m, ident = model(c), image(c)
        print(json.dumps({'image_id': ident, 'model': m}, indent=2), flush=True)
        print('Informational inspection only. Run compatibility.sh before planning GPU inference.', flush=True)
        for flags in (['--version'], ['capabilities', '--json'], ['info', m['container_path']]):
            run(['docker', 'run', '--rm', '--network', 'none', *mount(m),
                 '--entrypoint', BINARY, ident, *flags])
        return
    if command in ('doctor', 'start', 'gpu-check'):
        with lock():
            m, ident = model(c), image(c)
            require_compatible(m)
            if container(c):
                raise ValueError('TensorFold container already exists; inspect its status before stopping/replacing it')
            ports(c)
            gpus = gpu_preflight(c)
            probe(m, ident)
            print(json.dumps({'image_id': ident, 'model': m, 'gpus': gpus, 'context': c['TF_CONTEXT'],
                              'parallel': c['TF_PARALLEL']}, indent=2), flush=True)
            if command == 'gpu-check':
                gpu_checks(c, m, ident)
            if command == 'start':
                folder = ROOT / 'logs/tensorfold' / datetime.now(timezone.utc).strftime('run-%Y%m%dT%H%M%SZ')
                folder.mkdir(parents=True, exist_ok=False)
                args = launch(c, m, ident)
                (folder / 'launch.json').write_text(json.dumps({'model': m, 'config': c, 'command': args}, indent=2) + '\n')
                run(args)
                print('Started. Run ./tensorfold/smoke.sh --wait 1200 for readiness and acceptance.')
        return
    if command in ('status', 'logs', 'stop'):
        with lock():
            data = container(c)
            if data is None:
                print('TensorFold container does not exist')
                return
            if command == 'status':
                print(json.dumps(data['State'], indent=2))
            elif command == 'stop':
                run(['docker', 'stop', c['TF_CONTAINER']])
                run(['docker', 'rm', c['TF_CONTAINER']])
            else:
                if extra not in ([], ['--follow']):
                    raise ValueError('logs accepts only --follow')
                # Release lifecycle lock before following an indefinite log stream.
        if command == 'logs':
            run(['docker', 'logs', '--tail', '150', *extra, data['Id']])
        return
    raise ValueError(f'Unknown command: {command}')


if __name__ == '__main__':
    try:
        main(sys.argv[1:])
    except (ValueError, RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
