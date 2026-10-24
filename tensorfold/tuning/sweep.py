#!/usr/bin/env python3
"""Serial, resumable served-model screen. Controls only the owned TensorFold container."""
import argparse
import csv
import io
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

from run import ROOT, environment


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def headroom(path, devices, minimum):
    # Read only flushed complete lines; include startup peaks conservatively.
    raw = path.read_text().rsplit('\n', 1)[0]
    rows = list(csv.DictReader(io.StringIO(raw)))
    free = {str(device): [] for device in devices}
    for row in rows:
        device = row['index'].strip()
        if device in free:
            free[device].append(int(row['memory_free_MiB'].strip()))
    if any(not values for values in free.values()):
        raise ValueError('Missing GPU telemetry for the capacity headroom gate')
    minima = {device: min(values) for device, values in free.items()}
    return {'required_free_MiB': minimum, 'sampled_min_free_MiB': minima,
            'passed': all(value >= minimum for value in minima.values()),
            'scope': 'One-second samples across startup and completed rungs; transient peaks between samples are not captured'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile-file', type=Path, default=Path(__file__).with_name('profiles.json'))
    p.add_argument('--profiles', nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--mtp', choices=['on', 'off', 'both'], default='on')
    p.add_argument('--teacher-histories', type=Path)
    p.add_argument('--quality', action='store_true')
    p.add_argument('--baseline-image', action='store_true')
    p.add_argument('--cache-check', action='store_true')
    p.add_argument('--skip-benchmark', action='store_true')
    p.add_argument('--context-stage-timeout', type=int, default=1800)
    p.add_argument('--minimum-free-mib', type=int, default=0,
                   help='Run context targets as separate rungs and stop if sampled free memory falls below this floor')
    p.add_argument('--context-targets', nargs='+', type=int, default=[2048, 8192])
    p.add_argument('--cases', nargs='+', default=['short-code', 'short-prose', 'prefill-8192', 'prefill-32768'])
    args = p.parse_args()
    profiles = json.loads(args.profile_file.read_text())
    if args.minimum_free_mib < 0 or (args.minimum_free_mib and args.cache_check):
        p.error('Memory floor must be nonnegative and cannot be combined with cache checks')
    if args.context_stage_timeout < 1:
        p.error('Context stage timeout must be positive')
    if args.repeats < 1 or any(x not in profiles['profiles'] for x in args.profiles):
        p.error('Invalid profile or repeat count')
    if args.cache_check and (not args.skip_benchmark or args.teacher_histories or args.quality):
        p.error('Cache qualification requires --skip-benchmark and no teacher/quality stages')
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (ROOT / '.state/tuning-sweep.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol = {'profiles_sha256': sha(args.profile_file),
                'client_sha256': sha(ROOT / 'benchmarks/run.py'),
                'workloads_sha256': sha(ROOT / 'logs/tuning/workloads.json'),
                'repeats': args.repeats, 'mtp': args.mtp, 'cases': args.cases,
                'teacher_histories_sha256': sha(args.teacher_histories) if args.teacher_histories else None,
                'quality': args.quality, 'cache_check': args.cache_check, 'baseline_image': args.baseline_image, 'skip_benchmark': args.skip_benchmark, 'context_targets': args.context_targets}
    if args.context_stage_timeout != 1800:
        protocol['context_stage_timeout_seconds'] = args.context_stage_timeout
    if args.minimum_free_mib:
        protocol['minimum_free_MiB'] = args.minimum_free_mib
    had_failure = False
    for number, profile in enumerate(args.profiles):
        folder = args.output / f'{number:02}-{profile}'
        done = folder / 'complete.json'
        if done.exists():
            prior = json.loads(done.read_text())
            if prior['protocol'] != protocol or prior['profile'] != profile:
                raise ValueError('Completed run protocol differs; use a new output directory')
            had_failure = had_failure or prior.get('status') != 'pass'
            print(f'SKIP completed {profile}: {prior.get("status")}', flush=True)
            continue
        if folder.exists():
            raise ValueError(f'Incomplete attempt preserved at {folder}; inspect before choosing a new output directory')
        folder.mkdir()
        env = environment(profile, baseline=args.baseline_image, profiles=profiles)
        recipe = [sys.executable, str(ROOT / 'tensorfold/scripts/recipe.py')]
        before = subprocess.check_output(['docker', 'ps', '-a', '--format', '{{.Names}}'], text=True).splitlines()
        if env['TF_CONTAINER'] in before:
            raise ValueError('Owned TensorFold container already exists; inspect before replacing')
        stop = threading.Event()
        def telemetry():
            with (folder / 'gpu.csv').open('w') as out:
                out.write('utc,index,memory_used_MiB,memory_free_MiB,utilization_pct,temperature_C,power_W,sm_clock_MHz,memory_clock_MHz,clock_event_reasons\n')
                while not stop.is_set():
                    r = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw,clocks.sm,clocks.mem,clocks_event_reasons.active', '--format=csv,noheader,nounits'], capture_output=True, text=True)
                    now = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
                    for line in r.stdout.splitlines():
                        out.write(now + ',' + line + '\n')
                    out.flush()
                    stop.wait(1)
        thread = threading.Thread(target=telemetry, daemon=True)
        thread.start()
        started = time.time()
        status = {'profile': profile, 'protocol': protocol, 'started_unix': started}
        print(f'START {profile}', flush=True)
        try:
            commands = [
                ('start', [*recipe, 'start']),
                ('smoke', [*recipe, 'smoke', '--wait', '1200', '--suite']),
                ('context', [sys.executable, str(ROOT / 'tensorfold/scripts/context_check.py'), '--targets', *map(str, args.context_targets), '--compare-plain', '--output', str(folder / 'context')]),
                ('benchmark', [sys.executable, str(ROOT / 'benchmarks/run.py'), '--backend', 'tensorfold', '--mtp', args.mtp, '--workloads', str(ROOT / 'logs/tuning/workloads.json'), '--output', str(folder / 'bench'), '--repeats', str(args.repeats), '--cases', *args.cases]),
            ]
            if args.minimum_free_mib:
                rungs = [(f'context-{target}', [sys.executable, str(ROOT / 'tensorfold/scripts/context_check.py'),
                          '--targets', str(target), '--compare-plain', '--output', str(folder / 'context' / str(target))])
                         for target in args.context_targets]
                at = next(i for i, (stage, _) in enumerate(commands) if stage == 'context')
                commands[at:at + 1] = rungs
            if args.cache_check:
                commands = [(stage, command) for stage, command in commands if stage != 'context']
                commands.append(('cache', [sys.executable, str(ROOT / 'tensorfold/scripts/cache_check.py'), '--output', str(folder / 'cache')]))
            if args.skip_benchmark:
                commands = [pair for pair in commands if pair[0] != 'benchmark']
            if args.teacher_histories:
                commands.append(('teacher', [sys.executable, str(Path(__file__).with_name('teacher_force.py')), 'run', str(args.teacher_histories), str(folder / 'teacher')]))
            if args.quality:
                commands.append(('quality', [str(ROOT / 'quality/run.sh'), '--backend', 'tensorfold', '--output', str(folder / 'quality')]))
            for stage, command in commands:
                print(f'{profile}: {stage}', flush=True)
                with (folder / f'{stage}.log').open('w') as out:
                    subprocess.run(command, cwd=ROOT, env=env, stdout=out, stderr=subprocess.STDOUT, check=True, timeout=7200 if stage == 'quality' else args.context_stage_timeout if stage.startswith('context') else 1800)
                if args.minimum_free_mib and (stage == 'smoke' or stage.startswith('context')):
                    gate = headroom(folder / 'gpu.csv', env['TF_GPUS'].split(','), args.minimum_free_mib)
                    gate['after_stage'] = stage
                    (folder / f'{stage}-headroom.json').write_text(json.dumps(gate, indent=2) + '\n')
                    if not gate['passed']:
                        raise ValueError(f'Capacity headroom floor failed: {gate["sampled_min_free_MiB"]}')
                if stage == 'start':
                    state = json.loads(subprocess.check_output(['docker', 'inspect', env['TF_CONTAINER']], text=True))[0]
                    actual = dict(item.split('=', 1) for item in state['Config']['Env'])
                    launch = {'image_id': state['Image'], 'profile': profile, 'protocol': protocol,
                              'settings': {k: v for k, v in actual.items() if k.startswith(('TF_', 'NCCL_')) or k == 'MODEL_PATH'}}
                    (folder / 'launch-manifest.json').write_text(json.dumps(launch, indent=2) + '\n')
            rows = [] if args.skip_benchmark else [json.loads(x) for x in (folder / 'bench/requests.jsonl').read_text().splitlines()]
            hashes = {}
            for row in rows:
                hashes.setdefault(row['case'], set()).add(row['response']['runtime']['token_sha256'])
            status['token_hashes'] = {key: sorted(value) for key, value in hashes.items()}
            status['repeat_and_mtp_identical'] = all(len(value) == 1 for value in hashes.values()) if rows else None
            status['status'] = 'pass'
            if status['repeat_and_mtp_identical'] is False:
                status['status'] = 'nonidentical-output-investigate'
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError) as exc:
            status.update(status='failed', error=str(exc))
        finally:
            with (folder / 'server.log').open('w') as out:
                subprocess.run(['docker', 'logs', env['TF_CONTAINER']], stdout=out, stderr=subprocess.STDOUT)
            with (folder / 'stop.log').open('w') as out:
                subprocess.run([*recipe, 'stop'], env=env, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT, check=True, timeout=120)
            stop.set(); thread.join(timeout=10)
        had_failure = had_failure or status['status'] != 'pass'
        status['elapsed_seconds'] = time.time() - started
        done.write_text(json.dumps(status, indent=2) + '\n')
        print(f'END {profile}: {status["status"]} {status["elapsed_seconds"]:.1f}s', flush=True)
    print('Sweep finished; TensorFold stopped. Restore the vLLM recipe when qualification is complete.', flush=True)
    return 1 if had_failure else 0


if __name__ == '__main__':
    raise SystemExit(main())
