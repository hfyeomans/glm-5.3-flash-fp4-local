#!/usr/bin/env python3
"""Serial, resumable served-model screen. Controls only the owned TensorFold container."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

from run import ROOT, PROFILES, environment


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profiles', nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--mtp', choices=['on', 'off', 'both'], default='on')
    p.add_argument('--cases', nargs='+', default=['short-code', 'short-prose', 'prefill-8192', 'prefill-32768'])
    args = p.parse_args()
    if args.repeats < 1 or any(x not in PROFILES['profiles'] for x in args.profiles):
        p.error('Invalid profile or repeat count')
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (ROOT / '.state/tuning-sweep.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol = {'profiles_sha256': sha(Path(__file__).with_name('profiles.json')),
                'client_sha256': sha(ROOT / 'benchmarks/run.py'),
                'workloads_sha256': sha(ROOT / 'logs/tuning/workloads.json'),
                'repeats': args.repeats, 'mtp': args.mtp, 'cases': args.cases}
    for number, profile in enumerate(args.profiles):
        folder = args.output / f'{number:02}-{profile}'
        done = folder / 'complete.json'
        if done.exists():
            prior = json.loads(done.read_text())
            if prior['protocol'] != protocol or prior['profile'] != profile:
                raise ValueError('Completed run protocol differs; use a new output directory')
            print(f'SKIP completed {profile}', flush=True)
            continue
        if folder.exists():
            raise ValueError(f'Incomplete attempt preserved at {folder}; inspect before choosing a new output directory')
        folder.mkdir()
        env = environment(profile)
        recipe = [sys.executable, str(ROOT / 'tensorfold/scripts/recipe.py')]
        before = subprocess.check_output(['docker', 'ps', '-a', '--format', '{{.Names}}'], text=True).splitlines()
        if env['TF_CONTAINER'] in before:
            raise ValueError('Owned TensorFold container already exists; inspect before replacing')
        stop = threading.Event()
        def telemetry():
            with (folder / 'gpu.csv').open('w') as out:
                out.write('utc,index,memory_used_MiB,memory_free_MiB,utilization_pct,temperature_C,power_W\n')
                while not stop.is_set():
                    r = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw', '--format=csv,noheader,nounits'], capture_output=True, text=True)
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
                ('context', [sys.executable, str(ROOT / 'tensorfold/scripts/context_check.py'), '--targets', '2048', '8192', '--compare-plain', '--output', str(folder / 'context')]),
                ('benchmark', [sys.executable, str(ROOT / 'benchmarks/run.py'), '--backend', 'tensorfold', '--mtp', args.mtp, '--workloads', str(ROOT / 'logs/tuning/workloads.json'), '--output', str(folder / 'bench'), '--repeats', str(args.repeats), '--cases', *args.cases]),
            ]
            for stage, command in commands:
                print(f'{profile}: {stage}', flush=True)
                with (folder / f'{stage}.log').open('w') as out:
                    subprocess.run(command, cwd=ROOT, env=env, stdout=out, stderr=subprocess.STDOUT, check=True, timeout=1800)
            rows = [json.loads(x) for x in (folder / 'bench/requests.jsonl').read_text().splitlines()]
            hashes = {}
            for row in rows:
                hashes.setdefault(row['case'], set()).add(row['response']['runtime']['token_sha256'])
            status['token_hashes'] = {key: sorted(value) for key, value in hashes.items()}
            status['repeat_and_mtp_identical'] = all(len(value) == 1 for value in hashes.values())
            status['status'] = 'pass'
            if not status['repeat_and_mtp_identical']:
                status['status'] = 'nonidentical-output-investigate'
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError) as exc:
            status.update(status='failed', error=str(exc))
        finally:
            with (folder / 'server.log').open('w') as out:
                subprocess.run(['docker', 'logs', env['TF_CONTAINER']], stdout=out, stderr=subprocess.STDOUT)
            with (folder / 'stop.log').open('w') as out:
                subprocess.run([*recipe, 'stop'], env=env, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT, check=True, timeout=120)
            stop.set(); thread.join(timeout=10)
        status['elapsed_seconds'] = time.time() - started
        done.write_text(json.dumps(status, indent=2) + '\n')
        print(f'END {profile}: {status["status"]} {status["elapsed_seconds"]:.1f}s', flush=True)
    print('Sweep finished; TensorFold stopped. Restore the vLLM recipe when qualification is complete.', flush=True)


if __name__ == '__main__':
    main()
