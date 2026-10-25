#!/usr/bin/env python3
"""Read-only launch checks; report every blocker in one pass."""
import csv
import io
import json
import math
import os
import platform
import shutil
import socket
import subprocess
import sys
from model import inspect_model
from draft_model import inspect_draft


def main():
    errors = []
    def check(ok, message):
        print(('PASS ' if ok else 'FAIL ') + message)
        if not ok: errors.append(message)
    def run(*args):
        return subprocess.run(args, capture_output=True, text=True, timeout=30)
    check(platform.system() == 'Linux' and platform.machine() == 'x86_64', 'Linux x86_64 host')
    for command in ['docker', 'nvidia-smi', 'flock']:
        check(shutil.which(command) is not None, command + ' installed')
    if shutil.which('docker'):
        info = run('docker', 'info', '--format', '{{json .Runtimes}}')
        check(info.returncode == 0, 'Docker daemon accessible')
        if info.returncode == 0:
            check('nvidia' in json.loads(info.stdout), 'NVIDIA Docker runtime registered')
            image = run('docker', 'image', 'inspect', os.environ['IMAGE'], '--format', '{{.Id}}')
            check(image.returncode == 0, 'Serving image available locally (run ./prepare.sh if absent)')
            containers = run('docker', 'ps', '--format', '{{.Names}}\t{{.Image}}')
            print('Running containers:\n' + (containers.stdout.strip() or '(none)'))
    if shutil.which('nvidia-smi'):
        gpu = run('nvidia-smi', '--query-gpu=index,name,memory.total,memory.free,driver_version', '--format=csv,noheader,nounits')
        check(gpu.returncode == 0, 'NVIDIA driver accessible')
        if gpu.returncode == 0:
            rows = {r[0].strip(): [v.strip() for v in r] for r in csv.reader(io.StringIO(gpu.stdout))}
            for selected in os.environ['GPUS'].split(','):
                row = rows.get(selected)
                check(row is not None, f'GPU {selected} exists')
                if row:
                    _, name, total, free, driver = row
                    check('RTX PRO 6000' in name and 'Blackwell' in name, f'GPU {selected}: {name}, driver {driver}')
                    threshold = max(int(os.environ['MIN_GPU_FREE_MIB']), math.ceil(float(total)*float(os.environ['GPU_MEMORY_UTILIZATION'])))
                    available = float(free) >= threshold
                    message = f'GPU {selected}: {free}/{total} MiB free, need {threshold} MiB'
                    check(available, message if available else message + '; release other GPU services before launch')
    try:
        m = inspect_model(os.environ['HF_HUB_CACHE'], os.environ['MODEL_REVISION'], os.environ.get('MODEL_DIR') or None)
        check(True, f'Offline NVIDIA checkpoint: {m["path"]}')
        print(f'     {m["shards"]} shards; {m["bytes"]/2**30:.2f} GiB; revision {m["revision"]}')
    except (ValueError, OSError, KeyError) as exc:
        check(False, str(exc))
    if os.environ.get('DRAFTER') == 'dflash2':
        try:
            d = inspect_draft(os.environ['HF_HUB_CACHE'], os.environ.get('DFLASH2_MODEL_DIR') or None)
            check(True, f'Offline DFlash2 draft: {d["path"]}; {d["license"]}; revision {d["revision"]}')
            print('     Experimental mode; runtime fit/accuracy must be measured. Target memory budget is unchanged.')
        except (ValueError, OSError, KeyError) as exc:
            check(False, str(exc))
    host, port = os.environ['HOST'], int(os.environ['PORT'])
    try:
        family = socket.AF_INET6 if ':' in host else socket.AF_INET
        with socket.socket(family, socket.SOCK_STREAM) as sock:
            # Match the server bind: recently closed connections must not block restart.
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
        check(True, f'API bind {host}:{port} available')
    except OSError as exc:
        check(False, f'API bind {host}:{port}: {exc}')
    print(f'\n{len(errors)} launch blocker(s). No containers or GPU settings changed.')
    return bool(errors)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        sys.exit('ERROR: ' + str(exc))
