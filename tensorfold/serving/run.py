#!/usr/bin/env python3
"""Run a pinned serving profile through the existing guarded recipe."""
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'tensorfold/scripts'))
import recipe


def main():
    if len(sys.argv) < 3:
        raise SystemExit('Usage: run.py PROFILE COMMAND [ARGS...]')
    manifest = json.loads((HERE / 'profiles.json').read_text())
    profile = sys.argv[1]
    if profile not in manifest['profiles']:
        raise SystemExit('Unknown serving profile')
    env = dict(os.environ, **recipe.DEFAULTS)
    env.update(manifest['common'])
    env.update(manifest['profiles'][profile])
    env.update(TF_IMAGE=manifest['image'], TF_PATCH_SHA256=manifest['patch_sha256'],
               TF_TUNED_TREE=manifest['source_tree'])
    os.execve(sys.executable, [sys.executable, str(ROOT / 'tensorfold/scripts/recipe.py'), *sys.argv[2:]], env)


if __name__ == '__main__':
    main()
