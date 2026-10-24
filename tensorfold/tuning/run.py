#!/usr/bin/env python3
"""Apply a pinned local experiment profile to an owned TensorFold recipe operation."""
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROFILES = json.loads((HERE / 'profiles.json').read_text())


def environment(name):
    if name not in PROFILES['profiles']:
        raise ValueError('Unknown experiment profile')
    # Set all recipe defaults explicitly so ambient tuning values cannot leak between trials.
    sys.path.insert(0, str(ROOT / 'tensorfold/scripts'))
    import recipe
    env = dict(os.environ, **recipe.DEFAULTS)
    env.update(PROFILES['common'])
    env.update(PROFILES['profiles'][name])
    env.update(TF_IMAGE=PROFILES['image'], TF_PATCH_SHA256=PROFILES['patch_sha256'],
               TF_TUNED_TREE=PROFILES['source_tree'])
    return env


def main():
    if len(sys.argv) < 3:
        raise SystemExit('Usage: python3 tensorfold/tuning/run.py PROFILE RECIPE_COMMAND [arguments]')
    env = environment(sys.argv[1])
    os.execve(sys.executable, [sys.executable, str(ROOT / 'tensorfold/scripts/recipe.py'), *sys.argv[2:]], env)


if __name__ == '__main__':
    main()
