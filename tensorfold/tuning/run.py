#!/usr/bin/env python3
"""Apply a pinned local experiment profile to an owned TensorFold recipe operation."""
import argparse
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROFILES = json.loads((HERE / 'profiles.json').read_text())


def environment(name, baseline=False):
    if name not in PROFILES['profiles']:
        raise ValueError('Unknown experiment profile')
    # Set all recipe defaults explicitly so ambient tuning values cannot leak between trials.
    sys.path.insert(0, str(ROOT / 'tensorfold/scripts'))
    import recipe
    env = dict(os.environ, **recipe.DEFAULTS)
    env.update(PROFILES['common'])
    env.update(PROFILES['profiles'][name])
    if not baseline:
        env.update(TF_IMAGE=PROFILES['image'], TF_PATCH_SHA256=PROFILES['patch_sha256'],
                   TF_TUNED_TREE=PROFILES['source_tree'])
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-image', action='store_true')
    parser.add_argument('profile', choices=PROFILES['profiles'])
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.command:
        parser.error('A recipe command is required')
    env = environment(args.profile, baseline=args.baseline_image)
    os.execve(sys.executable, [sys.executable, str(ROOT / 'tensorfold/scripts/recipe.py'), *args.command], env)


if __name__ == '__main__':
    main()
