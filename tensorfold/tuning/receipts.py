"""Validate the complete comparison set without modifying its receipts."""
import json


def completed_runs(output, protocol=None):
    runs = []
    image_id = None
    for path in sorted(output.glob('*/complete.json')):
        run = json.loads(path.read_text())
        recorded = run.get('protocol')
        if not isinstance(recorded, dict):
            raise ValueError(f'Missing completed run protocol: {path}')
        if protocol is None:
            protocol = recorded
        if recorded != protocol:
            raise ValueError(f'Completed run protocol differs at {path}; use a new output directory')
        manifest = path.with_name('launch-manifest.json')
        if manifest.exists():
            launch = json.loads(manifest.read_text())
            if launch.get('protocol') != recorded or launch.get('profile') != run.get('profile'):
                raise ValueError(f'Launch manifest protocol/profile differs at {manifest}')
            actual_image = launch.get('image_id')
            if not actual_image or (image_id is not None and actual_image != image_id):
                raise ValueError(f'Completed run image differs or is missing at {manifest}')
            image_id = actual_image
        elif run.get('status') == 'pass':
            raise ValueError(f'Passing run image manifest is missing: {manifest}')
        runs.append((path, run))
    return runs, image_id
