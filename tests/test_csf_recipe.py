"""Experiment manifests must select matching source pins and reset ambient tuning."""
import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('tuning_profiles', ROOT / 'tensorfold/tuning/run.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
receipt_spec = importlib.util.spec_from_file_location('tuning_receipts', ROOT / 'tensorfold/tuning/receipts.py')
receipts = importlib.util.module_from_spec(receipt_spec)
receipt_spec.loader.exec_module(receipts)


class ProfileSelectionTests(unittest.TestCase):
    def test_explicit_manifest_owns_settings_and_image_identity(self):
        manifest = {'common': {'TF_PROMPT_CACHE_GIB': '0'},
                    'profiles': {'resident': {'TF_TUNE_SCALE_LAYOUT': '2'}},
                    'image': 'candidate', 'patch_sha256': 'a' * 64, 'source_tree': 'b' * 40}
        env = runner.environment('resident', profiles=manifest)
        self.assertEqual((env['TF_IMAGE'], env['TF_PATCH_SHA256'], env['TF_TUNED_TREE']),
                         ('candidate', 'a' * 64, 'b' * 40))
        self.assertEqual(env['TF_TUNE_SCALE_LAYOUT'], '2')
        with self.assertRaises(ValueError):
            runner.environment('resident')

    def test_historical_profile_does_not_inherit_resident_mode(self):
        with patch.dict(os.environ, {'TF_TUNE_SCALE_LAYOUT': '2'}):
            self.assertEqual(runner.environment('control')['TF_TUNE_SCALE_LAYOUT'], '0')


class SweepStatusTests(unittest.TestCase):
    def test_failed_launch_and_failed_resume_both_return_failure(self):
        import contextlib
        import io
        import json
        import subprocess
        import sys
        import tempfile
        sweep_spec = importlib.util.spec_from_file_location('csf_sweep', ROOT / 'tensorfold/tuning/sweep.py')
        sweep = importlib.util.module_from_spec(sweep_spec)
        with patch.dict(sys.modules, {'run': runner, 'receipts': receipts}):
            sweep_spec.loader.exec_module(sweep)
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            (base / '.state').mkdir()
            manifest = base / 'profiles.json'
            manifest.write_text(json.dumps({'profiles': {'candidate': {}}}))
            output = base / 'results'
            argv = ['sweep', '--profile-file', str(manifest), '--profiles', 'candidate', '--output', str(output)]

            def fail_start(command, **kwargs):
                if command[-1] == 'start':
                    raise subprocess.CalledProcessError(1, command)
                return subprocess.CompletedProcess(command, 0)

            with patch.object(sweep, 'ROOT', base), patch.object(sweep, 'sha', return_value='fixed'), \
                 patch.object(sweep, 'environment', return_value={'TF_CONTAINER': 'owned', 'TF_IMAGE': 'test:image'}), \
                 patch.object(sweep.threading, 'Thread') as thread, \
                 patch.object(sweep.subprocess, 'check_output', side_effect=lambda cmd, **kw: 'sha256:test' if cmd[1] == 'image' else '') as ps, \
                 patch.object(sweep.subprocess, 'run', side_effect=fail_start), patch('sys.argv', argv), \
                 contextlib.redirect_stdout(io.StringIO()):
                thread.return_value.is_alive.return_value = False
                self.assertEqual(sweep.main(), 1)
                receipt = json.loads((output / '00-candidate/complete.json').read_text())
                self.assertEqual(receipt['status'], 'failed')
                ps.reset_mock()
                self.assertEqual(sweep.main(), 1)
                ps.assert_not_called()


class CapacityHeadroomTests(unittest.TestCase):
    def test_each_card_and_missing_samples_are_checked(self):
        import sys
        import tempfile
        spec = importlib.util.spec_from_file_location('capacity_sweep', ROOT / 'tensorfold/tuning/sweep.py')
        sweep = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'run': runner, 'receipts': receipts}):
            spec.loader.exec_module(sweep)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'gpu.csv'
            path.write_text('utc,index,memory_free_MiB\nt0,0,2000\nt0,1,1800\nt1,0,1500\nt1,1,')
            result = sweep.headroom(path, ['0', '1'], 1536)
            self.assertFalse(result['passed'])
            self.assertEqual(result['sampled_min_free_MiB'], {'0': 1500, '1': 1800})
            self.assertTrue(sweep.headroom(path, ['0', '1'], 1400)['passed'])
            with self.assertRaisesRegex(ValueError, 'Missing GPU telemetry'):
                sweep.headroom(path, ['0', '2'], 1400)


CANCEL_FIXTURE_SNIPPET = ('sock.sendall(b"POST /v1/chat/completions HTTP/1.1\\r\\nHost: x\\r\\nContent-Type: application/json\\r\\nContent-Length: "'
                          ' + str(len(payload)).encode() + b"\\r\\n\\r\\n" + payload)\n')


def load_cancel_sweep():
    import sys
    spec = importlib.util.spec_from_file_location('cancel_sweep', ROOT / 'tensorfold/tuning/sweep.py')
    sweep = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {'run': runner, 'receipts': receipts}):
        spec.loader.exec_module(sweep)
    return sweep


class CancelFixtureTests(unittest.TestCase):
    def test_host_adapter_preserves_body_and_rejects_unknown_source(self):
        import tempfile
        import types
        sweep = load_cancel_sweep()
        original = CANCEL_FIXTURE_SNIPPET
        with tempfile.TemporaryDirectory() as folder:
            source, target = [Path(folder) / name for name in ('original.py', 'adapted.py')]
            source.write_text(original)
            metadata = sweep.prepare_cancel_fixture(source, target, expected_sha=sweep.sha(source))
            sent = []
            exec(compile(target.read_text(), str(target), 'exec'),
                 {'sock': types.SimpleNamespace(sendall=sent.append), 'payload': b'{}',
                  'HOST': types.SimpleNamespace(netloc='127.0.0.1:8001')})
            self.assertIn(b'\r\nHost: 127.0.0.1:8001\r\n', sent[0])
            self.assertTrue(sent[0].endswith(b'Content-Length: 2\r\n\r\n{}'))
            self.assertEqual(source.read_text(), original)
            self.assertNotEqual(metadata['upstream_sha256'], metadata['sha256'])
            self.assertEqual(metadata['expected_upstream_sha256'], sweep.sha(source))
            source.write_text('print("different fixture")\n')
            with self.assertRaisesRegex(ValueError, 'Unrecognized upstream'):
                sweep.prepare_cancel_fixture(source, target, expected_sha=sweep.sha(source))

    def test_unpinned_fixture_source_is_refused_before_host_execution(self):
        import tempfile
        sweep = load_cancel_sweep()
        with tempfile.TemporaryDirectory() as folder:
            source, target = [Path(folder) / name for name in ('original.py', 'adapted.py')]
            source.write_text(CANCEL_FIXTURE_SNIPPET)
            with self.assertRaisesRegex(ValueError, 'does not match the pinned'):
                sweep.prepare_cancel_fixture(source, target)
            self.assertFalse(target.exists())


class ServingPlacementTests(unittest.TestCase):
    def test_only_1m_serving_profile_offloads_embeddings(self):
        import json
        manifest = json.loads((ROOT / 'tensorfold/serving/profiles.json').read_text())
        for name in manifest['profiles']:
            env = runner.environment(name, profiles=manifest)
            self.assertEqual(env['TF_TUNE_HOST_EMBED'], '1' if name == 'host-1m' else '0')
        self.assertEqual(runner.environment('gpu-512k', profiles=manifest)['TF_CONTEXT'], '524288')
