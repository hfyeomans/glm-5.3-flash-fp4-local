"""CPU-only receipt and telemetry regression tests; all external commands are mocked."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
TUNING = ROOT / 'tensorfold/tuning'


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, TUNING / filename)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules):
        sys.modules.pop('run', None)
        sys.modules.pop('receipts', None)
        with patch.object(sys, 'path', [str(TUNING), *sys.path]):
            spec.loader.exec_module(module)
    return module


sweep = load('sweep_receipt_tests', 'sweep.py')
report = load('report_receipt_tests', 'report.py')


class SweepTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for path in ['.state', 'benchmarks', 'logs/tuning']:
            (self.root / path).mkdir(parents=True)
        (self.root / 'benchmarks/run.py').write_text('# client')
        (self.root / 'logs/tuning/workloads.json').write_text('{}')
        self.profile_file = self.root / 'profiles.json'
        self.profile_file.write_text(json.dumps({'profiles': {'control': {}, 'next': {}}}))
        self.output = self.root / 'results'
        self.calls = []
        self.image = 'sha256:original'
        self.query = lambda: subprocess.CompletedProcess([], 0, '0,1,2,3,4,5,6,7,none\n', '')
        self.on_stop = lambda: None
        self.cleanup_failure = None

    def run_sweep(self, profile='control', extra=()):
        def run(cmd, **kwargs):
            self.calls.append(cmd)
            if cmd[0] == 'nvidia-smi':
                return self.query()
            if cmd[-1] == 'stop':
                self.on_stop()
            if self.cleanup_failure and (cmd[1] == 'logs' or cmd[-1] == 'stop'):
                raise subprocess.TimeoutExpired(cmd, kwargs['timeout'])
            return subprocess.CompletedProcess(cmd, 0)

        def output(cmd, **kwargs):
            self.calls.append(cmd)
            if cmd[1] == 'ps':
                return ''
            if cmd[1:3] == ['image', 'inspect']:
                return self.image + '\n'
            return json.dumps([{'Image': self.image, 'Config': {'Env': []}}])

        argv = ['sweep', '--profile-file', str(self.profile_file), '--profiles',
                *([profile] if isinstance(profile, str) else profile),
                '--output', str(self.output), '--skip-benchmark', *extra]
        stdout = io.StringIO()
        with patch.object(sys, 'argv', argv), patch.object(sweep, 'ROOT', self.root), \
                patch.object(sweep, 'environment', return_value={'TF_CONTAINER': 'owned', 'TF_IMAGE': 'test:image'}), \
                patch.object(sweep.subprocess, 'run', side_effect=run), \
                patch.object(sweep.subprocess, 'check_output', side_effect=output), \
                contextlib.redirect_stdout(stdout):
            result = sweep.main()
        self.stdout = stdout.getvalue()
        return result

    def receipt(self, profile='control'):
        return self.output / f'00-{profile}/complete.json'

    def run_report(self):
        with patch.object(sys, 'argv', ['report', str(self.output), '--output', str(self.root / 'report')]), \
                contextlib.redirect_stdout(io.StringIO()):
            report.main()

    def test_delayed_telemetry_error_cannot_pass(self):
        started = threading.Event()
        release = threading.Event()
        real_thread = threading.Thread
        threads = []

        def query():
            started.set()
            release.wait(1)
            return subprocess.CompletedProcess([], 1, '', 'delayed telemetry error')

        def factory(*args, **kwargs):
            thread = real_thread(*args, **kwargs)
            join = thread.join
            # Scale the old 10s cleanup budget below the simulated delayed query.
            thread.join = lambda timeout=None: join(0.01 if timeout == 10 else 0.5)
            threads.append(thread)
            return thread

        timer = threading.Timer(0.05, release.set)
        self.query = query
        self.on_stop = lambda: (started.wait(1), timer.start())
        try:
            with patch.object(sweep.threading, 'Thread', side_effect=factory):
                result = self.run_sweep()
            receipt = json.loads(self.receipt().read_text())
        finally:
            release.set()
            for thread in threads:
                thread.join(1)
            if timer.ident is not None:
                timer.join()
        self.assertEqual(result, 1)
        self.assertEqual(receipt['status'], 'failed')
        self.assertIn('delayed telemetry error', receipt['error'])

    def test_unfinished_telemetry_fails_with_bounded_cleanup_and_stops_sweep(self):
        started = threading.Event()
        release = threading.Event()
        finished = threading.Event()

        def query():
            started.set()
            release.wait(2)
            finished.set()
            return subprocess.CompletedProcess([], 0, '0,1,2,3,4,5,6,7,none\n', '')

        self.query = query
        self.on_stop = lambda: self.assertTrue(started.wait(1))
        try:
            with patch.object(sweep, 'TELEMETRY_JOIN_TIMEOUT', 0.01):
                result = self.run_sweep(['control', 'next'])
            receipt = json.loads(self.receipt().read_text())
            self.assertEqual(result, 1)
            self.assertFalse(finished.is_set())
            self.assertEqual(receipt['status'], 'failed')
            self.assertIn('cleanup deadline', receipt['error'])
            self.assertFalse((self.output / '01-next').exists())
        finally:
            release.set()
            self.assertTrue(finished.wait(1))

    def test_cleanup_timeouts_still_stop_and_join_telemetry(self):
        self.cleanup_failure = True
        self.assertEqual(self.run_sweep(), 1)
        self.assertIn('check stop.log before assuming TensorFold stopped', self.stdout)
        self.assertNotIn('Sweep finished; TensorFold stopped.', self.stdout)
        receipt = json.loads(self.receipt().read_text())
        self.assertEqual(receipt['status'], 'failed')
        self.assertIn('server.log:', receipt['error'])
        self.assertIn('stop.log:', receipt['error'])
        self.assertTrue(any(cmd[-1] == 'stop' for cmd in self.calls))

    def test_report_rejects_missing_passing_image_manifest(self):
        self.assertEqual(self.run_sweep(), 0)
        (self.output / '00-control/launch-manifest.json').unlink()
        with self.assertRaisesRegex(ValueError, '[Ii]mage'):
            self.run_report()

    def test_report_rejects_manifest_protocol_disagreement(self):
        self.assertEqual(self.run_sweep(), 0)
        path = self.output / '00-control/launch-manifest.json'
        value = json.loads(path.read_text())
        value['protocol']['cases'] = ['other']
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, '[Pp]rotocol'):
            self.run_report()

    def test_resume_disjoint_profile_rejects_different_workload(self):
        self.assertEqual(self.run_sweep(), 0)
        original = self.receipt().read_bytes()
        self.calls.clear()
        with self.assertRaisesRegex(ValueError, '[Pp]rotocol'):
            self.run_sweep('next', ['--cases', 'different-workload'])
        self.assertEqual(self.receipt().read_bytes(), original)
        self.assertFalse((self.output / '00-next').exists())
        self.assertEqual(self.calls, [])

    def test_resume_rejects_changed_image_before_start(self):
        self.assertEqual(self.run_sweep(), 0)
        original = self.receipt().read_bytes()
        self.image = 'sha256:changed'
        self.calls.clear()
        with self.assertRaisesRegex(ValueError, '[Ii]mage'):
            self.run_sweep('next')
        self.assertEqual(self.receipt().read_bytes(), original)
        self.assertFalse(any(cmd[-1] == 'start' for cmd in self.calls))
        self.assertFalse((self.output / '00-next').exists())

    def test_report_rejects_mixed_protocols(self):
        self.assertEqual(self.run_sweep(), 0)
        self.assertEqual(self.run_sweep('next'), 0)
        path = self.receipt('next')
        value = json.loads(path.read_text())
        value['protocol']['workloads_sha256'] = 'different'
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, '[Pp]rotocol'):
            self.run_report()
        self.assertFalse((self.root / 'report').exists())

    def test_report_rejects_mixed_images(self):
        self.assertEqual(self.run_sweep(), 0)
        self.assertEqual(self.run_sweep('next'), 0)
        path = self.output / '00-next/launch-manifest.json'
        value = json.loads(path.read_text())
        value['image_id'] = 'sha256:changed'
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, '[Ii]mage'):
            self.run_report()
        self.assertFalse((self.root / 'report').exists())

    def test_compatible_resume_and_report_preserve_receipts(self):
        self.assertEqual(self.run_sweep(), 0)
        original = self.receipt().read_bytes()
        self.assertEqual(self.run_sweep('next'), 0)
        self.calls.clear()
        self.assertEqual(self.run_sweep(), 0)
        self.assertEqual(self.calls, [])
        self.run_report()
        self.assertEqual(self.receipt().read_bytes(), original)
        self.assertEqual(len(json.loads((self.root / 'report/screen.json').read_text())), 2)


if __name__ == '__main__':
    unittest.main()
