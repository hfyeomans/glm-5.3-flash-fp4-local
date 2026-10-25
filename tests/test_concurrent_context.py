"""Prevent queued/evicted requests and reused cache from qualifying simultaneous context."""
import importlib.util
import json
import sys
import tempfile
from unittest import mock
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('concurrent_context', Path(__file__).resolve().parents[1] / 'benchmarks/concurrent_context.py')
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class ConcurrentEvidenceTests(unittest.TestCase):
    def test_external_drafter_cannot_enter_mtp_capacity_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'result'
            state = {'State': {'Running': True}, 'Config': {
                'Labels': {'local.glm53.recipe': 'glm53-fp4-local-v1'},
                'Cmd': ['vllm', 'serve', '/target/' + bench.REVISION,
                        '--speculative-config', '{"method":"dflash"}']},
                'HostConfig': {'DeviceRequests': [{'DeviceIDs': ['0', '1']}]}}
            argv = ['concurrent_context.py', '--window', '8192', '--clients', '1', '--output', str(output)]
            with mock.patch.object(sys, 'argv', argv), mock.patch.object(
                    bench.subprocess, 'check_output', return_value=json.dumps([state])), mock.patch.object(
                    bench, 'metrics') as metrics:
                with self.assertRaisesRegex(ValueError, 'refuses DFlash2'):
                    bench.main()
                metrics.assert_not_called()
            self.assertFalse(output.exists())

    def test_dflash2_capacity_mode_binds_window_and_endpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'result'
            state = {'Image': 'sha256:745feb916ee816f5409fd58386114be4f22fbede7f9c0e0a7b8c4fe8dac0fb29',
                'State': {'Running': True}, 'Config': {
                'Labels': {'local.glm53.recipe': 'glm53-fp4-local-v1'},
                'Cmd': ['vllm', 'serve', '/target/' + bench.REVISION, '--port', '8002',
                        '--max-model-len', '131072', '--speculative-config',
                        '{"method":"dflash","model":"/cache/models--incoai--GLM-5.3-Flash-DFlash2/snapshots/bf582e4eacc1810f76656d1811693ff6c6737d2a"}']},
                'HostConfig': {'DeviceRequests': [{'DeviceIDs': ['0', '1']}]}}
            for base, error in [('http://127.0.0.1:8000', 'Endpoint differs'),
                                ('http://127.0.0.1:8002', 'window exceeds')]:
                argv = ['concurrent_context.py', '--drafter', 'dflash2', '--container', 'probe',
                        '--base-url', base, '--window', '262144', '--clients', '1', '--output', str(output)]
                with mock.patch.object(sys, 'argv', argv), mock.patch.object(
                        bench.subprocess, 'check_output', return_value=json.dumps([state])), mock.patch.object(
                        bench, 'metrics') as metrics:
                    with self.assertRaisesRegex(ValueError, error):
                        bench.main()
                    metrics.assert_not_called()
            self.assertFalse(output.exists())

    def data(self):
        rows = [dict(client=str(i), prompt_tokens=1000, done=True, text='answer',
                     usage={'prompt_tokens': 1000, 'completion_tokens': 128,
                            'prompt_tokens_details': {'cached_tokens': 0}},
                     finish_reason='length', retrieved=[True]*3, start_monotonic=0,
                     first_text_monotonic=10+i, last_text_monotonic=20+i,
                     end_monotonic=21+i) for i in range(2)]
        before = {'vllm:num_preemptions_total': 7, 'vllm:request_success_total': 5}
        after = {'vllm:num_preemptions_total': 7, 'vllm:request_success_total': 7}
        samples = [{'metrics': {'vllm:num_requests_running': 2, 'vllm:num_requests_waiting': 0},
                    'gpus': [{'index': 0, 'free_mib': 1600}, {'index': 1, 'free_mib': 1700}]}]
        return rows, samples, before, after

    def test_two_running_requests_do_not_prove_overlapping_decode(self):
        rows, samples, before, after = self.data()
        rows[1].update(first_text_monotonic=22, last_text_monotonic=25)
        report = bench.summarize(rows, samples, before, after, True, 128)
        self.assertTrue(report['requests_completed_cleanly'])
        self.assertFalse(report['overlapping_decode_without_preemption'])

    def test_preemption_prevents_residency_claim_even_with_overlapping_intervals(self):
        rows, samples, before, after = self.data()
        self.assertTrue(bench.summarize(rows, samples, before, after, True, 128)['overlapping_decode_without_preemption'])
        after['vllm:num_preemptions_total'] += 1
        self.assertFalse(bench.summarize(rows, samples, before, after, True, 128)['overlapping_decode_without_preemption'])

    def test_cached_or_wrong_length_requests_do_not_pass(self):
        for mode in ('cached', 'wrong_count', 'missing_detail'):
            rows, samples, before, after = self.data()
            if mode == 'cached':
                rows[0]['usage']['prompt_tokens_details']['cached_tokens'] = 256
            elif mode == 'wrong_count':
                rows[0]['usage']['prompt_tokens'] -= 1
            else:
                rows[0]['usage'].pop('prompt_tokens_details')
            with self.subTest(mode=mode):
                self.assertFalse(bench.summarize(rows, samples, before, after, True, 128)['requests_completed_cleanly'])

    def test_invisible_token_intervals_and_token_accounting(self):
        rows, samples, before, after = self.data()
        for i, row in enumerate(rows):
            row.update(token_ids_required=True, server_prompt_ids_match=True,
                       emitted_token_count=128, first_token_monotonic=10+i,
                       last_token_monotonic=20+i, last_text_monotonic=10.1+i)
        r = bench.summarize(rows, samples, before, after, True, 128)
        self.assertTrue(r['overlapping_decode_without_preemption'])
        self.assertEqual(r['decode_interval_basis'], 'token_ids')
        rows[0]['emitted_token_count'] -= 1
        self.assertFalse(bench.summarize(rows, samples, before, after, True, 128)['requests_completed_cleanly'])

    def test_cross_client_markers_fail_isolation(self):
        rows, samples, before, after = self.data()
        rows[0].update(expected=['alpha-secret'], text='alpha-secret and beta-secret')
        rows[1].update(expected=['beta-secret'], text='beta-secret')
        self.assertFalse(bench.summarize(rows, samples, before, after, True, 128)['request_isolation_passed'])

    def test_metric_labels_sum_and_missing_scheduler_metrics_fail(self):
        raw = ('vllm:num_requests_running{engine="0"} 1\n'
               'vllm:num_requests_running{engine="1"} 1\n'
               'vllm:num_requests_waiting 0\nvllm:num_preemptions_total 3\n')
        self.assertEqual(bench.parse_metrics(raw)['vllm:num_requests_running'], 2)
        with self.assertRaises(ValueError):
            bench.parse_metrics('vllm:num_requests_running 2\n')


if __name__ == '__main__':
    unittest.main()
