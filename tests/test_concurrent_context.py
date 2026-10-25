"""Prevent queued/evicted requests and reused cache from qualifying simultaneous context."""
import importlib.util
import json
import sys
import tempfile
import types
from unittest import mock
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('concurrent_context', Path(__file__).resolve().parents[1] / 'benchmarks/concurrent_context.py')
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class MtpProfileTests(unittest.TestCase):
    def fixture(self, window=262144, clients=4):
        return {'Image': 'sha256:745feb916ee816f5409fd58386114be4f22fbede7f9c0e0a7b8c4fe8dac0fb29',
                'State': {'Running': True}, 'Config': {
                    'Labels': {'local.glm53.recipe': 'glm53-fp4-local-v1'},
                    'Cmd': ['vllm', 'serve', '/cache/models--nvidia--GLM-5.3-Flash-FP4-Local/snapshots/' + bench.REVISION,
                            '--host', '127.0.0.1', '--port', '8002',
                            '--max-model-len', str(window), '--max-num-seqs', str(clients),
                            '--gpu-memory-utilization', '0.985', '--kv-cache-memory-bytes', '7650410496',
                            '--attention-backend', 'B12X', '--linear-backend', 'b12x', '--moe-backend', 'b12x',
                            '--tensor-parallel-size', '2', '--decode-context-parallel-size', '2',
                            '--kv-cache-dtype', 'fp8', '--dtype', 'bfloat16',
                            '--quantization', 'modelopt_mixed', '--block-size', '256',
                            '--speculative-config', json.dumps({'method': 'mtp', 'num_speculative_tokens': 3,
                                'draft_sample_method': 'probabilistic', 'rejection_sample_method': 'standard',
                                'moe_backend': 'b12x', 'attention_backend': 'B12X'})]},
                'HostConfig': {'DeviceRequests': [{'DeviceIDs': ['0', '1']}]}}

    def test_capacity_accepts_exact_262k_four_and_1m_one_profiles(self):
        for window, clients in ((262144, 4), (1048576, 1)):
            with self.subTest(window=window, clients=clients):
                bench.validate_mtp_profile(self.fixture(window, clients), 'http://127.0.0.1:8002',
                                           'capacity', window, clients)

    def test_capacity_launch_still_fails_default_baseline_guard(self):
        with self.assertRaisesRegex(ValueError, 'Unexpected baseline setting'):
            bench.validate_mtp_profile(self.fixture(), 'http://127.0.0.1:8002', 'baseline', 262144, 4)
        bench.validate_mtp_profile(self.fixture(1048576, 8), 'http://127.0.0.1:8002', 'baseline', 262144, 4)

    def test_capacity_refuses_different_launch_limits_and_budgets(self):
        for key, replacement in (('--max-model-len', '1048576'), ('--max-num-seqs', '8'),
                                 ('--gpu-memory-utilization', '0.99'), ('--kv-cache-memory-bytes', '3221225472')):
            state = self.fixture()
            state['Config']['Cmd'][state['Config']['Cmd'].index(key) + 1] = replacement
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'Unexpected capacity setting: ' + key):
                bench.validate_mtp_profile(state, 'http://127.0.0.1:8002', 'capacity', 262144, 4)

    def test_capacity_refuses_changed_target_attention_backend(self):
        state = self.fixture()
        command = state['Config']['Cmd']
        command[command.index('--attention-backend') + 1] = 'TRITON_ATTN'
        with self.assertRaisesRegex(ValueError, 'Unexpected capacity setting: --attention-backend'):
            bench.validate_mtp_profile(state, 'http://127.0.0.1:8002', 'capacity', 262144, 4)

    def test_capacity_requires_target_kernel_distribution_and_precision_settings(self):
        replacements = {'--attention-backend': 'TRITON_ATTN', '--linear-backend': 'triton',
                        '--moe-backend': 'triton', '--tensor-parallel-size': '1',
                        '--decode-context-parallel-size': '1', '--kv-cache-dtype': 'auto',
                        '--dtype': 'float16', '--quantization': 'fp8', '--block-size': '128'}
        for key, replacement in replacements.items():
            for mode in ('changed', 'missing'):
                state = self.fixture()
                command = state['Config']['Cmd']
                index = command.index(key)
                if mode == 'changed':
                    command[index + 1] = replacement
                else:
                    del command[index:index + 2]
                with self.subTest(key=key, mode=mode), self.assertRaisesRegex(
                        ValueError, 'Unexpected capacity setting: ' + key):
                    bench.validate_mtp_profile(state, 'http://127.0.0.1:8002', 'capacity', 262144, 4)

    def test_capacity_refuses_wrong_image_checkpoint_spec_and_endpoint(self):
        variants = []
        state = self.fixture(); state['Image'] = 'sha256:other'; variants.append((state, 'pinned image'))
        state = self.fixture(); state['Config']['Cmd'][2] = '/other/' + bench.REVISION; variants.append((state, 'NVIDIA target'))
        state = self.fixture(); state['Config']['Cmd'][2] += '-other'; variants.append((state, 'NVIDIA target'))
        state = self.fixture(); state['Config']['Cmd'][4] = '0.0.0.0'; variants.append((state, 'loopback'))
        for change in ({'num_speculative_tokens': 7}, {'attention_backend': 'TRITON_ATTN'},
                       {'draft_sample_method': 'greedy'}, {'model': '/other/draft'}):
            state = self.fixture()
            config = json.loads(state['Config']['Cmd'][-1]); config.update(change)
            state['Config']['Cmd'][-1] = json.dumps(config)
            variants.append((state, 'MTP3/B12X'))
        for state, error in variants:
            with self.subTest(error=error, state=state), self.assertRaisesRegex(ValueError, error):
                bench.validate_mtp_profile(state, 'http://127.0.0.1:8002', 'capacity', 262144, 4)
        for base in ('http://127.0.0.1:8000', 'https://localhost', 'http://example.com:8002'):
            with self.subTest(base=base), self.assertRaises(ValueError):
                bench.validate_mtp_profile(self.fixture(), base, 'capacity', 262144, 4)

    def test_invalid_capacity_is_refused_before_requests_or_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'out'
            for invalid_setting in ('launch_limits', 'target_attention'):
                state = self.fixture(1048576, 8) if invalid_setting == 'launch_limits' else self.fixture()
                if invalid_setting == 'target_attention':
                    command = state['Config']['Cmd']
                    command[command.index('--attention-backend') + 1] = 'TRITON_ATTN'
                argv = ['concurrent', '--mtp-profile', 'capacity', '--base-url', 'http://127.0.0.1:8002',
                        '--window', '262144', '--clients', '4', '--output', str(output)]
                with self.subTest(invalid_setting=invalid_setting), mock.patch.object(sys, 'argv', argv), mock.patch.object(
                        bench.subprocess, 'check_output', return_value=json.dumps([state])), mock.patch.object(bench, 'metrics') as metrics:
                    with self.assertRaisesRegex(ValueError, 'Unexpected capacity setting'):
                        bench.main()
                    metrics.assert_not_called()
                self.assertFalse(output.exists())

    def test_selected_profile_is_recorded_in_manifest(self):
        for profile, window, clients in (('capacity', 262144, 4), ('baseline', 1048576, 8)):
            with self.subTest(profile=profile), tempfile.TemporaryDirectory() as tmp:
                model_path = Path(tmp) / 'model'; model_path.mkdir()
                (model_path / 'tokenizer.json').write_text('{}')
                (model_path / 'chat_template.jinja').write_text('template')
                output = Path(tmp) / 'out'
                state = self.fixture(window, clients); state['Id'] = 'test-container'
                argv = ['concurrent', '--mtp-profile', profile, '--base-url', 'http://127.0.0.1:8002',
                        '--window', '262144', '--clients', '4', '--output', str(output)]
                transformers = types.ModuleType('transformers'); transformers.AutoTokenizer = mock.Mock()
                case = {'prompt_tokens': 261888, 'token_sha256': 'test', 'request': {}}
                with mock.patch.object(sys, 'argv', argv), mock.patch.dict(sys.modules, {'transformers': transformers}), \
                        mock.patch.object(bench.subprocess, 'check_output', side_effect=[json.dumps([state]), json.dumps({'path': str(model_path)})]), \
                        mock.patch.object(bench, 'prompt_case', return_value=case), mock.patch.object(bench, 'metrics', side_effect=[
                            {'vllm:num_requests_running': 0, 'vllm:num_requests_waiting': 0}, RuntimeError('stop after manifest')]):
                    with self.assertRaisesRegex(RuntimeError, 'stop after manifest'):
                        bench.main()
                manifest = json.loads((output / 'manifest.json').read_text())
                self.assertEqual(manifest['mtp_profile'], profile)
                self.assertEqual(manifest['drafter'], 'mtp')
                self.assertEqual(manifest['command'], state['Config']['Cmd'])


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
