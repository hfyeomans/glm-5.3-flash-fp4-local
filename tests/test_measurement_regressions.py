"""Exercise endpoint attribution and cross-engine reports without serving a model."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.object(sys, 'path', [str((ROOT / path).parent), *sys.path]):
        spec.loader.exec_module(module)
    return module


bench = load('measurement_run', 'benchmarks/run.py')
report = load('measurement_report', 'benchmarks/report.py')
quality = load('measurement_quality', 'quality/run.py')
decode = load('measurement_decode', 'tensorfold/scripts/decode_bench.py')
concurrent = load('measurement_concurrent', 'benchmarks/concurrent_context.py')
dflash = load('measurement_dflash', 'experiments/dflash2/bench.py')


class EndpointTests(unittest.TestCase):
    def test_vllm_command_port_without_environment_port_reaches_measurement(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'workloads.json'
            workload = {'protocol': 'shared-cold-v1', 'model_revision': 'revision',
                        'seed': 1729, 'max_tokens': 256,
                        'cases': [{'name': 'probe', 'prompt': 'hello', 'tokens': [1],
                                   'prompt_tokens': 1, 'prompt_sha256': 'p', 'token_sha256': 't'}]}
            workload['workloads_sha256'] = bench.digest(workload)
            source.write_text(json.dumps(workload))
            state = {'Image': 'vllm-image', 'State': {'Running': True}, 'Config': {
                'Labels': {'local.glm53.recipe': 'glm53-fp4-local-v1'}, 'Env': [],
                'Cmd': ['vllm', 'serve', '/target', '--port', '8123']}}
            argv = ['run', '--backend', 'vllm', '--mtp', 'off', '--workloads', str(source),
                    '--base-url', 'http://127.0.0.1:8123', '--output', str(Path(tmp) / 'out')]
            for port in ('8123', '8000'):
                argv[argv.index('--base-url') + 1] = 'http://127.0.0.1:' + port
                argv[-1] = str(Path(tmp) / ('out-' + port))
                error, message = (RuntimeError, 'reached tokenizer') if port == '8123' else (ValueError, 'endpoint port')
                with self.subTest(port=port), mock.patch.object(sys, 'argv', argv), mock.patch.object(
                        bench.subprocess, 'check_output', side_effect=[json.dumps([state]), '{"container_path":"/target"}']), \
                        mock.patch.object(bench, 'call', side_effect=RuntimeError('reached tokenizer')) as call:
                    with self.assertRaisesRegex(error, message):
                        bench.main()
                    self.assertEqual(call.call_count, int(port == '8123'))

    def test_quality_command_port_without_environment_port_reaches_task_loading(self):
        tokenizer = mock.Mock()
        modules = {name: types.ModuleType(name) for name in (
            'transformers', 'lm_eval', 'lm_eval.api.model', 'lm_eval.tasks',
            'lm_eval.api.task', 'lm_eval.tasks._yaml_loader', 'lm_eval.utils', 'langdetect')}
        modules['transformers'].AutoTokenizer = tokenizer
        modules['lm_eval.api.model'].LM = object
        modules['lm_eval'].evaluator = mock.Mock()
        modules['lm_eval.tasks'].TaskManager = mock.Mock(side_effect=RuntimeError('reached tasks'))
        modules['lm_eval.api.task'].ConfigurableTask = mock.Mock()
        modules['lm_eval.tasks._yaml_loader'].load_yaml = mock.Mock()
        modules['lm_eval.utils'].handle_non_serializable = mock.Mock()
        modules['langdetect'].DetectorFactory = mock.Mock()
        direct = json.dumps({'vcs_info': {'commit_id': quality.PROFILE['harness_commit']}})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'chat_template.jinja').write_text('template')
            model = {'path': tmp, 'container_path': '/target'}
            state = {'Image': 'image', 'State': {'Running': True}, 'Config': {
                'Env': [], 'Cmd': ['vllm', 'serve', '/target', '--port', '8123']}}
            argv = ['quality', '--backend', 'vllm', '--base-url', 'http://localhost:8123',
                    '--output', str(root / 'out')]
            for port in ('8123', '8000'):
                argv[argv.index('--base-url') + 1] = 'http://localhost:' + port
                argv[-1] = str(root / ('out-' + port))
                with self.subTest(port=port), mock.patch.object(sys, 'argv', argv), mock.patch.dict(sys.modules, modules), \
                        mock.patch.dict(quality.os.environ), mock.patch.object(quality.subprocess, 'check_output',
                        side_effect=[json.dumps(model), json.dumps([state])]), mock.patch.object(
                        quality.importlib.metadata, 'distribution') as dist, mock.patch.object(
                        quality, 'call', return_value={'data': [{'id': 'glm-5.3-flash-fp4-local'}]}) as call:
                    dist.return_value.read_text.return_value = direct
                    message = 'reached tasks' if port == '8123' else 'endpoint port'
                    with self.assertRaisesRegex(RuntimeError, message):
                        quality.main()
                    self.assertEqual(call.call_count, int(port == '8123'))

    def test_concurrent_mtp_rejects_an_endpoint_with_the_wrong_effective_port(self):
        state = {'State': {'Running': True}, 'Config': {
            'Labels': {'local.glm53.recipe': 'glm53-fp4-local-v1'},
            'Cmd': ['vllm', 'serve', '/target/' + concurrent.REVISION, '--port', '80',
                    '--speculative-config', '{"method":"mtp"}', '--max-model-len', '1048576',
                    '--max-num-seqs', '8', '--gpu-memory-utilization', '0.985',
                    '--kv-cache-memory-bytes', '7650410496']},
            'HostConfig': {'DeviceRequests': [{'DeviceIDs': ['0', '1']}]}}
        with tempfile.TemporaryDirectory() as tmp:
            for base in ('http://127.0.0.1:8000', 'https://localhost'):
                argv = ['concurrent', '--window', '8192', '--clients', '1', '--base-url', base,
                        '--output', str(Path(tmp) / 'out')]
                with self.subTest(base=base), mock.patch.object(sys, 'argv', argv), mock.patch.object(
                        concurrent.subprocess, 'check_output', return_value=json.dumps([state])), mock.patch.object(
                        concurrent, 'metrics', side_effect=AssertionError('HTTP attempted')) as metrics:
                    with self.assertRaisesRegex(ValueError, 'Endpoint differs'):
                        concurrent.main()
                    metrics.assert_not_called()

    def test_dflash_rejects_https_default_port_mismatch(self):
        state = {'Image': dflash.IMAGE, 'State': {'Running': True}, 'Config': {
            'Labels': {'local.glm53.recipe': 'glm53-fp4-local-v1'},
            'Cmd': ['vllm', 'serve', '/target/' + dflash.REVISION, '--port', '80']},
            'HostConfig': {'DeviceRequests': [{'DeviceIDs': ['0', '1']}]}}
        with self.assertRaisesRegex(ValueError, 'Endpoint differs'):
            dflash.validate_server(state, 'none', 'https://localhost')

    def test_decode_rejects_remote_endpoint_before_http(self):
        env = {'TF_CONTEXT': '16384', 'TF_PARALLEL': '1', 'TF_PROMPT_CACHE_GIB': '0',
               'TF_MEMORY_RESERVE_GIB': '', 'TF_PROFILE': '0', 'TF_NVFP4_DECODE_WARPS': '4', 'TF_PORT': '8001'}
        state = {'Image': 'image', 'State': {'Running': True},
                 'Config': {'Env': [f'{k}={v}' for k, v in env.items()]}}
        with tempfile.TemporaryDirectory() as tmp:
            for base, message in [('http://example.com:8001', 'Local'),
                                  ('http://127.0.0.1:8000', 'endpoint port')]:
                argv = ['decode', '--base-url', base, '--output', str(Path(tmp) / 'out')]
                with self.subTest(base=base), mock.patch.object(sys, 'argv', argv), mock.patch.object(decode.subprocess, 'check_output',
                        return_value=json.dumps([state])), mock.patch.object(decode, 'call', side_effect=AssertionError('HTTP attempted')) as call:
                    with self.assertRaisesRegex(ValueError, message):
                        decode.main()
                    call.assert_not_called()
                    self.assertFalse((Path(tmp) / 'out').exists())



class ReportTests(unittest.TestCase):
    def fixtures(self, root):
        folders = []
        for name, backend, mode in [('on', 'vllm', 'on'), ('off', 'vllm', 'off'), ('native', 'tensorfold', 'both')]:
            folder = root / name
            folder.mkdir()
            command = ['vllm', 'serve', '/locked', '--port', '8000', '--max-num-seqs', '8']
            if mode == 'on':
                command += ['--speculative-config', '{"method":"mtp","num_speculative_tokens":3}']
            manifest = dict(protocol='shared-cold-v1', workloads_sha256='workloads', model_revision='revision',
                            client_sha256='client', image_id=backend + '-image', backend=backend, mtp=mode,
                            settings={'command': command} if backend == 'vllm' else {'TF_CONTEXT': '16384', 'TF_PARALLEL': '1'},
                            repeats=3, cases=['probe'])
            modes = [True, False] if mode == 'both' else [mode == 'on']
            summary = [dict(case='probe', mtp=draft, n=3, prompt_tokens=8, decode_tokens_per_second=10,
                            prefill_tokens_per_second=100, prefill_seconds=.08, client_ttft_seconds=.1) for draft in modes]
            rows = [dict(case='probe', draft=draft, warmup=repeat == -1, repeat=repeat,
                         response={'usage': {'prompt_tokens': 8, 'completion_tokens': 256,
                                            'prompt_tokens_details': {'cached_tokens': 0}},
                                   'finish_reason': 'length', 'runtime': {'token_sha256': 'tokens'}})
                    for draft in modes for repeat in range(-1, 3)]
            (folder / 'manifest.json').write_text(json.dumps(manifest))
            (folder / 'summary.json').write_text(json.dumps(summary))
            (folder / 'requests.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
            folders.append(folder)
        return folders

    def change(self, folder, file, mutate):
        path = folder / file
        value = json.loads(path.read_text())
        mutate(value)
        path.write_text(json.dumps(value))

    def run_report(self, root, folders):
        with mock.patch.object(sys, 'argv', ['report', *map(str, folders), '--output', str(root / 'out')]), \
                contextlib.redirect_stdout(io.StringIO()):
            report.main()
        return json.loads((root / 'out/comparison.json').read_text())

    def test_cross_engine_images_and_mtp_option_difference_are_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = self.run_report(root, self.fixtures(root))
            self.assertEqual(len(result['cells']), 4)

    def test_split_case_runs_require_consistent_mtp_and_tensorfold_settings(self):
        for variant in ('equivalent_json', 'changed_depth', 'changed_native_settings'):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                folders = self.fixtures(root)
                extra = root / 'extra'
                extra.mkdir()
                second = self.fixtures(extra)
                for folder in second:
                    self.change(folder, 'manifest.json', lambda m: m.update(cases=['probe2']))
                    self.change(folder, 'summary.json', lambda rows: [r.update(case='probe2') for r in rows])
                    path = folder / 'requests.jsonl'
                    rows = [json.loads(line) for line in path.read_text().splitlines()]
                    for row in rows:
                        row['case'] = 'probe2'
                    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
                spec = '{"num_speculative_tokens": 3, "method": "mtp"}'
                if variant == 'changed_depth':
                    spec = '{"method":"mtp","num_speculative_tokens":4}'
                self.change(second[0], 'manifest.json', lambda m: m['settings']['command'].__setitem__(-1, spec))
                if variant == 'changed_native_settings':
                    self.change(second[2], 'manifest.json', lambda m: m['settings'].update(TF_PARALLEL='2'))
                if variant == 'equivalent_json':
                    self.assertEqual(len(self.run_report(root, folders + second)['cells']), 8)
                else:
                    error = 'MTP configurations' if variant == 'changed_depth' else 'tensorfold launch settings'
                    with self.assertRaisesRegex(ValueError, error):
                        self.run_report(root, folders + second)

    def test_incompatible_runs_are_rejected(self):
        mutations = [
            ('image_id', 1, lambda m: m.update(image_id='changed')),
            ('launch settings', 1, lambda m: m['settings']['command'].extend(['--max-model-len', '8192'])),
            ('launch settings', 0, lambda m: m['settings'].update(extra='changed')),
            ('MTP mode', 1, lambda m: m.update(mtp='on')),
            ('other drafters', 0, lambda m: m['settings']['command'].__setitem__(-1, '{"method":"dflash"}')),
            ('client_sha256', 1, lambda m: m.update(client_sha256='changed')),
            ('workloads_sha256', 2, lambda m: m.update(workloads_sha256='changed')),
        ]
        for error, index, mutate in mutations:
            with self.subTest(error=error, index=index), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                folders = self.fixtures(root)
                self.change(folders[index], 'manifest.json', mutate)
                with self.assertRaisesRegex(ValueError, error):
                    self.run_report(root, folders)
                self.assertFalse((root / 'out').exists())

    def test_manifest_cases_cannot_be_silently_omitted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folders = self.fixtures(root)
            for folder in folders:
                self.change(folder, 'manifest.json', lambda m: m['cases'].append('missing'))
            with self.assertRaisesRegex(ValueError, 'coverage'):
                self.run_report(root, folders)

    def test_missing_cache_telemetry_bad_finish_or_duplicate_repeats_are_rejected(self):
        for mode in ('missing_cache', 'cached', 'early_finish', 'duplicate_repeat'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                folders = self.fixtures(root)
                path = folders[0] / 'requests.jsonl'
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                if mode == 'missing_cache':
                    rows[1]['response']['usage'].pop('prompt_tokens_details')
                elif mode == 'cached':
                    rows[1]['response']['usage']['prompt_tokens_details']['cached_tokens'] = 1
                elif mode == 'early_finish':
                    rows[1]['response']['finish_reason'] = 'stop'
                else:
                    rows[1]['repeat'] = rows[2]['repeat']
                path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
                with self.assertRaises(ValueError):
                    self.run_report(root, folders)

    def test_mtp_option_difference_is_accepted_independently_of_image_rule(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folders = self.fixtures(root)
            for folder in folders:
                self.change(folder, 'manifest.json', lambda m: m.update(image_id='same-image'))
            self.assertEqual(len(self.run_report(root, folders)['cells']), 4)


if __name__ == '__main__':
    unittest.main()
