"""Guards added by the 2026-10-25 code hardening must fail closed offline."""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


context = load('context_check_guard', 'tensorfold/scripts/context_check.py')
teacher = load('teacher_force_guard', 'tensorfold/tuning/teacher_force.py')


class GuardTests(unittest.TestCase):
    def test_context_check_refuses_existing_output_directory(self):
        with tempfile.TemporaryDirectory() as d:
            output = Path(d) / 'out'
            output.mkdir()
            with patch('sys.argv', ['context_check', '--targets', '1024', '--output', str(output)]), \
                    contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as ctx:
                    context.main()
            self.assertIn('Refusing to reuse existing output directory', str(ctx.exception))

    def test_teacher_force_rejects_missing_cache_telemetry(self):
        cases = [{'task': 'gsm', 'doc_id': 0, 'prompt': 'p', 'ids': [1, 2, 3], 'score_from': 2}]
        value = {'source_sha256': 'x', 'cases': cases}
        value['sha256'] = teacher.digest({k: v for k, v in value.items() if k != 'sha256'})
        model_json = json.dumps({'container_path': '/model'})
        docker_json = json.dumps([{'State': {'Running': True},
                                   'Config': {'Env': ['MODEL_PATH=/model', 'TF_PROMPT_CACHE_GIB=0']},
                                   'Image': 'sha256:faketestimage'}])
        logprobs = [None,
                    {'2': {'logprob': -0.1, '9': {'logprob': -2.0}}},
                    {'3': {'logprob': -0.2, '9': {'logprob': -2.0}}}]

        def fake_check_output(cmd, *args, **kwargs):
            return model_json if 'model.py' in str(cmd[-1]) else docker_json

        def fake_call(path, body):
            if path == '/tokenize':
                return {'tokens': [1, 2, 3]}
            return {'choices': [{'prompt_logprobs': logprobs}],
                    'usage': {'prompt_tokens': 3}}

        with tempfile.TemporaryDirectory() as d:
            source = Path(d) / 'bank.json'
            source.write_text(json.dumps(value))
            with patch.object(teacher, 'subprocess') as fake_sub, \
                    patch.object(teacher, 'call', fake_call):
                fake_sub.check_output = fake_check_output
                with self.assertRaises(ValueError) as ctx:
                    teacher.run(Path(d) / 'bank.json', Path(d) / 'run')
        self.assertIn('prompt_tokens_details', str(ctx.exception))

    def test_squad_pin_matches_cached_bytes(self):
        resources = json.loads((ROOT / 'quality/resources.json').read_text())
        cache = Path(os.environ.get('HF_CACHE_ROOT', '/models_engrams/huggingface'))
        squad = cache / 'benchmarks/ruler-local-v1/squad.json'
        if not squad.is_file():
            self.skipTest('squad.json not cached on this host')
        self.assertEqual(hashlib.sha256(squad.read_bytes()).hexdigest(),
                         resources['squad_json_sha256'])


if __name__ == '__main__':
    unittest.main()
