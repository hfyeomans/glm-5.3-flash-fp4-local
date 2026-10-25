"""Long-context evidence must distinguish retrieval from mere HTTP success."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('context_check', ROOT / 'tensorfold/scripts/context_check.py')
context = importlib.util.module_from_spec(spec)
spec.loader.exec_module(context)


class ContextTests(unittest.TestCase):
    def test_reply_without_secrets_wrong_count_or_warm_cache_is_not_a_pass(self):
        for content, count, cached, expected_code in [
                ('one two three', 1024, 0, 0), ('only one', 1024, 0, 1),
                ('one two three', 1000, 0, 1), ('one two three', 1024, 100, 1)]:
            response = {'choices': [{'message': {'content': content}, 'finish_reason': 'stop'}],
                        'usage': {'prompt_tokens': count, 'prompt_tokens_details': {'cached_tokens': cached}}}
            with self.subTest(content=content, count=count, cached=cached), tempfile.TemporaryDirectory() as d:
                with patch.object(context, 'fit', return_value=({}, ['one', 'two', 'three'], [10, 50, 90], 1024, 100)), \
                     patch.object(context, 'call', return_value=response), \
                     patch('sys.argv', ['context_check', '--targets', '1024', '--output', str(Path(d) / 'out')]), \
                     contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(context.main(), expected_code)
                self.assertEqual(json.loads((Path(d) / 'out' / 'report.json').read_text())['passed'], expected_code == 0)
