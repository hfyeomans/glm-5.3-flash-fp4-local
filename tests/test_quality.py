"""Guard scoring boundaries and uncertainty calculation without model calls."""
import importlib.util
from pathlib import Path
import unittest
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('quality_common', ROOT / 'quality/common.py')
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)


class QualityTests(unittest.TestCase):
    def test_reasoning_is_not_graded_as_the_final_answer(self):
        self.assertEqual(common.visible_answer('The answer is 42. Still thinking'), ('', False))
        self.assertEqual(common.visible_answer('The answer is 42.</think> The answer is 17. '),
                         ('The answer is 17.', True))

    def test_wilson_interval_retains_uncertainty_at_perfect_score(self):
        low, high = common.wilson(128, 128)
        self.assertGreater(low, 0.97)
        self.assertLess(low, 1)
        self.assertAlmostEqual(high, 1)
        with self.assertRaises(ValueError):
            common.wilson(0, 0)

    def test_request_digest_ignores_dictionary_order_not_values(self):
        self.assertEqual(common.digest({'a': 1, 'b': 2}), common.digest({'b': 2, 'a': 1}))
        self.assertNotEqual(common.digest({'a': 1}), common.digest({'a': 2}))

class QualityComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        module_spec = importlib.util.spec_from_file_location('quality_report', ROOT / 'quality/report.py')
        cls.report = importlib.util.module_from_spec(module_spec)
        with patch.dict(sys.modules, {'common': common}):
            module_spec.loader.exec_module(cls.report)

    def test_pair_direction_and_disagreement_counts(self):
        out = self.report.paired_stats([1, 0, 1, 0], [1, 1, 1, 0], iterations=1000)
        self.assertEqual(out['difference_tensorfold_minus_vllm'], .25)
        self.assertEqual(out['tensorfold_only_correct'], 1)
        self.assertEqual(out['vllm_only_correct'], 0)
        self.assertEqual(out['both_incorrect'], 1)
        self.assertEqual(out['mcnemar_exact_two_sided_p'], 1)

    def test_changed_tokenization_invalidates_comparison(self):
        a = dict(backend='vllm', profile_sha256='p', template_sha256='t', selected_ids={'task': [4]},
                 lock_sha256='l', client_sha256='c', pilot=False)
        b = dict(a, backend='tensorfold')
        calls = {('task', 4): dict(request_sha256='r', prompt_token_sha256='a', prompt_tokens=8)}
        other = {('task', 4): dict(calls[('task', 4)], prompt_token_sha256='b')}
        with self.assertRaisesRegex(ValueError, 'requests/tokens differ'):
            self.report.validate_pair(a, b, calls, other)
