"""Protect timing isolation against concurrent traffic and token-count mismatches."""
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'benchmarks'))
spec = importlib.util.spec_from_file_location('shared_benchmark', ROOT / 'benchmarks/run.py')
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class TimingIsolationTests(unittest.TestCase):
    def sample(self):
        before = {name + suffix: 100.0 for name in bench.METRICS for suffix in ('_sum', '_count')}
        after = before.copy()
        values = (2.0, 3.0, 2.1, 8192, 256)
        for name, value in zip(bench.METRICS, values):
            after[name + '_count'] += 1
            after[name + '_sum'] += value
        return before, after

    def test_decode_excludes_first_token_and_is_separate_from_prefill(self):
        before, after = self.sample()
        got = bench.metric_delta(before, after, 8192, 256)
        self.assertAlmostEqual(got['decode_tokens_per_second'], 85)
        self.assertAlmostEqual(got['prefill_seconds'], 2)
        self.assertAlmostEqual(got['server_ttft_seconds'], 2.1)

    def test_every_histogram_must_isolate_one_request_with_matching_tokens(self):
        for name in bench.METRICS:
            for count in (0, 2):
                before, after = self.sample()
                after[name + '_count'] = before[name + '_count'] + count
                with self.subTest(name=name, count=count), self.assertRaisesRegex(ValueError, 'exactly one request'):
                    bench.metric_delta(before, after, 8192, 256)
        for name in bench.METRICS[-2:]:
            before, after = self.sample()
            after[name + '_sum'] += 1
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'tokens disagree'):
                bench.metric_delta(before, after, 8192, 256)


if __name__ == '__main__':
    unittest.main()
