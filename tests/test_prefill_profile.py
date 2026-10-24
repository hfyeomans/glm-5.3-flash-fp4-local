import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('prefill_report',ROOT/'tensorfold/tuning/prefill_report.py')
report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report)

class TraceCompletenessTests(unittest.TestCase):
    def test_missing_interleaved_and_foreign_rank_records_are_refused(self):
        header='info: glm prompt chunk rank=0 slot=0 start=0 rows=2048 wall_ms=450.0 spans=1\n'
        stage='info: glm prompt stage rank=0 slot=0 start=0 rows=2048 layer=3 piece=null site=ffn stage=experts stream=main ms=125.0\n'
        self.assertEqual(len(report.traces(header+stage,0,2048)),1)
        for broken in (header,header+stage.replace('rank=0','rank=1'),header+stage.replace('piece=null','piece=info: interleaved record')):
            with self.assertRaises(AssertionError):report.traces(broken,0,2048)
