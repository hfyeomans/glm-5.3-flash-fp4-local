"""Prevent draft, checkpoint and endpoint misattribution in GPU comparisons."""
import copy
import importlib.util
import io
from unittest import mock
import unittest
from pathlib import Path

spec=importlib.util.spec_from_file_location('dflash2_bench',Path(__file__).resolve().parents[1]/'experiments/dflash2/bench.py')
bench=importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class AttributionTests(unittest.TestCase):
    def fixture(self):
        return {'Image':bench.IMAGE,'State':{'Running':True},'Config':{
            'Labels':{'local.glm53.recipe':'glm53-fp4-local-v1'},
            'Cmd':['vllm','serve','/target/'+bench.REVISION,'--port','8002','--speculative-config',
                   '{"method":"dflash","model":"/cache/models--incoai--GLM-5.3-Flash-DFlash2/snapshots/'+bench.LOCK['revision']+'"}']},
            'HostConfig':{'DeviceRequests':[{'DeviceIDs':['0','1']}]}}

    def test_exact_draft_accepted_but_mtp_label_refused(self):
        state=self.fixture()
        self.assertEqual(bench.validate_server(state,'dflash2','http://127.0.0.1:8002')[1],[0,1])
        with self.assertRaisesRegex(ValueError,'drafter differs'):
            bench.validate_server(state,'mtp','http://127.0.0.1:8002')

    def test_wrong_endpoint_or_draft_cannot_receive_requests(self):
        for base in ('http://127.0.0.1:8000','http://example.com:8002'):
            with self.subTest(base=base), self.assertRaises(ValueError):
                bench.validate_server(self.fixture(),'dflash2',base)
        state=self.fixture(); state['Config']['Cmd'][-1]='{"method":"dflash","model":"another-draft"}'
        with self.assertRaisesRegex(ValueError,'locked DFlash2'):
            bench.validate_server(state,'dflash2','http://127.0.0.1:8002')

    def test_foreign_or_unpinned_service_is_refused(self):
        state=self.fixture()
        variants=[]
        other=copy.deepcopy(state); other['Image']='sha256:other'; variants.append(other)
        other=copy.deepcopy(state); other['Config']['Cmd'][2]='/other/checkpoint'; variants.append(other)
        other=copy.deepcopy(state); other['Config']['Labels']={}; variants.append(other)
        for other in variants:
            with self.assertRaises(ValueError):
                bench.validate_server(other,'dflash2','http://127.0.0.1:8002')


class MetricBoundaryTests(unittest.TestCase):
    def snapshot(self, count, running=0, waiting=0):
        return {bench.shared.METRICS[0]+'_count':count,
                'vllm:num_requests_running':running, 'vllm:num_requests_waiting':waiting}

    def test_reader_accepts_busy_gauges_but_start_boundary_refuses_them(self):
        data=b'vllm:num_requests_running 1\nvllm:num_requests_waiting 0\n'
        with mock.patch.object(bench.shared.urllib.request, 'urlopen', return_value=io.BytesIO(data)):
            totals=bench.read_metrics('http://127.0.0.1:8002')
        with self.assertRaisesRegex(ValueError, 'Other traffic'):
            bench.require_idle(totals)
        with mock.patch.object(bench.shared.urllib.request, 'urlopen', return_value=io.BytesIO(b'vllm:num_requests_running 0\n')):
            with self.assertRaisesRegex(ValueError, 'Missing scheduler gauge'):
                bench.read_metrics('http://127.0.0.1:8002')

    def test_completion_waits_for_both_count_and_idle_gauges(self):
        before=self.snapshot(10)
        samples=[self.snapshot(10,1),self.snapshot(11,1),self.snapshot(11,0,1),self.snapshot(11)]
        with mock.patch.object(bench, 'read_metrics', side_effect=samples), mock.patch.object(bench.time, 'sleep') as sleep:
            self.assertEqual(bench.completion_metrics('local',before),samples[-1])
            self.assertEqual(sleep.call_count,3)

    def test_overlap_and_persistent_busy_state_fail(self):
        before=self.snapshot(10)
        with mock.patch.object(bench, 'read_metrics', return_value=self.snapshot(12)):
            with self.assertRaisesRegex(ValueError, 'exactly one request'):
                bench.completion_metrics('local',before)
        with mock.patch.object(bench, 'read_metrics', return_value=self.snapshot(11,1)):
            with self.assertRaisesRegex(ValueError, 'idle scheduler boundary delayed'):
                bench.completion_metrics('local',before,timeout=0)


if __name__=='__main__':
    unittest.main()
