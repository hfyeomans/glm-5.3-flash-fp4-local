import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
def module(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'quality'/f'{name}.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

class RulerTests(unittest.TestCase):
    def test_aliases_are_alternatives_not_multiple_required_answers(self):
        score=module('long_context').score
        self.assertEqual(score('NYC',['New York','NYC'],aliases=True),100)
        self.assertEqual(score('seven',['seven','nine']),50)
        self.assertEqual(score('SEVEN\nnine',['seven','nine']),100)
        self.assertEqual(score('none',['seven','nine']),0)
        with self.assertRaises(ValueError):score('x',[])

    def test_sizing_keeps_whole_gold_example_and_refuses_impossible_fit(self):
        fit=module('ruler_bank').fit
        factory=lambda n: ('x'*n+' GOLD',['GOLD'],' answer:')
        count=lambda text,prefix: len(text+prefix)
        result=fit(factory,count,30,1,100)
        self.assertTrue(result[0].endswith(' GOLD'))
        self.assertLessEqual(result[3],30)
        self.assertGreater(count(*[factory(result[4]+1)[i] for i in (0,2)]),30)
        with self.assertRaises(ValueError):fit(factory,count,1,1,5)

class ReasoningWrapperTests(unittest.TestCase):
    def test_explicit_closure_is_required_when_template_ignores_switch(self):
        class Tokenizer:
            def apply_chat_template(self,messages,**kwargs):
                self.kwargs=kwargs
                return '[gMASK]<sop><|system|>Reasoning Effort: Low<|user|>'+messages[0]['content']+'<|assistant|><think>'
        tok=Tokenizer();render=module('ruler_bank').render_prompt
        result=render(tok,'question','Answer:')
        self.assertTrue(result.endswith('<think></think>\n\nAnswer:'))
        self.assertEqual(tok.kwargs['reasoning_effort'],'low')
        validate=module('long_context').validate_prompt
        validate(result)
        with self.assertRaises(ValueError):validate(result.replace('</think>\n\n',''))
        with self.assertRaises(ValueError):validate(result.replace('Effort: Low','Effort: Max'))

    def test_unrecognized_template_fails_closed(self):
        class Tokenizer:
            def apply_chat_template(self,*args,**kwargs):return '<|assistant|>'
        with self.assertRaises(ValueError):module('ruler_bank').render_prompt(Tokenizer(),'question','Answer:')

class ReducedCohortTests(unittest.TestCase):
    def test_selection_keeps_all_short_controls_and_fixed_first_long_indices(self):
        bank={'protocol':{'samples_per_task':4},'cases':[{'id':f'{length}-{i}','length':length,'sample':i} for length in (32768,1048576) for i in range(4)]}
        select=module('long_context').select_cases
        selected=select(bank,[32768,1048576],2)
        self.assertEqual([c['id'] for c in selected],['32768-0','32768-1','32768-2','32768-3','1048576-0','1048576-1'])
        self.assertEqual(len(select(bank,[32768,1048576])),8)
        with self.assertRaises(ValueError):select(bank,[32768,1048576],0)

class ReuseProtocolTests(unittest.TestCase):
    def test_invalid_reasoning_client_and_different_engine_cannot_be_reused(self):
        import json
        import tempfile
        mod=module('long_context')
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)
            base={'bank_sha256':'bank','official_scorer_sha256':'scorer','image_id':'old','settings':{},'interface':'raw','minimum_free_MiB':1536}
            (source/'manifest.json').write_text(json.dumps(dict(base,client_sha256='invalid-prefix-client')))
            with self.assertRaisesRegex(ValueError,'Unqualified source client'):
                mod.import_completed(source,source,[],base,source)
            (source/'manifest.json').write_text(json.dumps(dict(base,client_sha256=mod.REUSE_CLIENT_SHA256)))
            with self.assertRaisesRegex(ValueError,'image_id'):
                mod.import_completed(source,source,[],dict(base,image_id='new-kernel-image'),source)

class ReporterGateTests(unittest.TestCase):
    def test_reporter_uses_manifest_floor_and_devices(self):
        import hashlib
        import json
        import sys
        import tempfile
        from unittest.mock import patch
        def load(name,path):
            spec=importlib.util.spec_from_file_location(name,path)
            mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
        run_mod=load('report_gate_run',ROOT/'tensorfold/tuning/run.py')
        receipts_mod=load('report_gate_receipts',ROOT/'tensorfold/tuning/receipts.py')
        with patch.dict(sys.modules,{'run':run_mod,'receipts':receipts_mod}):
            sweep_mod=load('report_gate_sweep',ROOT/'tensorfold/tuning/sweep.py')
        lc_mod=module('long_context')
        with patch.dict(sys.modules,{'run':run_mod,'receipts':receipts_mod,'sweep':sweep_mod,'long_context':lc_mod}):
            report=load('report_gate_report',ROOT/'quality/report_long.py')
        sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as folder:
            run=Path(folder)/'run';bank=Path(folder)/'bank'
            tasks=[f'task{i}' for i in range(13)];cases=[];rows=[];groups=[]
            for length in (32768,1048576):
                for task in tasks:
                    for sample in range(4):
                        ident=f'{length}-{task}-{sample}'
                        cases.append({'id':ident,'length':length,'sample':sample,'task':task})
                        rows.append({'id':ident,'task':task,'length':length,'prompt_tokens':length,'score':100,
                                     'full_credit':True,'truncated':False,'reasoning_marker':False,
                                     'near_window':length==1048576,'response':{'choices':[{'text':'answer'}]}})
                    groups.append({'configured_context':length,'task':task,'n':4,'mean_score':100.0,'full_credit':4,
                                   'near_window_cases':0 if length==1048576 and task in ('task11','task12') else 4,
                                   'input_tokens_min':length,'input_tokens_max':length,'truncated':0,
                                   'reasoning_markers':0,'full_credit_clean_completion':4})
            bank.mkdir();(bank/'manifest.json').write_text(json.dumps(
                {'protocol':{'name':'local-ruler-13-v2','samples_per_task':4,'tasks':tasks},'cases':cases}))
            (run/'long-context').mkdir(parents=True)
            manifest={'bank_sha256':sha(bank/'manifest.json'),'client_sha256':sha(ROOT/'quality/long_context.py'),
                      'official_scorer_sha256':'scorer','image_id':'img','settings':{'TF_CONTEXT':'1048576'},
                      'cases':[c['id'] for c in cases],'interface':'raw','minimum_free_MiB':1200,
                      'gpu_devices':['2','3'],'sampling_plan':{'long_samples_per_task':4},'response_diagnostics':'d'}
            requests=run/'long-context/requests.jsonl'
            requests.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            results={'complete':True,'cases':len(cases),'groups':groups,'manifest':manifest,
                     'requests_sha256':sha(requests)}
            (run/'complete.json').write_text(json.dumps({'status':'pass'}))
            (run/'launch-manifest.json').write_text('{}')
            (run/'server.log').write_text('')
            (run/'long-context/manifest.json').write_text('{}')
            (run/'long-context/results.json').write_text(json.dumps(results))
            (run/'gpu.csv').write_text('utc,index,memory_free_MiB\nt0,2,2000\nt0,3,1900\n')
            receipt=Path(folder)/'receipt.json';out=Path(folder)/'report.md'
            with patch('sys.argv',['report_long.py','--run',str(run),'--bank',str(bank),
                                   '--receipt',str(receipt),'--report',str(out)]):
                report.main()
            packed=json.loads(receipt.read_text())
            self.assertEqual(packed['headroom']['required_free_MiB'],1200)
            self.assertEqual(sorted(packed['headroom']['sampled_min_free_MiB']),['2','3'])
            self.assertIn('The 1,200 MiB floor passed',out.read_text())
