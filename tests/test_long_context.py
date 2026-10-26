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
    def setUp(self):
        import hashlib
        import json
        import sys
        import tempfile
        from unittest.mock import patch
        self.json=json
        self.patch=patch
        self.sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
        def load(name,path):
            spec=importlib.util.spec_from_file_location(name,path)
            mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
        run_mod=load('report_gate_run',ROOT/'tensorfold/tuning/run.py')
        receipts_mod=load('report_gate_receipts',ROOT/'tensorfold/tuning/receipts.py')
        with patch.dict(sys.modules,{'run':run_mod,'receipts':receipts_mod}):
            sweep_mod=load('report_gate_sweep',ROOT/'tensorfold/tuning/sweep.py')
        lc_mod=module('long_context')
        with patch.dict(sys.modules,{'run':run_mod,'receipts':receipts_mod,'sweep':sweep_mod,'long_context':lc_mod}):
            self.report=load('report_gate_report',ROOT/'quality/report_long.py')
        folder=tempfile.TemporaryDirectory();self.addCleanup(folder.cleanup)
        self.run=Path(folder.name)/'run';self.bank=Path(folder.name)/'bank'
        self.receipt=Path(folder.name)/'receipt.json';self.out=Path(folder.name)/'report.md'
        self.tasks=[f'task{i}' for i in range(13)];self.cases=[];self.rows=[];self.groups=[]
        for length in (32768,1048576):
            for task in self.tasks:
                near_window=length==1048576 and task not in ('task11','task12')
                for sample in range(4):
                    ident=f'{length}-{task}-{sample}'
                    self.cases.append({'id':ident,'length':length,'sample':sample,'task':task})
                    self.rows.append({'id':ident,'task':task,'length':length,'prompt_tokens':length,'score':100,
                                 'full_credit':True,'truncated':False,'reasoning_marker':False,
                                 'near_window':near_window,'response':{'choices':[{'text':'answer'}]}})
                self.groups.append({'configured_context':length,'task':task,'n':4,'mean_score':100.0,'full_credit':4,
                               'near_window_cases':4 if near_window else 0,
                               'input_tokens_min':length,'input_tokens_max':length,'truncated':0,
                               'reasoning_markers':0,'full_credit_clean_completion':4})
        self.bank.mkdir();(self.bank/'manifest.json').write_text(json.dumps(
            {'protocol':{'name':'local-ruler-13-v2','samples_per_task':4,'tasks':self.tasks},'cases':self.cases}))
        (self.run/'long-context').mkdir(parents=True)
        self.manifest={'bank_sha256':self.sha(self.bank/'manifest.json'),'client_sha256':self.sha(ROOT/'quality/long_context.py'),
                  'official_scorer_sha256':'scorer','image_id':'img','settings':{'TF_CONTEXT':'1048576'},
                  'cases':[c['id'] for c in self.cases],'interface':'raw','minimum_free_MiB':1200,
                  'gpu_devices':['2','3'],'sampling_plan':{'long_samples_per_task':4},'response_diagnostics':'d'}
        self.results={'complete':True,'cases':len(self.cases),'groups':self.groups,'manifest':self.manifest}
        (self.run/'complete.json').write_text(json.dumps({'status':'pass'}))
        (self.run/'launch-manifest.json').write_text('{}')
        (self.run/'server.log').write_text('')
        # An unrelated GPU below the floor must not affect the selected devices.
        (self.run/'gpu.csv').write_text('utc,index,memory_free_MiB\nt0,0,10\nt0,2,2000\nt0,3,1900\n')
        self.write_run()

    def write_run(self):
        requests=self.run/'long-context/requests.jsonl'
        requests.write_text(''.join(self.json.dumps(r)+'\n' for r in self.rows))
        self.results['requests_sha256']=self.sha(requests)
        (self.run/'long-context/manifest.json').write_text(self.json.dumps(self.manifest))
        (self.run/'long-context/results.json').write_text(self.json.dumps(self.results))

    def controls_only(self):
        self.rows=[r for r in self.rows if r['length']==32768]
        self.results['groups']=[g for g in self.groups if g['configured_context']==32768]
        self.results['cases']=len(self.rows)
        self.manifest['cases']=[r['id'] for r in self.rows]
        self.write_run()

    def invoke(self,*extra):
        with self.patch('sys.argv',['report_long.py','--run',str(self.run),'--bank',str(self.bank),
                                   '--receipt',str(self.receipt),'--report',str(self.out),*extra]):
            self.report.main()

    def test_reporter_uses_manifest_floor_and_devices(self):
        self.invoke()
        packed=self.json.loads(self.receipt.read_text())
        self.assertEqual(packed['headroom']['required_free_MiB'],1200)
        self.assertEqual(sorted(packed['headroom']['sampled_min_free_MiB']),['2','3'])
        self.assertIn('The 1,200 MiB floor passed',self.out.read_text())
        self.assertEqual(packed['near_1m']['n'],44)
        self.assertEqual(packed['near_1m']['clean_full_credit'],44)
        (self.run/'long-context/manifest.json').write_text('{"tampered":true}')
        with self.assertRaisesRegex(AssertionError,'Embedded manifest'):
            self.invoke()

    def test_short_run_requires_explicit_controls_mode(self):
        self.controls_only()
        with self.assertRaises(AssertionError):
            self.invoke()
        self.assertFalse(self.receipt.exists())

    def test_controls_only_report(self):
        self.controls_only()
        self.invoke('--controls-only')
        packed=self.json.loads(self.receipt.read_text())
        self.assertEqual(packed['cohort_requests'],52)
        self.assertEqual(packed['short_controls']['clean_full_credit'],52)
        self.assertIsNone(packed['near_1m'])
        self.assertNotIn('both groups',packed['context_label_semantics'])
        self.assertEqual(packed['headroom']['required_free_MiB'],1200)
        self.assertEqual(sorted(packed['headroom']['sampled_min_free_MiB']),['2','3'])
        self.assertIn('The 1,200 MiB floor passed',self.out.read_text())
        self.assertIn('Near-1M cases were not run',self.out.read_text())
        self.assertNotIn('user reduced',self.out.read_text())
        evidence={e['path']:e['sha256'] for e in packed['evidence']}
        path=self.run/'long-context/requests.jsonl'
        self.assertEqual(evidence[str(path)],self.sha(path))

    def test_controls_only_accepts_fractional_scores(self):
        self.controls_only()
        scores=[100/3,100/7,100/11,100/13]
        for row,score in zip(self.rows,scores):
            row.update(score=score,full_credit=False)
        group=self.results['groups'][0]
        group.update(mean_score=sum(scores)/len(scores),full_credit=0,full_credit_clean_completion=0)
        self.write_run()
        self.invoke('--controls-only')
        packed=self.json.loads(self.receipt.read_text())
        self.assertEqual(packed['short_controls']['clean_full_credit'],48)
        self.assertEqual(len(packed['all_nonclean_or_incorrect_cases']),4)
        self.assertAlmostEqual(packed['short_controls']['mean_task_score'],(1200+group['mean_score'])/13)

    def test_controls_only_rejects_incomplete_unknown_and_long_cases(self):
        self.controls_only()
        original_rows=list(self.rows)
        for kind in ('missing','unknown','long','wrong_task','wrong_length'):
            with self.subTest(kind=kind):
                self.rows=[dict(r) for r in original_rows]
                if kind=='missing':self.rows.pop()
                elif kind=='unknown':self.rows[0]['id']='unknown'
                elif kind=='long':self.rows.append(dict(self.rows[0],id='1048576-task0-0',length=1048576))
                elif kind=='wrong_task':self.rows[0]['task']='unknown'
                else:self.rows[0]['length']=1048576
                self.manifest['cases']=[r['id'] for r in self.rows]
                self.results['cases']=len(self.rows)
                self.write_run()
                with self.assertRaises(AssertionError):self.invoke('--controls-only')
                self.assertFalse(self.receipt.exists())

    def test_controls_only_preserves_manifest_and_hash_gates(self):
        self.controls_only()
        manifest_path=self.run/'long-context/manifest.json'
        manifest_path.write_text('{"tampered":true}')
        with self.assertRaisesRegex(AssertionError,'Embedded manifest'):self.invoke('--controls-only')
        self.write_run()
        requests=self.run/'long-context/requests.jsonl'
        requests.write_text(requests.read_text()+'\n')
        with self.assertRaises(AssertionError):self.invoke('--controls-only')
        self.write_run()
        for key in ('bank_sha256','client_sha256'):
            with self.subTest(key=key):
                original=self.manifest[key]
                self.manifest[key]='wrong'
                self.write_run()
                with self.assertRaises(AssertionError):self.invoke('--controls-only')
                self.manifest[key]=original
        self.results['complete']=False
        self.write_run()
        with self.assertRaises(AssertionError):self.invoke('--controls-only')

    def test_controls_only_rejects_inconsistent_groups(self):
        self.controls_only()
        original=[dict(g) for g in self.results['groups']]
        for key,value in (('task','unknown'),('configured_context',1048576),('n',3),('mean_score',50),
                          ('full_credit_clean_completion',3),('input_tokens_min',1)):
            with self.subTest(key=key):
                self.results['groups']=[dict(g) for g in original]
                self.results['groups'][0][key]=value
                self.write_run()
                with self.assertRaises(AssertionError):self.invoke('--controls-only')
                self.assertFalse(self.receipt.exists())

    def test_controls_only_rejects_tampered_group_counters(self):
        self.controls_only()
        original=[dict(g) for g in self.results['groups']]
        for counter in ('full_credit','near_window_cases','truncated','reasoning_markers'):
            with self.subTest(counter=counter):
                self.receipt.unlink(missing_ok=True)
                self.out.unlink(missing_ok=True)
                self.results['groups']=[dict(g) for g in original]
                self.results['groups'][0][counter]+=1
                self.write_run()
                with self.assertRaises(AssertionError):self.invoke('--controls-only')
                self.assertFalse(self.receipt.exists())

    def test_controls_only_rejects_selected_gpu_below_floor(self):
        self.controls_only()
        (self.run/'gpu.csv').write_text('utc,index,memory_free_MiB\nt0,2,1199\nt0,3,1900\n')
        with self.assertRaises(AssertionError):self.invoke('--controls-only')

class PhysicalGateTests(unittest.TestCase):
    def test_physical_check_scopes_to_selected_devices(self):
        check=module('long_context').physical_shortfalls
        free={'0':900,'2':2000,'3':1900}
        self.assertIsNone(check(free,['2','3'],1536))
        self.assertIn('GPU 0',check(free,['0','2'],1536))
        self.assertIn('absent',check({'0':900},['2','3'],1536))
