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
