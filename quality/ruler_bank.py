#!/usr/bin/env python3
"""Frozen local RULER-derived screen: pinned generators, exact rendered-token budgets."""
import argparse
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
HARNESS = 'd6de81643928d653435c431bae19945d41d32520'
RULER = 'c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a'
ESSAYS = '792d672b77d1a67720f7e9820d646358fc18792e'
TASKS = ['niah_single_1', 'niah_single_2', 'niah_single_3', 'niah_multikey_1',
         'niah_multikey_2', 'niah_multikey_3', 'niah_multivalue', 'niah_multiquery',
         'vt', 'cwe', 'fwe', 'qa_squad', 'qa_hotpot']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit(factory, count, budget, low, high):
    """Keep a verified-fitting whole example; never truncate task content or gold evidence."""
    best = None
    while low <= high:
        size = (low + high) // 2
        text, answers, prefix = factory(size)
        n = count(text, prefix)
        if n <= budget:
            best = (text, answers, prefix, n, size)
            low = size + 1
        else:
            high = size - 1
    if best is None:
        raise ValueError('No intact task example fits the token budget')
    return best



def render_prompt(tokenizer, text, prefix):
    # This checkpoint template always opens <think>; enable_thinking is ignored.
    prompt = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': text}], tokenize=False,
        add_generation_prompt=True, reasoning_effort='low', clear_thinking=False)
    if not prompt.endswith('<|assistant|><think>'):
        raise ValueError('Expected the pinned checkpoint template to open reasoning')
    return prompt + '</think>\n\n' + prefix


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--lengths', nargs='+', type=int, default=[32768, 1048576])
    p.add_argument('--samples', type=int, default=4)
    p.add_argument('--tasks', nargs='+', choices=TASKS, default=TASKS)
    args = p.parse_args()
    if args.samples < 1 or any(n < 8192 or n > 1048576 for n in args.lengths):
        p.error('Invalid size or sample count')
    args.output.mkdir(parents=True, exist_ok=True)
    cache = Path('/models_engrams/huggingface')
    os.environ.update(HF_HOME=str(cache), HF_HUB_CACHE=str(cache/'hub'), HF_DATASETS_CACHE=str(cache/'datasets'),
                      HF_HUB_DISABLE_IMPLICIT_TOKEN='1', NLTK_DATA=str(cache/'benchmarks/nltk'), TOKENIZERS_PARALLELISM='false')
    sys.path.insert(0, str(ROOT/'.state/long-quality-deps'))
    import numpy as np
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from lm_eval.tasks.ruler import prepare_niah as niah, niah_utils, vt_utils as vt, cwe_utils as cwe, fwe_utils as fwe, qa_utils as qa
    direct = json.loads(importlib.metadata.distribution('lm_eval').read_text('direct_url.json'))
    assert direct['vcs_info']['commit_id'] == HARNESS
    assert subprocess.check_output(['git','-C',str(ROOT/'.state/ruler-upstream'),'rev-parse','HEAD'],text=True).strip() == RULER
    assert importlib.metadata.version('wonderwords') == '2.2.0'
    model = json.loads(subprocess.check_output([sys.executable,str(ROOT/'scripts/model.py')],text=True))
    tok = AutoTokenizer.from_pretrained(model['path'],local_files_only=True,trust_remote_code=False)
    template = (Path(model['path'])/'chat_template.jinja').read_text()
    def render(text,prefix):
        return render_prompt(tok, text, prefix)
    def count(text,prefix):
        return len(tok.encode(render(text,prefix),add_special_tokens=False))
    data_dir = cache/'benchmarks/ruler-local-v1';data_dir.mkdir(parents=True,exist_ok=True)
    resources = json.loads((ROOT/'quality/resources.json').read_text())
    def download(url):
        path = data_dir/('hotpot.json' if 'hotpot' in url else 'squad.json')
        if not path.exists():
            if 'hotpot' in url:
                src=cache/'hub/datasets--namlh2004--hotpotqa/snapshots/7e54db4656209750ff487f6fdf8e39a66dba136b/hotpot_dev_distractor_v1.json'
                path.write_bytes(src.read_bytes())
            else:
                with urllib.request.urlopen(url,timeout=120) as response:payload=response.read()
                if hashlib.sha256(payload).hexdigest() != resources['squad_json_sha256']:
                    raise ValueError('Downloaded squad.json does not match the pinned hash in quality/resources.json')
                path.write_bytes(payload)
        return json.loads(path.read_text())
    qa.download_json = download
    essays = re.sub(r'\s+',' ',' '.join(load_dataset('baber/paul_graham_essays',revision=ESSAYS,split='train')['text'])).split()
    # Repeat only the distractor corpus if needed for the local extension beyond published lengths.
    essay_repeats = max(1, math.ceil(max(args.lengths)*1.5/len(essays)))
    haystack = essays * essay_repeats
    corpora={'qa_squad':qa.read_squad(),'qa_hotpot':qa.read_hotpotqa()}
    indices={name:random.Random(1729).sample(range(len(data[0])),args.samples) for name,data in corpora.items()}
    sources={str(Path(m.__file__).name):sha(Path(m.__file__)) for m in (niah,niah_utils,vt,cwe,fwe,qa)}
    protocol={'name':'local-ruler-13-v2','harness_commit':HARNESS,'official_scorer_commit':RULER,'model_revision':model['revision'] if 'revision' in model else '76348ddb2be2e35e39ecfc00e75dc07703f4f3d5',
              'model_path':model['path'],'tokenizer_sha256':sha(Path(model['path'])/'tokenizer.json'),'template_sha256':hashlib.sha256(template.encode()).hexdigest(),
              'client_sha256':sha(Path(__file__)),'lengths':args.lengths,'samples_per_task':args.samples,'tasks':args.tasks,'seed':1729,'max_output_tokens':256,
              'essay_revision':ESSAYS,'essay_repeats':essay_repeats,'qa_indices':indices,'sources':sources,'data_sha256':{p.name:sha(p) for p in sorted(data_dir.glob('*.json'))},
              'adaptations':['Pinned upstream task-construction functions with deterministic per-case seeds and binary sizing of complete examples.','Shared NVIDIA low-reasoning chat template with explicitly closed think prefix, assistant answer prefix; 256-token output cap for all tasks.','Four cases/task/length, not a full published RULER run. QA indices shared across lengths; other length controls need not share identical golds.','Essay distractors repeated if needed at 1M. CWE vocabulary cap retained; undersized examples explicitly excluded from near-window macro summaries.','No silent truncation, retry-based answer selection or kernel/precision changes.']}
    manifest=args.output/'manifest.json'
    if manifest.exists():assert json.loads(manifest.read_text())['protocol']==protocol,'Generation protocol changed'
    else:manifest.write_text(json.dumps({'protocol':protocol,'cases':[]},indent=2)+'\n')
    cases=[]
    for length in args.lengths:
        budget=length-4096
        for task in args.tasks:
            for sample in range(args.samples):
                ident=f'{task}-{length}-{sample}'
                dest=args.output/(ident+'.json.gz')
                seed=1729+TASKS.index(task)*1009+sample*17
                def reset():
                    random.seed(seed);np.random.seed(seed);cwe.RNG.seed(seed)
                def factory(size):
                    reset()
                    if task.startswith('niah_'):
                        typ='repeat' if task=='niah_single_1' else 'needle' if task in ['niah_multikey_2','niah_multikey_3'] else 'essay'
                        value='uuids' if task in ['niah_single_3','niah_multikey_3'] else 'numbers'
                        key='uuids' if task=='niah_multikey_3' else 'words'
                        keys=4 if task in ['niah_multikey_1','niah_multiquery'] else 1
                        vals=4 if task=='niah_multivalue' else 1; queries=4 if task=='niah_multiquery' else 1
                        h=haystack if typ=='essay' else niah.get_haystack(typ)
                        text,answers,query=niah.generate_input_output(size,h,type_haystack=typ,type_needle_k=key,type_needle_v=value,num_needle_k=keys,num_needle_v=vals,num_needle_q=queries,template=niah_utils.TEMPLATE,random_seed=seed)
                        prefix=(f'The special magic {value[:-1]} for {query} mentioned in the provided text is' if vals*queries==1 else f'The special magic {value} for {query} mentioned in the provided text are')
                        return text,answers,prefix
                    if task=='vt':
                        text,answers=vt.generate_input_output(size,1,4)
                        # The full variable-tracking answer prefix includes the known query/count, not gold variables.
                        at=text.rfind(' Answer: According to the chain(s)')
                        return vt_example+'\n\n'+text[:at],answers,text[at:].strip()
                    if task=='cwe':
                        _,text,answers=cwe.generate_input_output(size,length)
                        prefix=cwe.CONFIG['answer_prefix'];return text[:text.rfind(prefix)],answers,prefix.strip()
                    if task=='fwe':
                        text,answers,_=fwe.generate_input_output(10**10,tok,num_words=size,vocab_size=max(2,length//50))
                        prefix=fwe.CONFIG['answer_prefix'];return text[:text.rfind(prefix)],answers,prefix.strip()
                    qas,docs=corpora[task];text,answers=qa.generate_input_output(indices[task][sample],size,qas,docs)
                    return text,answers,qa.CONFIG['answer_prefix']
                if task=='vt':
                    random.seed(42);np.random.seed(42)
                    ex=vt.sys_vartrack_w_noise_random(tok,1,500,incremental=5)[0]
                    vt_example=ex['input']+' '+' '.join(ex['outputs'])
                if dest.exists():
                    payload=json.loads(gzip.decompress(dest.read_bytes()))
                else:
                    upper=len(cwe.WORDS) if task=='cwe' else len(corpora[task][1]) if task.startswith('qa_') else length
                    if task.startswith('niah_') and task not in ['niah_single_1','niah_multikey_2','niah_multikey_3']:upper=len(haystack)
                    text,answers,prefix,n,size=fit(factory,count,budget,10,upper)
                    prompt=render(text,prefix);ids=tok.encode(prompt,add_special_tokens=False)
                    assert len(ids)==n and n+256<=length and all(answers)
                    payload={'id':ident,'task':task,'length':length,'sample':sample,'seed':seed,'construction_size':size,'prompt_tokens':n,'near_window':n>=.9*length,
                             'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'token_sha256':hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),
                             'prompt':prompt,'answers':answers,'max_tokens':256}
                    dest.write_bytes(gzip.compress(json.dumps(payload,ensure_ascii=False).encode(),mtime=0))
                row={k:v for k,v in payload.items() if k not in ['prompt','answers']};row.update(file=dest.name,sha256=sha(dest))
                cases.append(row);manifest.write_text(json.dumps({'protocol':protocol,'cases':cases},indent=2)+'\n')
                print(ident,payload['prompt_tokens'],'near_window',payload['near_window'],flush=True)
    assert len(cases)==len(args.lengths)*len(args.tasks)*args.samples
    (args.output/'complete.json').write_text(json.dumps({'cases':len(cases),'manifest_sha256':sha(manifest)},indent=2)+'\n')


if __name__=='__main__':main()
