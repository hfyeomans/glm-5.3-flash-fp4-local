#!/usr/bin/env python3
"""Run the frozen local RULER-derived bank on the owned TensorFold service; no lifecycle changes."""
import argparse
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def call(path, body=None):
    req=urllib.request.Request('http://127.0.0.1:8001'+path, data=None if body is None else json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=1800) as r:return json.load(r)


def score(prediction, answers, aliases=False):
    if not answers or any(not a for a in answers):raise ValueError('Empty gold answers')
    text=re.sub(r'[\x00-\x1f]','\n',prediction.strip()).strip().lower()
    hits=[answer.lower() in text for answer in answers]
    return round(100*(max(hits) if aliases else sum(hits)/len(hits)),2)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bank',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--lengths',type=int,nargs='+',required=True)
    p.add_argument('--minimum-free-mib',type=int,default=1536)
    p.add_argument('--telemetry-file',type=Path)
    args=p.parse_args()
    bank=json.loads((args.bank/'manifest.json').read_text()); complete=json.loads((args.bank/'complete.json').read_text())
    assert complete['manifest_sha256']==sha(args.bank/'manifest.json')
    scorer_path=ROOT/'.state/ruler-upstream/scripts/eval/synthetic/constants.py'
    spec=importlib.util.spec_from_file_location('ruler_official_metrics',scorer_path);official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
    args.output.mkdir(parents=True,exist_ok=True)
    state=json.loads(subprocess.check_output(['docker','inspect','glm53-fp4-tensorfold'],text=True))[0]
    env=dict(x.split('=',1) for x in state['Config']['Env']);assert state['State']['Running']
    assert env['TF_PARALLEL']=='1' and env['TF_PROMPT_CACHE_GIB']=='0'
    assert env['MODEL_PATH'].endswith(bank['protocol']['model_revision'])
    assert int(env['TF_CONTEXT'])>=max(args.lengths)
    selected=[x for x in bank['cases'] if x['length'] in args.lengths]
    manifest={'bank_sha256':sha(args.bank/'manifest.json'),'client_sha256':sha(Path(__file__)),'official_scorer_sha256':sha(scorer_path),
              'image_id':state['Image'],'settings':{k:v for k,v in env.items() if k.startswith(('TF_','NCCL_')) or k=='MODEL_PATH'},
              'cases':[c['id'] for c in selected],'interface':'raw completion, shared locally rendered chat template, thinking off, greedy, MTP on, cold cache',
              'minimum_free_MiB':args.minimum_free_mib}
    dest=args.output/'manifest.json'
    if dest.exists():assert json.loads(dest.read_text())==manifest,'Resume protocol differs'
    else:dest.write_text(json.dumps(manifest,indent=2)+'\n')
    rows_path=args.output/'requests.jsonl'
    done={r['id']:r for r in map(json.loads,rows_path.read_text().splitlines())} if rows_path.exists() else {}
    if rows_path.exists() and len(done)!=len(rows_path.read_text().splitlines()):
        raise ValueError('Duplicate completed cases')
    for ident,row in done.items():
        expected=next(c for c in selected if c['id']==ident)
        assert row['prompt_sha256']==expected['prompt_sha256'] and row['token_sha256']==expected['token_sha256']
        assert row['score']==score(row['response']['choices'][0]['text'],row['answers'],row['task'].startswith('qa_'))
    for i,case in enumerate(selected):
        if case['id'] in done:continue
        file=args.bank/case['file'];assert sha(file)==case['sha256']
        doc=json.loads(gzip.decompress(file.read_bytes()));prompt=doc['prompt']
        assert hashlib.sha256(prompt.encode()).hexdigest()==doc['prompt_sha256']
        remote=call('/tokenize',{'model':'glm-5.3-flash-fp4-local','prompt':prompt,'add_special_tokens':False})
        assert remote['count']==doc['prompt_tokens']
        assert hashlib.sha256(json.dumps(remote['tokens'],separators=(',',':')).encode()).hexdigest()==doc['token_sha256']
        del remote
        raw_path=args.output/(doc['id']+'-response.json')
        if raw_path.exists():raise ValueError('An uncommitted response exists; inspect before retrying this case')
        body={'model':'glm-5.3-flash-fp4-local','prompt':prompt,'max_tokens':doc['max_tokens'],'temperature':0,'top_p':1,'seed':1729,'draft':True,'add_special_tokens':False}
        start=time.time();response=call('/v1/completions',body);seconds=time.time()-start
        raw_path.write_text(json.dumps(response)+'\n')
        assert response['tensorfold']['drafts'] is True
        assert response['tensorfold']['enable_thinking'] is False
        assert response['tensorfold']['sampling'] == 'greedy'
        assert response['choices'][0]['finish_reason'] in ('stop','length')
        usage=response['usage'];assert usage['prompt_tokens']==doc['prompt_tokens']
        assert usage.get('prompt_tokens_details',{}).get('cached_tokens',0)==0
        assert usage['total_tokens']<=int(env['TF_CONTEXT'])
        choice=response['choices'][0];text=choice['text'];metric=score(text,doc['answers'],doc['task'].startswith('qa_'))
        official_fn=official.string_match_part if doc['task'].startswith('qa_') else official.string_match_all
        normalized=re.sub(r'[\x00-\x1f]','\n',text.strip()).strip()
        assert metric==official_fn([normalized],[doc['answers']])
        row={k:v for k,v in doc.items() if k!='prompt'};row.update(started_unix=start,seconds=seconds,score=metric,full_credit=metric==100,
                    truncated=choice['finish_reason']=='length',response=response)
        with rows_path.open('a') as out:out.write(json.dumps(row)+'\n')
        done[doc['id']]=row
        print(f'{i+1}/{len(selected)} {doc["id"]} tokens={doc["prompt_tokens"]} score={metric} seconds={seconds:.2f} truncated={row["truncated"]}',flush=True)
        free=[int(x.split(',')[1]) for x in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()]
        if min(free)<args.minimum_free_mib:raise RuntimeError('Post-request physical headroom below the declared floor')
        if args.telemetry_file:
            sys.path.insert(0,str(ROOT/'tensorfold/tuning'))
            from sweep import headroom
            gate=headroom(args.telemetry_file,['0','1'],args.minimum_free_mib)
            (args.output/'headroom.json').write_text(json.dumps(gate,indent=2)+'\n')
            if not gate['passed']:raise RuntimeError('Sampled headroom floor failed during the run')
    assert set(done)==set(manifest['cases'])
    groups=[]
    for length in args.lengths:
        for task in bank['protocol']['tasks']:
            rows=[r for r in done.values() if r['length']==length and r['task']==task]
            assert len(rows)==bank['protocol']['samples_per_task']
            groups.append({'configured_context':length,'task':task,'n':len(rows),'mean_score':sum(r['score'] for r in rows)/len(rows),
                           'full_credit':sum(r['full_credit'] for r in rows),'near_window_cases':sum(r['near_window'] for r in rows),
                           'input_tokens_min':min(r['prompt_tokens'] for r in rows),'input_tokens_max':max(r['prompt_tokens'] for r in rows),
                           'truncated':sum(r['truncated'] for r in rows)})
    result={'complete':True,'cases':len(done),'groups':groups,'manifest':manifest,'requests_sha256':sha(rows_path),
            'scope':'Local RULER-derived four-example/task screen using NVIDIA metrics, not a full published leaderboard run. Report actual lengths; undersized tasks are not evidence at 1M. Short controls need not have identical synthetic golds; QA question IDs are shared. No inference about vLLM long-context accuracy.'}
    (args.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
