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


def validate_prompt(prompt):
    if not prompt.startswith('[gMASK]<sop><|system|>Reasoning Effort: Low<|user|>'):
        raise ValueError('Unexpected reasoning header')
    if not prompt.rsplit('<|assistant|>', 1)[-1].startswith('<think></think>\n\n'):
        raise ValueError('Raw prompt must explicitly close the reasoning prefix')



REUSE_CLIENT_SHA256 = 'edb5d4a0f3015840c9d35257af027dc7e66bde12e73a2e73725ee8f144a6c97a'


def select_cases(bank, lengths, long_samples=None):
    if long_samples is not None and not 1 <= long_samples <= bank['protocol']['samples_per_task']:
        raise ValueError('Long sample count outside frozen bank')
    return [c for c in bank['cases'] if c['length'] in lengths and
            (long_samples is None or c['length'] != max(lengths) or c['sample'] < long_samples)]


def import_completed(source, bank_dir, selected, manifest, output):
    previous=json.loads((source/'manifest.json').read_text())
    if previous['client_sha256'] != REUSE_CLIENT_SHA256:
        raise ValueError('Unqualified source client for reuse')
    for key in ('bank_sha256','official_scorer_sha256','image_id','settings','interface','minimum_free_MiB','gpu_devices'):
        if previous[key] != manifest[key]:raise ValueError('Reuse protocol differs: '+key)
    sys.path.insert(0,str(ROOT/'tensorfold/tuning'))
    from sweep import headroom
    gate=headroom(source.parent/'gpu.csv',previous.get('gpu_devices') or manifest['gpu_devices'],manifest['minimum_free_MiB'])
    if not gate['passed']:raise ValueError('Source run failed the memory floor')
    old=[json.loads(line) for line in (source/'requests.jsonl').read_text().splitlines()]
    by_id={r['id']:r for r in old};assert len(by_id)==len(old),'Duplicate source rows'
    imported=[]
    for case in selected:
        ident=case['id'];raw=source/(ident+'-response.json')
        if ident not in by_id:
            if raw.exists():raise ValueError('Uncommitted selected source response; inspect before retrying')
            continue
        row=by_id[ident];assert ident in previous['cases']
        file=bank_dir/case['file'];assert sha(file)==case['sha256']
        doc=json.loads(gzip.decompress(file.read_bytes()));validate_prompt(doc['prompt'])
        for key in ('prompt_sha256','token_sha256','prompt_tokens','answers','max_tokens','seed','task','sample','length'):
            assert row[key]==doc[key],('Changed frozen case',ident,key)
        assert json.loads(raw.read_text())==row['response']
        assert row['score']==score(row['response']['choices'][0]['text'],doc['answers'],doc['task'].startswith('qa_'))
        assert row['response']['tensorfold']['drafts'] is True and row['response']['tensorfold']['enable_thinking'] is False
        assert row['response']['tensorfold']['sampling']=='greedy'
        assert row['response']['usage']['prompt_tokens_details']['cached_tokens']==0
        saved=dict(row,reused_from=str(source),original_response_sha256=sha(raw))
        (output/raw.name).write_bytes(raw.read_bytes());imported.append(saved)
    temp=output/'requests.importing.jsonl'
    temp.write_text(''.join(json.dumps(r)+'\n' for r in imported));temp.replace(output/'requests.jsonl')
    (output/'reuse-headroom.json').write_text(json.dumps(gate,indent=2)+'\n')
    print(f'Reused {len(imported)} completed valid requests; no inference repeated',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bank',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--lengths',type=int,nargs='+',required=True)
    p.add_argument('--minimum-free-mib',type=int,default=1536)
    p.add_argument('--devices',help='Comma-separated GPU indices for the headroom gate; defaults to every local GPU')
    p.add_argument('--telemetry-file',type=Path)
    p.add_argument('--long-samples',type=int,help='Use the first N fixed sample indices at the longest nominal length; retain all shorter controls')
    p.add_argument('--reuse-run',type=Path,help='Import completed valid results from the pinned corrected client; never choose by score')
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
    devices=args.devices.split(',') if args.devices else [x.strip() for x in subprocess.check_output(['nvidia-smi','--query-gpu=index','--format=csv,noheader,nounits'],text=True).splitlines()]
    selected=select_cases(bank,args.lengths,args.long_samples)
    manifest={'bank_sha256':sha(args.bank/'manifest.json'),'client_sha256':sha(Path(__file__)),'official_scorer_sha256':sha(scorer_path),
              'image_id':state['Image'],'settings':{k:v for k,v in env.items() if k.startswith(('TF_','NCCL_')) or k=='MODEL_PATH'},
              'cases':[c['id'] for c in selected],'interface':'raw completion, shared locally rendered chat template, thinking off, greedy, MTP on, cold cache',
              'minimum_free_MiB':args.minimum_free_mib,'gpu_devices':devices,
              'response_diagnostics':'Report truncations and generated think markers separately; clean full credit also requires neither. Official RULER score remains unchanged.'}
    manifest['sampling_plan']={'long_samples_per_task':args.long_samples or bank['protocol']['samples_per_task'],'rule':'Lowest fixed sample indices at longest nominal length; all shorter controls retained, independent of scores'}
    if args.reuse_run:
        manifest['reuse']={'source':str(args.reuse_run),'manifest_sha256':sha(args.reuse_run/'manifest.json'),'requests_sha256':sha(args.reuse_run/'requests.jsonl')}
    dest=args.output/'manifest.json'
    if dest.exists():assert json.loads(dest.read_text())==manifest,'Resume protocol differs'
    else:dest.write_text(json.dumps(manifest,indent=2)+'\n')
    rows_path=args.output/'requests.jsonl'
    if args.reuse_run and not rows_path.exists():
        import_completed(args.reuse_run,args.bank,selected,manifest,args.output)
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
        validate_prompt(prompt)
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
        usage_details=usage.get('prompt_tokens_details') or {}
        assert 'cached_tokens' in usage_details and usage_details['cached_tokens']==0
        assert usage['total_tokens']<=int(env['TF_CONTEXT'])
        choice=response['choices'][0];text=choice['text'];metric=score(text,doc['answers'],doc['task'].startswith('qa_'))
        official_fn=official.string_match_part if doc['task'].startswith('qa_') else official.string_match_all
        normalized=re.sub(r'[\x00-\x1f]','\n',text.strip()).strip()
        assert metric==official_fn([normalized],[doc['answers']])
        row={k:v for k,v in doc.items() if k!='prompt'};row.update(started_unix=start,seconds=seconds,score=metric,full_credit=metric==100,
                    truncated=choice['finish_reason']=='length',reasoning_marker=('<think>' in text or '</think>' in text),response=response)
        with rows_path.open('a') as out:out.write(json.dumps(row)+'\n')
        done[doc['id']]=row
        print(f'{i+1}/{len(selected)} {doc["id"]} tokens={doc["prompt_tokens"]} score={metric} seconds={seconds:.2f} truncated={row["truncated"]}',flush=True)
        free=[int(x.split(',')[1]) for x in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()]
        if min(free)<args.minimum_free_mib:raise RuntimeError('Post-request physical headroom below the declared floor')
        if args.telemetry_file:
            sys.path.insert(0,str(ROOT/'tensorfold/tuning'))
            from sweep import headroom
            gate=headroom(args.telemetry_file,devices,args.minimum_free_mib)
            (args.output/'headroom.json').write_text(json.dumps(gate,indent=2)+'\n')
            if not gate['passed']:raise RuntimeError('Sampled headroom floor failed during the run')
    assert set(done)==set(manifest['cases'])
    groups=[]
    for length in args.lengths:
        for task in bank['protocol']['tasks']:
            rows=[r for r in done.values() if r['length']==length and r['task']==task]
            assert len(rows)==sum(c['length']==length and c['task']==task for c in selected)
            groups.append({'configured_context':length,'task':task,'n':len(rows),'mean_score':sum(r['score'] for r in rows)/len(rows),
                           'full_credit':sum(r['full_credit'] for r in rows),'near_window_cases':sum(r['near_window'] for r in rows),
                           'input_tokens_min':min(r['prompt_tokens'] for r in rows),'input_tokens_max':max(r['prompt_tokens'] for r in rows),
                           'truncated':sum(r['truncated'] for r in rows),'reasoning_markers':sum(r['reasoning_marker'] for r in rows),
                           'full_credit_clean_completion':sum(r['full_credit'] and not r['truncated'] and not r['reasoning_marker'] for r in rows)})
    result={'complete':True,'cases':len(done),'groups':groups,'manifest':manifest,'requests_sha256':sha(rows_path),
            'scope':'Local RULER-derived screen using NVIDIA metrics; per-group sample counts and any reused results are explicit, not a full published leaderboard run. Report actual lengths; undersized tasks are not evidence at 1M. Short controls need not have identical synthetic golds; QA question IDs are shared. No inference about vLLM long-context accuracy.'}
    (args.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
