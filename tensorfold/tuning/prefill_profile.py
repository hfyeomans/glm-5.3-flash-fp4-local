#!/usr/bin/env python3
"""Diagnostic-only prefill spans with per-rank logs; normal timing guard stays intact."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tensorfold/tuning'))
from run import environment
sys.path.insert(0,str(ROOT/'tensorfold/scripts'))
import recipe
sys.path.insert(0,str(ROOT/'benchmarks'))
from prepare import digest
import importlib.util
spec=importlib.util.spec_from_file_location('benchmark_helpers',ROOT/'benchmarks/run.py')
bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
    lock=(ROOT/'.state/tuning-sweep.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    profiles=json.loads((ROOT/'tensorfold/serving/profiles.json').read_text())
    env=environment('gpu-profile',profiles=profiles);os.environ.update(env)
    source=ROOT/'tensorfold/scripts/runner.sh';original=source.read_text()
    old='TF_GLM_RANK="$rank" TF_CUDA_DEVICE="$rank" /opt/tensorfold/bin/tensorfold-native "${args[@]}" &'
    new='TF_GLM_RANK="$rank" TF_CUDA_DEVICE="$rank" /opt/tensorfold/bin/tensorfold-native "${args[@]}" 2>"/profile-logs/rank-$rank.log" &'
    assert original.count(old)==1
    runner=a.output/'runner.sh';runner.write_text(original.replace(old,new));runner.chmod(0o755)
    c=recipe.config();assert recipe.container(c) is None,'Owned container already exists'
    saved=recipe.launch
    def diagnostic_launch(config,model,image):
        args=saved(config,model,image)
        return args[:-1]+['--mount',f'type=bind,src={runner},dst=/opt/recipe/runner.sh,readonly','--mount',f'type=bind,src={a.output},dst=/profile-logs',args[-1]]
    recipe.launch=diagnostic_launch
    result={'status':'failed','scope':'Instrumented diagnostics; not production throughput. Only stderr destination differs from the image launcher.'}
    started=time.time()
    try:
        recipe.main(['start'])
        recipe.main(['smoke','--wait','1200','--suite'])
        state=json.loads(subprocess.check_output(['docker','inspect',env['TF_CONTAINER']],text=True))[0]
        workload=json.loads((ROOT/'logs/tuning/workloads.json').read_text())
        assert digest({k:v for k,v in workload.items() if k!='workloads_sha256'})==workload['workloads_sha256']
        cases=[x for x in workload['cases'] if x['name'] in ['prefill-8192','prefill-32768']]
        assert len(cases)==2
        actual=dict(x.split('=',1) for x in state['Config']['Env'])
        assert actual['TF_PROFILE']=='1' and actual['TF_PROMPT_CACHE_GIB']=='0'
        assert workload['model_revision'] in actual['MODEL_PATH']
        manifest={'image_id':state['Image'],'settings':{k:v for k,v in actual.items() if k.startswith(('TF_','NCCL_')) or k=='MODEL_PATH'},'client_sha256':sha(Path(__file__)),'stream_helper_sha256':sha(ROOT/'benchmarks/run.py'),'original_runner_sha256':sha(source),'runner_sha256':sha(runner),'workloads_sha256':workload['workloads_sha256'],'cases':[x['name'] for x in cases],'repeats':3,'mtp':False}
        (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        base='http://127.0.0.1:8001'
        for case in cases:
            ids=bench.call(base,'/tokenize',{'model':'glm-5.3-flash-fp4-local','prompt':case['prompt'],'add_special_tokens':False})['tokens']
            assert ids==case['tokens']
        rows=[]
        with (a.output/'requests.jsonl').open('w') as out:
            for repeat in range(-1,3):
                order=cases.copy();random.Random(1729+repeat).shuffle(order)
                for case in order:
                    body={'model':'glm-5.3-flash-fp4-local','prompt':case['prompt'],'temperature':0,'top_p':1,'seed':workload['seed'],'max_tokens':workload['max_tokens'],'add_special_tokens':False,'stream':True,'stream_options':{'include_usage':True},'draft':False}
                    offsets={str(rank):(a.output/f'rank-{rank}.log').stat().st_size for rank in (0,1)}
                    response=bench.stream(base,body)
                    row={'case':case['name'],'repeat':repeat,'warmup':repeat<0,'rank_log_offsets':offsets,'prompt_sha256':case.get('prompt_sha256'),'response':response}
                    out.write(json.dumps(row)+'\n');out.flush();rows.append(row)
                    assert response['usage']['prompt_tokens']==len(case['tokens'])
                    assert response['usage']['completion_tokens']==workload['max_tokens']
                    profile_details=response['usage'].get('prompt_tokens_details') or {}
                    assert 'cached_tokens' in profile_details and profile_details['cached_tokens']==0
                    assert response['runtime']['drafts'] is False and response['runtime']['enable_thinking'] is False
                    print(case['name'],repeat,response['client_ttft_seconds'],flush=True)
        result['status']='pass';result['requests']=len(rows)
    finally:
        recipe.main(['stop'])
        result['elapsed_seconds']=time.time()-started
        (a.output/'complete.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
