#!/usr/bin/env python3
"""Validate complete per-rank prompt traces and report inclusive stage diagnostics."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import statistics


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def traces(text,rank,tokens):
    chunks=[]
    header=re.compile(r'info: glm prompt chunk rank=(\d+) slot=(\d+) start=(\d+) rows=(\d+) wall_ms=([\d.]+) spans=(\d+)$')
    stage=re.compile(r'info: glm prompt stage rank=(\d+) slot=(\d+) start=(\d+) rows=(\d+) layer=(\S+) piece=(\S+) site=(\S+) stage=(\S+) stream=(\S+) ms=([\d.]+)$')
    for line in text.splitlines():
        if 'glm prompt invalid' in line:raise ValueError('Invalid prompt trace')
        if 'glm prompt chunk' in line:
            m=header.fullmatch(line);assert m,'Malformed chunk header'
            assert int(m[1])==rank
            chunks.append({'rank':rank,'slot':int(m[2]),'start':int(m[3]),'rows':int(m[4]),'wall_ms':float(m[5]),'spans':int(m[6]),'events':[]})
        elif 'glm prompt stage' in line:
            m=stage.fullmatch(line);assert m,'Malformed stage record'
            assert chunks and (int(m[1]),int(m[2]),int(m[3]),int(m[4]))==(rank,chunks[-1]['slot'],chunks[-1]['start'],chunks[-1]['rows'])
            chunks[-1]['events'].append({'layer':m[5],'piece':m[6],'site':m[7],'stage':m[8],'stream':m[9],'ms':float(m[10])})
    assert sum(c['rows'] for c in chunks)==tokens,'Missing prompt chunks'
    assert [c['start'] for c in chunks]==list(range(0,tokens,2048))
    assert all(len(c['events'])==c['spans'] for c in chunks),'Incomplete stage records'
    return chunks


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--diagnostic',type=Path,required=True);p.add_argument('--controls',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    load=lambda p:json.loads(p.read_text())
    assert load(a.diagnostic/'complete.json')['status']=='pass'
    diag=[json.loads(s) for s in (a.diagnostic/'requests.jsonl').read_text().splitlines()]
    controls=[]
    for folder in a.controls:
        assert load(folder/'complete.json')['status']=='pass'
        controls.append([json.loads(s) for s in (folder/'bench/requests.jsonl').read_text().splitlines()])
    manifests=[load(folder/'bench/manifest.json') for folder in a.controls]
    dm=load(a.diagnostic/'manifest.json')
    for manifest in manifests:
        assert manifest['image_id']==dm['image_id'] and manifest['workloads_sha256']==dm['workloads_sha256']
        assert {k:v for k,v in manifest['settings'].items() if k!='TF_PROFILE'}=={k:v for k,v in dm['settings'].items() if k!='TF_PROFILE'}
    stage_totals=[]
    for rank in (0,1):
        raw=(a.diagnostic/f'rank-{rank}.log').read_bytes()
        for i,row in enumerate(diag):
            start=row['rank_log_offsets'][str(rank)];end=diag[i+1]['rank_log_offsets'][str(rank)] if i+1<len(diag) else len(raw)
            chunks=traces(raw[start:end].decode(),rank,row['response']['usage']['prompt_tokens'])
            sums=defaultdict(float)
            for chunk in chunks:
                for event in chunk['events']:sums[(event['stream'],event['site'],event['stage'])]+=event['ms']
            stage_totals.append({'case':row['case'],'repeat':row['repeat'],'rank':rank,'chunks':len(chunks),'wall_ms':sum(c['wall_ms'] for c in chunks),'stages':[{'stream':k[0],'site':k[1],'stage':k[2],'ms':v} for k,v in sorted(sums.items())]})
    cells=[]
    for case in dm['cases']:
        cr=[[r for r in rows if r['case']==case and not r['warmup']] for rows in controls]
        dr=[r for r in diag if r['case']==case and not r['warmup']]
        hashes={r['response']['runtime']['token_sha256'] for rows in [*controls,diag] for r in rows if r['case']==case}
        assert len(hashes)==1,'Instrumentation changed output hashes'
        summary={'case':case,'control_client_ttft_medians':[statistics.median(r['response']['client_ttft_seconds'] for r in rows) for rows in cr],'diagnostic_client_ttft_median':statistics.median(r['response']['client_ttft_seconds'] for r in dr),'control_prefill_medians':[statistics.median(r['response']['runtime']['prefill_seconds'] for r in rows) for rows in cr],'control_decode_tps_medians':[statistics.median(r['response']['runtime']['tokens_per_second'] for r in rows) for rows in cr],'all_output_hashes_identical':True,'output_sha256':next(iter(hashes)),'ranks':[]}
        for rank in (0,1):
            rows=[r for r in stage_totals if r['case']==case and r['rank']==rank and r['repeat']>=0]
            keys={(s['stream'],s['site'],s['stage']) for r in rows for s in r['stages']}
            stages=[]
            for key in keys:
                values=[sum(s['ms'] for s in r['stages'] if (s['stream'],s['site'],s['stage'])==key)/r['chunks'] for r in rows]
                stages.append({'stream':key[0],'site':key[1],'stage':key[2],'median_ms_per_chunk':statistics.median(values)})
            summary['ranks'].append({'rank':rank,'n':len(rows),'median_wall_ms_per_chunk':statistics.median(r['wall_ms']/r['chunks'] for r in rows),'stages':sorted(stages,key=lambda s:-s['median_ms_per_chunk'])})
        cells.append(summary)
    files=[a.diagnostic/'manifest.json',a.diagnostic/'requests.jsonl',a.diagnostic/'rank-0.log',a.diagnostic/'rank-1.log',*[f/'bench/requests.jsonl' for f in a.controls]]
    result={'cells':cells,'diagnostic_manifest':dm,'validated_requests':len(diag),'validated_rank_requests':len(stage_totals),'scope':'Instrumented CUDA-event spans are inclusive and may overlap across streams. Do not add FFN to its router/experts children, output_projection to shared_expert children, or side waits to main wall time. No causal attribution of the entire vLLM gap. Controls bracket the preserved failed profiling-client attempt; the successful separate-log diagnostic follows those controls.','evidence':[{'path':str(f),'sha256':sha(f)} for f in files]}
    a.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
