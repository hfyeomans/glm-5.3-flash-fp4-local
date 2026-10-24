#!/usr/bin/env python3
"""Correct the v1 reasoning wrapper without selecting or changing task bodies/golds."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from ruler_bank import render_prompt, sha, ROOT


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    parent=json.loads((a.source/'manifest.json').read_text())
    assert json.loads((a.source/'complete.json').read_text())['manifest_sha256']==sha(a.source/'manifest.json')
    assert parent['protocol']['name']=='local-ruler-13-v1'
    a.output.mkdir(parents=True,exist_ok=False)
    os.environ['TOKENIZERS_PARALLELISM']='false'
    from transformers import AutoTokenizer
    model=json.loads(subprocess.check_output([sys.executable,str(ROOT/'scripts/model.py')],text=True))
    tok=AutoTokenizer.from_pretrained(model['path'],local_files_only=True,trust_remote_code=False)
    assert sha(Path(model['path'])/'tokenizer.json')==parent['protocol']['tokenizer_sha256']
    protocol=dict(parent['protocol'],name='local-ruler-13-v2',client_sha256=sha(ROOT/'quality/ruler_bank.py'),
        parent_manifest_sha256=sha(a.source/'manifest.json'),rewrapper_sha256=sha(Path(__file__)),
        construction='All v1 task bodies, golds, seeds and selected questions preserved. Only render low-effort header and explicitly close think prefix, then retokenize; no task selection or inference outputs read.',
        adaptations=[x for x in parent['protocol']['adaptations'] if not x.startswith('Shared NVIDIA')]+['NVIDIA low-effort template; explicitly closed think prefix and unchanged assistant answer prefix. The wrapper adds a few tokens to the original roughly-4K margin.'])
    start='[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>';sep='<|assistant|><think>'
    cases=[]
    for c in parent['cases']:
        source=a.source/c['file'];assert sha(source)==c['sha256']
        d=json.loads(gzip.decompress(source.read_bytes()));old=d['prompt']
        assert old.startswith(start) and old.count(sep)==1
        text,prefix=old[len(start):].split(sep)
        prompt=render_prompt(tok,text,prefix)
        ids=tok.encode(prompt,add_special_tokens=False)
        assert len(ids)+d['max_tokens']<=d['length']
        d.update(prompt=prompt,prompt_tokens=len(ids),near_window=len(ids)>=.9*d['length'],
                 prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),token_sha256=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),
                 parent_prompt_sha256=c['prompt_sha256'],task_body_sha256=hashlib.sha256(text.encode()).hexdigest())
        dest=a.output/c['file'];dest.write_bytes(gzip.compress(json.dumps(d,ensure_ascii=False).encode(),mtime=0))
        row={k:v for k,v in d.items() if k not in ['prompt','answers']};row.update(file=dest.name,sha256=sha(dest));cases.append(row)
        print(d['id'],len(ids),'wrapper corrected; same task/golds',flush=True)
    manifest=a.output/'manifest.json';manifest.write_text(json.dumps({'protocol':protocol,'cases':cases},indent=2)+'\n')
    (a.output/'complete.json').write_text(json.dumps({'cases':len(cases),'manifest_sha256':sha(manifest)},indent=2)+'\n')


if __name__=='__main__':main()
