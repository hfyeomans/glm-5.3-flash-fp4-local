#!/usr/bin/env python3
"""Create shared raw-completion prompts with exact tokenizer-verified lengths."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REVISION = '76348ddb2be2e35e39ecfc00e75dc07703f4f3d5'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    model = json.loads(subprocess.check_output([sys.executable, str(ROOT / 'scripts/model.py')], text=True,
                                              env=dict(os.environ, MODEL_REVISION=REVISION)))
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model['path'], local_files_only=True, trust_remote_code=False)
    template = (Path(model['path']) / 'chat_template.jinja').read_text()

    def render(text):
        prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': text}], tokenize=False,
                    chat_template=template, add_generation_prompt=True, reasoning_effort='low', clear_thinking=False)
        if not prompt.endswith('<think>'):
            raise ValueError('Expected checkpoint template to open reasoning')
        return prompt + '</think>\n\n'

    def item(name, text):
        prompt = render(text)
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        return {'name': name, 'prompt': prompt, 'tokens': ids, 'prompt_tokens': len(ids),
                'prompt_sha256': digest(prompt), 'token_sha256': digest(ids)}

    cases = [item('short-code', 'Explain hash-table collisions and resizing with Python examples. Write at least 1000 words.'),
             item('short-prose', 'Explain database transactions, isolation and recovery. Write at least 1000 words.')]
    intro = 'Read these archive records before writing the requested tutorial.\n'
    question = '\nWrite a detailed tutorial of at least 1000 words on indexing this archive, validating records and recovering damaged indexes.'
    for target in (2048, 8192, 32768, 65536):
        lo, hi = 0, target // 5
        best = None
        while lo <= hi:
            count = (lo + hi) // 2
            records = '\n'.join(f'Record {i}: the archive entry is blue.' for i in range(count))
            candidate = item(f'prefill-{target}', intro + records + '\nPadding:' + question)
            if candidate['prompt_tokens'] <= target:
                best = (records, candidate); lo = count + 1
            else:
                hi = count - 1
        records, candidate = best
        padding = target - candidate['prompt_tokens']
        for _ in range(20):
            candidate = item(f'prefill-{target}', intro + records + '\nPadding:' + ' x' * padding + question)
            gap = target - candidate['prompt_tokens']
            if gap == 0:
                break
            padding += gap
            if padding < 0:
                raise ValueError('Padding underflow')
        if candidate['prompt_tokens'] != target:
            raise ValueError('Cannot fit exact prompt length')
        cases.append(candidate)
    result = {'protocol': 'cold-shared-v1', 'model_revision': REVISION,
              'template_sha256': hashlib.sha256(template.encode()).hexdigest(),
              'interface': 'raw completions; checkpoint low-reasoning template with explicitly closed think prefix',
              'max_tokens': 256, 'seed': 1729, 'cases': cases}
    result['workloads_sha256'] = digest(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as out:
        json.dump(result, out, ensure_ascii=False); out.write('\n')
    print(json.dumps({'workloads_sha256': result['workloads_sha256'],
                      'lengths': {x['name']: x['prompt_tokens'] for x in cases}}, indent=2))


if __name__ == '__main__':
    main()
