#!/usr/bin/env python3
"""Pinned lm-eval tasks/scorers, identical locally rendered prompts, local HTTP engines."""
import argparse
import importlib.metadata
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import urllib.request
from urllib.parse import urlparse

from common import digest, visible_answer

ROOT = Path(__file__).resolve().parents[1]
PROFILE = json.loads((ROOT / 'quality/profile.json').read_text())


def call(base, path, body=None):
    headers = {'Content-Type': 'application/json'}
    if os.environ.get('QUALITY_API_KEY'):
        headers['Authorization'] = 'Bearer ' + os.environ['QUALITY_API_KEY']
    request = urllib.request.Request(base + path, headers=headers,
                                    data=None if body is None else json.dumps(body).encode())
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['vllm', 'tensorfold'], required=True)
    parser.add_argument('--base-url')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=PROFILE['samples_per_task'],
                        help='Prefix of the fixed random selection, for pipeline pilots')
    args = parser.parse_args()
    if not 1 <= args.limit <= PROFILE['samples_per_task']:
        parser.error('limit must be 1..128')
    base = (args.base_url or ('http://127.0.0.1:8000' if args.backend == 'vllm' else 'http://127.0.0.1:8001')).rstrip('/')
    if urlparse(base).hostname not in ('localhost', '127.0.0.1', '::1'):
        parser.error('This recipe sends benchmark prompts only to a local server')
    args.output.mkdir(parents=True, exist_ok=False)
    cache = Path(os.environ.get('HF_CACHE_ROOT', '/models_engrams/huggingface'))
    os.environ.update(HF_HOME=str(cache), HF_HUB_CACHE=str(cache / 'hub'),
                      HF_DATASETS_CACHE=str(cache / 'datasets'), HF_HUB_DISABLE_IMPLICIT_TOKEN='1',
                      NLTK_DATA=str(cache / 'benchmarks/nltk'), TOKENIZERS_PARALLELISM='false')
    model_env = dict(os.environ, MODEL_REVISION=PROFILE['model_revision'])
    model = json.loads(subprocess.check_output([sys.executable, str(ROOT / 'scripts/model.py')], env=model_env, text=True))
    snapshot = Path(model['path'])
    from transformers import AutoTokenizer
    from lm_eval.api.model import LM
    from lm_eval import evaluator
    from lm_eval.tasks import TaskManager
    from lm_eval.api.task import ConfigurableTask
    from lm_eval.tasks._yaml_loader import load_yaml
    from lm_eval.utils import handle_non_serializable
    import langdetect
    langdetect.DetectorFactory.seed = PROFILE['sample_seed']
    direct = json.loads(importlib.metadata.distribution('lm_eval').read_text('direct_url.json'))
    if direct.get('vcs_info', {}).get('commit_id') != PROFILE['harness_commit']:
        raise RuntimeError('Evaluation harness is not the pinned commit; run quality/setup.sh')
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    template = (snapshot / 'chat_template.jinja').read_text()
    alias = 'glm-5.3-flash-fp4-local'
    listed = call(base, '/v1/models')
    if alias not in [x['id'] for x in listed['data']]:
        raise RuntimeError('Unexpected served model identity')
    container = 'glm53-fp4-tp2' if args.backend == 'vllm' else 'glm53-fp4-tensorfold'
    state = json.loads(subprocess.check_output(['docker', 'inspect', container], text=True))[0]
    image_id = state['Image']
    environment = dict(x.split('=', 1) for x in state['Config']['Env'])
    command = state['Config']['Cmd'] or []
    actual_path = command[2] if args.backend == 'vllm' and command[:2] == ['vllm', 'serve'] else environment.get('MODEL_PATH')
    if actual_path != model['container_path'] or not state['State']['Running']:
        raise RuntimeError('Container does not serve the locked checkpoint')
    settings = {k: v for k, v in environment.items() if k in ('TF_CONTEXT', 'TF_PARALLEL', 'TF_PROMPT_CACHE_GIB', 'TF_MEMORY_RESERVE_GIB', 'TF_PROFILE')}
    for option in ('--max-model-len', '--max-num-seqs', '--kv-cache-dtype', '--gpu-memory-utilization', '--kv-cache-memory-bytes', '--tensor-parallel-size', '--decode-context-parallel-size', '--load-format', '--speculative-config'):
        if option in command:
            settings[option] = command[command.index(option) + 1]
    expected_port = environment.get('PORT' if args.backend == 'vllm' else 'TF_PORT')
    if str(urlparse(base).port or (443 if urlparse(base).scheme == 'https' else 80)) != expected_port:
        raise RuntimeError('Benchmark endpoint port does not match the recorded container')
    manager = TaskManager()
    tasks, selected, task_docs = [], {}, {}
    for name, config in PROFILE['tasks'].items():
        task_config = load_yaml(manager.task_index[name].yaml_path, resolve_func=True)
        task_config['dataset_kwargs'] = dict(task_config.get('dataset_kwargs') or {}, revision=config['revision'])
        task = ConfigurableTask(config=task_config)
        docs = task.eval_docs
        if len(docs) != config['expected_size']:
            raise RuntimeError(f'Unexpected dataset size for {name}: {len(docs)}')
        tasks.append(task)
        selected[name] = sorted(random.Random(PROFILE['sample_seed']).sample(range(len(docs)), PROFILE['samples_per_task'])[:args.limit])
        task_docs[name] = docs
    manifest = {'profile': PROFILE, 'profile_sha256': digest(PROFILE), 'backend': args.backend,
                'base_url': base, 'container': container, 'image_id': image_id, 'model': model,
                'template_sha256': hashlib.sha256(template.encode()).hexdigest(), 'selected_ids': selected,
                'lock_sha256': hashlib.sha256((ROOT / 'quality/uv.lock').read_bytes()).hexdigest(),
                'client_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'engine_settings': settings,
                'pilot': args.limit != PROFILE['samples_per_task'],
                'interface': 'raw completions; shared checkpoint chat template; low reasoning; greedy; one request at a time',
                'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')

    class LocalQuality(LM):
        def __init__(self):
            super().__init__()
            self.count = 0

        @property
        def tokenizer_name(self):
            return 'glm53-pinned-shared-low-template'

        def chat_template(self, chat_template=False):
            return template

        def apply_chat_template(self, chat_history, add_generation_prompt=True):
            return tokenizer.apply_chat_template(chat_history, chat_template=template, tokenize=False,
                                                 add_generation_prompt=add_generation_prompt,
                                                 reasoning_effort='low', clear_thinking=False)

        def loglikelihood(self, requests):
            raise NotImplementedError('This profile contains generation tasks only')

        def loglikelihood_rolling(self, requests):
            raise NotImplementedError('Perplexity is not measured by this profile')

        def generate_until(self, requests):
            answers = []
            with (args.output / 'requests.jsonl').open('a') as evidence:
                for instance in requests:
                    doc_id = selected[instance.task_name][instance.doc_id]
                    if digest(instance.doc) != digest(task_docs[instance.task_name][doc_id]):
                        raise RuntimeError('Harness sample IDs do not match the selected dataset rows')
                    prompt, generation = instance.args
                    if not prompt.endswith('<think>'):
                        raise RuntimeError('Expected the shared low-reasoning prompt to open <think>')
                    ids = tokenizer.encode(prompt, add_special_tokens=False)
                    remote = call(base, '/tokenize', {'model': alias, 'prompt': prompt, 'add_special_tokens': False})
                    if remote['tokens'] != ids:
                        raise RuntimeError('Server tokenization differs from the pinned local tokenizer')
                    if len(ids) + PROFILE['max_output_tokens'] > PROFILE['max_input_plus_output_tokens']:
                        raise RuntimeError('Prompt would exceed the common input/output envelope; never truncate silently')
                    body = {'model': alias, 'prompt': prompt, 'max_tokens': PROFILE['max_output_tokens'],
                            'temperature': 0, 'top_p': 1, 'seed': PROFILE['sample_seed'],
                            'add_special_tokens': False, 'stop': generation.get('until', [])}
                    start = time.monotonic()
                    response = call(base, '/v1/completions', body)
                    seconds = time.monotonic() - start
                    choice = response['choices'][0]
                    if response['usage']['prompt_tokens'] != len(ids):
                        raise RuntimeError('Completion usage does not match the exact prompt token count')
                    answer, closed = visible_answer(choice['text'])
                    row = {'task': instance.task_name, 'doc_id': doc_id,
                           'request_sha256': digest(body), 'prompt_token_sha256': digest(ids),
                           'prompt_tokens': len(ids), 'body': body, 'response': response,
                           'seconds': seconds, 'reasoning_closed': closed,
                           'truncated': choice['finish_reason'] == 'length', 'scored_answer': answer}
                    evidence.write(json.dumps(row, ensure_ascii=False) + '\n')
                    evidence.flush()
                    answers.append(answer)
                    self.count += 1
                    print(f'{self.count}/{2 * args.limit} {instance.task_name}:{doc_id} '
                          f'{seconds:.1f}s tokens={response["usage"]["completion_tokens"]} '
                          f'closed={closed} finish={choice["finish_reason"]}', flush=True)
            return answers

    results = evaluator.simple_evaluate(model=LocalQuality(), tasks=tasks, samples=selected,
                                        apply_chat_template=True, fewshot_as_multiturn=False,
                                        bootstrap_iters=0, log_samples=True,
                                        random_seed=1729, numpy_random_seed=1729,
                                        torch_random_seed=1729, fewshot_random_seed=1729)
    (args.output / 'results.json').write_text(json.dumps(results, default=handle_non_serializable, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(results['results'], indent=2), flush=True)


if __name__ == '__main__':
    main()
