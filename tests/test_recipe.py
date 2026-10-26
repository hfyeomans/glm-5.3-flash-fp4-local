"""Operational boundaries using fixtures, fake Docker and local HTTP only."""
import contextlib
import io
import hashlib
import json
import os
import shutil
import shlex
import socket
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import model
import smoke

TEST_REV = '0'*40


def checkpoint(path, *, quant='modelopt'):
    path.mkdir(parents=True)
    (path/'config.json').write_text(json.dumps({'model_type':'glm5_next',
        'quantization_config':{'quant_method':quant,'encoding':'csf_32x2_v1'}}))
    (path/'model.safetensors.index.json').write_text(json.dumps({
        'metadata':{'total_size':8},'weight_map':{'tensor':'model-00001.safetensors'}}))
    header = json.dumps({'tensor': {'dtype': 'U8', 'shape': [8], 'data_offsets': [0, 8]}}).encode()
    (path/'model-00001.safetensors').write_bytes(len(header).to_bytes(8, 'little') + header + b'fixture!')
    for name in ['tokenizer.json','tokenizer_config.json']:
        (path/name).write_text('{}')
    return path


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.hub = self.base/'hub'
        self.snapshot = self.hub/'models--nvidia--GLM-5.3-Flash-FP4-Local'/'snapshots'/TEST_REV
        checkpoint(self.snapshot)

    def test_complete_snapshot_stays_inside_hub_mount(self):
        r = model.inspect_model(self.hub, TEST_REV)
        self.assertEqual(r['shards'],1)
        self.assertEqual(r['bytes'], (self.snapshot/'model-00001.safetensors').stat().st_size)
        self.assertTrue(r['container_path'].startswith(str(model.CONTAINER_HUB)))

    def test_plan_uses_same_automatically_discovered_local_copy(self):
        local = self.base/'nvidia/glm-5.3-flash-fp4-local'
        local.parent.mkdir()
        self.snapshot.rename(local)
        with patch.dict(os.environ, {'HF_HOME': str(self.base)}):
            planned = model.plan_model(self.hub, TEST_REV)
            actual = model.inspect_model(self.hub, TEST_REV)
        for key in ('path', 'mount_source', 'mount_target', 'container_path'):
            self.assertEqual(planned[key], actual[key])
        self.assertEqual(planned['container_path'], '/model')

    def test_plan_and_inspection_both_refuse_ambiguous_local_copies(self):
        checkpoint(self.base/'glm-5.3-flash-fp4-local')
        with patch.dict(os.environ, {'HF_HOME': str(self.base)}):
            for resolve in (model.plan_model, model.inspect_model):
                with self.subTest(resolve=resolve.__name__), self.assertRaisesRegex(ValueError, 'Multiple local copies'):
                    resolve(self.hub, TEST_REV)

    def test_plan_without_weights_uses_requested_snapshot_or_directory(self):
        shutil.rmtree(self.hub)
        with patch.dict(os.environ, {'HF_HOME': str(self.base)}):
            planned = model.plan_model(self.hub, TEST_REV)
            self.assertEqual(planned['path'], str(self.snapshot))
            self.assertEqual(planned['mount_source'], str(self.hub))
            local = self.base/'absent-model'
            planned = model.plan_model(self.hub, TEST_REV, local)
            self.assertEqual(planned['path'], str(local))
            self.assertEqual(planned['container_path'], '/model')

    def test_missing_shard_fails_without_downloading(self):
        (self.snapshot/'model-00001.safetensors').unlink()
        with self.assertRaisesRegex(ValueError,'Missing, empty or broken shard'):
            model.inspect_model(self.hub,TEST_REV)

    def test_original_fp8_checkpoint_is_rejected(self):
        c = json.loads((self.snapshot/'config.json').read_text())
        c['quantization_config']['quant_method']='fp8'
        (self.snapshot/'config.json').write_text(json.dumps(c))
        with self.assertRaisesRegex(ValueError,'expected NVIDIA'):
            model.inspect_model(self.hub,TEST_REV)

    def test_shard_symlink_must_be_visible_in_container(self):
        shard = self.snapshot/'model-00001.safetensors'
        shard.unlink()
        outside = self.base/'outside.safetensors'
        outside.write_bytes(b'fixture!')
        shard.symlink_to(outside)
        with self.assertRaisesRegex(ValueError,'symlink escapes'):
            model.inspect_model(self.hub,TEST_REV)

    def test_hub_relative_blob_links_are_supported(self):
        shard = self.snapshot/'model-00001.safetensors'
        blob = self.snapshot.parent.parent/'blobs'/'hash'
        blob.parent.mkdir();shard.rename(blob)
        shard.symlink_to('../../blobs/hash')
        self.assertEqual(model.inspect_model(self.hub,TEST_REV)['bytes'],blob.stat().st_size)

    def test_pinned_metadata_cannot_be_substituted(self):
        with self.assertRaisesRegex(ValueError,'metadata differs'):
            model.inspect_model(self.hub, model.REVISION, self.snapshot)

    def test_absolute_internal_link_is_rejected(self):
        shard = self.snapshot/'model-00001.safetensors'
        blob = self.snapshot/'blob'
        shard.rename(blob); shard.symlink_to(blob)
        with self.assertRaisesRegex(ValueError, 'absolute symlink'):
            model.inspect_model(self.hub, TEST_REV)

    def test_relative_link_through_absolute_directory_is_rejected(self):
        shard = self.snapshot/'model-00001.safetensors'
        blobs = self.snapshot/'blobs'; blobs.mkdir()
        shard.rename(blobs/'blob')
        (self.snapshot/'alias').symlink_to(blobs, target_is_directory=True)
        shard.symlink_to('alias/blob')
        with self.assertRaisesRegex(ValueError, 'absolute symlink'):
            model.inspect_model(self.hub, TEST_REV)

    def test_host_hub_alias_is_supported(self):
        alias = self.base/'hub-alias'
        alias.symlink_to(self.hub, target_is_directory=True)
        result = model.inspect_model(alias, TEST_REV)
        self.assertEqual(result['mount_source'], str(self.hub))

    def test_invalid_shard_offsets_are_rejected(self):
        shard = self.snapshot/'model-00001.safetensors'
        for extent in ([1,8], [0,9], [0,-1], [0,True]):
            with self.subTest(extent=extent):
                header=json.dumps({'tensor':{'dtype':'U8','shape':[8],'data_offsets':extent}}).encode()
                shard.write_bytes(len(header).to_bytes(8,'little')+header+b'fixture!')
                with self.assertRaises(ValueError):
                    model.inspect_model(self.hub, TEST_REV)

    def test_one_byte_truncation_fails_despite_header_overhead(self):
        shard = self.snapshot/'model-00001.safetensors'
        shard.write_bytes(shard.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError, 'Truncated|size'):
            model.inspect_model(self.hub, TEST_REV)

    def test_index_must_name_tensors_present_in_each_shard(self):
        index_path = self.snapshot/'model.safetensors.index.json'
        index = json.loads(index_path.read_text())
        index['weight_map'] = {'different_tensor': 'model-00001.safetensors'}
        index_path.write_text(json.dumps(index))
        with self.assertRaisesRegex(ValueError, 'tensor.*index|index.*tensor'):
            model.inspect_model(self.hub, TEST_REV)

    def test_alternate_revision_with_pinned_metadata_retains_revision(self):
        hashes = {name: hashlib.sha256((self.snapshot/name).read_bytes()).hexdigest()
                  for name in ['config.json', 'model.safetensors.index.json']}
        read_text = Path.read_text
        def fixture_lock(path, *args, **kwargs):
            if path.name == 'model-lock.json':
                return json.dumps({'metadata_sha256': hashes})
            return read_text(path, *args, **kwargs)
        with patch.object(Path, 'read_text', fixture_lock):
            self.assertEqual(model.inspect_model(self.hub, TEST_REV)['revision'], TEST_REV)
            self.assertEqual(model.inspect_model(self.hub, '1'*40, self.snapshot)['revision'],
                             'local-directory-unverified-revision')
            original_requested = model.inspect_model(self.hub, model.REVISION, self.snapshot)
            self.assertEqual(original_requested['revision'], 'local-directory-unverified-revision')
            self.assertEqual(original_requested['requested_revision'], model.REVISION)


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)/'recipe'
        self.root.mkdir()
        for p in ROOT.glob('*.sh'): shutil.copy2(p,self.root/p.name)
        shutil.copytree(ROOT/'scripts',self.root/'scripts',ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copytree(ROOT/'config',self.root/'config')
        self.bin = Path(self.tmp.name)/'bin';self.bin.mkdir()
        self.calls = Path(self.tmp.name)/'docker-calls.jsonl'
        names = ['HF_CACHE_ROOT','HF_HOME','HF_HUB_CACHE','MODEL_DIR','MODEL_REVISION','LOAD_FORMAT','MTP_ENABLED','DRAFTER','DFLASH2_MODEL_DIR','DFLASH2_NUM_SPECULATIVE_TOKENS', 'DFLASH2_KV_CACHE_DTYPE','IMAGE','CONTAINER_NAME',
                 'GPUS','HOST','PORT','SERVED_MODEL_NAME','RUNTIME_VOLUME','MAX_MODEL_LEN','MAX_NUM_SEQS',
                 'MAX_NUM_BATCHED_TOKENS','GPU_MEMORY_UTILIZATION','KV_CACHE_MEMORY_BYTES',
                 'MIN_GPU_FREE_MIB','WAIT_TIMEOUT','SMOKE_TIMEOUT','RESTART_POLICY','DRY_RUN','BASE_URL']
        self.env = {k:v for k,v in os.environ.items() if k not in names}
        self.env.update(PATH=str(self.bin)+os.pathsep+os.environ['PATH'],CALLS=str(self.calls),
                        PYTHONDONTWRITEBYTECODE='1',OWNER='foreign',HF_HOME='/old/cache',
                        HF_HUB_CACHE='/old/cache/hub',HF_TOKEN='secret-fixture-token',VLLM_API_KEY='secret-fixture-api')
        docker = self.bin/'docker'
        docker.write_text('#!'+sys.executable+r'''
import json,os,sys
with open(os.environ['CALLS'],'a') as f: f.write(json.dumps(sys.argv[1:])+'\n')
a=sys.argv[1:]
if a[0]=='info': print('{"nvidia":{}}');sys.exit(0)
if a[:2]==['container','inspect']:
    status=int(os.environ.get('ABSENT','0'))
    if status: print(os.environ.get('INSPECT_ERROR', 'Error response from daemon: No such container: '+a[-1]), file=sys.stderr)
    sys.exit(status)
if a[:2]==['image','inspect']: print('sha256:fixture');sys.exit(0)
if a[0]=='inspect':
    if 'Running' in a[2]: print(os.environ.get('FAKE_RUNNING', 'false'))
    elif 'Config.Cmd' in a[2]: print(json.dumps(['vllm','serve','/model','--port',os.environ.get('FAKE_SERVING_PORT', '8000')]))
    else: print(os.environ.get('OWNER','foreign'))
    sys.exit(0)
if a[0]=='ps': print('other-gpu-service\tother-image');sys.exit(0)
if a[0]=='logs':
    if os.environ.get('LOGS_ERROR'): sys.exit(1)
    print('fixture log');sys.exit(0)
if a[0]=='stop': sys.exit(0)
if a[0]=='rm': sys.exit(0)
if a[0]=='run': print('fixture-container-id');sys.exit(0)
sys.exit(70)
''')
        docker.chmod(0o755)
        gpu=self.bin/'nvidia-smi'
        gpu.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "0, NVIDIA RTX PRO 6000 Blackwell, 97887, 10, fixture" "1, NVIDIA RTX PRO 6000 Blackwell, 97887, 10, fixture"\n')
        gpu.chmod(0o755)

    def run_script(self, name, *args):
        return subprocess.run([str(self.root/name),*args],cwd=self.root,env=self.env,
                              text=True,capture_output=True,timeout=15)

    def commands(self):
        return [json.loads(line) for line in self.calls.read_text().splitlines()] if self.calls.exists() else []

    def test_dry_run_uses_correct_cache_without_side_effects_or_secrets(self):
        before = sorted(str(p.relative_to(self.root)) for p in self.root.rglob('*'))
        r = self.run_script('start.sh','--dry-run')
        self.assertEqual(r.returncode,0,r.stderr)
        self.assertIn('/models_engrams/huggingface/hub',r.stdout)
        self.assertNotIn('/old/cache',r.stdout)
        self.assertNotIn('secret-fixture',r.stdout+r.stderr)
        self.assertFalse(self.calls.exists())
        self.assertEqual(before,sorted(str(p.relative_to(self.root)) for p in self.root.rglob('*')))

    def test_mtp_off_removes_speculation_without_touching_budget_or_docker(self):
        baseline = self.run_script('start.sh', '--dry-run')
        self.assertIn('--speculative-config', baseline.stdout)
        self.env['MTP_ENABLED'] = '0'
        plain = self.run_script('start.sh', '--dry-run')
        self.assertEqual(plain.returncode, 0, plain.stderr)
        self.assertNotIn('--speculative-config', plain.stdout)
        self.assertIn('7650410496', plain.stdout)
        self.assertIn('0.985', plain.stdout)
        self.assertFalse(self.calls.exists())
        self.env['MTP_ENABLED'] = 'invalid'
        invalid = self.run_script('start.sh', '--dry-run')
        self.assertNotEqual(invalid.returncode, 0)
        self.assertFalse(self.calls.exists())

    def dry_command(self):
        result = self.run_script('start.sh', '--dry-run')
        self.assertEqual(result.returncode, 0, result.stderr)
        line = next(x for x in result.stdout.splitlines() if x.strip().startswith('docker run '))
        return shlex.split(line)

    def test_explicit_drafter_overrides_legacy_switch(self):
        self.env.update(DRAFTER='none', MTP_ENABLED='1')
        self.assertNotIn('--speculative-config', self.dry_command())
        self.env.update(DRAFTER='mtp', MTP_ENABLED='0')
        args = self.dry_command()
        self.assertEqual(json.loads(args[args.index('--speculative-config')+1])['method'], 'mtp')
        self.assertFalse(self.calls.exists())

    def test_dflash2_dry_run_pins_external_draft_without_changing_target_budget(self):
        self.env.update(DRAFTER='dflash2', MTP_ENABLED='0')
        args = self.dry_command()
        spec = json.loads(args[args.index('--speculative-config')+1])
        self.assertEqual(spec['method'], 'dflash')
        self.assertIn('models--incoai--GLM-5.3-Flash-DFlash2/snapshots/bf582e4eacc1810f76656d1811693ff6c6737d2a', spec['model'])
        self.assertEqual(spec['num_speculative_tokens'], 7)
        self.assertEqual(spec['draft_tensor_parallel_size'], 2)
        self.assertEqual(spec['attention_backend'], 'TRITON_ATTN')
        self.assertEqual(args[args.index('--attention-backend')+1], 'B12X')
        self.assertIsNone(spec['quantization'])
        self.assertEqual(spec['rejection_sample_method'], 'standard')
        self.assertEqual(args[args.index('--kv-cache-memory-bytes')+1], '7650410496')
        self.assertEqual(args[args.index('--decode-context-parallel-size')+1], '2')
        mounts = [args[i+1] for i,x in enumerate(args) if x=='--mount']
        self.assertEqual(sum('dst=/root/.cache/huggingface/hub' in m for m in mounts), 1)
        self.assertFalse(self.calls.exists())

    def test_draft_kv_precision_is_explicit_and_leaves_target_unchanged(self):
        self.env.update(DRAFTER='dflash2', DFLASH2_KV_CACHE_DTYPE='fp8')
        args = self.dry_command()
        spec = json.loads(args[args.index('--speculative-config')+1])
        self.assertEqual(spec['kv_cache_dtype'], 'fp8')
        self.assertIsNone(spec['quantization'])
        self.assertEqual(args[args.index('--quantization')+1], 'modelopt_mixed')
        self.assertEqual(args[args.index('--kv-cache-dtype')+1], 'fp8')
        self.env['DFLASH2_KV_CACHE_DTYPE']='int4'
        self.assertNotEqual(self.run_script('start.sh', '--dry-run').returncode, 0)
        self.assertFalse(self.calls.exists())

    def test_external_draft_mount_is_readonly_and_keeps_target_mount(self):
        self.env.update(DRAFTER='dflash2', DFLASH2_MODEL_DIR='/tmp/draft with spaces')
        args = self.dry_command()
        self.assertIn('type=bind,src=/tmp/draft with spaces,dst=/draft,readonly', args)
        self.assertTrue(any('dst=/root/.cache/huggingface/hub,readonly' in x for x in args))
        spec=json.loads(args[args.index('--speculative-config')+1])
        self.assertEqual(spec['model'],'/draft')

    def test_missing_external_draft_cannot_stop_existing_service(self):
        self.env.update(DRAFTER='dflash2', DFLASH2_MODEL_DIR=str(Path(self.tmp.name)/'absent'), OWNER='glm53-fp4-local-v1')
        result=self.run_script('start.sh','restart')
        self.assertNotEqual(result.returncode,0)
        self.assertIn('DFlash2 file missing',result.stderr)
        self.assertEqual(self.commands(),[])

    def test_invalid_draft_mode_and_depth_fail_before_docker(self):
        self.env['DRAFTER']='dflash'
        self.assertNotEqual(self.run_script('start.sh','--dry-run').returncode,0)
        self.env['DRAFTER']='dflash2'
        for value in ('0','8','garbage'):
            self.env['DFLASH2_NUM_SPECULATIVE_TOKENS']=value
            self.assertNotEqual(self.run_script('start.sh','--dry-run').returncode,0)
        self.assertEqual(self.commands(),[])

    def test_foreign_container_is_never_stopped_or_removed(self):
        r = self.run_script('start.sh','restart')
        self.assertNotEqual(r.returncode,0)
        self.assertIn('not owned by this recipe',r.stderr)
        self.assertFalse(any(c[0] in ['stop','rm','run'] for c in self.commands()))

    def test_stop_preserves_recipe_container_and_volumes(self):
        self.env['OWNER']='glm53-fp4-local-v1'
        r=self.run_script('stop.sh')
        self.assertEqual(r.returncode,0,r.stderr)
        self.assertTrue(any(c[0]=='stop' for c in self.commands()))
        self.assertFalse(any(c[0] in ['rm','volume','run'] for c in self.commands()))

    def test_stop_distinguishes_absence_from_inspect_failure(self):
        self.env['ABSENT']='1'
        result=self.run_script('stop.sh')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('is absent', result.stdout)
        self.env['INSPECT_ERROR']='permission denied connecting to Docker daemon'
        result=self.run_script('stop.sh')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('permission denied', result.stderr)
        self.assertNotIn('is absent', result.stdout)
        self.assertFalse(any(c[0]=='stop' for c in self.commands()))

    def test_stop_recognizes_exact_absence_error_variants(self):
        self.env['ABSENT'] = '1'
        for prefix in ('Error:', 'Error response from daemon:'):
            for kind in ('container', 'object'):
                error = f'{prefix} No such {kind}: glm53-fp4-tp2'
                with self.subTest(error=error):
                    self.env['INSPECT_ERROR'] = error
                    result = self.run_script('stop.sh')
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn('is absent', result.stdout)
        for error in ('Error: No such container: glm53-fp4-tp2-other',
                      'permission denied\nError: No such container: glm53-fp4-tp2'):
            with self.subTest(error=error):
                self.env['INSPECT_ERROR'] = error
                result = self.run_script('stop.sh')
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('Cannot inspect container', result.stderr)
        self.assertFalse(any(c[0] in ('run', 'rm', 'stop') for c in self.commands()))

    def test_launch_pins_the_json_file_logging_driver(self):
        args=self.dry_command()
        self.assertEqual(args[args.index('--log-driver')+1],'json-file')

    def test_open_bind_warns_without_api_key(self):
        self.env.pop('VLLM_API_KEY',None)
        r=self.run_script('start.sh','--dry-run')
        self.assertIn('unauthenticated API beyond loopback',r.stdout)
        self.env['VLLM_API_KEY']='fixture-key'
        r=self.run_script('start.sh','--dry-run')
        self.assertNotIn('unauthenticated API beyond loopback',r.stdout)

    def test_stop_succeeds_when_log_capture_fails(self):
        self.env['OWNER']='glm53-fp4-local-v1'
        self.env['LOGS_ERROR']='1'
        r=self.run_script('stop.sh')
        self.assertEqual(r.returncode,0,r.stderr)
        self.assertIn('Stopped',r.stdout)
        self.assertTrue(any(c[0]=='stop' for c in self.commands()))
        self.assertTrue(any(c[0]=='logs' for c in self.commands()))

    def test_start_stops_a_non_running_container_before_removal(self):
        self.env['OWNER']='glm53-fp4-local-v1'
        r=self.run_script('start.sh')
        recorded=[c[0] for c in self.commands()]
        self.assertEqual([x for x in recorded if x in ('stop','logs','rm')],['stop','logs','rm'])
        self.assertNotEqual(r.returncode,0)

    def test_start_refuses_inspect_errors_before_preflight(self):
        self.env.update(ABSENT='1', INSPECT_ERROR='daemon disconnected')
        result=self.run_script('start.sh')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('daemon disconnected', result.stderr)
        self.assertFalse(any(c[0] in ('run', 'rm', 'stop') for c in self.commands()))

    def test_manual_hub_dry_run_matches_model_and_draft_mounts(self):
        hub=Path(self.tmp.name)/'cache'/'hub'
        local=checkpoint(hub/'manual-model')
        self.env.update(HF_CACHE_ROOT=str(hub.parent), MODEL_DIR=str(local),
                        MODEL_REVISION=TEST_REV, DRAFTER='dflash2')
        args=self.dry_command()
        actual=model.inspect_model(hub, TEST_REV, local)
        self.assertEqual(args[args.index('serve')+1], actual['container_path'])
        mounts=[args[i+1] for i,x in enumerate(args) if x=='--mount']
        self.assertEqual(sum('dst=/root/.cache/huggingface/hub' in m for m in mounts), 1)
        self.assertIn('type=bind,src='+actual['mount_source']+',dst='+actual['mount_target']+',readonly', mounts)
        self.assertFalse(self.commands())

    def test_start_uses_configured_endpoint_despite_inherited_base_url(self):
        self.env.update(BASE_URL='http://unrelated.invalid:1234', HOST='127.0.0.1', PORT='8123',
                        OWNER='glm53-fp4-local-v1', FAKE_RUNNING='true', FAKE_SERVING_PORT='8123')
        (self.root/'scripts'/'smoke.py').write_text("import os; print(os.environ['BASE_URL'])\n")
        result=self.run_script('start.sh')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('http://127.0.0.1:8123', result.stdout)
        self.assertNotIn('unrelated.invalid', result.stdout)

    def test_running_container_port_is_checked_before_http_readiness(self):
        self.env.update(HOST='127.0.0.1', OWNER='glm53-fp4-local-v1', FAKE_RUNNING='true')
        marker = Path(self.tmp.name)/'http-accepted'
        (self.root/'scripts'/'smoke.py').write_text(
            'from pathlib import Path\nPath(' + repr(str(marker)) + ').touch()\n')
        for configured_port, succeeds in (('8000', True), ('8123', False)):
            with self.subTest(configured_port=configured_port):
                marker.unlink(missing_ok=True)
                self.env['PORT'] = configured_port
                result = self.run_script('start.sh')
                self.assertEqual(result.returncode == 0, succeeds, result.stderr)
                self.assertEqual(marker.exists(), succeeds)
                if not succeeds:
                    self.assertIn('serves port 8000', result.stderr)
                    self.assertIn('PORT=8123', result.stderr)
                    self.assertIn('./start.sh restart', result.stderr)
        self.assertFalse(any(c[0] in ('run', 'rm', 'stop') for c in self.commands()))

    def test_busy_gpus_prevent_launch(self):
        self.env['ABSENT']='1'
        local=checkpoint(Path(self.tmp.name)/'weights')
        self.env.update(MODEL_DIR=str(local),MODEL_REVISION=TEST_REV)
        r=self.run_script('start.sh')
        self.assertNotEqual(r.returncode,0)
        self.assertIn('10/97887 MiB free',r.stdout)
        self.assertFalse(any(c[0]=='run' for c in self.commands()))

    def test_literal_env_is_not_executed(self):
        sentinel=Path(self.tmp.name)/'must-not-exist'
        (self.root/'.env').write_text('SERVED_MODEL_NAME=$(touch '+str(sentinel)+')\n')
        r=self.run_script('start.sh','--dry-run')
        self.assertEqual(r.returncode,0,r.stderr)
        self.assertFalse(sentinel.exists())
        self.assertIn('touch',r.stdout)

    def test_preflight_rejects_a_live_port_listener(self):
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(('127.0.0.1',0));listener.listen()
            self.env.update(HOST='127.0.0.1',PORT=str(listener.getsockname()[1]))
            r=self.run_script('doctor.sh')
            self.assertIn('FAIL API bind 127.0.0.1:',r.stdout)

    def test_preflight_accepts_a_recently_closed_server_port(self):
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(('127.0.0.1',0));listener.listen()
            address=listener.getsockname()
            with socket.create_connection(address) as client:
                accepted,_=listener.accept()
                accepted.close()
                self.assertEqual(client.recv(1),b'')
        self.env.update(HOST=address[0],PORT=str(address[1]))
        r=self.run_script('doctor.sh')
        self.assertIn('PASS API bind 127.0.0.1:',r.stdout)

    def test_invalid_gpu_selection_fails_before_docker(self):
        self.env['GPUS']='0,0'
        r=self.run_script('start.sh','--dry-run')
        self.assertNotEqual(r.returncode,0)
        self.assertFalse(self.commands())


class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args): pass
    def send_json(self, data):
        b=json.dumps(data).encode()
        self.send_response(200);self.send_header('Content-Type','application/json')
        self.send_header('Content-Length',str(len(b)));self.end_headers();self.wfile.write(b)
    def do_GET(self):
        self.send_json({'data':[{'id':smoke.MODEL}]} if self.path=='/v1/models' else {})
    def do_POST(self):
        b=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if b.get('stream'):
            self.send_response(200);self.send_header('Content-Type','text/event-stream');self.end_headers()
            events=[{'choices':[{'delta':{'content':'The Moon orbits Earth.'}}]},
                    {'choices':[],'usage':{'completion_tokens':5}}]
            for event in events: self.wfile.write(('data: '+json.dumps(event)+'\n\n').encode())
            self.wfile.write(b'data: [DONE]\n\n');return
        if b.get('tools'):
            message={'content':None,'tool_calls':[{'id':'call-fixture','type':'function',
                'function':{'name':'get_temperature','arguments':'{"city":"Paris","unit":"celsius"}'}}]}
        else: message={'content':'{"answer":42}' if b.get('response_format') else 'ready'}
        # This pinned GLM template always starts <think>; disabling its parser leaks reasoning.
        if b.get('chat_template_kwargs',{}).get('enable_thinking') is False:
            message={'content':'Reasoning text. '+str(message.get('content')), 'reasoning':None}
        self.send_json({'choices':[{'message':message}],'usage':{'completion_tokens':5}})


class ClientTests(unittest.TestCase):
    def test_acceptance_suite_handles_text_json_tools_and_sse(self):
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            with patch.object(smoke,'BASE',f'http://127.0.0.1:{server.server_port}'):
                smoke.ready()
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertTrue(smoke.run_checks(suite=True))
                r=smoke.stream(smoke.payload('hi'))
                self.assertEqual(r['usage']['completion_tokens'],5)
                self.assertGreaterEqual(r['ttft_seconds'],0)
        finally:
            server.shutdown();server.server_close();thread.join()

    def test_readiness_rejects_dead_container_even_with_healthy_api(self):
        with patch.dict(os.environ, {'CONTAINER_NAME': 'fixture'}), \
                patch.object(smoke, 'request') as req, patch.object(smoke, 'call', return_value={'data':[{'id':smoke.MODEL}]}), \
                patch.object(smoke.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, 'false')):
            req.return_value.__enter__.return_value.status=200
            with self.assertRaisesRegex(RuntimeError, 'container'):
                smoke.ready()

    def test_readiness_checks_container_again_after_healthy_api(self):
        for states, succeeds in ((['true','true'],True), (['true','false'],False)):
            with self.subTest(states=states), patch.dict(os.environ, {'CONTAINER_NAME':'fixture'}), \
                    patch.object(smoke, 'request') as req, \
                    patch.object(smoke, 'call', return_value={'data':[{'id':smoke.MODEL}]}), \
                    patch.object(smoke.subprocess, 'run', side_effect=[subprocess.CompletedProcess([],0,s) for s in states]):
                req.return_value.__enter__.return_value.status=200
                if succeeds:
                    self.assertEqual(smoke.ready()['data'][0]['id'], smoke.MODEL)
                else:
                    with self.assertRaisesRegex(RuntimeError, 'container'):
                        smoke.ready()

    def test_malformed_tool_arguments_fail_without_aborting_suite(self):
        for arguments in ([], {'city':None,'unit':'celsius'}, {'city':42,'unit':'celsius'},
                          {'city':'Paris','unit':'celsius','extra':True}):
            with self.subTest(arguments=arguments), tempfile.TemporaryDirectory() as directory:
                replies=[{'choices':[{'message':{'content':'ready'}}]},
                         {'choices':[{'message':{'content':'{"answer":42}'}}]},
                         {'choices':[{'message':{'tool_calls':[{'function':{'name':'get_temperature',
                            'arguments':json.dumps(arguments)}}]}}]}]
                with patch.object(smoke, 'call', side_effect=replies), \
                        patch.object(smoke, 'stream', return_value={'content':'Moon'}) as stream, \
                        contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    self.assertFalse(smoke.run_checks(suite=True, output=Path(directory)))
                stream.assert_called_once()
                checks=json.loads((Path(directory)/'checks.json').read_text())
                self.assertEqual([(r['check'],r['passed']) for r in checks],
                                 [('completion',True),('json',True),('tools',False),('streaming',True)])


if __name__=='__main__': unittest.main()
