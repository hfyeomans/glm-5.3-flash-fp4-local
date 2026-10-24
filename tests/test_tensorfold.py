"""Guard the native recipe's isolation and two-rank failure cleanup."""
import contextlib
import importlib.util
import os
import json
import struct
from pathlib import Path
import socket
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('tf_recipe', ROOT / 'tensorfold/scripts/recipe.py')
tf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tf)


class TensorFoldTests(unittest.TestCase):
    def test_env_is_literal_and_does_not_read_parent_env_file(self):
        with tempfile.TemporaryDirectory() as d:
            env = Path(d) / '.env'
            marker = Path(d) / 'executed'
            env.write_text(f'MODEL_DIR=$(touch {marker})\nTF_PORT=8011\n')
            c = tf.config(env, {'TF_PORT': '8012', 'HF_HOME': '/wrong/cache'})
            self.assertEqual(c['TF_PORT'], '8012')
            self.assertEqual(c['HF_CACHE_ROOT'], '/models_engrams/huggingface')
            self.assertIn('$(touch ', c['MODEL_DIR'])
            self.assertFalse(marker.exists())
            env.write_text('HF_TOKEN=do-not-print-this\n')
            with self.assertRaises(ValueError) as caught:
                tf.config(env, {})
            self.assertNotIn('do-not-print-this', str(caught.exception))

    def test_rank_and_watchdog_port_collisions_refused(self):
        for env in ({'TF_GPUS': '0,0'}, {'TF_GPUS': 'all'},
                    {'TF_PORT': '29600'}, {'TF_PORT': '29601'},
                    {'TF_CONTEXT': '1048577'}, {'TF_CONTAINER': 'glm53-fp4-tp2'}):
            with self.subTest(env=env), self.assertRaises(ValueError):
                tf.config(Path('/nonexistent'), env)

    def test_foreign_container_never_stopped(self):
        c = tf.config(Path('/nonexistent'), {})
        response = subprocess.CompletedProcess([], 0, '[{"Config":{"Labels":{}},"Id":"foreign"}]', '')
        with patch.object(tf, 'config', return_value=c), patch.object(tf, 'lock', contextlib.nullcontext), \
             patch.object(tf.subprocess, 'run', return_value=response) as call:
            with self.assertRaisesRegex(ValueError, 'not owned'):
                tf.main(['stop'])
            self.assertEqual(call.call_count, 1)
            self.assertEqual(call.call_args.args[0][:3], ['docker', 'container', 'inspect'])

    def test_busy_gpu_refuses_start_before_mutations(self):
        c = tf.config(Path('/nonexistent'), {})
        with patch.object(tf, 'config', return_value=c), patch.object(tf, 'lock', contextlib.nullcontext), \
             patch.object(tf, 'model', return_value={}), patch.object(tf, 'image', return_value='sha256:pinned'), \
             patch.object(tf, 'container', return_value=None), patch.object(tf, 'ports'), \
             patch.object(tf, 'gpu_preflight', side_effect=ValueError('busy')), patch.object(tf, 'require_compatible'), \
             patch.object(tf, 'run') as call:
            with self.assertRaisesRegex(ValueError, 'busy'):
                tf.main(['start'])
            call.assert_not_called()

    def test_gpu_preflight_rejects_existing_compute_job(self):
        c = tf.config(Path('/nonexistent'), {})
        rows = '0, GPU-a, NVIDIA RTX PRO 6000 Blackwell, 97000\n1, GPU-b, NVIDIA RTX PRO 6000 Blackwell, 97000\n'
        results = [subprocess.CompletedProcess([], 0, rows),
                   subprocess.CompletedProcess([], 0, 'GPU-b, 12345\n')]
        with patch.object(tf, 'run', side_effect=results), self.assertRaisesRegex(ValueError, 'compute process'):
            tf.gpu_preflight(c)

    def test_live_http_port_is_refused(self):
        with socket.socket() as server:
            server.bind(('127.0.0.1', 0))
            server.listen()
            c = tf.config(Path('/nonexistent'), {'TF_PORT': str(server.getsockname()[1])})
            with self.assertRaisesRegex(ValueError, 'unavailable'):
                tf.ports(c)

    def test_launch_mounts_same_checkpoint_readonly_and_pinned_image(self):
        c = tf.config(Path('/nonexistent'), {})
        args = tf.launch(c, {'mount_source': '/cache with spaces/hub', 'mount_target': '/hf/hub',
                             'container_path': '/hf/hub/snapshots/pinned'}, 'sha256:verified')
        self.assertIn('type=bind,src=/cache with spaces/hub,dst=/hf/hub,readonly', args)
        self.assertIn('MODEL_PATH=/hf/hub/snapshots/pinned', args)
        self.assertEqual(args[-1], 'sha256:verified')
        self.assertNotIn('HF_TOKEN', ' '.join(args))
        self.assertNotIn('B12X', ' '.join(args))

    def test_compressed_scales_and_fp8_projections_are_blocked(self):
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            names = ('model.language_model.layers.0.self_attn.q_proj.weight', 'lm_head.weight')
            scale = 'model.language_model.layers.3.mlp.experts.0.gate_proj.weight_scale.csf32_codes'
            header = json.dumps({name: {'dtype': 'F8_E4M3'} for name in names}).encode()
            (directory / 'model.safetensors').write_bytes(struct.pack('<Q', len(header)) + header)
            (directory / 'model.safetensors.index.json').write_text(json.dumps(
                {'weight_map': {name: 'model.safetensors' for name in (*names, scale)}}))
            m = {'path': d, 'revision': 'fixture'}
            report = tf.compatibility(m)
            self.assertFalse(report['loader_contract_checks_passed'])
            self.assertEqual([i['code'] for i in report['issues']],
                             ['missing_plain_nvfp4_scale', 'unsupported_projection_dtype', 'unsupported_projection_dtype'])
            with self.assertRaisesRegex(ValueError, 'incompatible'):
                tf.require_compatible(m)

    def test_incompatible_checkpoint_blocks_launch_before_gpu_or_container_work(self):
        c = tf.config(Path('/nonexistent'), {})
        with patch.object(tf, 'config', return_value=c), patch.object(tf, 'lock', contextlib.nullcontext), \
             patch.object(tf, 'model', return_value={}), patch.object(tf, 'image', return_value='sha256:pinned'), \
             patch.object(tf, 'require_compatible', side_effect=ValueError('incompatible')), \
             patch.object(tf, 'gpu_preflight') as gpu, patch.object(tf, 'container') as container, \
             patch.object(tf, 'run') as call:
            with self.assertRaisesRegex(ValueError, 'incompatible'):
                tf.main(['start'])
            gpu.assert_not_called()
            container.assert_not_called()
            call.assert_not_called()

    def test_failed_rank_terminates_peer_and_preserves_failure_code(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            native = folder / 'native'
            native.write_text('''#!/usr/bin/env bash
set -eu
if [[ "$TF_GLM_RANK" == 0 ]]; then
    trap 'touch "$MARKER/stopped"; exit 0' TERM
    touch "$MARKER/ready"
    while true; do sleep 0.05; done
else
    while [[ ! -f "$MARKER/ready" ]]; do sleep 0.05; done
    exit 7
fi
''')
            native.chmod(0o755)
            runner = folder / 'runner.sh'
            source = (ROOT / 'tensorfold/scripts/runner.sh').read_text()
            runner.write_text(source.replace('/opt/tensorfold/bin/tensorfold-native', str(native))
                              .replace('/opt/tensorfold/bin/glm-loader-probe', '/bin/true'))
            env = dict(os.environ, MODEL_PATH='/model', TF_PORT='8001', TF_COMM_PORT='29600',
                       TF_CONTEXT='8192', TF_PARALLEL='1', MARKER=d)
            result = subprocess.run(['bash', str(runner)], env=env, capture_output=True, timeout=10, check=False)
            self.assertEqual(result.returncode, 7, result.stderr.decode())
            self.assertTrue((folder / 'stopped').exists())


if __name__ == '__main__':
    unittest.main()
