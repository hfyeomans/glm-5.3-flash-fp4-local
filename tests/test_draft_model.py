"""Pin DFlash2 identity and keep missing/unsafe checkpoints outside GPU startup."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from draft_model import inspect_draft, plan_draft


class DraftIdentityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.hub = self.root / 'hub'
        self.path = self.hub / 'models--incoai--GLM-5.3-Flash-DFlash2' / 'snapshots' / ('b' * 40)
        self.path.mkdir(parents=True)
        config = b'{"architectures":["DFlash2DraftModel"]}'
        header = json.dumps({'tensor': {'dtype': 'BF16', 'shape': [2], 'data_offsets': [0, 4]}}).encode()
        weights = len(header).to_bytes(8, 'little') + header + b'1234'
        (self.path / 'config.json').write_bytes(config)
        (self.path / 'model.safetensors').write_bytes(weights)
        self.lock = {'model_id': 'incoai/GLM-5.3-Flash-DFlash2', 'revision': 'b' * 40,
                     'license': 'CC-BY-NC-ND-4.0', 'config_sha256': hashlib.sha256(config).hexdigest(),
                     'weights_file': 'model.safetensors', 'weights_bytes': len(weights),
                     'weights_sha256': hashlib.sha256(weights).hexdigest(),
                     'safetensors_header_sha256': hashlib.sha256(header).hexdigest()}

    def inspect(self, **kwargs):
        return inspect_draft(self.hub, lock=self.lock, **kwargs)

    def test_pinned_hub_blob_and_full_hash(self):
        file = self.path / 'model.safetensors'
        blob = self.path.parents[1] / 'blobs' / self.lock['weights_sha256']
        blob.parent.mkdir(); file.rename(blob); file.symlink_to('../../blobs/' + blob.name)
        result = self.inspect(verify=True)
        self.assertTrue(result['weights_sha256_verified'])
        self.assertEqual(result['mount_source'], str(self.hub))
        self.assertIn('/root/.cache/huggingface/hub/', result['container_path'])

    def test_other_draft_metadata_is_rejected(self):
        (self.path / 'config.json').write_text('{"architectures":["DFlashDraftModel"]}')
        with self.assertRaisesRegex(ValueError, 'metadata differs'):
            self.inspect()

    def test_truncated_and_changed_header_are_rejected(self):
        file = self.path / 'model.safetensors'
        original = file.read_bytes()
        file.write_bytes(original[:-1])
        with self.assertRaisesRegex(ValueError, 'truncated'):
            self.inspect()
        changed = bytearray(original); changed[9] ^= 1; file.write_bytes(changed)
        with self.assertRaisesRegex(ValueError, 'tensor layout'):
            self.inspect()

    def test_same_size_payload_corruption_fails_full_verification(self):
        file = self.path / 'model.safetensors'
        data = file.read_bytes(); file.write_bytes(data[:-1] + b'x')
        self.assertFalse(self.inspect()['weights_sha256_verified'])
        with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
            self.inspect(verify=True)

    def test_escaped_blob_is_rejected(self):
        file = self.path / 'model.safetensors'
        outside = self.root / 'outside.safetensors'
        file.rename(outside); file.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, 'symlink escapes'):
            self.inspect()

    def test_missing_draft_refused_but_plan_needs_no_files(self):
        (self.path / 'model.safetensors').unlink()
        with self.assertRaisesRegex(ValueError, 'file missing'):
            self.inspect()
        result = plan_draft(self.hub, lock=self.lock)
        self.assertEqual(result['revision'], self.lock['revision'])


if __name__ == '__main__':
    unittest.main()
