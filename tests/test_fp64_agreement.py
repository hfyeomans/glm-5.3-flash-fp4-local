"""Compile the probe's actual agreement gate without its CUDA-only imports."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class AgreementTests(unittest.TestCase):
    def test_host_agreement_gate(self):
        zig = os.environ.get('ZIG') or shutil.which('zig')
        if not zig:
            self.skipTest('Set ZIG to run the CPU-host numerical gate tests')
        source = (ROOT / 'tensorfold/tuning/glm_mm_bench_fp64.zig').read_text()
        # Compile the exact production function and its Zig tests. The preceding
        # benchmark requires CUDA modules and upstream check.zig, absent on a host.
        gate = source[source.index('fn productsAgree('):]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'agreement.zig'
            path.write_text('const std = @import("std");\n' + gate)
            result = subprocess.run(
                [zig, 'test', str(path), '--cache-dir', str(Path(tmp) / 'cache'),
                 '--global-cache-dir', str(Path(tmp) / 'global-cache')],
                capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('All 3 tests passed.', result.stderr)


if __name__ == '__main__':
    unittest.main()
