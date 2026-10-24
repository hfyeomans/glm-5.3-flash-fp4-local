"""CPU checks for TP cuts and byte/record units in compressed-scale accounting."""
import importlib.util
from pathlib import Path
import struct
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('csf_inventory', ROOT / 'tensorfold/tuning/csf_inventory.py')
inventory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inventory)


def indexes(halves):
    pages, offsets, count = [], [], 0
    for left, right in halves:
        pages.append(2 * count)
        # Sparse records in block 0 and block 128; legal monotone prefix sequence.
        offsets.extend([0] + [left] * 128 + [left + right] * 128)
        count += left + right
    return (struct.pack('<' + 'I' * len(pages), *pages),
            struct.pack('<' + 'H' * len(offsets), *offsets), 2 * count)


class InventoryTests(unittest.TestCase):
    def test_row_cut_uses_page_origin_in_bytes(self):
        data = indexes([(1, 2), (3, 4), (5, 6), (7, 8)])
        self.assertEqual(inventory.exception_shares('gate_proj', 256, 256, *data), [20, 52])
        self.assertEqual(inventory.exception_shares('up_proj', 256, 256, *data), [20, 52])

    def test_column_cut_sums_half_each_page_in_records(self):
        data = indexes([(1, 2), (3, 4)])
        self.assertEqual(inventory.exception_shares('down_proj', 256, 128, *data), [8, 12])

    def test_no_exceptions(self):
        data = indexes([(0, 0), (0, 0)])
        self.assertEqual(inventory.exception_shares('down_proj', 256, 128, *data), [0, 0])

    def test_bad_extents_and_origins_fail(self):
        pages, offsets, total = indexes([(1, 2), (3, 4)])
        cases = [(pages[:-1], offsets, total), (pages, offsets[:-2], total),
                 (pages, offsets, total + 2), (pages, offsets, total + 1),
                 (struct.pack('<II', 0, 3), offsets, total)]
        for args in cases:
            with self.subTest(args=args), self.assertRaises(ValueError):
                inventory.exception_shares('down_proj', 256, 128, *args)

    def test_unsupported_cut_and_nonmonotone_boundary_fail(self):
        pages, offsets, total = indexes([(1, 2), (3, 4)])
        bad = bytearray(offsets)
        struct.pack_into('<H', bad, 2 * 128, 100)
        with self.assertRaises(ValueError):
            inventory.exception_shares('down_proj', 256, 128, pages, bad, total)
        with self.assertRaises(ValueError):
            inventory.exception_shares('unknown', 256, 128, pages, offsets, total)
        with self.assertRaises(ValueError):
            inventory.exception_shares('down_proj', 256, 64, pages, offsets, total)
