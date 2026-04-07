import unittest

import torch

from quantize.bitpacking import pack_cols, pack_rows, unpack_cols, unpack_rows


class BitPackingTest(unittest.TestCase):
    def test_row_roundtrip(self):
        for bits in (2, 4, 8):
            maxq = 2 ** bits - 1
            values = torch.randint(0, maxq + 1, (9, 5), dtype=torch.int64)
            packed = pack_rows(values, bits)
            unpacked = unpack_rows(packed, bits, rows=values.shape[0], cols=values.shape[1]).to(torch.int64)
            self.assertTrue(torch.equal(values, unpacked))

    def test_col_roundtrip(self):
        for bits in (2, 4, 8):
            maxq = 2 ** bits - 1
            values = torch.randint(0, maxq + 1, (3, 11), dtype=torch.int64)
            packed = pack_cols(values, bits)
            unpacked = unpack_cols(packed, bits, rows=values.shape[0], cols=values.shape[1]).to(torch.int64)
            self.assertTrue(torch.equal(values, unpacked))


if __name__ == "__main__":
    unittest.main()
