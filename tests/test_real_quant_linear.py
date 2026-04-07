import unittest

import torch
import torch.nn as nn

from quantize.int_linear_real import QuantLinear


class RealQuantLinearTest(unittest.TestCase):
    def test_zero_point_state_dict_roundtrip(self):
        linear = nn.Linear(5, 3, bias=False)
        quant = QuantLinear(bits=2, group_size=4, infeatures=5, outfeatures=3, bias=False)
        scales = torch.ones((2, 3), dtype=torch.float32)
        zero_points = torch.tensor([[1, 2, 3], [0, 1, 2]], dtype=torch.float32)
        quant.pack(linear, scales, zero_points)

        quant.zero_points.data.copy_(torch.tensor([[3, 2, 1], [1, 0, 2]], dtype=torch.float32))
        state = quant.state_dict()

        restored = QuantLinear(bits=2, group_size=4, infeatures=5, outfeatures=3, bias=False)
        restored.load_state_dict(state)
        self.assertTrue(torch.equal(restored.zero_points.round(), quant.zero_points.round()))

    def test_forward_matches_dequantized_weight_shape(self):
        linear = nn.Linear(5, 3, bias=False)
        quant = QuantLinear(bits=2, group_size=4, infeatures=5, outfeatures=3, bias=False)
        scales = torch.ones((2, 3), dtype=torch.float32)
        zero_points = torch.zeros((2, 3), dtype=torch.float32)
        quant.pack(linear, scales, zero_points)

        x = torch.randn(2, 5)
        y = quant(x)
        self.assertEqual(y.shape, (2, 3))


if __name__ == "__main__":
    unittest.main()
