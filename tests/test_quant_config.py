import unittest
from tempfile import TemporaryDirectory

from quantize.config import EfficientQATQuantConfig, QuantizationRule, QuantizationSpec, maybe_load_quant_config


class QuantConfigTest(unittest.TestCase):
    def test_override_resolution(self):
        config = EfficientQATQuantConfig(
            default=QuantizationSpec(bits=2, group_size=32),
            overrides=[
                QuantizationRule(
                    pattern="*.self_attn.o_proj",
                    spec=QuantizationSpec(bits=8, group_size=-1, granularity="per_channel", mapping="symmetric"),
                ),
                QuantizationRule(
                    pattern="model.embed_tokens",
                    spec=QuantizationSpec(bits=16, group_size=32, enabled=False),
                ),
            ],
        )

        default_spec = config.resolve("mlp.gate_proj", in_features=128)
        self.assertEqual(default_spec.bits, 2)
        self.assertEqual(default_spec.resolved_group_size(128), 32)

        per_channel_spec = config.resolve("decoder.self_attn.o_proj", in_features=96)
        self.assertEqual(per_channel_spec.bits, 8)
        self.assertEqual(per_channel_spec.granularity, "per_channel")
        self.assertEqual(per_channel_spec.resolved_group_size(96), 96)
        self.assertEqual(per_channel_spec.mapping, "symmetric")
        self.assertFalse(per_channel_spec.train_zero_point)

        skipped_spec = config.resolve("model.embed_tokens", in_features=128)
        self.assertFalse(skipped_spec.should_quantize)

    def test_maybe_load_accepts_file_path(self):
        config = EfficientQATQuantConfig(default=QuantizationSpec(bits=2, group_size=32))
        with TemporaryDirectory() as tmpdir:
            config_path = config.save(tmpdir)
            loaded = maybe_load_quant_config(str(config_path), default_bits=4, default_group_size=128)
        self.assertEqual(loaded.default.bits, 2)
        self.assertEqual(loaded.default.group_size, 32)


if __name__ == "__main__":
    unittest.main()
