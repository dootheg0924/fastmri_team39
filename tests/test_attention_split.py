import copy
import unittest

import torch
from torch import nn

from utils.learning.train_part import migrate_optimizer_state_by_name
from utils.model.fi_varnet import AttentionFeatureVarNetBlock, AttentionPE


class EncoderStub(nn.Module):
    def __init__(self, feature_chans):
        super().__init__()
        self.feature_chans = feature_chans


class AttentionSplitTest(unittest.TestCase):
    def test_split_attention_starts_equal_but_routes_gradients_to_one_expert(self):
        block = AttentionFeatureVarNetBlock(
            encoder=EncoderStub(2),
            decoder=nn.Identity(),
            feature_processor=nn.Identity(),
            attention_layer=AttentionPE(2),
            split_attention=True,
        )
        for acc4, acc8 in zip(
            block.attention_layer.parameters(),
            block.attention_layer_acc8.parameters(),
        ):
            self.assertTrue(torch.equal(acc4, acc8))
            self.assertNotEqual(acc4.data_ptr(), acc8.data_ptr())

        x = torch.randn(1, 2, 8, 8, requires_grad=True)
        block._attention_for_acceleration(8)(x, 8).sum().backward()
        self.assertTrue(
            any(parameter.grad is not None
                for parameter in block.attention_layer_acc8.parameters())
        )
        self.assertTrue(
            all(parameter.grad is None
                for parameter in block.attention_layer.parameters())
        )

    def test_split_requires_an_attention_layer(self):
        with self.assertRaisesRegex(ValueError, "requires an attention layer"):
            AttentionFeatureVarNetBlock(
                encoder=EncoderStub(2),
                decoder=nn.Identity(),
                feature_processor=nn.Identity(),
                attention_layer=None,
                split_attention=True,
            )


class ToyBlock(nn.Module):
    def __init__(self, split):
        super().__init__()
        self.feature_processor = nn.Linear(3, 3)
        self.attention_layer = nn.Linear(3, 3)
        self.attention_layer_acc8 = (
            copy.deepcopy(self.attention_layer) if split else None
        )


class ToyModel(nn.Module):
    def __init__(self, split):
        super().__init__()
        self.cascades = nn.ModuleList([ToyBlock(split)])


class OptimizerMigrationTest(unittest.TestCase):
    def test_adam_moments_are_cloned_to_both_attention_experts(self):
        source = ToyModel(split=False)
        source_optimizer = torch.optim.Adam(source.parameters(), lr=3e-4)
        source_optimizer.zero_grad(set_to_none=True)
        sum(parameter.square().sum() for parameter in source.parameters()).backward()
        source_optimizer.step()

        target = ToyModel(split=True)
        target_optimizer = torch.optim.Adam(target.parameters(), lr=9e-3)
        migrate_optimizer_state_by_name(
            source_optimizer.state_dict(),
            [name for name, _ in source.named_parameters()],
            target,
            target_optimizer,
        )

        source_state = source_optimizer.state[
            source.cascades[0].attention_layer.weight
        ]
        acc4_state = target_optimizer.state[
            target.cascades[0].attention_layer.weight
        ]
        acc8_state = target_optimizer.state[
            target.cascades[0].attention_layer_acc8.weight
        ]
        for key in ("step", "exp_avg", "exp_avg_sq"):
            self.assertTrue(torch.equal(source_state[key], acc4_state[key]))
            self.assertTrue(torch.equal(source_state[key], acc8_state[key]))
            self.assertNotEqual(acc4_state[key].data_ptr(), acc8_state[key].data_ptr())
        self.assertAlmostEqual(target_optimizer.param_groups[0]["lr"], 3e-4)


if __name__ == "__main__":
    unittest.main()
