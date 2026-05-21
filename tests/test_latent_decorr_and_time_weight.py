import unittest

import torch

from SAE.sae import SharedSAE
from train.losses import latent_covariance_decorrelation_loss


def _tiny_sae(*, use_time_branch: bool = True, time_branch_mode: str = "sincos_linear") -> SharedSAE:
    return SharedSAE(
        blocks=["a", "b", "c", "d"],
        d_model=4,
        n_dirs=8,
        top_k=2,
        auxk=2,
        dead_tokens_threshold=100,
        use_block_in_adapter=False,
        use_block_out_adapter=False,
        block_in_rank=2,
        block_in_alpha=2,
        block_out_rank=2,
        block_out_alpha=2,
        use_time_branch=use_time_branch,
        time_branch_mode=time_branch_mode,
        time_embed_dim=4,
        time_hidden_dim=4,
        use_spatial_branch=False,
        spatial_branch_mode="sincos_linear",
        spatial_embed_dim=4,
        spatial_hidden_dim=4,
    )


class LatentDecorrLossTest(unittest.TestCase):
    def test_diagonal_terms_do_not_contribute(self) -> None:
        z = torch.tensor(
            [
                [1.0, 0.0, 2.0],
                [2.0, 1.0, 0.5],
                [3.0, 0.5, 1.0],
            ]
        )
        got = latent_covariance_decorrelation_loss(
            {"a": z},
            top_k=3,
            eps=1e-4,
        )
        z_sel = z - z.mean(dim=0, keepdim=True)
        std = z_sel.pow(2).mean(dim=0).sqrt()
        z_norm = z_sel / std.unsqueeze(0)
        corr = z_norm.t() @ z_norm / float(max(1, int(z_norm.shape[0]) - 1))
        corr = corr - torch.diag_embed(torch.diagonal(corr))
        expected = torch.mean(corr.pow(2))
        self.assertTrue(torch.allclose(got, expected))

    def test_topq_one_matches_mean_pool(self) -> None:
        z_by_block = {
            "a": torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [4.0, 5.0, 6.0]]),
            "b": torch.tensor([[2.0, 1.0, 3.0], [3.0, 2.0, 4.0], [5.0, 4.0, 6.0]]),
            "c": torch.tensor([[3.0, 1.0, 2.0], [4.0, 2.0, 3.0], [6.0, 4.0, 5.0]]),
        }
        mean_loss = latent_covariance_decorrelation_loss(
            z_by_block,
            top_k=3,
            mode="block_pooled",
            pool="mean",
        )
        topq_loss = latent_covariance_decorrelation_loss(
            z_by_block,
            top_k=3,
            mode="block_pooled",
            pool="topq",
            pool_topq=1.0,
        )
        self.assertTrue(torch.allclose(mean_loss, topq_loss))

    def test_invalid_pool_raises(self) -> None:
        with self.assertRaises(ValueError):
            latent_covariance_decorrelation_loss({"a": torch.ones(3, 4)}, top_k=2, pool="bad")


class LearnedTimeWeightTest(unittest.TestCase):
    def test_neutral_sigmoid_zero_raw_is_one(self) -> None:
        sae = _tiny_sae()
        raw, weight = sae.get_learned_time_weight(
            timestep=torch.tensor([500.0]),
            feature_ids=[0, 1, 2],
            transform="neutral_sigmoid",
        )
        self.assertTrue(torch.allclose(raw, torch.zeros_like(raw)))
        self.assertTrue(torch.allclose(weight, torch.ones_like(weight)))

    def test_disabled_time_branch_returns_neutral_weight(self) -> None:
        sae = _tiny_sae(use_time_branch=False)
        raw, weight = sae.get_learned_time_weight(
            timestep=torch.tensor([500.0]),
            feature_ids=[0, 1, 2],
            transform="relu",
        )
        self.assertTrue(torch.allclose(raw, torch.zeros_like(raw)))
        self.assertTrue(torch.allclose(weight, torch.ones_like(weight)))

    def test_film_mode_uses_beta_shape(self) -> None:
        sae = _tiny_sae(time_branch_mode="sincos_film")
        raw, weight = sae.get_learned_time_weight(
            timestep=torch.tensor([500.0]),
            feature_ids=[0, 1, 2],
            transform="abs",
        )
        self.assertEqual(tuple(raw.shape), (3,))
        self.assertEqual(tuple(weight.shape), (3,))
        self.assertTrue(torch.all(weight >= 0))


if __name__ == "__main__":
    unittest.main()
