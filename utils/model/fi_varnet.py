"""FI-VarNet (Feature-Image VarNet) for the FastMRI challenge.

Ported from Giannakopoulos et al., "Accelerated MRI reconstructions via
variational network and feature domain learning", Scientific Reports 2024
(reference implementation: IliasGiannakopoulosLab/VarNet, varnets/VarNet.py).

Differences from the reference, driven by this repository's constraints:
- Reuses the baseline `varnet.py` building blocks (SensitivityModel,
  VarNetBlock, NormUnet) so the sensitivity path and image cascades are
  identical to the E2E-VarNet baseline.
- The per-sample acceleration is inferred from the mask's outer sampling
  stride at forward time (acc4/acc8 volumes are mixed with batch=1, and the
  actual sampled-line counts differ from the nominal file names).
- Attention is enabled only on a subset of feature cascades
  (`attention_cascades`) and gradient checkpointing is applied per cascade,
  both required to fit training in 8GB VRAM (GTX 1080).
- Batch size 1 is assumed (AttentionPE block reshape and NormStats).
"""

import math
from typing import List, NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

import fastmri
from unet import Unet
from varnet import NormUnet, SensitivityModel, VarNetBlock
from utils.common.utils import center_crop


def complex_to_chan_dim(x: Tensor) -> Tensor:
    b, c, h, w, two = x.shape
    assert two == 2
    return x.permute(0, 4, 1, 2, 3).reshape(b, 2 * c, h, w)


def chan_complex_to_last_dim(x: Tensor) -> Tensor:
    b, c2, h, w = x.shape
    assert c2 % 2 == 0
    c = c2 // 2
    return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1).contiguous()


def sens_expand(x: Tensor, sens_maps: Tensor) -> Tensor:
    """(B, 2, H, W) channel image -> (B, coils, H, W, 2) k-space."""
    return fastmri.fft2c(fastmri.complex_mul(chan_complex_to_last_dim(x), sens_maps))


def sens_reduce(kspace: Tensor, sens_maps: Tensor) -> Tensor:
    """(B, coils, H, W, 2) k-space -> (B, 2, H, W) channel image."""
    x = fastmri.ifft2c(kspace)
    return complex_to_chan_dim(
        fastmri.complex_mul(x, fastmri.complex_conj(sens_maps)).sum(dim=1, keepdim=True)
    )


class NormStats(nn.Module):
    def forward(self, data: Tensor) -> Tuple[Tensor, Tensor]:
        batch, chans, _, _ = data.shape
        if batch != 1:
            raise ValueError("NormStats expects batch size 1.")
        data = data.view(chans, -1)
        mean = data.mean(dim=1)
        variance = data.var(dim=1, unbiased=False)
        return mean, variance


class FeatureImage(NamedTuple):
    features: Tensor
    sens_maps: Tensor
    means: Tensor
    variances: Tensor
    mask: Tensor
    ref_kspace: Tensor


class FeatureEncoder(nn.Module):
    def __init__(self, in_chans: int = 2, feature_chans: int = 32):
        super().__init__()
        self.feature_chans = feature_chans
        self.encoder = nn.Conv2d(in_chans, feature_chans, kernel_size=5, padding=2, bias=True)

    def forward(self, image: Tensor, means: Tensor, variances: Tensor) -> Tensor:
        means = means.view(1, -1, 1, 1)
        variances = variances.view(1, -1, 1, 1)
        return self.encoder((image - means) * torch.rsqrt(variances))


class FeatureDecoder(nn.Module):
    def __init__(self, feature_chans: int = 32, out_chans: int = 2):
        super().__init__()
        self.feature_chans = feature_chans
        self.decoder = nn.Conv2d(feature_chans, out_chans, kernel_size=5, padding=2, bias=True)

    def forward(self, features: Tensor, means: Tensor, variances: Tensor) -> Tensor:
        means = means.view(1, -1, 1, 1)
        variances = variances.view(1, -1, 1, 1)
        return self.decoder(features) * torch.sqrt(variances) + means


class FeatureNormUnet(nn.Module):
    """NormUnet variant for real-valued feature maps (no complex<->channel packing).

    Per-channel normalization as in the reference `NormUNet(use_complex=False)`.
    """

    def __init__(self, chans: int, num_pools: int, in_chans: int, out_chans: int):
        super().__init__()
        self.unet = Unet(
            in_chans=in_chans,
            out_chans=out_chans,
            chans=chans,
            num_pool_layers=num_pools,
            drop_prob=0.0,
        )

    def norm(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        b, c, h, w = x.shape
        flat = x.view(b, c, -1)
        mean = flat.mean(dim=2).view(b, c, 1, 1)
        std = flat.std(dim=2).view(b, c, 1, 1)
        return (x - mean) / (std + 1e-8), mean, std

    def unnorm(self, x: Tensor, mean: Tensor, std: Tensor) -> Tensor:
        return x * (std + 1e-8) + mean

    def pad(self, x: Tensor):
        _, _, h, w = x.shape
        w_mult = ((w - 1) | 15) + 1
        h_mult = ((h - 1) | 15) + 1
        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
        x = F.pad(x, w_pad + h_pad)
        return x, (h_pad, w_pad, h_mult, w_mult)

    def unpad(self, x: Tensor, h_pad, w_pad, h_mult, w_mult) -> Tensor:
        return x[..., h_pad[0]: h_mult - h_pad[1], w_pad[0]: w_mult - w_pad[1]]

    def forward(self, x: Tensor) -> Tensor:
        x, mean, std = self.norm(x)
        x, pad_sizes = self.pad(x)
        x = self.unet(x)
        x = self.unpad(x, *pad_sizes)
        return self.unnorm(x, mean, std)


class AttentionPE(nn.Module):
    """Cartesian-aliasing block attention with sinusoidal positional encoding."""

    def __init__(self, in_chans: int):
        super().__init__()
        self.in_chans = in_chans
        self.norm = nn.InstanceNorm2d(in_chans)
        self.q = nn.Conv2d(in_chans, in_chans, kernel_size=1)
        self.k = nn.Conv2d(in_chans, in_chans, kernel_size=1)
        self.v = nn.Conv2d(in_chans, in_chans, kernel_size=1)
        self.proj_out = nn.Conv2d(in_chans, in_chans, kernel_size=1)
        self.dilated_conv = nn.Conv2d(in_chans, in_chans, kernel_size=3, padding=2, dilation=2)

    def reshape_to_blocks(self, x: Tensor, accel: int) -> Tensor:
        # (1, C, H, W) -> (H*W_pad/accel, C, accel): columns W/accel apart share
        # the same Cartesian aliasing block.
        chans = x.shape[1]
        pad_total = (accel - x.shape[3] % accel) % accel
        pad_right = pad_total // 2
        pad_left = pad_total - pad_right
        if pad_total > 0:
            x = F.pad(x, (pad_left, pad_right, 0, 0), "reflect")
        return (
            torch.stack(x.chunk(chunks=accel, dim=3), dim=-1)
            .view(chans, -1, accel)
            .permute(1, 0, 2)
            .contiguous()
        )

    def reshape_from_blocks(self, x: Tensor, image_size: Tuple[int, int], accel: int) -> Tensor:
        chans = x.shape[1]
        num_freq, num_phase = image_size
        x = (
            x.permute(1, 0, 2)
            .reshape(1, chans, num_freq, -1, accel)
            .permute(0, 1, 2, 4, 3)
            .reshape(1, chans, num_freq, -1)
        )
        padded_phase = x.shape[3]
        pad_total = padded_phase - num_phase
        pad_right = pad_total // 2
        pad_left = pad_total - pad_right
        return x[:, :, :, pad_left: padded_phase - pad_right]

    def get_positional_encodings(self, seq_len: int, embed_dim: int, device, dtype) -> Tensor:
        idx = torch.arange(embed_dim, device=device, dtype=dtype)
        freqs = 1.0 / (10000 ** (2 * torch.div(idx, 2, rounding_mode="floor") / embed_dim))
        positions = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(1)
        scaled = positions * freqs.unsqueeze(0)
        encodings = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=1)[:, :embed_dim]
        return encodings

    def forward(self, x: Tensor, accel: int) -> Tensor:
        im_size = (x.shape[2], x.shape[3])
        h_ = self.norm(x)

        pos_enc = self.get_positional_encodings(x.shape[2], x.shape[3], h_.device, h_.dtype)
        h_ = h_ + pos_enc

        q = self.dilated_conv(self.q(h_))
        k = self.dilated_conv(self.k(h_))
        v = self.dilated_conv(self.v(h_))

        c = q.shape[1]
        q = self.reshape_to_blocks(q, accel)
        k = self.reshape_to_blocks(k, accel)
        q = q.permute(0, 2, 1)
        w_ = torch.bmm(q, k) * (int(c) ** (-0.5))
        w_ = F.softmax(w_, dim=2)

        v = self.reshape_to_blocks(v, accel)
        w_ = w_.permute(0, 2, 1)
        h_ = torch.bmm(v, w_)
        h_ = self.reshape_from_blocks(h_, im_size, accel)

        return x + self.proj_out(h_)


class AttentionFeatureVarNetBlock(nn.Module):
    """Feature-space cascade: DC term + (optional) aliasing attention + U-Net regularizer."""

    def __init__(
        self,
        encoder: FeatureEncoder,
        decoder: FeatureDecoder,
        feature_processor: nn.Module,
        attention_layer: Optional[AttentionPE] = None,
        use_extra_feature_conv: bool = False,
        use_acc_film: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.feature_processor = feature_processor
        self.attention_layer = attention_layer
        self.use_image_conv = use_extra_feature_conv
        self.dc_weight = nn.Parameter(torch.ones(1))

        feature_chans = encoder.feature_chans
        self.input_norm = nn.InstanceNorm2d(feature_chans)

        if use_acc_film:
            # Acceleration-conditioned FiLM on the regularizer output: row 0 =
            # acc4, row 1 = acc8, each row stores [gamma_hat | beta] per feature
            # channel with gamma = 1 + gamma_hat. Zero init makes the block
            # exactly equal to the unconditioned block, so a non-FiLM
            # checkpoint can warm-start via load_state_dict(strict=False).
            self.acc_film = nn.Embedding(2, 2 * feature_chans)
            nn.init.zeros_(self.acc_film.weight)
        else:
            self.acc_film = None

        if use_extra_feature_conv:
            self.output_norm = nn.InstanceNorm2d(feature_chans)
            self.output_conv = nn.Sequential(
                nn.Conv2d(feature_chans, feature_chans, kernel_size=5, padding=2, bias=False),
                nn.InstanceNorm2d(feature_chans),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(feature_chans, feature_chans, kernel_size=5, padding=2, bias=False),
                nn.InstanceNorm2d(feature_chans),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.register_buffer("zero", torch.zeros(1, 1, 1, 1, 1))

    def encode_from_kspace(self, kspace: Tensor, feature_image: FeatureImage) -> Tensor:
        image = sens_reduce(kspace, feature_image.sens_maps)
        return self.encoder(image, feature_image.means, feature_image.variances)

    def decode_to_kspace(self, feature_image: FeatureImage) -> Tensor:
        image = self.decoder(
            feature_image.features, feature_image.means, feature_image.variances
        )
        return sens_expand(image, feature_image.sens_maps)

    def _apply_acc_film(self, features: Tensor, accel: int) -> Tensor:
        # Bucket boundary 6: stride detection yields 4 or 8 in this challenge;
        # anything ambiguous falls into the nearer bucket.
        row = self.acc_film.weight[1 if accel >= 6 else 0]
        gamma_hat, beta = row.chunk(2)
        return features * (1.0 + gamma_hat.view(1, -1, 1, 1)) + beta.view(1, -1, 1, 1)

    def compute_dc_term(self, feature_image: FeatureImage) -> Tensor:
        est_kspace = self.decode_to_kspace(feature_image)
        dc_residual = torch.where(
            feature_image.mask.bool(), est_kspace - feature_image.ref_kspace, self.zero
        )
        return self.dc_weight * self.encode_from_kspace(dc_residual, feature_image)

    def forward(self, feature_image: FeatureImage, accel: int) -> FeatureImage:
        feature_image = feature_image._replace(
            features=self.input_norm(feature_image.features)
        )

        new_features = feature_image.features - self.compute_dc_term(feature_image)

        if self.attention_layer is not None:
            feature_image = feature_image._replace(
                features=self.attention_layer(feature_image.features, accel)
            )

        regularization = self.feature_processor(feature_image.features)
        if self.acc_film is not None:
            regularization = self._apply_acc_film(regularization, accel)
        new_features = new_features - regularization

        if self.use_image_conv:
            new_features = self.output_norm(new_features)
            new_features = new_features + self.output_conv(new_features)

        return feature_image._replace(features=new_features)


class FIVarNet(nn.Module):
    """Feature-Image VarNet: feature-space cascades followed by image-space cascades.

    forward(masked_kspace, mask) matches the baseline VarNet call signature:
    masked_kspace (B, coils, H, W, 2), mask (B, 1, 1, W, 1) -> (B, 384, 384).
    """

    def __init__(
        self,
        num_cascades: int = 4,
        num_image_cascades: int = 2,
        sens_chans: int = 4,
        sens_pools: int = 4,
        chans: int = 12,
        pools: int = 4,
        acceleration: int = 4,
        attention_cascades: Optional[List[int]] = None,
        image_conv_cascades: Optional[List[int]] = None,
        kspace_mult_factor: float = 1e6,
        use_checkpoint: bool = True,
        use_acc_film: bool = False,
    ):
        super().__init__()
        if image_conv_cascades is None:
            image_conv_cascades = [i for i in range(num_cascades) if i % 3 == 0]
        if attention_cascades is None:
            attention_cascades = [0]

        self.acceleration = acceleration  # fallback when mask stride detection fails
        self.kspace_mult_factor = kspace_mult_factor
        self.use_checkpoint = use_checkpoint

        self.sens_net = SensitivityModel(sens_chans, sens_pools)
        self.encoder = FeatureEncoder(in_chans=2, feature_chans=chans)
        self.decoder = FeatureDecoder(feature_chans=chans, out_chans=2)
        self.decode_norm = nn.InstanceNorm2d(chans)
        self.norm_fn = NormStats()

        self.cascades = nn.ModuleList(
            [
                AttentionFeatureVarNetBlock(
                    encoder=self.encoder,
                    decoder=self.decoder,
                    feature_processor=FeatureNormUnet(
                        chans, pools, in_chans=chans, out_chans=chans
                    ),
                    attention_layer=AttentionPE(chans) if i in attention_cascades else None,
                    use_extra_feature_conv=(i in image_conv_cascades),
                    use_acc_film=use_acc_film,
                )
                for i in range(num_cascades)
            ]
        )
        self.image_cascades = nn.ModuleList(
            [VarNetBlock(NormUnet(chans, pools)) for _ in range(num_image_cascades)]
        )

    def _infer_acceleration(self, mask: Tensor) -> int:
        """Outer (non-ACS) sampling stride of the mask = Cartesian aliasing period.

        Consecutive sampled-line gaps inside the ACS are 1; outside they equal
        the stride. Falls back to `self.acceleration` when detection fails.
        """
        with torch.no_grad():
            line = mask[0, 0, 0, :, 0] > 0
            idx = torch.nonzero(line).flatten()
            if idx.numel() >= 2:
                diffs = idx[1:] - idx[:-1]
                outer = diffs[diffs > 1]
                if outer.numel() > 0:
                    vals, counts = torch.unique(outer, return_counts=True)
                    stride = int(vals[counts.argmax()].item())
                    if 2 <= stride <= 16:
                        return stride
        return self.acceleration

    def _checkpointing(self) -> bool:
        return self.use_checkpoint and self.training and torch.is_grad_enabled()

    def _run_cascade(self, cascade: nn.Module, fi: FeatureImage, accel: int) -> FeatureImage:
        if not self._checkpointing():
            return cascade(fi, accel)

        def run(features, sens_maps, means, variances, mask, ref_kspace):
            out = cascade(
                FeatureImage(
                    features=features,
                    sens_maps=sens_maps,
                    means=means,
                    variances=variances,
                    mask=mask,
                    ref_kspace=ref_kspace,
                ),
                accel,
            )
            return out.features

        features = checkpoint(
            run, fi.features, fi.sens_maps, fi.means, fi.variances, fi.mask,
            fi.ref_kspace, use_reentrant=False,
        )
        return fi._replace(features=features)

    def _run_image_cascade(self, cascade, kspace_pred, ref_kspace, mask, sens_maps):
        if not self._checkpointing():
            return cascade(kspace_pred, ref_kspace, mask, sens_maps)
        return checkpoint(
            cascade, kspace_pred, ref_kspace, mask, sens_maps, use_reentrant=False
        )

    def forward(
        self,
        masked_kspace: Tensor,
        mask: Tensor,
        num_low_frequencies: Optional[int] = None,
        crop_size: Optional[Tuple[int, int]] = None,
    ) -> Tensor:
        accel = self._infer_acceleration(mask)
        masked_kspace = masked_kspace * self.kspace_mult_factor

        if self._checkpointing():
            sens_maps = checkpoint(self.sens_net, masked_kspace, mask, use_reentrant=False)
        else:
            sens_maps = self.sens_net(masked_kspace, mask)

        image = sens_reduce(masked_kspace, sens_maps)
        means, variances = self.norm_fn(image)
        feature_image = FeatureImage(
            features=self.encoder(image, means, variances),
            sens_maps=sens_maps,
            means=means,
            variances=variances,
            mask=mask,
            ref_kspace=masked_kspace,
        )

        for cascade in self.cascades:
            feature_image = self._run_cascade(cascade, feature_image, accel)

        kspace_pred = sens_expand(
            self.decoder(
                self.decode_norm(feature_image.features),
                feature_image.means,
                feature_image.variances,
            ),
            feature_image.sens_maps,
        )

        for cascade in self.image_cascades:
            kspace_pred = self._run_image_cascade(
                cascade, kspace_pred, masked_kspace, mask, sens_maps
            )

        kspace_pred = kspace_pred / self.kspace_mult_factor
        result = fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(kspace_pred)), dim=1)
        return center_crop(result, 384, 384)
