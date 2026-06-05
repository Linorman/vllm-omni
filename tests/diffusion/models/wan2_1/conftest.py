# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import torch
from torch import nn


class StubTransformer(nn.Module):
    def __init__(self, *, in_channels: int = 4, out_channels: int = 4) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            patch_size=(1, 2, 2),
            in_channels=in_channels,
            out_channels=out_channels,
            image_dim=None,
            vace_in_channels=12,
            vace_layers=[0, 1, 2],
        )

    @property
    def dtype(self) -> torch.dtype:
        return torch.float32

    def forward(self, **kwargs):
        hidden_states = kwargs["hidden_states"]
        return (torch.zeros_like(hidden_states[:, : self.config.out_channels]),)


class StubVAE:
    dtype = torch.float32

    def __init__(self, z_dim: int = 4) -> None:
        self.config = SimpleNamespace(
            z_dim=z_dim,
            scale_factor_temporal=4,
            scale_factor_spatial=8,
            latents_mean=[0.0] * z_dim,
            latents_std=[1.0] * z_dim,
        )

    def encode(self, video: torch.Tensor):
        latent_frames = (video.shape[2] + self.config.scale_factor_temporal - 1) // self.config.scale_factor_temporal
        latent_height = video.shape[-2] // self.config.scale_factor_spatial
        latent_width = video.shape[-1] // self.config.scale_factor_spatial
        latents = torch.ones(
            video.shape[0],
            self.config.z_dim,
            latent_frames,
            latent_height,
            latent_width,
            dtype=video.dtype,
            device=video.device,
        )
        return SimpleNamespace(latents=latents)

    def decode(self, latents: torch.Tensor, return_dict: bool = False):
        del return_dict
        return (latents,)


@contextmanager
def noop_progress_bar(*args, **kwargs):
    del args, kwargs

    class Bar:
        def update(self) -> None:
            return None

    yield Bar()
