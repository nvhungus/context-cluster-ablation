from __future__ import annotations

import torch
from einops import rearrange
from torch import nn

from .layers import (
    ChannelGate,
    DropPath,
    channel_shuffle,
    make_local_branch,
    make_norm,
    pairwise_cosine,
    pairwise_hamming_simulated,
)


class Mlp(nn.Module):
    def __init__(self, channels: int, ratio: float = 2.0, drop: float = 0.0, act_layer: type[nn.Module] = nn.GELU):
        super().__init__()
        hidden = int(channels * ratio)
        self.fc1 = nn.Linear(channels, hidden)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x.permute(0, 3, 1, 2).contiguous()


class ContextClusterOp(nn.Module):
    """CoC token mixer with cosine or simulated Hamming similarity."""

    def __init__(
        self,
        dim: int,
        out_dim: int | None = None,
        proposal: tuple[int, int] = (2, 2),
        fold: tuple[int, int] = (1, 1),
        heads: int = 1,
        head_dim: int = 16,
        similarity: str = "cosine",
        hamming_scale: bool = False,
        store_assignments: bool = False,
    ) -> None:
        super().__init__()
        out_dim = out_dim or dim
        self.dim = dim
        self.out_dim = out_dim
        self.heads = heads
        self.head_dim = head_dim
        self.proposal = proposal
        self.fold = fold
        self.similarity = similarity.lower()
        self.store_assignments = store_assignments
        self.f = nn.Conv2d(dim, heads * head_dim, 1)
        self.v = nn.Conv2d(dim, heads * head_dim, 1)
        self.proj = nn.Conv2d(heads * head_dim, out_dim, 1)
        self.sim_alpha = nn.Parameter(torch.ones(1))
        self.sim_beta = nn.Parameter(torch.zeros(1))
        self.binary_scale = nn.Parameter(torch.ones(1)) if hamming_scale else None
        self.centers_proposal = nn.AdaptiveAvgPool2d(proposal)
        self.last_assignment: torch.Tensor | None = None

    def _similarity(self, centers: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        if self.similarity == "cosine":
            return pairwise_cosine(centers, points)
        if self.similarity in {"hamming", "hamming_sim", "hamming_simulated"}:
            return pairwise_hamming_simulated(centers, points, self.binary_scale)
        raise ValueError(f"Unsupported similarity: {self.similarity}")

    def _partition(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int, int] | None]:
        fold_h, fold_w = self.fold
        if fold_h <= 1 and fold_w <= 1:
            return x, None
        b, c, h, w = x.shape
        if h % fold_h != 0 or w % fold_w != 0:
            raise ValueError(f"Feature map {(h, w)} is not divisible by fold {self.fold}")
        x = rearrange(x, "b c (fh h) (fw w) -> (b fh fw) c h w", fh=fold_h, fw=fold_w)
        return x, (b, h // fold_h, w // fold_w, c)

    def _unpartition(self, x: torch.Tensor, meta: tuple[int, int, int, int] | None) -> torch.Tensor:
        if meta is None:
            return x
        b, _, _, _ = meta
        fold_h, fold_w = self.fold
        return rearrange(x, "(b fh fw) c h w -> b c (fh h) (fw w)", b=b, fh=fold_h, fw=fold_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = self.v(x)
        feat = self.f(x)
        feat = rearrange(feat, "b (heads c) h w -> (b heads) c h w", heads=self.heads)
        value = rearrange(value, "b (heads c) h w -> (b heads) c h w", heads=self.heads)
        feat, meta = self._partition(feat)
        value, _ = self._partition(value)

        b, c, h, w = feat.shape
        centers = self.centers_proposal(feat)
        value_centers = self.centers_proposal(value)
        center_h, center_w = centers.shape[-2:]
        centers_flat = centers.flatten(2).transpose(1, 2)
        points_flat = feat.flatten(2).transpose(1, 2)
        value_flat = value.flatten(2).transpose(1, 2)
        value_centers_flat = value_centers.flatten(2).transpose(1, 2)

        raw_sim = self._similarity(centers_flat, points_flat)
        sim = torch.sigmoid(self.sim_beta + self.sim_alpha * raw_sim)
        sim_max_idx = sim.max(dim=1, keepdim=True).indices
        if self.store_assignments:
            self.last_assignment = sim_max_idx.detach()
        mask = torch.zeros_like(sim)
        mask.scatter_(1, sim_max_idx, 1.0)
        sim = sim * mask

        aggregated = (
            (value_flat.unsqueeze(1) * sim.unsqueeze(-1)).sum(dim=2) + value_centers_flat
        ) / (sim.sum(dim=-1, keepdim=True) + 1.0)
        dispatched = (aggregated.unsqueeze(2) * sim.unsqueeze(-1)).sum(dim=1)
        out = dispatched.transpose(1, 2).reshape(b, c, h, w)
        out = self._unpartition(out, meta)
        out = rearrange(out, "(b heads) c h w -> b (heads c) h w", heads=self.heads)
        out = self.proj(out)
        return out

    def theoretical_bops(self, input_shape: tuple[int, int, int, int]) -> int:
        if self.similarity not in {"hamming", "hamming_sim", "hamming_simulated"}:
            return 0
        b, _, h, w = input_shape
        fold_h, fold_w = self.fold
        region_h = h // max(1, fold_h)
        region_w = w // max(1, fold_w)
        proposals = self.proposal[0] * self.proposal[1]
        regions = max(1, fold_h * fold_w)
        return int(b * self.heads * regions * proposals * region_h * region_w * self.head_dim)


class HybridClusterBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 2.0,
        proposal: tuple[int, int] = (2, 2),
        fold: tuple[int, int] = (1, 1),
        heads: int = 1,
        head_dim: int = 16,
        similarity: str = "cosine",
        local_branch: str = "identity",
        local_ratio: float = 0.0,
        channel_shuffle_enabled: bool = False,
        norm: str = "bn",
        drop: float = 0.0,
        drop_path: float = 0.0,
        layer_scale: float = 1e-5,
        hamming_scale: bool = False,
        mode: str = "cluster",
    ) -> None:
        super().__init__()
        self.mode = mode
        self.channel_shuffle_enabled = channel_shuffle_enabled
        self.norm1 = make_norm(norm, dim)
        self.norm2 = make_norm(norm, dim)
        self.drop_path = DropPath(drop_path)
        self.layer_scale_1 = nn.Parameter(layer_scale * torch.ones(dim))
        self.layer_scale_2 = nn.Parameter(layer_scale * torch.ones(dim))

        if mode == "local":
            self.cluster_dim = 0
            self.local_dim = dim
        else:
            self.local_dim = int(round(dim * local_ratio))
            self.local_dim = max(0, min(self.local_dim, dim))
            self.cluster_dim = dim - self.local_dim

        self.cluster = (
            ContextClusterOp(
                self.cluster_dim,
                self.cluster_dim,
                proposal=proposal,
                fold=fold,
                heads=heads,
                head_dim=head_dim,
                similarity=similarity,
                hamming_scale=hamming_scale,
            )
            if self.cluster_dim > 0
            else nn.Identity()
        )
        self.local = (
            make_local_branch(local_branch, self.local_dim, self.local_dim, norm=norm, act="gelu")
            if self.local_dim > 0
            else nn.Identity()
        )
        self.fuse = nn.Conv2d(dim, dim, 1) if (self.cluster_dim > 0 and self.local_dim > 0) else nn.Identity()
        self.mlp = Mlp(dim, ratio=mlp_ratio, drop=drop)

    def _token_mix(self, x: torch.Tensor) -> torch.Tensor:
        if self.cluster_dim == 0:
            return self.local(x)
        if self.local_dim == 0:
            return self.cluster(x)
        xc, xl = torch.split(x, [self.cluster_dim, self.local_dim], dim=1)
        out = torch.cat((self.cluster(xc), self.local(xl)), dim=1)
        if self.channel_shuffle_enabled:
            out = channel_shuffle(out, groups=2)
        return self.fuse(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self._token_mix(self.norm1(x))
        x = x + self.drop_path(self.layer_scale_1.view(1, -1, 1, 1) * y)
        y = self.mlp(self.norm2(x))
        x = x + self.drop_path(self.layer_scale_2.view(1, -1, 1, 1) * y)
        return x


class Stage(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        mlp_ratio: float,
        proposal: tuple[int, int],
        fold: tuple[int, int],
        heads: int,
        head_dim: int,
        similarity: str,
        local_branch: str,
        local_ratio: float,
        mode: str,
        channel_shuffle_enabled: bool,
        norm: str,
        drop: float,
        drop_path_rates: list[float],
        pruning_mask: bool = False,
    ) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            *[
                HybridClusterBlock(
                    dim=dim,
                    mlp_ratio=mlp_ratio,
                    proposal=proposal,
                    fold=fold,
                    heads=heads,
                    head_dim=head_dim,
                    similarity=similarity,
                    local_branch=local_branch,
                    local_ratio=local_ratio,
                    channel_shuffle_enabled=channel_shuffle_enabled,
                    norm=norm,
                    drop=drop,
                    drop_path=drop_path_rates[i],
                    mode=mode,
                )
                for i in range(depth)
            ]
        )
        self.gate = ChannelGate(dim) if pruning_mask else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gate(self.blocks(x))
