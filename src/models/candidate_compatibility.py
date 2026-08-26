from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CandidateConditionedCompatibility(nn.Module):
    def __init__(
        self,
        visual_dim: int,
        semantic_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.visual_dim = int(visual_dim)
        self.semantic_dim = int(semantic_dim)
        self.hidden_dim = int(hidden_dim)
        self.visual_encoder = nn.Sequential(
            nn.LayerNorm(self.visual_dim),
            nn.Linear(self.visual_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
        )
        self.semantic_encoder = nn.Sequential(
            nn.LayerNorm(self.semantic_dim),
            nn.Linear(self.semantic_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
        )
        self.pair_scorer = nn.Sequential(
            nn.Linear(4 * self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, 1),
        )
        self.logit_scale = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def _score_chunk(
        self,
        visual: torch.Tensor,
        semantic: torch.Tensor,
    ) -> torch.Tensor:
        batch = visual.shape[0]
        classes = semantic.shape[0]
        visual_expanded = visual[:, None, :].expand(batch, classes, -1)
        semantic_expanded = semantic[None, :, :].expand(batch, classes, -1)
        pair = torch.cat(
            (
                visual_expanded * semantic_expanded,
                torch.abs(visual_expanded - semantic_expanded),
                visual_expanded,
                semantic_expanded,
            ),
            dim=-1,
        )
        return self.pair_scorer(pair).squeeze(-1)

    def forward(
        self,
        visual_features: torch.Tensor,
        semantic_candidates: torch.Tensor,
        *,
        candidate_chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        if visual_features.dim() != 2 or visual_features.shape[-1] != self.visual_dim:
            raise ValueError("visual_features do not match the declared visual dimension")
        if semantic_candidates.dim() != 2 or semantic_candidates.shape[-1] != self.semantic_dim:
            raise ValueError("semantic_candidates do not match the declared semantic dimension")
        visual = F.normalize(self.visual_encoder(visual_features.float()), dim=-1)
        semantic = F.normalize(self.semantic_encoder(semantic_candidates.float()), dim=-1)
        chunk_size = max(
            1,
            int(candidate_chunk_size or semantic.shape[0]),
        )
        chunks = [
            self._score_chunk(visual, semantic[start : start + chunk_size])
            for start in range(0, semantic.shape[0], chunk_size)
        ]
        scale = self.logit_scale.clamp(min=-4.0, max=4.0).exp()
        return torch.cat(chunks, dim=1) * scale


class ImageOnlyTemperatureCompatibility(nn.Module):
    def __init__(self, visual_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.visual_dim = int(visual_dim)
        self.temperature = nn.Sequential(
            nn.LayerNorm(self.visual_dim),
            nn.Linear(self.visual_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(
        self,
        visual_features: torch.Tensor,
        semantic_candidates: torch.Tensor,
        *,
        candidate_chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        del candidate_chunk_size
        visual = F.normalize(visual_features.float(), dim=-1)
        semantic = F.normalize(semantic_candidates.float(), dim=-1)
        base = visual @ semantic.t()
        scale = self.temperature(visual_features.float()).clamp(-4.0, 4.0).exp()
        return base * scale


class GlobalTemperatureCompatibility(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward(
        self,
        visual_features: torch.Tensor,
        semantic_candidates: torch.Tensor,
        *,
        candidate_chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        del candidate_chunk_size
        visual = F.normalize(visual_features.float(), dim=-1)
        semantic = F.normalize(semantic_candidates.float(), dim=-1)
        return (visual @ semantic.t()) * self.logit_scale.clamp(-4.0, 4.0).exp()


def trainable_parameter_count(module: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad))
