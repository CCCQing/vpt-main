#!/usr/bin/env python3
import os
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def as_long_path(path: str) -> str:
    if os.name != "nt":
        return path
    abs_path = os.path.abspath(path)
    if abs_path.startswith("\\\\?\\"):
        return abs_path
    if abs_path.startswith("\\\\"):
        return "\\\\?\\UNC\\" + abs_path.lstrip("\\")
    return "\\\\?\\" + abs_path


def ensure_dir(path: str) -> None:
    os.makedirs(as_long_path(path), exist_ok=True)


def to_uint8_image(img: torch.Tensor) -> np.ndarray:
    """
    img: [C,H,W] tensor (any range). Returns uint8 [H,W,3].
    """
    if not torch.is_tensor(img):
        raise TypeError("img must be torch.Tensor")
    x = img.detach().float().cpu()
    if x.dim() != 3:
        raise ValueError(f"expect [C,H,W], got {tuple(x.shape)}")
    if x.shape[0] == 1:
        x = x.repeat(3, 1, 1)
    if x.shape[0] > 3:
        x = x[:3]
    x = x.permute(1, 2, 0).numpy()
    mn, mx = float(x.min()), float(x.max())
    if mx > mn:
        x = (x - mn) / (mx - mn)
    else:
        x = np.zeros_like(x)
    return (x * 255.0).clip(0, 255).astype(np.uint8)


def normalize_map(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mn, mx = float(x.min()), float(x.max())
    if mx - mn < eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


def resize_map_torch(m: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    t = torch.from_numpy(np.asarray(m, dtype=np.float32)).unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=out_hw, mode="bilinear", align_corners=False)
    return t[0, 0].cpu().numpy()


def overlay_heatmap(image_u8: np.ndarray, heat: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    h = normalize_map(heat)
    cmap = plt.get_cmap("jet")
    hm = (cmap(h)[..., :3] * 255.0).astype(np.uint8)
    out = (alpha * hm.astype(np.float32) + (1.0 - alpha) * image_u8.astype(np.float32))
    return out.clip(0, 255).astype(np.uint8)


def save_overlay(path: str, image_u8: np.ndarray, heat: np.ndarray, title: str = "") -> None:
    ensure_dir(os.path.dirname(path))
    ov = overlay_heatmap(image_u8, heat)
    plt.figure(figsize=(4, 4))
    plt.imshow(ov)
    if title:
        plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(as_long_path(path), dpi=160)
    plt.close()


def save_panel(path: str, images: List[np.ndarray], titles: Optional[List[str]] = None, ncols: int = 2) -> None:
    ensure_dir(os.path.dirname(path))
    n = len(images)
    if n == 0:
        return
    ncols = max(1, int(ncols))
    nrows = int(np.ceil(float(n) / float(ncols)))
    plt.figure(figsize=(4 * ncols, 4 * nrows))
    for i, im in enumerate(images):
        ax = plt.subplot(nrows, ncols, i + 1)
        ax.imshow(im)
        ax.axis("off")
        if titles is not None and i < len(titles):
            ax.set_title(str(titles[i]))
    plt.tight_layout()
    plt.savefig(as_long_path(path), dpi=160)
    plt.close()


def entropy_lastdim(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = p.float().clamp_min(eps)
    return -(p * p.log()).sum(dim=-1)


def attention_rollout(attn_weights: List[torch.Tensor], add_identity: bool = True) -> Optional[torch.Tensor]:
    """
    attn_weights: list of [B,H,L,L]. Returns rollout [B,L,L].
    """
    if not isinstance(attn_weights, list) or len(attn_weights) == 0:
        return None
    roll = None
    for a in attn_weights:
        if (not torch.is_tensor(a)) or a.dim() != 4:
            continue
        a_mean = a.float().mean(dim=1)  # [B,L,L]
        if add_identity:
            eye = torch.eye(a_mean.shape[-1], device=a_mean.device, dtype=a_mean.dtype).unsqueeze(0)
            a_mean = a_mean + eye
        a_mean = a_mean / a_mean.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        roll = a_mean if roll is None else torch.bmm(a_mean, roll)
    return roll


def append_csv_row(path: str, row: Dict[str, object], field_order: Optional[List[str]] = None) -> None:
    import csv

    ensure_dir(os.path.dirname(path))
    if field_order is None:
        field_order = list(row.keys())
    write_header = not os.path.exists(as_long_path(path))
    with open(as_long_path(path), "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=field_order)
        if write_header:
            w.writeheader()
        w.writerow(row)


def save_json(path: str, obj: Dict[str, object]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(as_long_path(path), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
