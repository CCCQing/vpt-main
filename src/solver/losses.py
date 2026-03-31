#!/usr/bin/env python3
"""
鎹熷け鍑芥暟瀹氫箟鏂囦欢锛?- 鍩虹鍒嗙被鎹熷け锛歋oftmaxLoss锛堟爣鍑嗗绫讳氦鍙夌喌锛?- 鍒嗙被 + 鎻愮ず瀵归綈缁勫悎鎹熷け锛歋oftmaxWithPromptAlignLoss锛圕E + L_avg锛?- RSimilarityClassifier锛氬熀浜庡叡浜蹇靛熀 R 鐨勭浉浼煎害鍒嗙被澶达紙涓嶆槸鎹熷け锛岃€屾槸涓€涓?head锛?"""


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional, List, Tuple


# ===========================
# 涓€浜涘皬宸ュ叿鍑芥暟
# ===========================
def norm1(u: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """
    绫讳技 L1 鐨勫綊涓€鍖栵細
        Norm1(u) = u / (sum_j u_j + eps)
    鐢ㄥ湪娉ㄦ剰鍔涘垎甯冧笂锛屾妸鏌愪竴缁村害涓婄殑鍊肩害鎴愨€滃拰涓?1鈥濈殑姒傜巼鍒嗗竷銆?    """
    denom = u.sum(dim=dim, keepdim=True) + eps
    return u / denom

def l_avg(
    attn_pv: torch.Tensor,
    attn_vs: torch.Tensor,
    eps: float = 1e-6,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    璁＄畻鍗曞眰鐨?L_avg^(l) 鎹熷け锛堟彁绀衡€撹瑙夆€撹涔変笁鑰呬腑鐨勨€滄彁绀衡€撹瑙夆€濆榻愰」锛夈€?
    绾﹀畾锛?        attn_pv: [B, T, N]锛屾彁绀?鈫?patch 鐨勬敞鎰忓姏 A_PV^(l)
                 B锛歜atch_size锛汿锛歱rompt 鏁帮紱N锛歱atch 鏁?        attn_vs: [B, N, M]锛宲atch 鈫?semantic 鐨勬敞鎰忓姏 A_VS^(l)
                 M锛氳涔?token / 灞炴€?slot 鏁?
    鍏紡瀵瑰簲鍏崇郴锛堝拰浣犺鎯抽噷鐨勯偅涓€娈碉級锛?    1锛塱mage-level 璇箟鍒嗗竷 蟺(x)锛氬姣忓紶鍥撅紝鎶婃墍鏈?patch鈫掕涔夋敞鎰忓姏鎸?patch 姹傚拰鍐嶅仛褰掍竴鍖栵細
            蟺(x) = Norm1( sum_n A_VS[n, :] )
       瀹炵幇锛歱i = norm1(attn_vs.sum(dim=1))        # [B, M]

    2锛塸atch 璐ｄ换搴?r(x)锛氫竴涓?patch 鍦ㄦ暣寮犲浘璇箟鍒嗗竷涓嬬殑閲嶈鎬э紝
       鎸?A_VS 涓?蟺(x) 鍋氬姞鏉冨拰锛屽啀鍋氫竴娆″綊涓€鍖栵細
            r = Norm1( A_VS * 蟺(x) )
       瀹炵幇锛歮atmul + squeeze锛屽緱鍒?[B, N]

    3锛夊钩鍧囨彁绀烘敞鎰忓姏 a_bar(x)锛氬鎻愮ず缁村害姹傚钩鍧囷細
            a_bar = mean_t A_PV[t, n]
       瀹炵幇锛歛ttn_pv.mean(dim=1) 鈫?[B, N]

    4锛塋_avg^(l)锛氬姣斺€滃钩鍧囨彁绀哄叧娉ㄧ殑 patch 鍒嗗竷 a_bar(x)鈥濆拰鈥滆涔夎矗浠诲害 r(x)鈥濓紝
       鍦?patch 缁村害涓婂仛 L2 璺濈锛?            L_avg(x) = sum_n (a_bar[n] - r[n])^2

    reduction:
        - "none": 杩斿洖 [B]锛屾瘡寮犲浘涓€涓?loss
        - "mean": batch 涓婂钩鍧?鈫?鏍囬噺
        - "sum":  batch 涓婃眰鍜?鈫?鏍囬噺
    """

    # 1) 蟺(x)锛氬 patch 缁村害姹傚拰鍐嶅仛 Norm1锛屽舰鐘?[B, M]
    pi = norm1(attn_vs.sum(dim=1), dim=-1, eps=eps)  # sum over patches

    # 2) r(x)锛欰_VS * 蟺(x)锛屽厛 [B, N, M] @ [B, M, 1] 鈫?[B, N, 1]锛屽啀 squeeze 鎴?[B, N]
    r = torch.matmul(attn_vs, pi.unsqueeze(-1)).squeeze(-1)
    r = norm1(r, dim=-1, eps=eps)

    # 3) a_bar(x)锛氬鎵€鏈?prompt 鍙栧钩鍧囷紝寰楀埌 [B, N]
    a_bar = attn_pv.mean(dim=1)

    # 4) L2 宸殑鍜岋細姣忎釜鏍锋湰涓€涓爣閲?[B]
    per_sample = ((a_bar - r) ** 2).sum(dim=-1)

    if reduction == "none":
        return per_sample
    if reduction == "sum":
        return per_sample.sum()
    # default: mean
    return per_sample.mean()


def l_avg_multi(
    attn_pv_dict: Dict[int, torch.Tensor],
    attn_vs_dict: Dict[int, torch.Tensor],
    layers: Any,
    eps: float = 1e-6,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    鎶婂灞傜殑 L_avg^(l) 鑱氬悎璧锋潵銆?
    鍙傛暟锛?        attn_pv_dict: {灞傚彿 l 鈫?[B, T, N] A_PV^(l)}
        attn_vs_dict: {灞傚彿 l 鈫?[B, N, M] A_VS^(l)}
        layers:       瑕佸弬涓庤绠楃殑灞傚垪琛紝濡?[0, 1, 2, ...]
    鍋氭硶锛?        - 瀵规瘡涓€灞傚崟鐙畻 l_avg(..., reduction="none") 寰楀埌 [B]
        - 鍦ㄢ€滃眰缁村害鈥濅笂鍙栧钩鍧囷紝寰楀埌姣忎釜鏍锋湰涓€涓暟
        - 鍐嶆寜 reduction 鍐冲畾鏄?batch-mean 杩樻槸 batch-sum
    """
    losses = []
    for l in layers:
        losses.append(l_avg(attn_pv_dict[l], attn_vs_dict[l], eps=eps, reduction="none"))
    # losses: [num_layers, B] 鈫?鍦ㄥ眰涓婂钩鍧?鈫?[B]
    stacked = torch.stack(losses, dim=0).mean(dim=0)
    if reduction == "none":
        return stacked
    if reduction == "sum":
        return stacked.sum()
    return stacked.mean()


def _extract_logits_and_aux(pred_logits: Any, kwargs: Optional[Dict[str, Any]]):
    """
    Unified extraction of:
      - logits: Tensor [B, C]
      - aux: optional dict containing attention tensors for align loss.
    """
    aux = None
    logits = pred_logits

    if isinstance(pred_logits, (list, tuple)) and len(pred_logits) > 0:
        logits = pred_logits[0]
        if len(pred_logits) > 1 and isinstance(pred_logits[1], Dict):
            aux = pred_logits[1]

    if isinstance(pred_logits, Dict) and "logits" in pred_logits:
        logits = pred_logits["logits"]
        aux = pred_logits

    if aux is None and kwargs is not None and isinstance(kwargs, Dict):
        aux = kwargs.get("aux")

    return logits, aux


def _effective_scale_from_model(model: Optional[nn.Module], logits: torch.Tensor) -> torch.Tensor:
    """
    Resolve effective scale tau used by r-similarity head.
    Fallback to 1.0 if unavailable.
    """
    tau = logits.new_tensor(1.0)
    if model is None:
        return tau
    r_head = getattr(model, "r_similarity_head", None)
    if r_head is None:
        return tau

    fixed_scale = float(getattr(r_head, "fixed_logit_scale", 0.0))
    if fixed_scale > 0:
        return logits.new_tensor(fixed_scale)

    logit_scale = getattr(r_head, "logit_scale", None)
    if logit_scale is not None:
        return logit_scale.exp().to(device=logits.device, dtype=logits.dtype)
    return tau


def _apply_additive_margin(logits: torch.Tensor, targets: torch.Tensor, margin: float, tau: torch.Tensor) -> torch.Tensor:
    """
    Additive-margin in logit space:
      z_y = z_y - tau * m
    """
    if margin <= 0:
        return logits
    if logits.dim() != 2:
        return logits
    z = logits.clone()
    idx = torch.arange(z.shape[0], device=z.device)
    z[idx, targets.long()] = z[idx, targets.long()] - (tau * float(margin))
    return z


def _compute_cm_loss_from_rhead_cache(
    model: Optional[nn.Module],
    raw_targets: Optional[torch.Tensor],
    logits: torch.Tensor,
) -> Optional[torch.Tensor]:
    """
    L_cm = mean || normalize(v_i) - normalize(s_{y_i}) ||_2^2
    using the same visual/semantic embeddings as r_similarity classification.
    """
    if model is None or raw_targets is None:
        return None
    r_head = getattr(model, "r_similarity_head", None)
    if r_head is None:
        return None

    visual = getattr(r_head, "_loss_last_visual", None)
    semantic_bank = getattr(r_head, "_loss_last_semantic", None)
    if visual is None or semantic_bank is None:
        return None
    if (not torch.is_tensor(visual)) or (not torch.is_tensor(semantic_bank)):
        return None
    if visual.dim() != 2 or semantic_bank.dim() != 2:
        return None
    if visual.shape[0] != logits.shape[0]:
        return None

    y = raw_targets.to(device=visual.device, dtype=torch.long)
    if y.shape[0] != visual.shape[0]:
        return None
    if y.min().item() < 0 or y.max().item() >= semantic_bank.shape[0]:
        return None

    s_pos = semantic_bank.index_select(0, y)
    v_n = F.normalize(visual, dim=-1)
    s_n = F.normalize(s_pos, dim=-1)
    return ((v_n - s_n) ** 2).sum(dim=-1).mean()


def _compute_role_migration_terms(
    aux: Optional[Dict[str, Any]],
    early_end: int,
    late_start: int,
) -> Dict[str, Optional[torch.Tensor]]:
    """
    Lightweight affinity-driven role losses from layer-wise aux:
      - early: encourage A_pv energy > A_ps energy
      - late:  encourage (A_ps + A_vs) > A_pv
      - avs_entropy: encourage sharper late A_vs
    """
    out: Dict[str, Optional[torch.Tensor]] = {
        "role_early": None,
        "role_late": None,
        "avs_entropy": None,
    }
    if aux is None or (not isinstance(aux, Dict)):
        return out
    attn_pv = aux.get("attn_pv")
    attn_vs = aux.get("attn_vs")
    attn_ps = aux.get("attn_ps")
    if not isinstance(attn_pv, Dict) or not isinstance(attn_vs, Dict):
        return out

    layers = sorted(set(attn_pv.keys()) & set(attn_vs.keys()))
    if len(layers) == 0:
        return out
    if isinstance(attn_ps, Dict):
        layers_ps = set(attn_ps.keys())
    else:
        layers_ps = set()

    early_layers = [l for l in layers if int(l) <= int(early_end)]
    late_layers = [l for l in layers if int(l) >= int(late_start)]
    if len(late_layers) == 0:
        late_layers = [layers[-1]]

    # Energy proxy: mean absolute attention value.
    def _energy(x: torch.Tensor) -> torch.Tensor:
        return x.float().abs().mean()

    if early_layers and len(layers_ps) > 0:
        e_pv = torch.stack([_energy(attn_pv[l]) for l in early_layers]).mean()
        e_ps = torch.stack([_energy(attn_ps[l]) for l in early_layers if l in attn_ps]).mean()
        out["role_early"] = F.relu(e_ps - e_pv)

    if late_layers and len(layers_ps) > 0:
        l_pv = torch.stack([_energy(attn_pv[l]) for l in late_layers]).mean()
        l_ps = torch.stack([_energy(attn_ps[l]) for l in late_layers if l in attn_ps]).mean()
        l_vs = torch.stack([_energy(attn_vs[l]) for l in late_layers]).mean()
        out["role_late"] = F.relu(l_pv - 0.5 * (l_ps + l_vs))

    # Late A_vs entropy (lower is sharper)
    if late_layers:
        ent = []
        for l in late_layers:
            x = attn_vs[l].float().clamp_min(1e-8)
            ent.append((-(x * x.log()).sum(dim=-1)).mean())
        out["avs_entropy"] = torch.stack(ent).mean()
    return out


def _compute_consistency_loss_from_rhead_cache(
    model: Optional[nn.Module],
    dist_type: str = "cosine",
) -> Optional[torch.Tensor]:
    if model is None:
        return None
    r_head = getattr(model, "r_similarity_head", None)
    if r_head is None:
        return None
    mu_s_final = getattr(r_head, "_loss_last_mu_s_final", None)
    h_y = getattr(r_head, "_loss_last_h_y", None)
    if torch.is_tensor(mu_s_final) and torch.is_tensor(h_y) and mu_s_final.shape == h_y.shape:
        delta_sem = mu_s_final - h_y
    else:
        delta_sem = getattr(r_head, "_loss_last_delta_sem", None)
    target = getattr(r_head, "_loss_last_cons_target", None)
    if (not torch.is_tensor(delta_sem)) or (not torch.is_tensor(target)):
        return None
    if delta_sem.numel() == 0 or target.numel() == 0:
        return None
    if delta_sem.shape != target.shape:
        return None
    if str(dist_type).lower() == "l2":
        return ((delta_sem - target) ** 2).sum(dim=-1).mean()
    # cosine (default)
    d = F.normalize(delta_sem, dim=-1)
    t = F.normalize(target, dim=-1)
    return (1.0 - (d * t).sum(dim=-1)).mean()


def _semantic_dist_from_attn(attn_pv: torch.Tensor, attn_vs: torch.Tensor, attn_ps: torch.Tensor, eps: float = 1e-6):
    """
    Build coarse semantic distributions:
      q_ind = Norm(mean_t((A_PV @ A_VS)[t, :]))
      q_dir = Norm(mean_t(A_PS[t, :]))
    """
    if attn_pv.dim() != 3 or attn_vs.dim() != 3 or attn_ps.dim() != 3:
        return None, None
    # [B, T, N] @ [B, N, M] -> [B, T, M]
    ind = torch.bmm(attn_pv, attn_vs)
    q_ind = norm1(ind.mean(dim=1), dim=-1, eps=eps)   # [B, M]
    q_dir = norm1(attn_ps.mean(dim=1), dim=-1, eps=eps)  # [B, M]
    return q_ind, q_dir


def _semantic_distribution_consistency_loss(
    aux: Optional[Dict[str, Any]],
    loss_type: str = "cosine",
    temp: float = 1.0,
    eps: float = 1e-6,
) -> Optional[torch.Tensor]:
    """
    Coarse semantic-distribution consistency between indirect and direct paths.
    """
    if aux is None or (not isinstance(aux, Dict)):
        return None
    attn_pv = aux.get("attn_pv")
    attn_vs = aux.get("attn_vs")
    attn_ps = aux.get("attn_ps")
    if attn_pv is None or attn_vs is None or attn_ps is None:
        return None

    q_ind_all = []
    q_dir_all = []

    if isinstance(attn_pv, Dict) and isinstance(attn_vs, Dict) and isinstance(attn_ps, Dict):
        layers = sorted(set(attn_pv.keys()) & set(attn_vs.keys()) & set(attn_ps.keys()))
        if not layers:
            return None
        for l in layers:
            qi, qd = _semantic_dist_from_attn(attn_pv[l], attn_vs[l], attn_ps[l], eps=eps)
            if qi is not None:
                q_ind_all.append(qi)
                q_dir_all.append(qd)
    elif torch.is_tensor(attn_pv) and torch.is_tensor(attn_vs) and torch.is_tensor(attn_ps):
        qi, qd = _semantic_dist_from_attn(attn_pv, attn_vs, attn_ps, eps=eps)
        if qi is not None:
            q_ind_all.append(qi)
            q_dir_all.append(qd)
    else:
        return None

    if len(q_ind_all) == 0:
        return None

    q_ind = torch.cat(q_ind_all, dim=0)
    q_dir = torch.cat(q_dir_all, dim=0)
    ltype = str(loss_type).lower()

    if ltype == "cosine":
        return (1.0 - F.cosine_similarity(q_ind, q_dir, dim=-1)).mean()

    t = max(float(temp), 1e-6)
    p = F.softmax(q_ind / t, dim=-1).clamp_min(eps)
    q = F.softmax(q_dir / t, dim=-1).clamp_min(eps)
    p = p / p.sum(dim=-1, keepdim=True)
    q = q / q.sum(dim=-1, keepdim=True)

    if ltype == "kl":
        return (p * (p.log() - q.log())).sum(dim=-1).mean()

    # default jsd
    m = 0.5 * (p + q)
    kl_pm = (p * (p.log() - m.log())).sum(dim=-1)
    kl_qm = (q * (q.log() - m.log())).sum(dim=-1)
    return 0.5 * (kl_pm + kl_qm).mean()


def _build_sem_route_mask(
    q_ind: torch.Tensor,
    q_dir: torch.Tensor,
    mask_type: str = "all_ones",
    topk: int = 8,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Build semantic-dimension mask/weights M with shape [B, M].
    """
    mtype = str(mask_type).lower()
    bsz, dim = q_ind.shape
    if mtype == "all_ones":
        return torch.ones_like(q_ind)

    k = max(1, min(int(topk), int(dim)))
    if mtype == "hard_topk":
        idx = torch.topk(q_ind, k=k, dim=-1).indices
        m = torch.zeros_like(q_ind)
        m.scatter_(1, idx, 1.0)
        return m

    if mtype == "soft_topk":
        # Continuous variant: emphasize values above the per-sample top-k threshold.
        kth = torch.topk(q_ind, k=k, dim=-1).values[:, -1].unsqueeze(-1)
        den = (q_ind.max(dim=-1, keepdim=True).values - kth).clamp_min(eps)
        return ((q_ind - kth) / den).clamp(0.0, 1.0)

    if mtype == "confidence_weighted":
        m = (q_ind * q_dir).clamp_min(0.0)
        return m / m.sum(dim=-1, keepdim=True).clamp_min(eps)

    # Fallback
    return torch.ones_like(q_ind)


def _semantic_route_consistency_loss(
    aux: Optional[Dict[str, Any]],
    mask_type: str = "hard_topk",
    topk: int = 8,
    gamma_ind: float = 0.0,
    gamma_dir: float = 1.0,
    eps: float = 1e-6,
) -> Optional[torch.Tensor]:
    """
    Unified soft E_a at coarse semantic-distribution level:
      q_ind = Norm(mean_t((A_PV @ A_VS)[t,:]))
      q_dir = Norm(mean_t(A_PS[t,:]))
      q_ind_tilde = (1-g_i)*sg(q_ind) + g_i*q_ind
      q_dir_tilde = (1-g_d)*sg(q_dir) + g_d*q_dir
      L_mdc = || M * (q_ind_tilde - q_dir_tilde) ||_2^2
    """
    if aux is None or (not isinstance(aux, Dict)):
        return None
    attn_pv = aux.get("attn_pv")
    attn_vs = aux.get("attn_vs")
    attn_ps = aux.get("attn_ps")
    if attn_pv is None or attn_vs is None or attn_ps is None:
        return None

    q_ind_all = []
    q_dir_all = []
    if isinstance(attn_pv, Dict) and isinstance(attn_vs, Dict) and isinstance(attn_ps, Dict):
        layers = sorted(set(attn_pv.keys()) & set(attn_vs.keys()) & set(attn_ps.keys()))
        if not layers:
            return None
        for l in layers:
            qi, qd = _semantic_dist_from_attn(attn_pv[l], attn_vs[l], attn_ps[l], eps=eps)
            if qi is not None:
                q_ind_all.append(qi)
                q_dir_all.append(qd)
    elif torch.is_tensor(attn_pv) and torch.is_tensor(attn_vs) and torch.is_tensor(attn_ps):
        qi, qd = _semantic_dist_from_attn(attn_pv, attn_vs, attn_ps, eps=eps)
        if qi is not None:
            q_ind_all.append(qi)
            q_dir_all.append(qd)
    else:
        return None

    if len(q_ind_all) == 0:
        return None

    q_ind = torch.cat(q_ind_all, dim=0)
    q_dir = torch.cat(q_dir_all, dim=0)

    gi = float(gamma_ind)
    gd = float(gamma_dir)
    q_ind_t = (1.0 - gi) * q_ind.detach() + gi * q_ind
    q_dir_t = (1.0 - gd) * q_dir.detach() + gd * q_dir

    m = _build_sem_route_mask(q_ind=q_ind.detach(), q_dir=q_dir.detach(), mask_type=mask_type, topk=topk, eps=eps)
    diff = m * (q_ind_t - q_dir_t)
    return (diff.pow(2).sum(dim=-1)).mean()


# ===========================
# 1. 鍩虹 Softmax 鍒嗙被鎹熷け
# ===========================
class SoftmaxLoss(nn.Module):
    """
    鏍囧噯鐨?Softmax + CrossEntropy 鍒嗙被鎹熷け銆?    """
    def __init__(self, cfg=None):
        # 杩欓噷 cfg 鏆傛椂娌＄敤锛屼絾淇濈暀鎺ュ彛鏂逛究浠ュ悗鎵╁睍锛堟瘮濡備粠 cfg 閲岃 class_weight 绛夛級
        super().__init__()

    def is_single(self):
        """Return True for single-branch classification loss."""
        return True

    def is_local(self):
        """
        杩斿洖 False锛岃〃绀鸿繖涓嶆槸鈥滃眬閮?patch-level 鎹熷け鈥濓紝
        鑰屾槸瀵规暣寮犲浘鐗?/ 鏁翠釜鏍锋湰鐨勫叏灞€鍒嗙被鎹熷け銆?        Trainer 閲屼細鐢ㄨ繖涓爣蹇楁潵鍖哄垎锛氭槸鍚﹂渶瑕佹妸 model / inputs 涔熶紶缁?loss銆?        """
        return False

    def loss(self, logits, targets, per_cls_weights, kwargs=None):
        """
        浣跨敤 F.cross_entropy 璁＄畻澶氱被浜ゅ弶鐔垫崯澶便€?
        鍙傛暟:
            logits: [B, C]锛屾ā鍨嬭緭鍑虹殑绫诲埆鍒嗘暟锛堟湭杩?softmax锛?            targets: [B]锛屾暣鍨嬬被鍒?id锛?~C-1锛?            per_cls_weights: 闀垮害涓?C 鐨勭被鍒潈閲嶅垪琛ㄦ垨寮犻噺
                             涓€鑸潵鑷?dataset.get_class_weights(...)锛?                             涓哄叏 1 鍒欒〃绀轰笉鍋氶噸鍔犳潈銆?            kwargs: 褰撳墠娌＄敤锛岄鐣欐墿灞曘€?
        杩斿洖:
            鏍囬噺鎹熷け鍊硷紙瀵?batch 鍋氬钩鍧囷級銆?        """
        # 鎶?python list 杞垚鏀惧湪鍚屼竴璁惧涓婄殑 tensor
        weight = torch.tensor(per_cls_weights, device=logits.device)

        # reduction="none": keep per-sample losses first.
        loss = F.cross_entropy(logits, targets, weight, reduction="none")

        # Batch mean.
        return torch.sum(loss) / targets.shape[0]

    def forward(self, pred_logits, targets, per_cls_weights, kwargs=None):
        return self.loss(pred_logits, targets, per_cls_weights, kwargs)


# ===========================
# 2. Softmax + 鎻愮ず瀵归綈缁勫悎鎹熷け
# ===========================
class SoftmaxWithPromptAlignLoss(nn.Module):
    """
    鍩虹鍒嗙被浜ゅ弶鐔?+ 鎻愮ず-瑙嗚娉ㄦ剰鍔涘榻愭鍒欙紙L_avg锛夈€?
    - 涓诲共浠嶆槸鏍囧噯 softmax 浜ゅ弶鐔碉紙SoftmaxLoss锛夈€?    - 鑻ユ彁渚?attn_pv / attn_vs锛堝崟灞傛垨澶氬眰锛変笖鏉冮噸 alpha>0锛?      鍒欓澶栧彔鍔?L_avg 瀵归綈椤癸細
          total = CE + alpha * L_avg
    - 鍏煎涓夌 logits 琛ㄨ揪褰㈠紡锛?        * 鐩存帴 tensor
        * (logits, aux) 鍏冪粍
        * {"logits": tensor, ...} 瀛楀吀
      aux 涓瓨鏀句粠鍓嶅悜浼犲洖鏉ョ殑浜插拰 / 娉ㄦ剰鍔涚瓑杈呭姪淇℃伅銆?    """

    def __init__(self, cfg=None):
        super().__init__()
        # Align loss weight alpha (0 means CE only).
        self.alpha = getattr(cfg.SOLVER, "LOSS_ALPHA", 0.0) if cfg is not None else 0.0
        # Reuse existing SoftmaxLoss implementation.
        self.cls_loss = SoftmaxLoss(cfg)

    def is_single(self):
        return True

    def is_local(self):
        return False

    def _extract_logits_and_aux(self, pred_logits: Any, kwargs: Optional[Dict[str, Any]]):
        """
        鍏煎澶氱 logits/aux 琛ㄨ揪褰㈠紡锛岀粺涓€鎶藉彇鍑猴細
            - logits: [B, C]
            - aux: dict 鎴?None锛堥噷闈㈠彲鑳芥湁 attn_pv / attn_vs锛?        """

        aux = None
        logits = pred_logits

        # 鎯呭喌涓€锛氫紶杩涙潵鐨勬槸 (logits, aux) 鎴?[logits, aux]
        if isinstance(pred_logits, (list, tuple)) and len(pred_logits) > 0:
            logits = pred_logits[0]
            # 鑻ョ浜屼釜鍏冪礌鏄?dict锛屽垯璁や负瀹冩槸 aux 淇℃伅
            if len(pred_logits) > 1 and isinstance(pred_logits[1], Dict):
                aux = pred_logits[1]

        # 鎯呭喌浜岋細浼犺繘鏉ョ殑鏄?{"logits": tensor, ...}
        if isinstance(pred_logits, Dict) and "logits" in pred_logits:
            logits = pred_logits["logits"]
            aux = pred_logits   # 鏁翠釜瀛楀吀閮藉綋鎴?aux

        # 鍏滃簳锛氬鏋?aux 杩樻槸 None锛屽彲浠ヤ粠 kwargs["aux"] 閲屽彇
        if aux is None and kwargs is not None and isinstance(kwargs, Dict):
            aux = kwargs.get("aux")

        return logits, aux

    def _compute_align_loss(self, aux: Optional[Dict[str, Any]]):
        """
        浠?aux 涓В鏋愭敞鎰忓姏寮犻噺锛岃绠?L_avg 瀵归綈椤广€?
        鏈熸湜 aux 鑷冲皯鍖呭惈锛?            aux["attn_pv"]: [B, T, N] 鎴?{灞?鈫?[B, T, N]}
            aux["attn_vs"]: [B, N, M] 鎴?{灞?鈫?[B, N, M]}
        鑻ユ壘涓嶅埌鎴?alpha <= 0锛屽垯杩斿洖 None锛岃〃绀轰笉鍔犲榻愭鍒欍€?        """

        if aux is None or self.alpha <= 0:
            return None

        attn_pv = aux.get("attn_pv") if isinstance(aux, Dict) else None
        attn_vs = aux.get("attn_vs") if isinstance(aux, Dict) else None

        if attn_pv is None or attn_vs is None:
            return None

        # 澶氬眰褰㈠紡锛氱敤 l_avg_multi 鑱氬悎
        if isinstance(attn_pv, Dict) and isinstance(attn_vs, Dict):
            layers = sorted(set(attn_pv.keys()) & set(attn_vs.keys()))
            if not layers:
                return None
            return l_avg_multi(attn_pv, attn_vs, layers=layers, reduction="mean")

        # 鍗曞眰 tensor 褰㈠紡
        if torch.is_tensor(attn_pv) and torch.is_tensor(attn_vs):
            return l_avg(attn_pv, attn_vs, reduction="mean")

        return None

    def loss(self, logits, targets, per_cls_weights, kwargs: Optional[Dict[str, Any]] = None):
        """
        缁勫悎鎹熷け锛?          1. 鍏堟娊鍑?logits / aux锛?          2. 鐢?SoftmaxLoss 绠楀熀纭€ CE锛?          3. 瑙嗘儏鍐靛彔鍔?alpha * L_avg銆?        """
        logits, aux = self._extract_logits_and_aux(logits, kwargs)

        # 鍩虹 CE 鍒嗙被鎹熷け
        base = self.cls_loss.loss(logits, targets, per_cls_weights, kwargs)

        # 瀵归綈姝ｅ垯椤?
        align = self._compute_align_loss(aux)

        if align is None:
            return base
        return base + self.alpha * align

    def forward(self, pred_logits, targets, per_cls_weights, kwargs: Optional[Dict[str, Any]] = None):
        return self.loss(pred_logits, targets, per_cls_weights, kwargs)


# ===========================
# 3. 鎹熷け鍚嶅埌绫荤殑鏄犲皠
# ===========================
LOSS = {
    "softmax": SoftmaxLoss,
    "softmax_prompt_align": SoftmaxWithPromptAlignLoss,
    "softmax_margin_cm": None,
    "softmax_margin_cm_prompt_align": None,
}


class SoftmaxMarginCMLoss(nn.Module):
    """
    L = L_cls_am + lambda_cm * L_cm
    """
    def __init__(self, cfg=None):
        super().__init__()
        self.margin = float(getattr(cfg.SOLVER, "LOSS_MARGIN", 0.05)) if cfg is not None else 0.05
        self.lambda_cm = float(getattr(cfg.SOLVER, "LOSS_CM_WEIGHT", 0.05)) if cfg is not None else 0.05
        self.sem_route_weight = float(getattr(cfg.SOLVER, "LOSS_SEM_ROUTE_WEIGHT", 0.01)) if cfg is not None else 0.01
        self.sem_route_mask_type = str(getattr(cfg.SOLVER, "LOSS_SEM_ROUTE_MASK_TYPE", "hard_topk")) if cfg is not None else "hard_topk"
        self.sem_route_topk = int(getattr(cfg.SOLVER, "LOSS_SEM_ROUTE_TOPK", 8)) if cfg is not None else 8
        self.sem_route_gamma_ind = float(getattr(cfg.SOLVER, "LOSS_SEM_ROUTE_GAMMA_IND", 0.0)) if cfg is not None else 0.0
        self.sem_route_gamma_dir = float(getattr(cfg.SOLVER, "LOSS_SEM_ROUTE_GAMMA_DIR", 1.0)) if cfg is not None else 1.0
        self.sem_route_start_epoch = int(getattr(cfg.SOLVER, "LOSS_SEM_ROUTE_START_EPOCH", 10)) if cfg is not None else 10
        self.hn_margin_enable = bool(getattr(cfg.SOLVER, "LOSS_HN_MARGIN_ENABLE", False)) if cfg is not None else False
        self.hn_margin_weight = float(getattr(cfg.SOLVER, "LOSS_HN_MARGIN_WEIGHT", 0.05)) if cfg is not None else 0.05
        self.hn_margin_value = float(getattr(cfg.SOLVER, "LOSS_HN_MARGIN_VALUE", 0.1)) if cfg is not None else 0.1
        self.hn_margin_start_epoch = int(getattr(cfg.SOLVER, "LOSS_HN_MARGIN_START_EPOCH", 0)) if cfg is not None else 0
        self.hn_detach_neg = bool(getattr(cfg.SOLVER, "LOSS_HN_DETACH_NEG", True)) if cfg is not None else True
        self.role_early_weight = float(getattr(cfg.SOLVER, "LOSS_ROLE_EARLY_WEIGHT", 0.0)) if cfg is not None else 0.0
        self.role_late_weight = float(getattr(cfg.SOLVER, "LOSS_ROLE_LATE_WEIGHT", 0.0)) if cfg is not None else 0.0
        self.agr_res_weight = float(getattr(cfg.SOLVER, "LOSS_AGR_RES_WEIGHT", 0.0)) if cfg is not None else 0.0
        self.avs_ent_weight = float(getattr(cfg.SOLVER, "LOSS_AVS_ENT_WEIGHT", 0.0)) if cfg is not None else 0.0
        self.cons_weight = float(getattr(cfg.SOLVER, "LOSS_CONS_WEIGHT", 0.0)) if cfg is not None else 0.0
        self.anchor_cons_weight = float(getattr(cfg.SOLVER, "LOSS_ANCHOR_CONS_WEIGHT", 0.0)) if cfg is not None else 0.0
        self.free_kd_weight = float(getattr(cfg.SOLVER, "LOSS_FREE_KD_WEIGHT", 0.0)) if cfg is not None else 0.0
        role_cfg = getattr(cfg.MODEL, "ROLE_MIGRATION", None) if cfg is not None else None
        self.role_early_end = int(getattr(role_cfg, "EARLY_END", 3)) if role_cfg is not None else 3
        self.role_late_start = int(getattr(role_cfg, "LATE_START", 9)) if role_cfg is not None else 9
        cons_cfg = getattr(cfg.MODEL, "CONSISTENCY", None) if cfg is not None else None
        self.consistency_dist = str(getattr(cons_cfg, "DIST", "cosine")).lower() if cons_cfg is not None else "cosine"
        diag_cfg = getattr(cfg.SOLVER, "DIAG", None) if cfg is not None else None
        self.diag_strict = bool(getattr(diag_cfg, "STRICT_CHECKS", False)) if diag_cfg is not None else False
        self.diag_print_wiring = bool(getattr(diag_cfg, "PRINT_LOSS_WIRING", False)) if diag_cfg is not None else False
        self._diag_printed = False
        self._last_hn_stats: Dict[str, float] = {}

    def is_single(self):
        return True

    def is_local(self):
        return False

    def loss(self, pred_logits, targets, per_cls_weights, kwargs: Optional[Dict[str, Any]] = None):
        logits, aux = _extract_logits_and_aux(pred_logits, kwargs)
        if not torch.is_tensor(logits):
            raise TypeError("SoftmaxMarginCMLoss expects tensor logits.")

        model = kwargs.get("model", None) if isinstance(kwargs, Dict) else None
        raw_targets = kwargs.get("raw_targets", None) if isinstance(kwargs, Dict) else None
        curr_epoch = int(kwargs.get("epoch", 0)) if isinstance(kwargs, Dict) else 0
        tau = _effective_scale_from_model(model, logits)
        logits_am = _apply_additive_margin(logits, targets, self.margin, tau)

        weight = torch.tensor(per_cls_weights, device=logits_am.device)
        ce = F.cross_entropy(logits_am, targets, weight, reduction="mean")
        cm = _compute_cm_loss_from_rhead_cache(model=model, raw_targets=raw_targets, logits=logits_am)
        if self.diag_strict and self.lambda_cm > 0 and cm is None:
            raise RuntimeError("CM loss enabled but CM term is unavailable (cache/targets mismatch).")
        total = ce if (cm is None or self.lambda_cm <= 0) else (ce + self.lambda_cm * cm)

        # HNMargin in the same seen-candidate subspace as the actually used logits/targets.
        # logits are scaled similarity; recover similarity scores by dividing tau.
        hn_term = None
        if self.hn_margin_enable and self.hn_margin_weight > 0 and curr_epoch >= self.hn_margin_start_epoch:
            tau_safe = tau.clamp_min(1e-12) if torch.is_tensor(tau) else max(float(tau), 1e-12)
            score = logits / tau_safe
            idx = torch.arange(score.shape[0], device=score.device)
            pos_score = score[idx, targets.long()]
            neg_score_mat = score.detach().clone() if self.hn_detach_neg else score.clone()
            neg_score_mat[idx, targets.long()] = -1e9
            hn_score, _ = neg_score_mat.max(dim=1)
            train_margin = pos_score - hn_score
            hn_term = F.relu(float(self.hn_margin_value) - train_margin).mean()
            total = total + self.hn_margin_weight * hn_term
            with torch.no_grad():
                self._last_hn_stats = {
                    "hn_margin_loss": float(hn_term.item()),
                    "pos_score_mean": float(pos_score.mean().item()),
                    "hn_score_mean": float(hn_score.mean().item()),
                    "train_margin_mean": float(train_margin.mean().item()),
                    "p_train_margin_lt_0": float((train_margin < 0).float().mean().item()),
                    "p_train_margin_lt_neg1": float((train_margin < -1).float().mean().item()),
                    "hn_detach_neg": bool(self.hn_detach_neg),
                }
        else:
            self._last_hn_stats = {}

        if self.sem_route_weight > 0 and curr_epoch >= self.sem_route_start_epoch:
            sem_route = _semantic_route_consistency_loss(
                aux=aux,
                mask_type=self.sem_route_mask_type,
                topk=self.sem_route_topk,
                gamma_ind=self.sem_route_gamma_ind,
                gamma_dir=self.sem_route_gamma_dir,
            )
            if sem_route is not None:
                total = total + self.sem_route_weight * sem_route
            elif self.diag_strict:
                raise RuntimeError("SemRoute enabled but sem-route term is unavailable (missing attn_ps/aux).")

        role_terms = _compute_role_migration_terms(
            aux=aux,
            early_end=self.role_early_end,
            late_start=self.role_late_start,
        )
        if self.role_early_weight > 0 and role_terms["role_early"] is not None:
            total = total + self.role_early_weight * role_terms["role_early"]
        if self.role_late_weight > 0 and role_terms["role_late"] is not None:
            total = total + self.role_late_weight * role_terms["role_late"]
        if self.avs_ent_weight > 0 and role_terms["avs_entropy"] is not None:
            total = total + self.avs_ent_weight * role_terms["avs_entropy"]

        # AGR residual norm regularizer: keep semantic delta small.
        if self.agr_res_weight > 0 and model is not None:
            r_head = getattr(model, "r_similarity_head", None)
            delta_sem = getattr(r_head, "_loss_last_delta_sem", None) if r_head is not None else None
            if torch.is_tensor(delta_sem) and delta_sem.numel() > 0:
                agr_res = (delta_sem ** 2).sum(dim=-1).mean()
                total = total + self.agr_res_weight * agr_res
                self._last_hn_stats["agr_res_loss"] = float(agr_res.detach().item())

        # AENet-style lightweight consistency on semantic increment only.
        if self.cons_weight > 0:
            cons = _compute_consistency_loss_from_rhead_cache(model=model, dist_type=self.consistency_dist)
            if cons is not None:
                total = total + self.cons_weight * cons
                self._last_hn_stats["consistency_loss"] = float(cons.detach().item())

        # Optional ablation 1: anchor-token consistency to class anchor h_y.
        if self.anchor_cons_weight > 0 and model is not None:
            r_head = getattr(model, "r_similarity_head", None)
            sem_state = getattr(r_head, "_runtime_semantic_state", None) if r_head is not None else None
            if isinstance(sem_state, dict):
                a_tok = sem_state.get("anchor_tokens")
                h_y = sem_state.get("h_y")
                if torch.is_tensor(a_tok) and torch.is_tensor(h_y) and a_tok.dim() == 3 and h_y.dim() == 2 and a_tok.shape[0] == h_y.shape[0]:
                    h = F.normalize(h_y, dim=-1).unsqueeze(1).expand_as(a_tok)
                    a = F.normalize(a_tok, dim=-1)
                    anchor_cons = (1.0 - (a * h).sum(dim=-1)).mean()
                    total = total + self.anchor_cons_weight * anchor_cons
                    self._last_hn_stats["anchor_cons_loss"] = float(anchor_cons.detach().item())

        # Optional ablation 2: free-token KD to semantic increment direction.
        if self.free_kd_weight > 0 and model is not None:
            r_head = getattr(model, "r_similarity_head", None)
            sem_state = getattr(r_head, "_runtime_semantic_state", None) if r_head is not None else None
            if isinstance(sem_state, dict):
                f_tok = sem_state.get("free_tokens")
                delta_sem = sem_state.get("delta_sem")
                if torch.is_tensor(f_tok) and torch.is_tensor(delta_sem) and f_tok.dim() == 3 and delta_sem.dim() == 2 and f_tok.shape[0] == delta_sem.shape[0] and f_tok.shape[1] > 0:
                    t = F.normalize(delta_sem.detach(), dim=-1).unsqueeze(1).expand_as(f_tok)
                    f = F.normalize(f_tok, dim=-1)
                    free_kd = (1.0 - (f * t).sum(dim=-1)).mean()
                    total = total + self.free_kd_weight * free_kd
                    self._last_hn_stats["free_kd_loss"] = float(free_kd.detach().item())

        if role_terms["role_early"] is not None:
            self._last_hn_stats["role_early_loss"] = float(role_terms["role_early"].detach().item())
        if role_terms["role_late"] is not None:
            self._last_hn_stats["role_late_loss"] = float(role_terms["role_late"].detach().item())
        if role_terms["avs_entropy"] is not None:
            self._last_hn_stats["avs_entropy"] = float(role_terms["avs_entropy"].detach().item())

        if self.diag_print_wiring and (not self._diag_printed):
            print(
                "[diag-loss] ce_targets[min,max]=({},{}) raw_targets[min,max]=({},{}) "
                "cm_enabled={} cm_available={} sem_route_enabled={} sem_route_start={} "
                "hn_enabled={} hn_start={} hn_detach_neg={} epoch={} tau={:.6f}".format(
                    int(targets.min().item()),
                    int(targets.max().item()),
                    int(raw_targets.min().item()) if torch.is_tensor(raw_targets) else -1,
                    int(raw_targets.max().item()) if torch.is_tensor(raw_targets) else -1,
                    bool(self.lambda_cm > 0),
                    bool(cm is not None),
                    bool(self.sem_route_weight > 0),
                    int(self.sem_route_start_epoch),
                    bool(self.hn_margin_enable and self.hn_margin_weight > 0),
                    int(self.hn_margin_start_epoch),
                    bool(self.hn_detach_neg),
                    int(curr_epoch),
                    float(tau.item()) if torch.is_tensor(tau) else float(tau),
                )
            )
            self._diag_printed = True
        return total

    def forward(self, pred_logits, targets, per_cls_weights, kwargs: Optional[Dict[str, Any]] = None):
        return self.loss(pred_logits, targets, per_cls_weights, kwargs)


class SoftmaxMarginCMPromptAlignLoss(SoftmaxMarginCMLoss):
    """
    L = L_cls_am + lambda_cm * L_cm + alpha * L_avg
    """
    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.alpha = float(getattr(cfg.SOLVER, "LOSS_ALPHA", 0.0)) if cfg is not None else 0.0

    def _compute_align_loss(self, aux: Optional[Dict[str, Any]]):
        if aux is None or self.alpha <= 0:
            return None
        attn_pv = aux.get("attn_pv") if isinstance(aux, Dict) else None
        attn_vs = aux.get("attn_vs") if isinstance(aux, Dict) else None
        if attn_pv is None or attn_vs is None:
            return None
        if isinstance(attn_pv, Dict) and isinstance(attn_vs, Dict):
            layers = sorted(set(attn_pv.keys()) & set(attn_vs.keys()))
            if not layers:
                return None
            return l_avg_multi(attn_pv, attn_vs, layers=layers, reduction="mean")
        if torch.is_tensor(attn_pv) and torch.is_tensor(attn_vs):
            return l_avg(attn_pv, attn_vs, reduction="mean")
        return None

    def loss(self, pred_logits, targets, per_cls_weights, kwargs: Optional[Dict[str, Any]] = None):
        logits, aux = _extract_logits_and_aux(pred_logits, kwargs)
        base = super().loss(logits, targets, per_cls_weights, kwargs=kwargs)
        align = self._compute_align_loss(aux)
        if align is None:
            return base
        return base + self.alpha * align


LOSS["softmax_margin_cm"] = SoftmaxMarginCMLoss
LOSS["softmax_margin_cm_prompt_align"] = SoftmaxMarginCMPromptAlignLoss

# ===========================
# 4. 鍩轰簬鍏变韩 R 鐨勭浉浼煎害鍒嗙被澶达紙涓嶆槸鎹熷け锛?# ===========================
class RSimilarityClassifier(nn.Module):
    """
    Use semantic prototypes (raw/refined/fused) for visual-semantic similarity classification.
    """

    def __init__(
        self,
        class_attr: torch.Tensor,
        hidden_size: int,
        proj_dim: int = None,
        use_cosine: bool = True,
        logit_scale_init: float = 10.0,
        visual_proj_enable: bool = False,
        fixed_logit_scale: float = 0.0,
        use_proto_per_sample: bool = False,
        shuffle_prototypes: bool = False,
        semantic_score_source: str = "auto",
        semantic_score_mode: str = "global_refined",
        semantic_score_topk: int = 5,
        semantic_score_alpha: float = 0.5,
        semantic_score_train_include_gt: bool = True,
        role_migration_enable: bool = False,
        role_migration_early_end: int = 3,
        role_migration_late_start: int = 9,
        agr_enable: bool = False,
        agr_topk: int = 5,
        agr_alpha: float = 0.2,
        agr_fuse_alpha: float = 0.5,
        agr_train_include_gt: bool = True,
        consistency_enable: bool = False,
        consistency_proj: str = "linear",
        consistency_dist: str = "cosine",
        debug_trace_once: bool = False,
        debug_shapes: bool = False,
    ) -> None:
        super().__init__()
        if proj_dim is None or proj_dim <= 0:
            proj_dim = hidden_size

        self.register_buffer("class_attr", class_attr.float())
        self.num_classes = class_attr.shape[0]
        self.attr_dim = int(class_attr.shape[-1])
        self.hidden_size = int(hidden_size)

        self.visual_proj_enabled = visual_proj_enable
        out_dim = proj_dim if self.visual_proj_enabled else hidden_size
        self.visual_proj = nn.Linear(hidden_size, out_dim) if self.visual_proj_enabled else None
        self.semantic_proj = nn.Linear(hidden_size, out_dim)
        # Stable semantic anchor: h_c = Linear(s_raw_c)
        self.semantic_anchor = nn.Linear(self.attr_dim, hidden_size)

        self.use_cosine = use_cosine
        self.fixed_logit_scale = float(fixed_logit_scale)
        self.use_proto_per_sample = bool(use_proto_per_sample)
        self.shuffle_prototypes = bool(shuffle_prototypes)

        self.semantic_score_source = str(semantic_score_source).lower()
        self.semantic_score_mode = str(semantic_score_mode).lower()
        self.semantic_score_topk = int(max(1, semantic_score_topk))
        self.semantic_score_alpha = float(semantic_score_alpha)
        self.semantic_score_train_include_gt = bool(semantic_score_train_include_gt)
        self.role_migration_enable = bool(role_migration_enable)
        self.role_migration_early_end = int(role_migration_early_end)
        self.role_migration_late_start = int(role_migration_late_start)
        self.agr_enable = bool(agr_enable)
        self.agr_topk = int(max(1, agr_topk))
        self.agr_alpha = float(agr_alpha)
        self.agr_fuse_alpha = float(agr_fuse_alpha)
        self.agr_train_include_gt = bool(agr_train_include_gt)
        self.consistency_enable = bool(consistency_enable)
        self.consistency_proj = str(consistency_proj).lower()
        self.consistency_dist = str(consistency_dist).lower()
        if self.consistency_enable:
            if self.consistency_proj == "mlp":
                self.consistency_head = nn.Sequential(
                    nn.Linear(self.attr_dim, hidden_size),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_size, out_dim),
                )
            else:
                self.consistency_head = nn.Linear(self.attr_dim, out_dim)
        else:
            self.consistency_head = None
        self.debug_trace_once = bool(debug_trace_once)
        self.debug_shapes = bool(debug_shapes)

        self._debug_sem_source_logged = False
        self._shape_debug_logged = False
        self._debug_last_raw_sim = None
        self._debug_last_raw_sim_raw = None
        self._debug_last_raw_sim_ref = None
        self._debug_last_scaled_logits = None
        self._loss_last_visual = None
        self._loss_last_semantic = None
        self._loss_last_semantic_raw = None
        self._loss_last_semantic_ref = None
        self._loss_last_scale = None
        self._loss_last_source = "refined"
        self._loss_last_delta_sem = None
        self._loss_last_cons_target = None
        self._loss_last_delta_sem_norm = None
        self._loss_last_mu_s_final = None
        self._loss_last_h_y = None
        self._runtime_targets = None
        self._runtime_token_sequence = None
        self._runtime_affinities = None
        self._runtime_semantic_state = None
        self._last_score_stats: Dict[str, Any] = {}

        if use_cosine:
            self.logit_scale = nn.Parameter(torch.log(torch.tensor(logit_scale_init, dtype=torch.float32)))
        else:
            self.logit_scale = None

    def _class_prototypes_raw(self) -> torch.Tensor:
        attr = self.class_attr
        return self.semantic_anchor(attr)

    def _class_prototypes_refined(self) -> torch.Tensor:
        # Legacy shared semantic concept branch was removed.
        # Refined behavior is handled by side-branch/runtime states and AGR scoring.
        return self._class_prototypes_raw()

    def _collect_role_summary_from_affinity(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        affinities = self._runtime_affinities if isinstance(self._runtime_affinities, list) else None
        if not affinities:
            return out

        def _entropy(x: torch.Tensor) -> torch.Tensor:
            p = x.clamp_min(1e-8)
            return -(p * p.log()).sum(dim=-1).mean()

        apv_energy: List[float] = []
        aps_energy: List[float] = []
        avs_energy: List[float] = []
        apv_entropy: List[float] = []
        per_layer: Dict[int, Dict[str, float]] = {}
        for li, aff in enumerate(affinities):
            if not isinstance(aff, dict):
                continue
            st: Dict[str, float] = {}
            apv = aff.get("Apv")
            aps = aff.get("Aps")
            avs = aff.get("Avs")
            if torch.is_tensor(apv):
                x = apv.float()
                e = float(x.abs().mean().item())
                h = float(_entropy(x).item())
                apv_energy.append(e)
                apv_entropy.append(h)
                st["apv_energy"] = e
                st["apv_entropy"] = h
            if torch.is_tensor(aps):
                e = float(aps.float().abs().mean().item())
                aps_energy.append(e)
                st["aps_energy"] = e
            if torch.is_tensor(avs):
                e = float(avs.float().abs().mean().item())
                avs_energy.append(e)
                st["avs_energy"] = e
            if st:
                per_layer[int(li)] = st

        if len(per_layer) == 0:
            return out
        out["affinity_per_layer"] = per_layer
        out["affinity_apv_energy_mean"] = float(sum(apv_energy) / max(1, len(apv_energy)))
        out["affinity_aps_energy_mean"] = float(sum(aps_energy) / max(1, len(aps_energy)))
        out["affinity_avs_energy_mean"] = float(sum(avs_energy) / max(1, len(avs_energy)))
        out["affinity_apv_entropy_mean"] = float(sum(apv_entropy) / max(1, len(apv_entropy)))

        max_layer = max(per_layer.keys())
        early_ids = [i for i in per_layer.keys() if i <= min(self.role_migration_early_end, max_layer)]
        late_ids = [i for i in per_layer.keys() if i >= min(self.role_migration_late_start, max_layer)]

        def _mean_for(layer_ids: List[int], key: str) -> Optional[float]:
            vals = [float(per_layer[i][key]) for i in layer_ids if key in per_layer[i]]
            if len(vals) == 0:
                return None
            return float(sum(vals) / len(vals))

        if early_ids:
            out["affinity_early_apv_energy"] = _mean_for(early_ids, "apv_energy")
            out["affinity_early_aps_energy"] = _mean_for(early_ids, "aps_energy")
            out["affinity_early_avs_energy"] = _mean_for(early_ids, "avs_energy")
        if late_ids:
            out["affinity_late_apv_energy"] = _mean_for(late_ids, "apv_energy")
            out["affinity_late_aps_energy"] = _mean_for(late_ids, "aps_energy")
            out["affinity_late_avs_energy"] = _mean_for(late_ids, "avs_energy")
            out["affinity_late_apv_entropy"] = _mean_for(late_ids, "apv_entropy")
        return out

    def _agr_refine_candidates(
        self,
        visual_tokens: torch.Tensor,
        raw_sim_raw: torch.Tensor,
        raw_proto: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[float], Optional[float], Optional[float]]:
        """
        Late-stage AGR:
          - coarse retrieval on raw scores
          - top-k candidates get refined prototype via Avs-weighted residual
          - non-candidates keep raw score
        """
        bsz, n_cls = raw_sim_raw.shape
        k = int(min(max(1, self.agr_topk), n_cls))
        topk_idx = torch.topk(raw_sim_raw, k=k, dim=1).indices
        candidate_mask = torch.zeros_like(raw_sim_raw, dtype=torch.bool)
        candidate_mask.scatter_(1, topk_idx, True)
        pre_union_mask = candidate_mask.clone()
        coarse_contains_gt = None
        post_union_contains_gt = None
        runtime_targets = self._runtime_targets if torch.is_tensor(self._runtime_targets) else None
        if runtime_targets is not None and runtime_targets.numel() == bsz:
            t = runtime_targets.to(raw_sim_raw.device, non_blocking=True).long().view(-1)
            valid = (t >= 0) & (t < n_cls)
            if valid.any():
                row = torch.arange(bsz, device=raw_sim_raw.device)[valid]
                coarse_contains_gt = float(pre_union_mask[row, t[valid]].float().mean().item())
                if self.training and self.agr_train_include_gt:
                    candidate_mask[row, t[valid]] = True
                post_union_contains_gt = float(candidate_mask[row, t[valid]].float().mean().item())

        # Avs from last layer (if available), fallback: uniform over patches.
        avs_last = None
        if isinstance(self._runtime_affinities, list):
            for aff in reversed(self._runtime_affinities):
                if isinstance(aff, dict) and torch.is_tensor(aff.get("Avs")):
                    avs_last = aff["Avs"]
                    break
        if torch.is_tensor(avs_last) and avs_last.dim() == 4:
            avs_last = avs_last.mean(dim=1)  # [B, N, M]
        if torch.is_tensor(avs_last) and avs_last.dim() == 3 and avs_last.shape[2] == n_cls:
            avs_cls_patch = avs_last.transpose(1, 2).contiguous()  # [B, C, N]
        else:
            # uniform fallback when affinity unavailable
            avs_cls_patch = visual_tokens.new_full((bsz, n_cls, visual_tokens.shape[1]), 1.0 / max(1, visual_tokens.shape[1]))

        delta = torch.bmm(avs_cls_patch, visual_tokens)  # [B, C, D]
        mu_s_final = raw_proto.unsqueeze(0) + float(self.agr_alpha) * delta
        return mu_s_final, candidate_mask, coarse_contains_gt, post_union_contains_gt, float(delta.norm(dim=-1).mean().item())

    def _select_classifier_prototypes(self, raw_proto: torch.Tensor, ref_proto: torch.Tensor):
        if self.role_migration_enable:
            return raw_proto, "raw_anchor"
        source = self.semantic_score_source
        if source == "auto":
            source = "raw"
        if source == "raw":
            return raw_proto, "raw"
        if source == "fused":
            a = float(self.semantic_score_alpha)
            return a * ref_proto + (1.0 - a) * raw_proto, "fused"
        return ref_proto, "refined"

    def forward(self, cls_feat: torch.Tensor) -> torch.Tensor:
        proto_raw = self._class_prototypes_raw()
        proto_ref = self._class_prototypes_refined()
        prototypes, sem_source = self._select_classifier_prototypes(proto_raw, proto_ref)

        if self.shuffle_prototypes and prototypes.shape[0] > 1:
            perm = torch.randperm(prototypes.shape[0], device=prototypes.device)
            prototypes = prototypes.index_select(0, perm)
            proto_raw = proto_raw.index_select(0, perm)
            proto_ref = proto_ref.index_select(0, perm)

        visual = self.visual_proj(cls_feat) if self.visual_proj else cls_feat
        semantic = self.semantic_proj(prototypes)
        semantic_raw = self.semantic_proj(proto_raw)
        semantic_ref = self.semantic_proj(proto_ref)

        if self.debug_shapes and (not self._shape_debug_logged):
            print(
                "[SHAPE-DEBUG] RSimilarityClassifier.forward visual_feature={} semantic_prototype={} "
                "visual_proj={} semantic_proj={} logits={}".format(
                    tuple(cls_feat.shape) if torch.is_tensor(cls_feat) else None,
                    tuple(prototypes.shape) if torch.is_tensor(prototypes) else None,
                    tuple(visual.shape) if torch.is_tensor(visual) else None,
                    tuple(semantic.shape) if torch.is_tensor(semantic) else None,
                    (int(visual.shape[0]), int(semantic.shape[0])) if (torch.is_tensor(visual) and torch.is_tensor(semantic)) else None,
                )
            )
            self._shape_debug_logged = True

        if self.use_cosine:
            visual = F.normalize(visual, dim=-1)
            semantic = F.normalize(semantic, dim=-1)
            semantic_raw = F.normalize(semantic_raw, dim=-1)
            semantic_ref = F.normalize(semantic_ref, dim=-1)

            raw_sim = visual @ semantic.t()
            raw_sim_raw = visual @ semantic_raw.t()
            raw_sim_ref = visual @ semantic_ref.t()

            if self.fixed_logit_scale > 0:
                scale = raw_sim.new_tensor(self.fixed_logit_scale)
            else:
                scale = self.logit_scale.exp()

            score_mode = self.semantic_score_mode
            if self.role_migration_enable and score_mode != "global_raw":
                score_mode = "affinity_role_migration"
            if self.training and score_mode not in {"coarse_to_fine", "affinity_role_migration", "agr_c2f"}:
                score_mode = "global_refined" if sem_source == "refined" else ("global_raw" if sem_source == "raw" else "global_refined")
            coarse_topk_contains_gt = None
            coarse_topk_contains_gt_postunion = None

            if score_mode == "global_raw":
                logits = raw_sim_raw * scale
                sem_source = "raw"
                self._last_score_stats = {
                    "semantic_score_mode": score_mode,
                    "semantic_score_topk": int(self.semantic_score_topk),
                    "semantic_score_alpha": float(self.semantic_score_alpha),
                    "semantic_score_train_include_gt": bool(self.semantic_score_train_include_gt),
                }
            elif score_mode == "global_refined":
                logits = raw_sim_ref * scale
                sem_source = "refined"
                self._last_score_stats = {
                    "semantic_score_mode": score_mode,
                    "semantic_score_topk": int(self.semantic_score_topk),
                    "semantic_score_alpha": float(self.semantic_score_alpha),
                    "semantic_score_train_include_gt": bool(self.semantic_score_train_include_gt),
                }
            elif score_mode in {"affinity_role_migration", "agr_c2f"}:
                # New path: raw anchors as coarse retrieval + late-stage AGR candidate refinement.
                token_seq = self._runtime_token_sequence if torch.is_tensor(self._runtime_token_sequence) else None
                if token_seq is not None and token_seq.dim() == 3:
                    patch_tokens = token_seq[:, 1:, :] if token_seq.shape[1] > 1 else token_seq
                else:
                    patch_tokens = cls_feat.unsqueeze(1)
                if self.visual_proj is not None:
                    patch_tokens = self.visual_proj(patch_tokens)
                patch_tokens = F.normalize(patch_tokens, dim=-1)

                mu_s_final_all, candidate_mask, coarse_topk_contains_gt, coarse_topk_contains_gt_postunion, delta_norm = self._agr_refine_candidates(
                    visual_tokens=patch_tokens,
                    raw_sim_raw=raw_sim_raw,
                    raw_proto=semantic_raw,
                )
                if self.use_proto_per_sample:
                    # Legacy ablation path: build per-sample mixed bank (raw + candidate refined).
                    proto_per_sample = semantic_raw.unsqueeze(0).expand_as(mu_s_final_all).clone()
                    proto_per_sample[candidate_mask] = mu_s_final_all[candidate_mask]
                    refined_bank = proto_per_sample
                else:
                    # Default path: mu_s_final is the only image-conditioned semantic object.
                    refined_bank = mu_s_final_all
                refined_sim = torch.einsum("bd,bcd->bc", visual, F.normalize(refined_bank, dim=-1))
                a = float(self.agr_fuse_alpha)
                blended = raw_sim_raw + a * (refined_sim - raw_sim_raw)
                final_sim = torch.where(candidate_mask, blended, raw_sim_raw)
                logits = final_sim * scale
                sem_source = "affinity_role_migration(anchor+agr)"

                with torch.no_grad():
                    role_stats = self._collect_role_summary_from_affinity()
                    self._last_score_stats = {
                        "semantic_score_mode": "affinity_role_migration",
                        "semantic_score_topk": int(self.agr_topk),
                        "semantic_score_alpha": float(self.agr_fuse_alpha),
                        "semantic_score_train_include_gt": bool(self.agr_train_include_gt),
                        "use_proto_per_sample": bool(self.use_proto_per_sample),
                        "coarse_topk_contains_gt_rate": coarse_topk_contains_gt,
                        "coarse_topk_contains_gt_rate_postunion": coarse_topk_contains_gt_postunion,
                        "candidate_source": "raw_topk",
                        "agr_alpha": float(self.agr_alpha),
                        "agr_delta_norm_mean": float(delta_norm) if delta_norm is not None else None,
                    }
                    self._last_score_stats.update(role_stats)

                # Cache per-sample semantic delta for consistency loss on GT only.
                self._loss_last_delta_sem = None
                self._loss_last_cons_target = None
                self._loss_last_delta_sem_norm = None
                self._loss_last_mu_s_final = None
                self._loss_last_h_y = None
                runtime_targets = self._runtime_targets if torch.is_tensor(self._runtime_targets) else None
                if runtime_targets is not None and runtime_targets.numel() == logits.shape[0]:
                    t = runtime_targets.to(logits.device).long().view(-1)
                    valid = (t >= 0) & (t < proto_raw.shape[0])
                    if valid.any():
                        ridx = torch.arange(logits.shape[0], device=logits.device)[valid]
                        raw_y = semantic_raw.index_select(0, t[valid])
                        fin_y = mu_s_final_all[ridx, t[valid]]
                        self._loss_last_h_y = raw_y
                        self._loss_last_mu_s_final = fin_y
                        delta_sem = fin_y - raw_y
                        self._loss_last_delta_sem = delta_sem
                        self._loss_last_delta_sem_norm = float(delta_sem.norm(dim=-1).mean().item())
                        if self.consistency_head is not None:
                            attr_y = self.class_attr.index_select(0, t[valid]).to(delta_sem.device)
                            target = self.consistency_head(attr_y)
                            target = F.normalize(target, dim=-1) if self.use_cosine else target
                            self._loss_last_cons_target = target
            elif score_mode == "coarse_to_fine":
                k = int(min(max(1, self.semantic_score_topk), raw_sim_raw.shape[1]))
                topk_idx = torch.topk(raw_sim_raw, k=k, dim=1).indices
                candidate_mask = torch.zeros_like(raw_sim_raw, dtype=torch.bool)
                candidate_mask.scatter_(1, topk_idx, True)

                coarse_topk_contains_gt = None
                coarse_recall_at_1 = None
                coarse_recall_at_3 = None
                coarse_recall_at_5 = None
                coarse_recall_at_10 = None
                coarse_gap_mean = None
                coarse_gap_median = None
                candidate_delta_mean = None
                candidate_delta_gt_mean = None
                candidate_delta_hn_mean = None
                candidate_delta_gap = None
                candidate_delta_gap_available_ratio = None
                runtime_targets = self._runtime_targets if torch.is_tensor(self._runtime_targets) else None
                if runtime_targets is not None and runtime_targets.numel() == raw_sim_raw.shape[0]:
                    t = runtime_targets.to(raw_sim_raw.device, non_blocking=True).long().view(-1)
                    valid = (t >= 0) & (t < raw_sim_raw.shape[1])
                    if valid.any():
                        row = torch.arange(raw_sim_raw.shape[0], device=raw_sim_raw.device)[valid]
                        topk_mask_pre = candidate_mask.clone()
                        coarse_topk_contains_gt = float(topk_mask_pre[row, t[valid]].float().mean().item())
                        # coarse recall@k diagnostics (always pre-union GT)
                        def _recall_at(kk: int) -> float:
                            kk = int(min(max(1, kk), raw_sim_raw.shape[1]))
                            idxk = torch.topk(raw_sim_raw[row], k=kk, dim=1).indices
                            return float((idxk == t[valid].view(-1, 1)).any(dim=1).float().mean().item())
                        coarse_recall_at_1 = _recall_at(1)
                        coarse_recall_at_3 = _recall_at(3)
                        coarse_recall_at_5 = _recall_at(5)
                        coarse_recall_at_10 = _recall_at(10)
                        # coarse gap true - hardest-negative in raw space
                        pos_raw = raw_sim_raw[row, t[valid]]
                        neg_raw = raw_sim_raw[row].clone()
                        neg_raw.scatter_(1, t[valid].view(-1, 1), -1e9)
                        hn_raw, hn_raw_idx = neg_raw.max(dim=1)
                        coarse_gap = pos_raw - hn_raw
                        coarse_gap_mean = float(coarse_gap.mean().item())
                        coarse_gap_median = float(torch.median(coarse_gap).item())
                        # candidate delta over pre-union raw_topk candidate set
                        delta_all = raw_sim_ref[row] - raw_sim_raw[row]
                        mask_pre = topk_mask_pre[row]
                        if mask_pre.any():
                            candidate_delta_mean = float(delta_all[mask_pre].mean().item())
                        gt_in = mask_pre.gather(1, t[valid].view(-1, 1)).squeeze(1)
                        if gt_in.any():
                            candidate_delta_gt_mean = float(
                                delta_all.gather(1, t[valid].view(-1, 1)).squeeze(1)[gt_in].mean().item()
                            )
                        hn_in = mask_pre.gather(1, hn_raw_idx.view(-1, 1)).squeeze(1)
                        if hn_in.any():
                            candidate_delta_hn_mean = float(
                                delta_all.gather(1, hn_raw_idx.view(-1, 1)).squeeze(1)[hn_in].mean().item()
                            )
                        both = gt_in & hn_in
                        candidate_delta_gap_available_ratio = float(both.float().mean().item())
                        if both.any():
                            gt_delta = delta_all.gather(1, t[valid].view(-1, 1)).squeeze(1)
                            hn_delta = delta_all.gather(1, hn_raw_idx.view(-1, 1)).squeeze(1)
                            candidate_delta_gap = float((gt_delta[both] - hn_delta[both]).mean().item())
                        if self.training and self.semantic_score_train_include_gt:
                            candidate_mask[row, t[valid]] = True
                            coarse_topk_contains_gt_postunion = float(candidate_mask[row, t[valid]].float().mean().item())

                a = float(self.semantic_score_alpha)
                blended = raw_sim_raw + a * (raw_sim_ref - raw_sim_raw)
                final_sim = torch.where(candidate_mask, blended, raw_sim_raw)
                logits = final_sim * scale
                sem_source = "coarse_to_fine(raw+residual_refined)"
                self._last_score_stats = {
                    "semantic_score_mode": score_mode,
                    "semantic_score_topk": int(self.semantic_score_topk),
                    "semantic_score_alpha": float(self.semantic_score_alpha),
                    "semantic_score_train_include_gt": bool(self.semantic_score_train_include_gt),
                    "coarse_topk_contains_gt_rate": coarse_topk_contains_gt,
                    "coarse_topk_contains_gt_rate_postunion": coarse_topk_contains_gt_postunion,
                    "coarse_recall_at_k": coarse_topk_contains_gt,
                    "coarse_recall_at_1": coarse_recall_at_1,
                    "coarse_recall_at_3": coarse_recall_at_3,
                    "coarse_recall_at_5": coarse_recall_at_5,
                    "coarse_recall_at_10": coarse_recall_at_10,
                    "coarse_gap_mean": coarse_gap_mean,
                    "coarse_gap_median": coarse_gap_median,
                    "candidate_source": "raw_topk",
                    "candidate_delta_mean": candidate_delta_mean,
                    "candidate_delta_gt_mean": candidate_delta_gt_mean,
                    "candidate_delta_hn_mean": candidate_delta_hn_mean,
                    "candidate_delta_gap": candidate_delta_gap,
                    "candidate_delta_gap_available_ratio": candidate_delta_gap_available_ratio,
                }
            else:
                logits = raw_sim * scale
                coarse_topk_contains_gt = None
                self._last_score_stats = {
                    "semantic_score_mode": score_mode,
                    "semantic_score_topk": int(self.semantic_score_topk),
                    "semantic_score_alpha": float(self.semantic_score_alpha),
                    "semantic_score_train_include_gt": bool(self.semantic_score_train_include_gt),
                }
                self._loss_last_delta_sem = None
                self._loss_last_cons_target = None
                self._loss_last_delta_sem_norm = None
                self._loss_last_mu_s_final = None
                self._loss_last_h_y = None

            if self.debug_trace_once and (not self._debug_sem_source_logged):
                scale_scalar = float(scale.item()) if torch.is_tensor(scale) else float(scale)
                fixed_mode = bool(self.fixed_logit_scale > 0)
                print(
                    "[trace] node=C.semantic_scoring semantic_score_mode={} topk={} alpha={:.4f} train_include_gt={} "
                    "classifier_semantic_source={} coarse_topk_contains_gt(pre_union)={} coarse_topk_contains_gt(post_union)={} "
                    "raw_semantic_shape={} refined_semantic_shape={} classifier_semantic_shape={} "
                    "raw_semantic_norm_mean={:.6f} refined_semantic_norm_mean={:.6f} classifier_semantic_norm_mean={:.6f} "
                    "effective_logit_scale={:.6f} whether_fixed_logit_scale={}".format(
                        score_mode,
                        int(self.semantic_score_topk),
                        float(self.semantic_score_alpha),
                        bool(self.semantic_score_train_include_gt),
                        sem_source,
                        "NA" if coarse_topk_contains_gt is None else "{:.4f}".format(float(coarse_topk_contains_gt)),
                        "NA" if coarse_topk_contains_gt_postunion is None else "{:.4f}".format(float(coarse_topk_contains_gt_postunion)),
                        tuple(proto_raw.shape),
                        tuple(proto_ref.shape),
                        tuple(proto_raw.shape),
                        float(proto_raw.float().norm(dim=-1).mean().item()),
                        float(proto_ref.float().norm(dim=-1).mean().item()),
                        float(proto_raw.float().norm(dim=-1).mean().item()),
                        scale_scalar,
                        fixed_mode,
                    )
                )
                self._debug_sem_source_logged = True

            self._debug_last_raw_sim = raw_sim.detach()
            self._debug_last_raw_sim_raw = raw_sim_raw.detach()
            self._debug_last_raw_sim_ref = raw_sim_ref.detach()
            self._debug_last_scaled_logits = logits.detach()
            self._loss_last_visual = visual
            # Main classifier bank for losses/monitor stays on raw anchors for seen/unseen symmetry.
            self._loss_last_semantic = semantic_raw
            self._loss_last_semantic_raw = semantic_raw
            self._loss_last_semantic_ref = semantic_ref
            self._loss_last_scale = scale
            self._loss_last_source = sem_source
            sem_state = self._runtime_semantic_state if isinstance(self._runtime_semantic_state, dict) else None
            if sem_state is not None:
                mu_s_final = sem_state.get("mu_s_final")
                h_y = sem_state.get("h_y")
                delta_sem = sem_state.get("delta_sem")
                if torch.is_tensor(mu_s_final) and torch.is_tensor(h_y):
                    if (delta_sem is None) or (not torch.is_tensor(delta_sem)):
                        delta_sem = mu_s_final - h_y
                    self._loss_last_mu_s_final = mu_s_final
                    self._loss_last_h_y = h_y
                    self._loss_last_delta_sem = delta_sem
                    self._loss_last_delta_sem_norm = float(delta_sem.norm(dim=-1).mean().item())
                    self._loss_last_cons_target = None
                    runtime_targets = self._runtime_targets if torch.is_tensor(self._runtime_targets) else None
                    if self.consistency_head is not None and runtime_targets is not None and runtime_targets.numel() == delta_sem.shape[0]:
                        t = runtime_targets.to(delta_sem.device).long().view(-1)
                        valid = (t >= 0) & (t < self.class_attr.shape[0])
                        if valid.any():
                            attr_y = self.class_attr.index_select(0, t[valid]).to(delta_sem.device)
                            target = self.consistency_head(attr_y)
                            target = F.normalize(target, dim=-1) if self.use_cosine else target
                            self._loss_last_cons_target = target
            if self._loss_last_delta_sem is None and (not self.role_migration_enable):
                # Fallback for non-AGR modes: use refined-vs-raw delta for GT if available.
                runtime_targets = self._runtime_targets if torch.is_tensor(self._runtime_targets) else None
                if runtime_targets is not None and runtime_targets.numel() == logits.shape[0]:
                    t = runtime_targets.to(logits.device).long().view(-1)
                    valid = (t >= 0) & (t < semantic_raw.shape[0])
                    if valid.any():
                        y = t[valid]
                        d = semantic_ref.index_select(0, y) - semantic_raw.index_select(0, y)
                        self._loss_last_h_y = semantic_raw.index_select(0, y)
                        self._loss_last_mu_s_final = semantic_ref.index_select(0, y)
                        self._loss_last_delta_sem = d
                        self._loss_last_delta_sem_norm = float(d.norm(dim=-1).mean().item())
                        if self.consistency_head is not None:
                            attr_y = self.class_attr.index_select(0, y).to(d.device)
                            target = self.consistency_head(attr_y)
                            target = F.normalize(target, dim=-1) if self.use_cosine else target
                            self._loss_last_cons_target = target
            if self._loss_last_delta_sem is None:
                self._loss_last_mu_s_final = None
                self._loss_last_h_y = None
        else:
            logits = visual @ semantic.t()
            self._debug_last_raw_sim = logits.detach()
            self._debug_last_scaled_logits = logits.detach()
            self._loss_last_visual = visual
            self._loss_last_semantic = semantic_raw
            self._loss_last_semantic_raw = semantic_raw
            self._loss_last_semantic_ref = semantic_ref
            self._loss_last_scale = logits.new_tensor(1.0)
            self._loss_last_source = sem_source
            self._loss_last_delta_sem = None
            self._loss_last_cons_target = None
            self._loss_last_delta_sem_norm = None
            self._loss_last_mu_s_final = None
            self._loss_last_h_y = None

        return logits

def build_loss(cfg):
    """
    Build loss module from cfg.SOLVER.LOSS.
    """
    loss_name = cfg.SOLVER.LOSS
    assert loss_name in LOSS, f'loss name {loss_name} is not supported'
    loss_fn = LOSS[loss_name]
    if not loss_fn:
        return None
    return loss_fn(cfg)
