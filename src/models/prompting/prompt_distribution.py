"""
Pre-ViT prompt distribution modules for ZSL/GZSL.

鍦?ViT 缂栫爜鍣?*涔嬪墠* 鏋勯€犳彁绀哄垎甯冪殑妯″潡銆?

鏁翠綋鎬濊矾锛?
- 瀵规棭鏈熺殑瑙嗚 token `V_raw` 鍋氭睜鍖栵紝寰楀埌涓€涓綆缁寸殑瑙嗚缁熻鍚戦噺 `h_v`锛?
- 鐢ㄤ竴涓€滃悗楠屽ご鈥濅及璁￠珮鏂悗楠?q(z|x) 鐨勫潎鍊?mu 涓?log 鏂瑰樊 logvar锛屽苟浣跨敤閲嶅弬鏁板寲鎶€宸ч噰鏍?z锛?
- 灏嗛殣鍙橀噺 z锛堝彲浠ヤ笌璇箟灞炴€ф嫾鎺ワ級瑙ｇ爜涓轰竴缁?prompt tokens锛?
  杩欎簺 prompt tokens 浼氬湪杈撳叆搴忓垪缁村害涓婁笌 [CLS]銆乸atch tokens 杩涜鎷兼帴閫佸叆 ViT銆?

璁捐鐩爣锛?
- 鑳藉鐩存帴鎻掑叆鐜版湁 VPT 鐨?prompt 娉ㄥ叆娴佺▼锛圼CLS] + prompt + patch锛夛紝
  ViT 鍙渶瑕佺煡閬?prompt_len锛岃€屼笉闇€瑕佺煡閬?prompt 鏄浣曠敱鍒嗗竷鐢熸垚鐨勩€?
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


def _reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    浣跨敤閲嶅弬鏁板寲鎶€宸т粠楂樻柉鍚庨獙涓噰鏍凤細
        z = mu + sigma * eps,
        鍏朵腑 sigma = exp(0.5 * logvar), eps ~ N(0, I)

    杩欐牱閲囨牱 z 鐨勮繃绋嬪 mu銆乴ogvar 鏄彲瀵肩殑锛屾柟渚垮仛 KL 绾︽潫銆?
    """
    std = torch.exp(0.5 * logvar)        # 鏍囧噯宸?sigma锛屼繚鎸侀潪璐?
    eps = torch.randn_like(std)          # 涓?std 鍚屽舰鐘剁殑鏍囧噯姝ｆ€佸櫔澹?
    return mu + eps * std               # 閲囨牱寰楀埌 z


class VisualStatsEncoder(nn.Module):
    """
    灏嗗師濮嬭瑙?token 搴忓垪 V_raw 缂栫爜涓轰竴涓叏灞€缁熻鍚戦噺 h_v 鐨勬ā鍧椼€?

    鏀寔鐨勬睜鍖栨ā寮忥細
    - "gap"   : global average pooling锛屽叏灞€骞冲潎姹犲寲锛?
    - "gem"   : generalized mean pooling锛屽甫鍙涔犳寚鏁扮殑骞夸箟鍧囧€兼睜鍖栵紱
    - "attnpool": 鍗曟煡璇㈢殑娉ㄦ剰鍔涙睜鍖栵紙绫讳技 CLIP 鐨?AttentionPool2d 鎬濊矾锛夛紱
    - "gated" : gated sum锛岀粰姣忎釜 token 瀛︿竴涓?gate锛屽啀鍔犳潈姹傚拰褰掍竴鍖栥€?
    """

    def __init__(self, dim: int, pool: str = "gap"):
        """
        鍙傛暟锛?
            dim  : 姣忎釜瑙嗚 token 鐨勭淮搴?D锛?
            pool : 姹犲寲绫诲瀷瀛楃涓诧紝瑙佷笂銆?
        """
        super().__init__()
        self.pool = pool
        if pool == "attnpool":
            # 娉ㄦ剰鍔涙睜鍖栦腑浣跨敤鐨?query 鍚戦噺锛岀淮搴︿笌 token 鐩稿悓
            self.query = nn.Parameter(torch.randn(dim))
        elif pool == "gem":
            # GeM 鐨勬寚鏁?p锛屽彲瀛︿範锛屽垵濮嬪寲涓?3.0锛堝父瑙佺殑缁忛獙鍊硷級
            self.p = nn.Parameter(torch.ones(1) * 3.0)
        elif pool == "gated":
            # Gated pooling 浣跨敤鐨?gate 绾挎€у眰锛氬姣忎釜 token 杈撳嚭涓€涓爣閲?gate
            self.gate = nn.Linear(dim, 1)
        elif pool != "gap":
            # 闈炴硶鐨勬睜鍖栫被鍨嬬粰鍑烘姤閿?
            raise ValueError(f"Unsupported pooling mode: {pool}")

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        杈撳叆锛?
            tokens: [B, L, D]锛孊 涓?batch 澶у皬锛孡 涓?token 鏁帮紝D 涓洪€氶亾缁村害銆?

        杈撳嚭锛?
            h_v: [B, D]锛屾瘡涓牱鏈搴斾竴涓叏灞€瑙嗚缁熻鍚戦噺銆?
        """
        if self.pool == "gap":
            # 绠€鍗曠殑 Token 缁村钩鍧?
            return tokens.mean(dim=1)

        if self.pool == "gem":
            # GeM: (1/L * sum(x^p))^(1/p)
            # 杩欓噷浣跨敤涓€涓叡浜殑 p 鍙傛暟锛屽苟瀵硅緭鍏ュ仛 clamp 闃叉鏁板€奸棶棰?
            p = torch.clamp(self.p, min=1e-3)
            # 鍏堝 token 缁存眰骞冲潎锛屽啀鍋?1/p 娆″箓
            return torch.pow(tokens.clamp(min=1e-6).mean(dim=1), 1.0 / p)

        if self.pool == "attnpool":
            # 鍗曟煡璇㈡敞鎰忓姏姹犲寲锛?
            # 瀵规瘡涓?token 璁＄畻涓?query 鐨勭偣绉紝鍋?softmax 寰楁潈閲嶏紝鍐嶆寜鏉冮噸鍔犳潈姹傚拰
            q = self.query.to(tokens.dtype)                 # 淇濊瘉涓?tokens 鐨?dtype 涓€鑷达紙鍏煎 AMP锛?
            # [B, L, D] @ [D] -> [B, L]
            attn = torch.matmul(tokens, q) / math.sqrt(tokens.size(-1))
            weights = attn.softmax(dim=1)                   # 鍦?token 缁村害鍋?softmax
            # 鎸夋潈閲嶅 tokens 鍔犳潈姹傚拰锛歴um_l w_l * token_l
            return torch.einsum("bl, bld -> bd", weights, tokens)

        if self.pool == "gated":
            # Gated pooling:
            # 鐢ㄤ竴灞傜嚎鎬у眰浜х敓 gate锛屽啀缁忚繃 sigmoid 鏄犲皠鍒?(0,1)
            gates = torch.sigmoid(self.gate(tokens))        # [B, L, 1]
            # 瀵规瘡涓?token 涔樹笂 gate 绯绘暟
            gated_tokens = tokens * gates                   # [B, L, D]
            # 褰掍竴鍖栧洜瀛愶細鎵€鏈?gate 鐨勫拰锛岄槻姝㈠叏 0 鐢?clamp
            denom = gates.sum(dim=1).clamp(min=1e-6)        # [B, 1]
            # 鍔犳潈鍜岄櫎浠ュ洜瀛?-> 绫讳技鈥滃姞鏉冨钩鍧団€?
            return gated_tokens.sum(dim=1) / denom          # [B, D]

        # 鐞嗚涓婁笉搴旇鍒拌繖閲岋紝鍥犱负闈炴硶妯″紡鍦?__init__ 涓凡缁忔姏寮傚父
        raise ValueError(f"Unsupported pooling mode: {self.pool}")


class PosteriorHead(nn.Module):
    """
    鍚庨獙鎺ㄦ柇澶达紙amortized posterior head锛夛細

    杈撳叆涓€涓粺璁″悜閲?h锛堜緥濡?h_v锛夛紝杈撳嚭涓よ矾锛?
        - mu     : 楂樻柉鍚庨獙鐨勫潎鍊煎悜閲?
        - logvar : 楂樻柉鍚庨獙瀵硅鍗忔柟宸殑 log 鏂瑰樊

    鍐呴儴缁撴瀯涓轰袱涓嫭绔嬬殑 MLP锛坢u_head 鍜?logvar_head锛夛紝鍏变韩杈撳叆 h銆?
    """

    def __init__(self, in_dim: int, hidden_dim: int, latent_dim: int):
        """
        鍙傛暟锛?
            in_dim    : 杈撳叆缁熻鍚戦噺 h 鐨勭淮搴︼紱
            hidden_dim: 涓棿闅愯棌灞傜淮搴︼紱
            latent_dim: 娼滃彉閲?z 鐨勭淮搴︺€?
        """
        super().__init__()
        # 鍧囧€煎垎鏀?
        self.mu_head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        # log 鏂瑰樊鍒嗘敮
        self.logvar_head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        杈撳叆锛?
            h: [B, in_dim] 瑙嗚缁熻鍚戦噺

        杈撳嚭锛?
            mu:     [B, latent_dim]
            logvar: [B, latent_dim]
        """
        return self.mu_head(h), self.logvar_head(h)


class PromptGenerator(nn.Module):
    """
    灏嗘綔鍙橀噺 z锛堜互鍙婂彲閫夌殑璇箟灞炴€у悜閲忥級瑙ｇ爜涓轰竴缁?prompt tokens 鐨勬ā鍧椼€?

    Args:
        latent_dim:  闅愬彉閲?z 鐨勭淮搴︼紱
        prompt_dim:  杈撳嚭鐨?prompt token 缁村害锛屽簲涓?ViT 鐨?hidden_size 涓€鑷达紱
        prompt_len:  闇€瑕佺敓鎴愮殑 prompt token 涓暟锛?
        hidden_dim:  瑙ｇ爜鍣ㄥ唴閮ㄧ殑闅愯棌灞傜淮搴︼紱
        semantic_dim: 鑻ヤ笉涓?None锛屽垯琛ㄧず灏嗚涔夊睘鎬ф嫾鎺ュ埌 z 涓婅繘琛屾潯浠剁敓鎴愶紝
                      璇箟鍚戦噺鐨勭淮搴︺€?
    """

    def __init__(
        self,
        latent_dim: int,
        prompt_dim: int,
        prompt_len: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.prompt_len = prompt_len
        self.fusion = nn.Linear(latent_dim, hidden_dim)
        # 鍚庣画 MLP锛歨idden -> hidden -> (prompt_len * prompt_dim)
        self.mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, prompt_len * prompt_dim),
        )
        self.prompt_dim = prompt_dim

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        杈撳叆锛?
            z:         [B, latent_dim] 娼滃彉閲忔牱鏈紱
            semantics: [B, semantic_dim] 鎴?None锛岃涔夋潯浠跺悜閲忋€?

        杈撳嚭锛?
            prompts: [B, prompt_len, prompt_dim] 鐢熸垚鐨勬彁绀?tokens銆?
        """
        h = self.fusion(z)                       # [B, hidden_dim]
        # MLP 瑙ｇ爜鍒?(prompt_len * prompt_dim)
        prompts = self.mlp(h)                        # [B, prompt_len * prompt_dim]
        # 鍐?reshape 鎴?[B, L_p, D]
        prompts = prompts.view(-1, self.prompt_len, self.prompt_dim)
        return prompts


class PreViTPromptDistributor(nn.Module):
    """
    灏嗘棭鏈熻瑙?tokens锛堜互鍙婂彲閫夌殑璇箟灞炴€э級鏄犲皠涓?prompt tokens 鐨勭鍒扮灏佽妯″潡銆?

    鍏稿瀷璋冪敤鏂瑰紡锛?
        prompts, stats = module(V_raw, S_raw)

    鍏朵腑锛?
        - V_raw: [B, L, D]锛屼负 ViT patch+pos 缂栫爜鍚庣殑瑙嗚 tokens锛堝彲鍙彇绗竴灞傛垨鑻ュ共灞傝緭鍑猴級锛?
        - S_raw: [B, d_s] 鎴?[B, T, d_s]锛屼负鍘熷璇箟灞炴€э紙鍙€夛級锛?
        - prompts: [B, prompt_len, D]锛屽彲鐩存帴鎷煎埌 ViT 鐨勮緭鍏ュ簭鍒楅噷锛?
        - stats: dict锛屽寘鍚?mu/logvar/z/h_v锛岀敤浜庡仛 KL 姝ｅ垯鎴栧垎鏋愩€?
    """

    def __init__(
        self,
        dim: int,
        prompt_len: int,
        latent_dim: int,
        hidden_dim: int,
        pool: str = "gap",
    ):
        """
        鍙傛暟锛?
            dim               : 瑙嗚 token 鐨勯€氶亾缁村害 D锛堝嵆 ViT hidden_size锛夛紱
            prompt_len        : 鐢熸垚鐨?prompt token 涓暟锛?
            latent_dim        : 娼滃彉閲?z 缁村害锛?
            hidden_dim        : 鍚庨獙澶翠笌鐢熸垚鍣ㄤ腑鐨勯殣钘忕淮搴︼紱
            pool              : 瑙嗚姹犲寲鏂瑰紡锛屼紶缁?VisualStatsEncoder锛?
            semantic_dim      : 鍘熷璇箟灞炴€х淮搴︼紙鑻ユ湁锛夛紱
            semantic_proj_dim : 鑻ヤ笉涓?None锛屽垯鍏堝皢璇箟浠?semantic_dim 鏄犲皠鍒拌缁村害锛?
                                鍐嶄笌 z 鎷兼帴杩涘叆 PromptGenerator銆?
        """
        super().__init__()
        # Debug/ablation switch: if True, bypass reparameterized sampling and use z=mu.
        self.disable_sampling = False
        # 鍦ㄥ仛缁熻鍓嶅厛瀵?V_raw 鍋?LayerNorm锛岀浉褰撲簬 鈥淟N(V_raw)鈥?鐨勬楠?
        self.norm = nn.LayerNorm(dim)
        # 瑙嗚缁熻缂栫爜鍣細LN 鍚庣殑 V_raw -> h_v
        self.visual_encoder = VisualStatsEncoder(dim, pool=pool)
        # 鍚庨獙澶达細h_v -> (mu, logvar)
        self.posterior = PosteriorHead(dim, hidden_dim, latent_dim)
        # 鎻愮ず鐢熸垚鍣細z (+ 璇箟) -> prompt tokens
        self.prompt_generator = PromptGenerator(
            latent_dim=latent_dim,
            prompt_dim=dim,
            prompt_len=prompt_len,
            hidden_dim=hidden_dim,
        )


    def forward(
        self, V_raw: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        杈撳叆锛?
            V_raw: [B, L, D]锛岃瑙?tokens锛堥€氬父鏄?ViT 鐨?patch+pos 缂栫爜杈撳嚭锛夛紱
            S_raw: [B, d_s] 鎴?[B, T, d_s]锛堝彲閫夛級锛屽師濮嬭涔夊睘鎬с€?

        杈撳嚭锛?
            prompts: [B, prompt_len, D]锛岀敓鎴愮殑 prompt tokens锛?
            stats:   dict锛屽寘鍚嫢骞蹭腑闂撮噺锛?
                     - "mu"   : [B, latent_dim]锛岄珮鏂悗楠屽潎鍊硷紱
                     - "logvar": [B, latent_dim]锛宭og 鏂瑰樊锛?
                     - "z"    : [B, latent_dim]锛岄噰鏍峰緱鍒扮殑娼滃彉閲忥紱
                     - "h_v"  : [B, D]锛岃瑙夌粺璁″悜閲忋€?
        """
        # 1) 瀵硅瑙?tokens 鍋?LN
        V_norm = self.norm(V_raw)                  # [B, L, D]
        # 2) 瑙嗚姹犲寲锛屽緱鍒?h_v
        h_v = self.visual_encoder(V_norm)          # [B, D]
        # 3) 鍚庨獙澶磋緭鍑?mu 涓?logvar
        mu, logvar = self.posterior(h_v)           # 鍚勪负 [B, latent_dim]
        # 4) 閲嶅弬鏁板寲閲囨牱 z
        if self.disable_sampling:
            z = mu
        else:
            z = _reparameterize(mu, logvar)            # [B, latent_dim]

        prompts = self.prompt_generator(z)  # [B, prompt_len, D]

        # 7) 鎵撳寘涓棿缁熻閲忥紝渚夸簬鍦ㄥ閮ㄦ瀯閫?KL loss 鎴栧彲瑙嗗寲
        stats = {"mu": mu, "logvar": logvar, "z": z, "h_v": h_v}
        return prompts, stats


__all__ = [
    "PreViTPromptDistributor",
    "VisualStatsEncoder",
    "PosteriorHead",
    "PromptGenerator",
    "generate_prompt_init",
]

def generate_prompt_init(
    distributor: PreViTPromptDistributor,
    V_raw: torch.Tensor,
    reduce: str = "mean",
) -> torch.Tensor:
    """鍒╃敤棰?ViT 鎻愮ず鍒嗗竷妯″潡鐢熸垚涓€娆℃€х殑 prompt 鍒濆鍖栧紶閲忋€?

    Args:
        distributor: 棰勫厛鏋勫缓濂界殑 ``PreViTPromptDistributor`` 瀹炰緥銆?
        V_raw: 褰㈢姸 (B, L, D) 鐨勬棭鏈熻瑙?token锛堝惈浣嶇疆缂栫爜锛夛紝閫氬父鍙?
            鍙栬缁冮泦鐨勪竴涓?batch 杩涜鍒濆鍖栥€?
        S_raw: 锛堝彲閫夛級褰㈢姸 (B, M, d_sem) 鐨勮涔夊睘鎬ф垨绫诲師鍨嬨€?
        reduce: 灏?batch 缁村悎骞朵负鍗曚釜鍒濆鍖栧悜閲忕殑鏂瑰紡锛岀洰鍓嶆敮鎸?"mean"
            鍜?"first"锛屽垎鍒〃绀哄 batch 骞冲潎鎴栧彇绗竴鏉℃牱鏈€?

    Returns:
        prompt_init: 褰㈢姸 (1, prompt_len, D) 鐨勫紶閲忥紝鍙洿鎺ヤ紶缁?
            ``PromptedVisionTransformer(prompt_init=...)`` 鐢ㄤ簬涓€娆℃€у垵濮嬪寲銆?
    """
    with torch.no_grad():
        prompts, _ = distributor(V_raw)
        if reduce == "first":
            prompts = prompts[:1]
        elif reduce == "mean":
            prompts = prompts.mean(dim=0, keepdim=True)
        else:
            raise ValueError(f"Unsupported reduce mode: {reduce}")
        return prompts.detach()



