#!/usr/bin/env python3

"""
ViT-related models ViT 绯诲垪妯″瀷瀹氫箟涓庡皝瑁呫€?
Note: models return logits instead of prob
娉ㄦ剰锛氬悇妯″瀷鐨?forward 杩斿洖鐨勬槸鈥渓ogits鈥濓紙鏈繃 softmax锛夛紝鏂逛究閰嶅悎浜ゅ弶鐔电瓑鎹熷け銆?
"""
import torch
import torch.nn as nn

from collections import OrderedDict
from torchvision import models

from .build_vit_backbone import (
    build_vit_sup_models
)
try:
    from .build_vit_backbone import build_swin_model, build_mocov3_model, build_mae_model
except ImportError:
    build_swin_model = None
    build_mocov3_model = None
    build_mae_model = None
from .mlp import MLP
from ..utils import logging
logger = logging.get_logger("visual_prompt")

from ..solver.losses import RSimilarityClassifier
from ..utils.param_logging import log_trainable_parameters


class ViT(nn.Module):
    """
    ViT-related model.

    杩欎釜绫绘槸鈥滅粺涓€鐨?ViT 澶栧３鈥濓細
    - 璐熻矗鏍规嵁 cfg 鏋勫缓涓嶅悓绫诲瀷鐨?ViT 涓诲共锛堢洃鐫?/ 鑷洃鐫?/ 甯?prompt / 甯?adapter 绛夛級锛?
    - 鏍规嵁 TRANSFER_TYPE 鎺у埗鈥滃摢浜涘弬鏁伴渶瑕佽缁冦€佸摢浜涜鍐荤粨鈥濓紱
    - 鍙€?side 鍒嗘敮锛堟梺璺?AlexNet 鐗瑰緛锛夛紱
    - 椤跺眰缁熶竴鎺ヤ竴涓?MLP 澶村仛鍒嗙被銆?

    鐪熸鐨?ViT 缂栫爜鍣ㄦ湰浣撳湪 self.enc 閲岋紝鐢?build_vit_sup_models 鏋勫缓銆?
    """
    def __init__(self, cfg, load_pretrain=True, vis=False):
        """
        鍙傛暟:
          cfg            : 鍏ㄥ眬閰嶇疆瀵硅薄
          load_pretrain  : 鏄惁浠庨璁粌鏉冮噸鍒濆鍖?backbone
          vis            : 鍙鍖?璋冭瘯鐩稿叧寮€鍏筹紙鐢辨瀯寤哄嚱鏁伴€忎紶锛?
        """
        super(ViT, self).__init__()

        # 鍏堢紦瀛?cfg锛屽悗缁瀯寤轰富骞?鍐荤粨绛栫暐鍜屾棩蹇楁墦鍗伴兘浼氫娇鐢?
        self.cfg = cfg

        # 濡傛灉 TRANSFER_TYPE 閲屽寘鍚?"prompt"锛屽垯闇€瑕佷紶鍏?prompt 閰嶇疆锛涘惁鍒欎笉浣跨敤 prompt
        if "prompt" in cfg.MODEL.TRANSFER_TYPE:
            prompt_cfg = cfg.MODEL.PROMPT
        else:
            prompt_cfg = None

        # self.froze_enc 鏍囪鈥滄槸鍚︽妸缂栫爜鍣ㄦ暣浣撶湅浣滄槸鍐荤粨鐨勨€?
        # - 褰?transfer_type 涓嶆槸 end2end 涓斾笉鍚?prompt 鏃讹紙渚嬪 linear銆乧ls銆乤dapter銆乸artial 绛夛級锛?
        #   榛樿璁や负鏄€滃喕涓诲共銆佸彧璁ご閮?灞€閮ㄢ€濓紝杩欓噷缃?True锛?
        # - 褰撴槸 end2end 鎴栧寘鍚?prompt 鐨勬ā寮忎笅锛坧rompt / cls+prompt锛夛紝缃?False锛?
        #   鍚庣画浼氭寜鏇寸粏绮掑害鎺у埗鍝簺鍙傛暟 requires_grad銆?
        if cfg.MODEL.TRANSFER_TYPE != "end2end" and "prompt" not in cfg.MODEL.TRANSFER_TYPE:
            # linear, cls, tiny-tl, parital, adapter
            self.froze_enc = True
        else:
            # prompt, end2end, cls+prompt
            self.froze_enc = False

        # adapter 妯″紡鏃讹紝鎵嶉渶瑕佽鍙?ADAPTER 閰嶇疆锛屽叾浣欐儏鍐?adapter_cfg 缃?None
        if cfg.MODEL.TRANSFER_TYPE == "adapter":
            adapter_cfg = cfg.MODEL.ADAPTER
        else:
            adapter_cfg = None

        # ===== 鏍稿績锛氭瀯寤?ViT 涓诲共锛屽苟鏍规嵁 transfer_type 璁剧疆鍙傛暟鍙缁冩€?=====
        self.build_backbone(
            prompt_cfg, cfg, adapter_cfg, load_pretrain, vis=vis)

        # 鍙€夛細鏋勫缓 side 鍒嗘敮锛堢敤 AlexNet 鍋氭梺璺壒寰侊級
        self.setup_side()

        # 鏈€缁堝垎绫诲ご锛歁LP(杈撳叆缁村害 = feat_dim, 杈撳嚭缁村害 = 绫诲埆鏁?
        self.setup_head(cfg)
        self.r_similarity_head = None
        self.debug_trace_once = bool(getattr(cfg.SOLVER, "DEBUG_TRACE_ONCE", False))
        self._debug_head_route_logged = False


    # --------------------------------------------------------------------- #
    #  side 鍒嗘敮锛氭梺璺壒寰侊紙AlexNet锛夛紝鐢ㄤ簬 "side" 杩佺Щ绫诲瀷
    # --------------------------------------------------------------------- #
    def setup_side(self):
        """
        side 鍒嗘敮锛堟梺璺壒寰侊級鍙湪 TRANSFER_TYPE == "side" 鏃跺惎鐢ㄣ€?

        鍋氭硶锛?
        - 浣跨敤 torchvision 鐨?AlexNet 棰勮缁冪壒寰侀儴鍒嗭紙features + avgpool锛夛紱
        - 灏嗚緭鍑?(B, 9216) 绾挎€ф姇褰卞埌鍜?ViT 涓诲共鐩稿悓鐨?feat_dim锛?
        - 鍦?forward 閲岋紝浣跨敤涓€涓彲璁粌鏍囬噺 alpha锛岄€氳繃 sigmoid(alpha) 鎶婁富骞插拰 side 鐨勭壒寰佸仛鍑哥粍鍚堬細
              x = 蟽(alpha) * vit_feat + (1 - 蟽(alpha)) * side_feat
        """
        if self.cfg.MODEL.TRANSFER_TYPE != "side":
            self.side = None
        else:
            # 鍙涔犵殑铻嶅悎绯绘暟锛屽垵濮嬪寲涓?0锛屽悗缁€氳繃 sigmoid 鍘嬪埌 (0,1)
            self.side_alpha = nn.Parameter(torch.tensor(0.0))

            # AlexNet 浣滀负 side 鍒嗘敮
            m = models.alexnet(pretrained=True)
            self.side = nn.Sequential(OrderedDict([
                ("features", m.features),   # 鍗风Н鐗瑰緛
                ("avgpool", m.avgpool),     # 骞冲潎姹犲寲鍒板浐瀹氱┖闂?
            ]))

            # AlexNet 灞曞钩鐗瑰緛缁村害涓?9216锛堝吀鍨?6*6*256锛夛紝鎶曞奖鍒颁笌 ViT 涓€鑷寸殑 self.feat_dim
            self.side_projection = nn.Linear(9216, self.feat_dim, bias=False)

    # --------------------------------------------------------------------- #
    #  鏋勫缓 ViT 涓诲共骞惰缃喕缁撶瓥鐣?
    # --------------------------------------------------------------------- #
    def build_backbone(self, prompt_cfg, cfg, adapter_cfg, load_pretrain, vis):
        """
        鏋勫缓 ViT 涓诲共锛屽苟渚濇嵁 TRANSFER_TYPE 璁剧疆鍚勫眰鍙傛暟鐨?requires_grad銆?

        姝ラ锛?
          1) 浣跨敤 build_vit_sup_models 鏋勫缓缂栫爜鍣?self.enc 浠ュ強鍏剁壒寰佺淮搴?self.feat_dim锛?
          2) 鏍规嵁 transfer_type 鐨勪笉鍚岋紝鍦?self.enc.named_parameters() 涓婁慨鏀?requires_grad锛?
             閫氳繃瀛楃涓插尮閰嶅弬鏁板悕鐨勬柟寮忥紝绮剧‘鍦拌В鍐?鍐荤粨瀵瑰簲鐨勫眰鎴栨ā鍧椼€?
        """
        transfer_type = cfg.MODEL.TRANSFER_TYPE

        # enc: 鍏蜂綋鐨?ViT 缂栫爜鍣紙VisionTransformer 鎴?PromptedVisionTransformer 绛夛級
        # feat_dim: 缂栫爜鍣ㄦ渶鍚庤緭鍑虹殑鐗瑰緛缁村害锛堜竴鑸槸 hidden_size锛屽 768锛?
        self.enc, self.feat_dim = build_vit_sup_models(
            cfg.DATA.FEATURE,         # 棰勮缁冨悕绉帮紝濡?"imagenet21k_sup_vitb16"
            cfg.DATA.CROPSIZE,        # 杈撳叆瑁佸壀灏哄锛堝 224锛?
            prompt_cfg,               # prompt 瀛愰厤缃紙鍙兘涓?None锛?
            cfg.MODEL.MODEL_ROOT,     # 瀛樻斁棰勮缁冩潈閲嶇殑璺緞
            adapter_cfg,              # adapter 瀛愰厤缃紙鍙兘涓?None锛?
            load_pretrain,            # 鏄惁鍔犺浇棰勮缁冩潈閲?
            vis                       # 鍙鍖?璋冭瘯寮€鍏?
        )
        # ====== 涓嬫柟鍒嗘敮鎺у埗鈥滃弬鏁板彲璁粌鎬р€?======
        # 绾﹀畾锛?
        # - partial-k锛氫粎寰皟鏈€鍚?k 灞?encoder block锛堜互鍙?layernorm锛?
        # - linear/side锛氬畬鍏ㄥ喕缁撶紪鐮佸櫒锛屼粎璁粌椤跺眰绾挎€?铻嶅悎澶?
        # - tinytl-bias锛氬彧璁粌 bias 鍙傛暟
        # - prompt锛氬彧璁粌 prompt 鐩稿叧鍙傛暟锛堝彲閫夛細below 鏃朵篃璁粌 patch embed锛?
        # - prompt+bias锛氳缁?prompt 涓庡叏閮?bias
        # - prompt-noupdate锛歱rompt 涔熶笉璁粌锛堝畬鍏ㄥ喕缁擄級
        # - cls锛氬彧璁粌 cls_token
        # - cls-reinit锛氶噸缃?cls_token 鍚庯紝鍙缁?cls_token
        # - cls+prompt / cls-reinit+prompt锛氬悓鏃惰缁?prompt 涓?cls_token
        # - adapter锛氬彧璁粌 adapter 妯″潡
        # - end2end锛氬叏閲忚缁?
        # linear, prompt, cls, cls+prompt, partial_1

        # ---------- partial-k锛氬彧寰皟鏈€鍚庤嫢骞插眰 block + encoder_norm ----------
        if transfer_type == "partial-1":
            total_layer = len(self.enc.transformer.encoder.layer) # 鎬诲眰鏁?L
            # tuned_params = [
            #     "transformer.encoder.layer.{}".format(i-1) for i in range(total_layer)]
            # 浠呭厑璁糕€滄渶鍚?1 灞?block + encoder_norm鈥濇洿鏂帮紝鍏朵綑鍐荤粨
            for k, p in self.enc.named_parameters():
                if "transformer.encoder.layer.{}".format(total_layer - 1) not in k and "transformer.encoder.encoder_norm" not in k: # noqa
                    p.requires_grad = False

        elif transfer_type == "partial-2":
            total_layer = len(self.enc.transformer.encoder.layer)
            for k, p in self.enc.named_parameters():
                # 鍙繚鐣欏€掓暟绗?銆佺2灞?block + encoder_norm
                if "transformer.encoder.layer.{}".format(total_layer - 1) not in k and "transformer.encoder.layer.{}".format(total_layer - 2) not in k and "transformer.encoder.encoder_norm" not in k: # noqa
                    p.requires_grad = False

        elif transfer_type == "partial-4":
            total_layer = len(self.enc.transformer.encoder.layer)
            for k, p in self.enc.named_parameters():
                # 鍙繚鐣欏€掓暟 4 灞?block + encoder_norm
                if "transformer.encoder.layer.{}".format(total_layer - 1) not in k and "transformer.encoder.layer.{}".format(total_layer - 2) not in k and "transformer.encoder.layer.{}".format(total_layer - 3) not in k and "transformer.encoder.layer.{}".format(total_layer - 4) not in k and "transformer.encoder.encoder_norm" not in k: # noqa
                    p.requires_grad = False

        # ---------- linear / side锛氬畬鍏ㄥ喕缁撲富骞诧紝鍙缁冪嚎鎬уご鎴?side ----------
        elif transfer_type == "linear" or transfer_type == "side":
            # 绾嚎鎬у井璋冩垨 side 铻嶅悎锛屽畬鍏ㄥ喕缁?backbone
            for k, p in self.enc.named_parameters():
                p.requires_grad = False

        # ---------- tinytl-bias锛氬彧璁粌鎵€鏈?bias ----------
        elif transfer_type == "tinytl-bias":
            # TinyTL锛氫粎璁粌 bias锛屾樉钁楅檷浣庡彲璁粌鍙傛暟閲忎笌鏄惧瓨鍗犵敤
            for k, p in self.enc.named_parameters():
                if 'bias' not in k:
                    p.requires_grad = False

        # ---------- prompt锛坆elow锛夛細鍙缁?prompt + patch embedding ----------
        elif transfer_type == "prompt" and prompt_cfg.LOCATION == "below":
            # ViT 鈥渂elow 妯″紡鈥濓細鍦ㄨ緭鍏ラ€氶亾缁存垨 patch embedding 涓婂紩鍏?prompt
            # 杩欓噷鍏佽 prompt 鍜?patch_embeddings锛坵eight, bias锛夋洿鏂帮紝鍏朵綑鍐荤粨
            for k, p in self.enc.named_parameters():
                if "prompt" not in k and "embeddings.patch_embeddings.weight" not in k  and "embeddings.patch_embeddings.bias" not in k:
                    p.requires_grad = False

        # ---------- prompt锛氬彧璁粌 prompt 妯″潡 ----------
        elif transfer_type == "prompt":
            # 浠呰缁?prompt 鐩稿叧鍙傛暟锛堝鍓嶇疆/涓棿鎻掑叆鐨勮櫄鎷?token锛?
            # 鈿狅笍 鏈」鐩繕瀛樺湪璇箟姒傚康妲?璺ㄦā鎬佹敞鎰忕瓑鍙涔犳ā鍧楋紝鍙傛暟鍚嶉€氬父鍖呭惈
            #    "semantic" / "concept" / "semantic_attn"锛岄渶瑕侀殢鎻愮ず涓€璧疯В鍐伙紱
            #    鍚﹀垯瀹冧滑浼氳璇喕浣忥紝姊害鏃犳硶钀藉埌鍏变韩姒傚康鍩烘垨浜插拰璋冨埗涓娿€?
            trainable_keys = ("prompt", "semantic", "concept", "semantic_attn")
            for k, p in self.enc.named_parameters():
                if not any(key in k for key in trainable_keys):
                    p.requires_grad = False

        # ---------- prompt+bias锛氳缁?prompt + 鎵€鏈?bias ----------
        elif transfer_type == "prompt+bias":
            # 璁粌 prompt 涓庢墍鏈?bias
            for k, p in self.enc.named_parameters():
                if "prompt" not in k and 'bias' not in k:
                    p.requires_grad = False

        # ---------- prompt-noupdate锛歱rompt 涔熶笉鏇存柊锛堝叏鍐荤粨锛岀敤浜?ablation锛?----------
        elif transfer_type == "prompt-noupdate":
            # prompt 涔熶笉鏇存柊锛堝叏鍐荤粨锛屽父鐢ㄤ簬 ablation锛?
            for k, p in self.enc.named_parameters():
                p.requires_grad = False

        # ---------- cls锛氬彧璁粌 cls_token ----------
        elif transfer_type == "cls":
            # 浠呰缁?cls_token锛堝叾浠栧叏鍐荤粨锛?
            for k, p in self.enc.named_parameters():
                if "cls_token" not in k:
                    p.requires_grad = False

        # ---------- cls-reinit锛氶噸缃?cls_token 鍐嶅彧璁粌 cls_token ----------
        elif transfer_type == "cls-reinit":
            # 鍏堥噸缃?cls_token锛屽啀浠呰缁?cls_token
            nn.init.normal_(
                self.enc.transformer.embeddings.cls_token,
                std=1e-6
            )
            # 鍐嶅彧璁粌 cls_token
            for k, p in self.enc.named_parameters():
                if "cls_token" not in k:
                    p.requires_grad = False

        # ---------- cls+prompt锛氬悓鏃惰缁?cls_token 鍜?prompt ----------
        elif transfer_type == "cls+prompt":
            # 鍚屾椂璁粌 cls_token 涓?prompt
            for k, p in self.enc.named_parameters():
                if "prompt" not in k and "cls_token" not in k:
                    p.requires_grad = False

        # ---------- cls-reinit+prompt锛氶噸缃?cls_token + 璁粌 cls_token 鍜?prompt ----------
        elif transfer_type == "cls-reinit+prompt":
            # 閲嶇疆 cls_token锛屽苟鍚屾椂璁粌 cls_token 涓?prompt
            nn.init.normal_(
                self.enc.transformer.embeddings.cls_token,
                std=1e-6
            )
            for k, p in self.enc.named_parameters():
                if "prompt" not in k and "cls_token" not in k:
                    p.requires_grad = False
        
        # ---------- adapter锛氬彧璁粌 adapter 妯″潡 ----------
        elif transfer_type == "adapter":
            # 浠呰缁冩敞鍏ュ埌鍚勫眰鐨?adapter 妯″潡锛屽叾浣欏喕缁?
            for k, p in self.enc.named_parameters():
                if "adapter" not in k:
                    p.requires_grad = False

        # ---------- end2end锛氭墍鏈夊弬鏁伴兘鍙缁?----------
        elif transfer_type == "end2end":
            # 绔埌绔叏閲忔洿鏂帮紙涓嶅仛鍐荤粨锛?
            logger.info("Enable all parameters update during training")

        # ---------- 鍏朵粬鏈敮鎸佺殑绫诲瀷 ----------
        else:
            raise ValueError("transfer type {} is not supported".format(
                transfer_type))

        # 鍙€夛細鎵撳嵃鍙缁冨弬鏁扮粺璁★紝渚夸簬纭鍐荤粨绛栫暐鏄惁绗﹀悎棰勬湡
        if self.cfg.MODEL.LOG_TRAINABLE or self.cfg.SOLVER.DBG_TRAINABLE:
            self._log_trainable_parameters()

    def _log_trainable_parameters(self):

        log_trainable_parameters(self, logger, max_examples_per_group=10)

    def log_trainable_parameters(self):

        self._log_trainable_parameters()

    def setup_head(self, cfg):

        self.head = MLP(
            input_dim=self.feat_dim,
            mlp_dims=[self.feat_dim] * self.cfg.MODEL.MLP_NUM + \
                [cfg.DATA.NUMBER_CLASSES], # noqa
            special_bias=True
        )

    def attach_r_similarity_head(self, class_attributes):

        # 1) 鑻ラ厤缃腑娌℃湁寮€鍚?R-similarity 鍔熻兘锛屽垯鐩存帴杩斿洖锛?
        #    淇濇寔浣跨敤榛樿鐨?self.head MLP 鍒嗙被澶达紝涓嶅仛浠讳綍鏀瑰姩銆?
        if not self.cfg.MODEL.R_SIMILARITY.ENABLE:
            return

        # 2) 浠庣紪鐮佸櫒涓彇鍑哄叡浜蹇靛熀妯″潡锛?
        #    - self.enc 閫氬父鏄?VisionTransformer 鎴?PromptedVisionTransformer
        #    - 鍏跺唴閮ㄧ殑 transformer 閲岃嫢鍚敤浜?SharedConceptAligner锛屽垯浼氭寕鍦?semantic_concept 涓?
        #    - getattr(getattr(...)) 鐨勫啓娉曟槸锛氳嫢涓棿浠绘剰涓€灞備笉瀛樺湪锛屽搴旇繑鍥?None 鑰屼笉鏄姤閿?
        concept = getattr(getattr(self.enc, "transformer", None), "semantic_concept", None)
        if concept is None:
            # 鑻ユ湭鎵惧埌 semantic_concept锛岃鏄庝綘娌℃湁鍦?backbone 涓惎鐢ㄥ叡浜蹇靛熀锛?
            # 姝ゆ椂鏋勫缓 R-similarity 澶存病鏈夋剰涔夛紝鐩存帴鎶ラ敊鎻愮ず閰嶇疆涓嶄竴鑷淬€?
            raise ValueError("R-similarity head requires semantic concept aligner to be enabled")

        # 3) 妫€鏌ュ苟瑙勮寖绫诲睘鎬х煩闃碉細
        #    - 璁粌 R-similarity 澶村繀椤绘湁绫荤骇璇箟灞炴€э紝鍚﹀垯鏃犳硶鏋勯€犺涔夊師鍨?
        if class_attributes is None:
            raise ValueError("class_attributes must be provided when R-similarity is enabled")
        #    - 鑻ヤ紶鍏ョ殑鏄?numpy锛屽垯杞垚 torch.Tensor锛涜嫢鏈韩鏄?Tensor锛屽垯鐩存帴澶嶇敤
        if not isinstance(class_attributes, torch.Tensor):
            class_attributes = torch.from_numpy(class_attributes)
        #    - 缁熶竴涓?float 绫诲瀷锛岄伩鍏嶅悗缁弬涓庤绠楁椂鍑虹幇 dtype 鍐茬獊
        class_attributes = class_attributes.float()

        # 4) 灏嗚涔夊睘鎬у紶閲忔尓鍒颁笌妯″瀷鐩稿悓鐨?device 涓婏紙GPU/CPU 涓€鑷达級锛?
        #    - next(self.parameters()) 鍙栨ā鍨嬩腑浠绘剰涓€涓弬鏁帮紝鑾峰彇褰撳墠鎵€鍦?device
        device = next(self.parameters()).device
        class_attributes = class_attributes.to(device)

        # 5) 浠庨厤缃腑璇诲嚭鎶曞奖缁村害绛夎秴鍙傛暟锛屽苟鐪熸瀹炰緥鍖?RSimilarityClassifier锛?
        #    - proj_dim: 璇箟/瑙嗚鍦?R 绌洪棿涓殑瀵归綈缁村害锛堣嫢涓?None/<=0锛屽垯鍐呴儴浼氶€€鍖栦负 hidden_size锛?
        proj_dim = self.cfg.MODEL.R_SIMILARITY.PROJ_DIM
        self.r_similarity_head = RSimilarityClassifier(
            concept,                                            # 鍏变韩姒傚康鍩烘ā鍧楋紙鐢ㄤ簬鎶婅涔夋槧灏勫埌 R 绌洪棿锛?
            class_attributes,                                   # 绫荤骇璇箟灞炴€х煩闃?[C, d_s]
            hidden_size=self.feat_dim,                          # ViT 杈撳嚭鐨勭壒寰佺淮搴︼紙閫氬父绛変簬 hidden_size锛?
            proj_dim=proj_dim,                                  # R 绌洪棿涓娇鐢ㄧ殑缁村害
            use_cosine=self.cfg.MODEL.R_SIMILARITY.USE_COSINE,  # 鏄惁鐢ㄤ綑寮︾浉浼煎害璁＄畻 logits
            logit_scale_init=self.cfg.MODEL.R_SIMILARITY.LOGIT_SCALE_INIT,  # 鍒濆娓╁害/缂╂斁鍥犲瓙
            visual_proj_enable=self.cfg.MODEL.R_SIMILARITY.VISUAL_PROJ_ENABLE,  # 鏄惁瀵硅瑙変晶鍐嶅仛涓€灞傛姇褰?
            fixed_logit_scale=getattr(self.cfg.MODEL.R_SIMILARITY, "FIXED_LOGIT_SCALE", 0.0),
            shuffle_prototypes=bool(getattr(getattr(self.cfg.SOLVER, "DIAG", None), "SHUFFLE_PROTOTYPES", False)),
            semantic_score_source=str(getattr(self.cfg.MODEL, "SEMANTIC_SCORE_SOURCE", "auto")),
            semantic_score_mode=str(getattr(self.cfg.MODEL, "SEMANTIC_SCORE_MODE", "global_refined")),
            semantic_score_topk=int(getattr(self.cfg.MODEL, "SEMANTIC_SCORE_TOPK", 5)),
            semantic_score_alpha=float(getattr(self.cfg.MODEL, "SEMANTIC_SCORE_ALPHA", 1.0)),
            debug_trace_once=bool(getattr(self.cfg.SOLVER, "DEBUG_TRACE_ONCE", False)),
        ).to(device)                    # 鎶婃暣涓垎绫诲ご绉诲姩鍒颁笌涓绘ā鍨嬬浉鍚岀殑 device 涓婏紝淇濊瘉鍓嶅悜/鍙嶅悜閮藉湪鍚屼竴璁惧鎵ц


    # --------------------------------------------------------------------- #
    #  鍒嗙被澶达細缁熶竴鐢ㄤ竴涓?MLP
    # --------------------------------------------------------------------- #
    def forward(self, x, return_feature=False, semantics=None):

        # 1) side 鍒嗘敮锛堝彲閫夛級
        if self.side is not None:
            side_output = self.side(x)
            side_output = side_output.view(side_output.size(0), -1)
            side_output = self.side_projection(side_output)

        # 2) 鑻?enc 琚爣璁颁负鈥滄暣浣撳喕缁撯€濓紝涓斿綋鍓嶇‘瀹炲湪璁粌闃舵锛屽垯鎶?enc 鍒囧埌 eval()
        #    鈥斺€?杩欐牱鍏朵腑鐨?Dropout/LayerNorm/BatchNorm 琛屼负鍥哄畾涓嬫潵锛屾洿鎺ヨ繎鈥滃浐瀹氱壒寰佲€濈殑璇箟銆?
        if self.froze_enc and self.enc.training:
            self.enc.eval()

        # 3) 涓诲共缂栫爜鍣ㄨ幏鍙栧叏灞€鐗瑰緛锛堥€氬父鏄?CLS 鎴?GAP 鍚庣殑 embedding锛?
        x = self.enc(x, semantics=semantics)  # batch_size x self.feat_dim
        feature_norm_mean = None
        if torch.is_tensor(x) and x.dim() == 2:
            with torch.no_grad():
                feature_norm_mean = float(x.float().norm(dim=-1).mean().item())

        # 4) 鑻ユ湁 side 鍒嗘敮锛岀敤涓€涓爣閲?alpha锛堢粡 sigmoid 鍚庯級铻嶅悎涓诲共涓?side 鐗瑰緛
        if self.side is not None:
            alpha_squashed = torch.sigmoid(self.side_alpha) # 鏍囬噺 鈭?(0,1)
            x = alpha_squashed * x + (1 - alpha_squashed) * side_output

        # 5) 鑻ュ彧闇€瑕佺壒寰侊紝涓嶉渶瑕佸垎绫?logits锛屽垯鐩存帴杩斿洖鐗瑰緛锛堢浜屼釜杩斿洖鍊间负鍚屾牱鐨?x锛屽吋瀹规棫鎺ュ彛锛?
        if return_feature:
            return x, x

        # 6) 閫氳繃 MLP 鎴?R-similarity 澶磋緭鍑?logits
        if self.r_similarity_head is not None:
            x = self.r_similarity_head(x)
            logits_source = "r_similarity_head"
        else:
            x = self.head(x)
            logits_source = "head"

        if self.debug_trace_once and not self._debug_head_route_logged:
            trace_id = getattr(self, "_debug_trace_id", "trace=NA")
            prompt_noop_info = None
            transformer = getattr(self.enc, "transformer", None)
            if transformer is not None:
                prompt_noop_info = getattr(transformer, "_last_prompt_noop_info", None)
            logger.info(
                "[trace] %s node=C.vit_models.forward use_r_similarity_head=%s logits_source=%s logits_shape=%s "
                "final_cls_or_pooled_feature_norm=%s prompt_noop_info=%s",
                trace_id,
                bool(self.r_similarity_head is not None),
                logits_source,
                tuple(x.shape) if torch.is_tensor(x) else None,
                feature_norm_mean,
                prompt_noop_info if isinstance(prompt_noop_info, dict) else None,
            )
            self._debug_head_route_logged = True

        return x
    
    def forward_cls_layerwise(self, x, semantics=None):

        cls_embeds = self.enc.forward_cls_layerwise(x)
        return cls_embeds

    def get_features(self, x):
        """
        get a (batch_size, self.feat_dim) feature
        浠呮彁鍙?(batch_size, feat_dim) 鐨勫叏灞€鐗瑰緛锛屼笉杩囧垎绫诲ご銆?
        """
        x = self.enc(x)  # batch_size x self.feat_dim
        return x

    def forward_with_affinity(self, x, affinity_config, semantics=None, vis=False):
        """
        甯︿翰鍜岃緭鍑虹殑鍓嶅悜鎺ュ彛锛氫笌 enc/backbone 鐨?forward_with_affinity 骞宠銆?

        杩斿洖:
          - vis=False: logits, affinities
          - vis=True:  logits, attn_weights, affinities
        """
        if not hasattr(self.enc, "forward_with_affinity"):
            raise AttributeError("backbone does not implement forward_with_affinity")

        if vis:
            feats, attn_weights, affinities = self.enc.forward_with_affinity(
                x, affinity_config, semantics=semantics, vis=vis
            )
        else:
            feats, affinities = self.enc.forward_with_affinity(
                x, affinity_config, semantics=semantics
            )
            attn_weights = None

        # 涓?forward 瀵归綈锛歟nc 杈撳嚭鍙兘鏄?[B, 1+N, D] 鎴?[B, D]锛屽彇 CLS 鍚庢帴澶撮儴
        feats = feats[:, 0] if feats.dim() == 3 else feats
        logits = self.r_similarity_head(feats) if self.r_similarity_head is not None else self.head(feats)

        if not vis:
            return logits, affinities
        return logits, attn_weights, affinities


# ===================================================================== #
#  Swin Transformer：继承 ViT 外壳，只重写 build_backbone（构建 Swin + 冻结规则）
# ===================================================================== #
class Swin(ViT):
    """Swin-related model.
    Swin Transformer 版本，复用 ViT 的整体框架，仅重写 build_backbone 与冻结逻辑。"""

    def __init__(self, cfg):
        super(Swin, self).__init__(cfg)

    def build_backbone(self, prompt_cfg, cfg, adapter_cfg, load_pretrain, vis):
        """
        构建 Swin 主干，并根据 TRANSFER_TYPE 选择性冻结。
        Swin 的模块命名与 ViT 不同，这里针对其层级结构（layers/blocks）做了适配。
        """
        if build_swin_model is None:
            raise ImportError("build_swin_model is unavailable in this repository build.")
        transfer_type = cfg.MODEL.TRANSFER_TYPE
        self.enc, self.feat_dim = build_swin_model(
            cfg.DATA.FEATURE, cfg.DATA.CROPSIZE,
            prompt_cfg, cfg.MODEL.MODEL_ROOT
        )

        # linear, prompt, cls, cls+prompt, partial_1
        # ====== Swin 的冻结策略分支 ======
        if transfer_type == "partial-1":
            # 仅训练最后一层的最后一个 block，以及最终的 norm
            total_layer = len(self.enc.layers)
            total_blocks = len(self.enc.layers[-1].blocks)
            for k, p in self.enc.named_parameters():
                if "layers.{}.blocks.{}".format(total_layer - 1, total_blocks - 1) not in k and "norm.weight" != k and "norm.bias" != k: # noqa
                    p.requires_grad = False

        elif transfer_type == "partial-2":
            # 仅训练最后一层（整个 stage）和最终 norm
            total_layer = len(self.enc.layers)
            for k, p in self.enc.named_parameters():
                if "layers.{}".format(total_layer - 1) not in k and "norm.weight" != k and "norm.bias" != k: # noqa
                    p.requires_grad = False

        elif transfer_type == "partial-4":
            # 训练最后一层 + 倒数第二层的后若干模块（包括 downsample）+ 最终 norm
            total_layer = len(self.enc.layers)
            total_blocks = len(self.enc.layers[-2].blocks)

            for k, p in self.enc.named_parameters():
                if "layers.{}".format(total_layer - 1) not in k and "layers.{}.blocks.{}".format(total_layer - 2, total_blocks - 1) not in k and "layers.{}.blocks.{}".format(total_layer - 2, total_blocks - 2) not in k and "layers.{}.downsample".format(total_layer - 2) not in k and "norm.weight" != k and "norm.bias" != k: # noqa
                    p.requires_grad = False

        elif transfer_type == "linear" or transfer_type == "side":
            for k, p in self.enc.named_parameters():
                p.requires_grad = False

        elif transfer_type == "tinytl-bias":
            for k, p in self.enc.named_parameters():
                if 'bias' not in k:
                    p.requires_grad = False

        elif transfer_type == "prompt" and prompt_cfg.LOCATION in ["below"]:
            # Swin 下，below 场景放开 patch_embed；其余层仅训练 prompt
            for k, p in self.enc.named_parameters():
                if "prompt" not in k and "patch_embed" not in k:
                    p.requires_grad = False

        elif transfer_type == "prompt":
            for k, p in self.enc.named_parameters():
                if "prompt" not in k:
                    p.requires_grad = False

        elif transfer_type == "prompt+bias":
            for k, p in self.enc.named_parameters():
                if "prompt" not in k and 'bias' not in k:
                    p.requires_grad = False

        elif transfer_type == "end2end":
            logger.info("Enable all parameters update during training")

        else:
            raise ValueError("transfer type {} is not supported".format(
                transfer_type))


# ===================================================================== #
#  自监督 ViT：MoCo v3 / MAE 封装（同样继承 ViT）
# ===================================================================== #
class SSLViT(ViT):
    """moco-v3 and mae model.
     自监督预训练（MoCo v3 / MAE）版本的 ViT 封装，构建函数会选用相应的 build_fn。"""

    def __init__(self, cfg):
        super(SSLViT, self).__init__(cfg)

    def build_backbone(self, prompt_cfg, cfg, adapter_cfg, load_pretrain, vis):
        """
        根据 cfg.DATA.FEATURE 选择构建 MoCo v3 或 MAE 的 ViT 主干，
        随后按照 TRANSFER_TYPE 冻结/解冻参数。
        """
        if "moco" in cfg.DATA.FEATURE:
            if build_mocov3_model is None:
                raise ImportError("build_mocov3_model is unavailable in this repository build.")
            build_fn = build_mocov3_model
        elif "mae" in cfg.DATA.FEATURE:
            if build_mae_model is None:
                raise ImportError("build_mae_model is unavailable in this repository build.")
            build_fn = build_mae_model

        self.enc, self.feat_dim = build_fn(
            cfg.DATA.FEATURE, cfg.DATA.CROPSIZE,
            prompt_cfg, cfg.MODEL.MODEL_ROOT, adapter_cfg=adapter_cfg
        )

        transfer_type = cfg.MODEL.TRANSFER_TYPE
        # linear, prompt, cls, cls+prompt, partial_1
        # ====== 自监督 ViT 的冻结策略分支 ======
        if transfer_type == "partial-1":
            total_layer = len(self.enc.blocks)
            for k, p in self.enc.named_parameters():
                if "blocks.{}".format(total_layer - 1) not in k and "fc_norm" not in k and k != "norm": # noqa
                    p.requires_grad = False
        elif transfer_type == "partial-2":
            total_layer = len(self.enc.blocks)
            for k, p in self.enc.named_parameters():
                if "blocks.{}".format(total_layer - 1) not in k and "blocks.{}".format(total_layer - 2) not in k and "fc_norm" not in k and k != "norm": # noqa
                    p.requires_grad = False

        elif transfer_type == "partial-4":
            total_layer = len(self.enc.blocks)
            for k, p in self.enc.named_parameters():
                if "blocks.{}".format(total_layer - 1) not in k and "blocks.{}".format(total_layer - 2) not in k and "blocks.{}".format(total_layer - 3) not in k and "blocks.{}".format(total_layer - 4) not in k and "fc_norm" not in k and k != "norm": # noqa
                    p.requires_grad = False

        elif transfer_type == "linear" or transfer_type == "sidetune":
            # 这里的 "sidetune" 对应其他项目里的一类旁路微调策略；本实现中与 linear 等价为冻结主干
            for k, p in self.enc.named_parameters():
                p.requires_grad = False

        elif transfer_type == "tinytl-bias":
            for k, p in self.enc.named_parameters():
                if 'bias' not in k:
                    p.requires_grad = False

        elif transfer_type == "prompt+bias":
            for k, p in self.enc.named_parameters():
                if "prompt" not in k and 'bias' not in k:
                    p.requires_grad = False

        elif transfer_type == "prompt" and prompt_cfg.LOCATION == "below":
            # below：同前，放开 patch_embed 的 conv（proj）权重与偏置
            for k, p in self.enc.named_parameters():
                if "prompt" not in k and "patch_embed.proj.weight" not in k  and "patch_embed.proj.bias" not in k:
                    p.requires_grad = False

        elif transfer_type == "prompt":
            for k, p in self.enc.named_parameters():
                if "prompt" not in k:
                    p.requires_grad = False

        elif transfer_type == "end2end":
            logger.info("Enable all parameters update during training")
        
        # adapter
        # adapter：仅训练 adapter 模块
        elif transfer_type == "adapter":
            for k, p in self.enc.named_parameters():
                if "adapter" not in k:
                    p.requires_grad = False

        else:
            raise ValueError("transfer type {} is not supported".format(
                transfer_type))


