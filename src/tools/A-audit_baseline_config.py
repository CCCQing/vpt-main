#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path
from typing import List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.configs.config import get_cfg
from src.utils.reproducibility import seed_streams


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")
    parser.add_argument("--build-model", action="store_true")
    args, opts = parser.parse_known_args()
    args.opts = opts
    return args


def load_cfg(config_file: str, opts: List[str]):
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    local_path = REPO_ROOT / "src" / "configs" / "local_path.yaml"
    if local_path.is_file():
        cfg.merge_from_file(str(local_path))
    if opts:
        cfg.merge_from_list(opts)
    return cfg


def resolved_stage(cfg) -> str:
    prompt = cfg.MODEL.PROMPT
    if not bool(prompt.ENABLE):
        return "B0"
    if str(prompt.BACKEND).lower() == "dynamic" and not bool(prompt.DEEP):
        return "B1"
    if str(prompt.BACKEND).lower() == "vpt_deep" and bool(prompt.DEEP):
        return "B2"
    return "invalid"


def static_checks(cfg) -> Tuple[str, List[str]]:
    stage = resolved_stage(cfg)
    failures = []

    expected = {
        "MODEL.CLASSIFIER": (str(cfg.MODEL.CLASSIFIER).lower(), "vspcn_baseline"),
        "SOLVER.MAIN_LOSS": (str(cfg.SOLVER.MAIN_LOSS).lower(), "vspcn"),
        "MODEL.PROMPT.INIT_SOURCE": (str(cfg.MODEL.PROMPT.INIT_SOURCE).lower(), "learned"),
        "MODEL.PROMPT.DISTRIBUTOR.ENABLE": (bool(cfg.MODEL.PROMPT.DISTRIBUTOR.ENABLE), False),
        "MODEL.PROMPT.DISTRIBUTOR.FACTORIZED_ENABLE": (bool(cfg.MODEL.PROMPT.DISTRIBUTOR.FACTORIZED_ENABLE), False),
        "MODEL.SEMANTIC_TOKENS.ENABLE": (bool(cfg.MODEL.SEMANTIC_TOKENS.ENABLE), False),
        "MODEL.AFFINITY.ENABLE": (bool(cfg.MODEL.AFFINITY.ENABLE), False),
        "MODEL.ATTENTION_MEDIATION.ENABLE": (bool(cfg.MODEL.ATTENTION_MEDIATION.ENABLE), False),
        "MODEL.CONSISTENCY.ENABLE": (bool(cfg.MODEL.CONSISTENCY.ENABLE), False),
        "MODEL.GRAPH_PROB_PRIOR.ENABLE": (bool(cfg.MODEL.GRAPH_PROB_PRIOR.ENABLE), False),
        "MODEL.GRAPH_PROB_PRIOR.LOSS_WEIGHT": (float(cfg.MODEL.GRAPH_PROB_PRIOR.LOSS_WEIGHT), 0.0),
        "SOLVER.LOSS_VSPCN_AR_WEIGHT": (float(cfg.SOLVER.LOSS_VSPCN_AR_WEIGHT), 0.0),
        "SOLVER.LOSS_CM_WEIGHT": (float(cfg.SOLVER.LOSS_CM_WEIGHT), 0.0),
        "SOLVER.LOSS_SEM_MED_WEIGHT": (float(cfg.SOLVER.LOSS_SEM_MED_WEIGHT), 0.0),
        "SOLVER.LOSS_SPV_WEIGHT": (float(cfg.SOLVER.LOSS_SPV_WEIGHT), 0.0),
        "SOLVER.LOSS_ATTR_WEIGHT": (float(cfg.SOLVER.LOSS_ATTR_WEIGHT), 0.0),
        "SOLVER.LOSS_PROMPT_KL_WEIGHT": (float(cfg.SOLVER.LOSS_PROMPT_KL_WEIGHT), 0.0),
        "SOLVER.LOSS_AGR_RES_WEIGHT": (float(cfg.SOLVER.LOSS_AGR_RES_WEIGHT), 0.0),
        "SOLVER.LOSS_CONS_WEIGHT": (float(cfg.SOLVER.LOSS_CONS_WEIGHT), 0.0),
        "SOLVER.VIS.ENABLE": (bool(cfg.SOLVER.VIS.ENABLE), False),
        "SOLVER.SAVE_TRAINABLE_FINAL_CHECKPOINT": (bool(cfg.SOLVER.SAVE_TRAINABLE_FINAL_CHECKPOINT), True),
        "MONITOR.ENABLE": (bool(cfg.MONITOR.ENABLE), True),
        "MONITOR.AFFINITY.ENABLE": (bool(cfg.MONITOR.AFFINITY.ENABLE), False),
        "MONITOR.NUMERICAL_GUARD.ENABLE": (bool(cfg.MONITOR.NUMERICAL_GUARD.ENABLE), True),
        "MONITOR.OPTIMIZER_SANITY.ENABLE": (bool(cfg.MONITOR.OPTIMIZER_SANITY.ENABLE), True),
        "MONITOR.PREDICTION_HEALTH.ENABLE": (bool(cfg.MONITOR.PREDICTION_HEALTH.ENABLE), True),
        "MONITOR.CLASS_ERROR.ENABLE": (bool(cfg.MONITOR.CLASS_ERROR.ENABLE), True),
        "MONITOR.CALIBRATION.ENABLE": (bool(cfg.MONITOR.CALIBRATION.ENABLE), True),
        "MONITOR.PROBE.ENABLE": (bool(cfg.MONITOR.PROBE.ENABLE), True),
        "MONITOR.MODULE_EFFECT.ENABLE": (bool(cfg.MONITOR.MODULE_EFFECT.ENABLE), True),
    }

    for name, (actual, target) in expected.items():
        if actual != target:
            failures.append(f"{name}: expected {target!r}, got {actual!r}")

    if stage == "invalid":
        failures.append("prompt configuration is not one of B0/B1/B2")
    if int(cfg.MODEL.PROMPT.NUM_TOKENS) != 5:
        failures.append("MODEL.PROMPT.NUM_TOKENS must be 5 for the initial A-series protocol")
    if str(cfg.DATA.XLSA.PROTOCOL_MODE).lower() != "final_gzsl":
        failures.append("DATA.XLSA.PROTOCOL_MODE must be 'final_gzsl' for the A-series")
    if not bool(cfg.MODEL.R_SIMILARITY.ENABLE):
        failures.append("MODEL.R_SIMILARITY.ENABLE must remain true until the current training entry is refactored")
    if cfg.SEED is None:
        failures.append("SEED must be set for the A-series reproducibility protocol")
    return stage, failures


def constructed_model_checks(cfg, stage: str) -> Tuple[List[str], List[str]]:
    import torch

    from src.models.build_model import build_model

    model, device = build_model(cfg)
    attributes = torch.zeros((int(cfg.DATA.NUMBER_CLASSES), 312), device=device)
    model.attach_r_similarity_head(attributes)
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    failures = []

    if not any(name.startswith("r_similarity_head.prototype_proj") for name in trainable):
        failures.append("VSPCN prototype projection is not trainable")
    if any("semantic_token" in name for name in trainable):
        failures.append("semantic-token parameters remain trainable")
    if any("attention_mediation" in name for name in trainable):
        failures.append("attention-mediation parameters remain trainable")
    if any("prompt_init_provider" in name for name in trainable):
        failures.append("prompt-distributor parameters remain trainable")

    prompt_names = [name for name in trainable if "prompt_embeddings" in name]
    deep_names = [name for name in trainable if "deep_prompt_embeddings" in name]
    if stage == "B0" and prompt_names:
        failures.append("B0 unexpectedly has trainable prompt parameters")
    if stage == "B1" and (not prompt_names or deep_names):
        failures.append("B1 must train input prompt embeddings only")
    if stage == "B2" and (not prompt_names or not deep_names):
        failures.append("B2 must train both input and deep prompt embeddings")
    allowed_prefixes = {
        "B0": ("r_similarity_head.prototype_proj.",),
        "B1": ("enc.transformer.prompt_embeddings", "r_similarity_head.prototype_proj."),
        "B2": (
            "enc.transformer.prompt_embeddings",
            "enc.transformer.deep_prompt_embeddings",
            "r_similarity_head.prototype_proj.",
        ),
    }
    unexpected = [
        name for name in trainable
        if not any(name.startswith(prefix) for prefix in allowed_prefixes[stage])
    ]
    if unexpected:
        failures.append("unexpected trainable parameters: {}".format(", ".join(unexpected)))
    return failures, trainable


def main():
    args = parse_args()
    cfg = load_cfg(args.config_file, args.opts)
    stage, failures = static_checks(cfg)
    trainable = []
    if not failures and args.build_model:
        model_failures, trainable = constructed_model_checks(cfg, stage)
        failures.extend(model_failures)

    print(f"baseline stage: {stage}")
    print("semantic matcher: VSPCN attribute projection plus CLS-prototype dot product")
    print("main loss: CE only; VSPCN AR weight is zero")
    streams = seed_streams(cfg.SEED)
    print(
        "random streams: master={} classifier_init={} prompt_init={} data_order={}".format(
            cfg.SEED,
            streams["classifier_init"],
            streams["prompt_init"],
            streams["data_order"],
        )
    )
    print("compatibility exception: MODEL.R_SIMILARITY.ENABLE=true is required by train.py, but CLASSIFIER=vspcn_baseline selects VSPCNBaselineClassifier")
    print(f"constructed-model check: {'run' if args.build_model else 'not run'}")
    if trainable:
        print("trainable parameters: {}".format(", ".join(trainable)))
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)
    print("PASS: configuration satisfies the A-series CE baseline gate")


if __name__ == "__main__":
    main()
