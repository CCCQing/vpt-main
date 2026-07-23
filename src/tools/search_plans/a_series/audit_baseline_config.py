#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path
from typing import List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.configs.config import get_cfg
from src.utils.reproducibility import seed_streams


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")
    parser.add_argument("--build-model", action="store_true")
    parser.add_argument(
        "--compare-config",
        action="append",
        default=[],
        metavar="CONFIG",
        help="Additional A-series config to compare for shared initialization and data-order streams.",
    )
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
        return "A0"
    if str(prompt.BACKEND).lower() == "dynamic" and not bool(prompt.DEEP):
        return "A1"
    if str(prompt.BACKEND).lower() == "vpt_deep" and bool(prompt.DEEP):
        return "A2"
    return "invalid"


def static_checks(cfg) -> Tuple[str, List[str]]:
    stage = resolved_stage(cfg)
    failures = []

    expected = {
        "MODEL.CLASSIFIER": (str(cfg.MODEL.CLASSIFIER).lower(), "r_similarity"),
        "MODEL.R_SIMILARITY.SCORE_MODE": (str(cfg.MODEL.R_SIMILARITY.SCORE_MODE).lower(), "dot"),
        "MODEL.R_SIMILARITY.LEARNABLE_SCALE": (bool(cfg.MODEL.R_SIMILARITY.LEARNABLE_SCALE), False),
        "SOLVER.RSIM.ALIGN_MODE": (str(cfg.SOLVER.RSIM.ALIGN_MODE).lower(), "none"),
        "SOLVER.RSIM.ALIGN_WEIGHT": (float(cfg.SOLVER.RSIM.ALIGN_WEIGHT), 0.0),
        "MODEL.PROMPT.INIT_SOURCE": (str(cfg.MODEL.PROMPT.INIT_SOURCE).lower(), "learned"),
        "MODEL.PROMPT.DISTRIBUTOR.ENABLE": (bool(cfg.MODEL.PROMPT.DISTRIBUTOR.ENABLE), False),
        "MODEL.SEMANTIC_TOKENS.ENABLE": (bool(cfg.MODEL.SEMANTIC_TOKENS.ENABLE), False),
        "MODEL.AFFINITY.ENABLE": (bool(cfg.MODEL.AFFINITY.ENABLE), False),
        "MODEL.ATTENTION_MEDIATION.ENABLE": (bool(cfg.MODEL.ATTENTION_MEDIATION.ENABLE), False),
        "MODEL.GRAPH_PROB_PRIOR.ENABLE": (bool(cfg.MODEL.GRAPH_PROB_PRIOR.ENABLE), False),
        "MODEL.GRAPH_PROB_PRIOR.LOSS_WEIGHT": (float(cfg.MODEL.GRAPH_PROB_PRIOR.LOSS_WEIGHT), 0.0),
        "SOLVER.LOSS_SEM_MED_WEIGHT": (float(cfg.SOLVER.LOSS_SEM_MED_WEIGHT), 0.0),
        "SOLVER.LOSS_SPV_WEIGHT": (float(cfg.SOLVER.LOSS_SPV_WEIGHT), 0.0),
        "SOLVER.LOSS_ATTR_WEIGHT": (float(cfg.SOLVER.LOSS_ATTR_WEIGHT), 0.0),
        "SOLVER.LOSS_PROMPT_KL_WEIGHT": (float(cfg.SOLVER.LOSS_PROMPT_KL_WEIGHT), 0.0),
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
        failures.append("prompt configuration is not one of A0/A1/A2")
    if int(cfg.MODEL.PROMPT.NUM_TOKENS) != 5:
        failures.append("MODEL.PROMPT.NUM_TOKENS must be 5 for the initial A-series protocol")
    if str(cfg.DATA.XLSA.PROTOCOL_MODE).lower() != "final_gzsl":
        failures.append("DATA.XLSA.PROTOCOL_MODE must be 'final_gzsl' for the A-series")
    if cfg.SEED is None:
        failures.append("SEED must be set for the A-series reproducibility protocol")
    if int(cfg.NUM_GPUS) != 1:
        failures.append("NUM_GPUS must be 1 for the current A-series single-GPU contract")
    if int(cfg.NUM_SHARDS) != 1:
        failures.append("NUM_SHARDS must be 1 for the current A-series single-GPU contract")
    if int(cfg.DATA.NUM_WORKERS) != 4:
        failures.append("DATA.NUM_WORKERS must be 4 for the current A-series deterministic multi-worker loader")
    if bool(cfg.CUDNN_BENCHMARK):
        failures.append("CUDNN_BENCHMARK must be false for the current A-series single-GPU contract")
    streams = seed_streams(cfg.SEED)
    if any(streams[name] is None for name in ("classifier_init", "prompt_init", "data_order")):
        failures.append("classifier_init, prompt_init, and data_order streams must all be derived")
    elif len(set(streams.values())) != len(streams):
        failures.append("classifier_init, prompt_init, and data_order streams must be distinct")
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
        failures.append("R-similarity prototype projection is not trainable")
    if any("semantic_token" in name for name in trainable):
        failures.append("semantic-token parameters remain trainable")
    if any("attention_mediation" in name for name in trainable):
        failures.append("attention-mediation parameters remain trainable")
    if any("prompt_init_provider" in name for name in trainable):
        failures.append("prompt-distributor parameters remain trainable")

    prompt_names = [name for name in trainable if "prompt_embeddings" in name]
    deep_names = [name for name in trainable if "deep_prompt_embeddings" in name]
    if stage == "A0" and prompt_names:
        failures.append("A0 unexpectedly has trainable prompt parameters")
    if stage == "A1" and (not prompt_names or deep_names):
        failures.append("A1 must train input prompt embeddings only")
    if stage == "A2" and (not prompt_names or not deep_names):
        failures.append("A2 must train both input and deep prompt embeddings")
    allowed_prefixes = {
        "A0": ("r_similarity_head.prototype_proj.",),
        "A1": ("enc.transformer.prompt_embeddings", "r_similarity_head.prototype_proj."),
        "A2": (
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


def cross_config_stream_checks(configs) -> List[str]:
    """Check the A-series random streams that must stay aligned across stages."""
    failures = []
    records = [
        {
            "path": path,
            "stage": resolved_stage(cfg),
            "streams": seed_streams(cfg.SEED),
        }
        for path, cfg in configs
    ]
    for stream_name in ("classifier_init", "data_order"):
        values = {record["streams"][stream_name] for record in records}
        if len(values) != 1:
            failures.append(
                "{} must match across compared A-series configs: {}".format(
                    stream_name,
                    ", ".join(
                        "{}={}".format(Path(record["path"]).name, record["streams"][stream_name])
                        for record in records
                    ),
                )
            )
    prompt_records = [record for record in records if record["stage"] in {"A1", "A2"}]
    if len(prompt_records) >= 2:
        prompt_values = {record["streams"]["prompt_init"] for record in prompt_records}
        if len(prompt_values) != 1:
            failures.append(
                "prompt_init must match across compared A1/A2 configs: {}".format(
                    ", ".join(
                        "{}={}".format(Path(record["path"]).name, record["streams"]["prompt_init"])
                        for record in prompt_records
                    ),
                )
            )
    return failures


def main():
    args = parse_args()
    cfg = load_cfg(args.config_file, args.opts)
    stage, failures = static_checks(cfg)
    compared_configs = [(args.config_file, cfg)]
    for config_file in args.compare_config:
        compared_cfg = load_cfg(config_file, [])
        compared_stage, compared_failures = static_checks(compared_cfg)
        failures.extend(
            "{}: {}".format(Path(config_file).name, failure)
            for failure in compared_failures
        )
        compared_configs.append((config_file, compared_cfg))
        print("compared baseline stage: {} ({})".format(compared_stage, config_file))
    if len(compared_configs) > 1:
        failures.extend(cross_config_stream_checks(compared_configs))
    trainable = []
    if not failures and args.build_model:
        model_failures, trainable = constructed_model_checks(cfg, stage)
        failures.extend(model_failures)

    print(f"baseline stage: {stage}")
    print("semantic matcher: R-similarity attribute projection plus CLS-prototype dot product")
    print("classification loss: CE only; RSIM alignment is disabled")
    streams = seed_streams(cfg.SEED)
    print(
        "random streams: master={} classifier_init={} prompt_init={} data_order={}".format(
            cfg.SEED,
            streams["classifier_init"],
            streams["prompt_init"],
            streams["data_order"],
        )
    )
    print("classifier route: r_similarity with SCORE_MODE=dot")
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
