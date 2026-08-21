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
        help="Additional A/B-series config to compare for shared initialization and data-order streams.",
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
        return (
            "B"
            if bool(prompt.DISTRIBUTOR.DEEP_RESIDUAL.ENABLE)
            else "A2"
        )
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
        "MODEL.PROMPT.DISTRIBUTOR.ENABLE": (
            bool(cfg.MODEL.PROMPT.DISTRIBUTOR.ENABLE),
            stage == "B",
        ),
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
        "SOLVER.BASE_LR": (float(cfg.SOLVER.BASE_LR), 0.0006),
        "SOLVER.TOTAL_EPOCH": (int(cfg.SOLVER.TOTAL_EPOCH), 15),
        "MONITOR.ENABLE": (bool(cfg.MONITOR.ENABLE), True),
        "MONITOR.AFFINITY.ENABLE": (bool(cfg.MONITOR.AFFINITY.ENABLE), False),
        "MONITOR.NUMERICAL_GUARD.ENABLE": (bool(cfg.MONITOR.NUMERICAL_GUARD.ENABLE), True),
        "MONITOR.OPTIMIZER_SANITY.ENABLE": (bool(cfg.MONITOR.OPTIMIZER_SANITY.ENABLE), True),
        "MONITOR.PREDICTION_HEALTH.ENABLE": (bool(cfg.MONITOR.PREDICTION_HEALTH.ENABLE), True),
        "MONITOR.CLASS_ERROR.ENABLE": (bool(cfg.MONITOR.CLASS_ERROR.ENABLE), True),
        "MONITOR.CALIBRATION.ENABLE": (bool(cfg.MONITOR.CALIBRATION.ENABLE), True),
        "MONITOR.TRAIN_EVAL.ENABLE": (bool(cfg.MONITOR.TRAIN_EVAL.ENABLE), False),
        "MONITOR.TRAIN_EVAL.EVERY_N": (int(cfg.MONITOR.TRAIN_EVAL.EVERY_N), 1),
        "MONITOR.LOSS_COMPONENT_TRAJECTORY.ENABLE": (
            bool(cfg.MONITOR.LOSS_COMPONENT_TRAJECTORY.ENABLE), True
        ),
        "MONITOR.PREDICTION_TRANSITION_TRAJECTORY.ENABLE": (
            bool(cfg.MONITOR.PREDICTION_TRANSITION_TRAJECTORY.ENABLE), True
        ),
        "MONITOR.MILESTONE_PROBE.ENABLE": (
            bool(cfg.MONITOR.MILESTONE_PROBE.ENABLE), False
        ),
        "MONITOR.PROBE.ENABLE": (bool(cfg.MONITOR.PROBE.ENABLE), True),
        "MONITOR.PROBE.REQUIRE_FULL_CLASS_COVERAGE": (
            bool(cfg.MONITOR.PROBE.REQUIRE_FULL_CLASS_COVERAGE), True
        ),
        "MONITOR.PROBE.REQUIRE_PER_CLASS_QUOTA": (
            bool(cfg.MONITOR.PROBE.REQUIRE_PER_CLASS_QUOTA), True
        ),
        "MONITOR.PROBE.ALLOW_MAX_SAMPLES_TRUNCATION": (
            bool(cfg.MONITOR.PROBE.ALLOW_MAX_SAMPLES_TRUNCATION), False
        ),
        "MONITOR.PROBE.SEMANTIC_INTERVENTION.ENABLE": (
            bool(cfg.MONITOR.PROBE.SEMANTIC_INTERVENTION.ENABLE), True
        ),
        "MONITOR.PROBE.TARGET_RELEVANCE.ENABLE": (
            bool(cfg.MONITOR.PROBE.TARGET_RELEVANCE.ENABLE), True
        ),
        "MONITOR.PROBE.PROMPT_ANALYSIS.ENABLE": (
            bool(cfg.MONITOR.PROBE.PROMPT_ANALYSIS.ENABLE), True
        ),
        "MONITOR.PROBE.PROMPT_ANALYSIS.CONTENT_ENABLE": (
            bool(cfg.MONITOR.PROBE.PROMPT_ANALYSIS.CONTENT_ENABLE), True
        ),
        "MONITOR.PROBE.PROMPT_ANALYSIS.SOURCE_DECOMPOSITION_ENABLE": (
            bool(
                cfg.MONITOR.PROBE.PROMPT_ANALYSIS.SOURCE_DECOMPOSITION_ENABLE
            ),
            True,
        ),
        "MONITOR.PROBE.PROMPT_ANALYSIS.FLIP_ENABLE": (
            bool(cfg.MONITOR.PROBE.PROMPT_ANALYSIS.FLIP_ENABLE), False
        ),
        "MONITOR.PROBE.PROMPT_ANALYSIS.ROLE_PROFILE_EXPORT_ENABLE": (
            bool(
                cfg.MONITOR.PROBE.PROMPT_ANALYSIS.ROLE_PROFILE_EXPORT_ENABLE
            ),
            False,
        ),
        "MONITOR.PROBE.EXPLANATION_VALIDITY.ENABLE": (
            bool(cfg.MONITOR.PROBE.EXPLANATION_VALIDITY.ENABLE), False
        ),
        "MONITOR.PROBE.BAYESIAN_OBJECT_SELECTION.ENABLE": (
            bool(cfg.MONITOR.PROBE.BAYESIAN_OBJECT_SELECTION.ENABLE), False
        ),
        "MONITOR.PROBE.ATTRIBUTE_CONCEPT_ENABLE": (
            bool(cfg.MONITOR.PROBE.ATTRIBUTE_CONCEPT_ENABLE), True
        ),
        "MONITOR.PROBE.ATTRIBUTE_CONCEPT_PATCH_RATIO": (
            float(cfg.MONITOR.PROBE.ATTRIBUTE_CONCEPT_PATCH_RATIO), 0.2
        ),
        "MONITOR.PROBE.PATCH_SEMANTIC_TRANSPORT.ENABLE": (
            bool(cfg.MONITOR.PROBE.PATCH_SEMANTIC_TRANSPORT.ENABLE), True
        ),
        "MONITOR.PROBE.PATCH_SEMANTIC_TRANSPORT.COST": (
            str(cfg.MONITOR.PROBE.PATCH_SEMANTIC_TRANSPORT.COST).lower(),
            "cosine",
        ),
        "MONITOR.PROBE.PATCH_SEMANTIC_TRANSPORT.PATCH_RATIO": (
            float(cfg.MONITOR.PROBE.PATCH_SEMANTIC_TRANSPORT.PATCH_RATIO),
            0.2,
        ),
        "MONITOR.MODULE_EFFECT.ENABLE": (bool(cfg.MONITOR.MODULE_EFFECT.ENABLE), True),
        "MONITOR.MODULE_EFFECT.PROMPT_READ_BLOCK": (
            bool(cfg.MONITOR.MODULE_EFFECT.PROMPT_READ_BLOCK), True
        ),
        "MONITOR.MODULE_EFFECT.PROMPT_WRITE_BLOCK": (
            bool(cfg.MONITOR.MODULE_EFFECT.PROMPT_WRITE_BLOCK), True
        ),
        "MONITOR.MODULE_EFFECT.PROMPT_SELECTION_UNIFORM": (
            bool(cfg.MONITOR.MODULE_EFFECT.PROMPT_SELECTION_UNIFORM), True
        ),
        "MONITOR.MODULE_EFFECT.PATCH_PROMPT_SELECTION_UNIFORM": (
            bool(cfg.MONITOR.MODULE_EFFECT.PATCH_PROMPT_SELECTION_UNIFORM), True
        ),
        "MONITOR.MODULE_EFFECT.PROMPT_VALUE_GLOBALIZE": (
            bool(cfg.MONITOR.MODULE_EFFECT.PROMPT_VALUE_GLOBALIZE), True
        ),
        "MONITOR.MODULE_EFFECT.LAYERWISE_PROMPT_READ_BLOCK": (
            bool(cfg.MONITOR.MODULE_EFFECT.LAYERWISE_PROMPT_READ_BLOCK), True
        ),
        "MONITOR.MODULE_EFFECT.LAYERWISE_PROMPT_READ_LAYERS": (
            [int(item) for item in cfg.MONITOR.MODULE_EFFECT.LAYERWISE_PROMPT_READ_LAYERS],
            [0, 3, 6, 9],
        ),
        "MONITOR.MODULE_EFFECT.PROMPT_CONTEXT_SWAP": (
            bool(cfg.MONITOR.MODULE_EFFECT.PROMPT_CONTEXT_SWAP), True
        ),
        "MONITOR.MODULE_EFFECT.PROMPT_CONTEXT_SWAP_LAYER": (
            int(cfg.MONITOR.MODULE_EFFECT.PROMPT_CONTEXT_SWAP_LAYER), 9
        ),
        "MONITOR.MODULE_EFFECT.PROMPT_CONTEXT_SWAP_SEED": (
            int(cfg.MONITOR.MODULE_EFFECT.PROMPT_CONTEXT_SWAP_SEED), 57721
        ),
        "MONITOR.MODULE_EFFECT.ATTRIBUTE_CONCEPT_PATCH_BLOCK": (
            bool(cfg.MONITOR.MODULE_EFFECT.ATTRIBUTE_CONCEPT_PATCH_BLOCK), True
        ),
        "MONITOR.MODULE_EFFECT.ATTRIBUTE_CONCEPT_RANDOM_SEED": (
            int(cfg.MONITOR.MODULE_EFFECT.ATTRIBUTE_CONCEPT_RANDOM_SEED), 271828
        ),
        "MONITOR.MODULE_EFFECT.TRANSPORT_PATCH_BLOCK": (
            bool(cfg.MONITOR.MODULE_EFFECT.TRANSPORT_PATCH_BLOCK), True
        ),
        "MONITOR.MODULE_EFFECT.TRANSPORT_RANDOM_SEED": (
            int(cfg.MONITOR.MODULE_EFFECT.TRANSPORT_RANDOM_SEED), 161803
        ),
        "MONITOR.MODULE_EFFECT.INSTANCE_PROMPT_ZERO": (
            bool(cfg.MONITOR.MODULE_EFFECT.INSTANCE_PROMPT_ZERO), False
        ),
        "MONITOR.MODULE_EFFECT.DOMAIN_PROMPT_ZERO": (
            bool(cfg.MONITOR.MODULE_EFFECT.DOMAIN_PROMPT_ZERO), False
        ),
        "MONITOR.MODULE_EFFECT.BOTH_PROMPT_ZERO": (
            bool(cfg.MONITOR.MODULE_EFFECT.BOTH_PROMPT_ZERO), False
        ),
        "MONITOR.MODULE_EFFECT.INSTANCE_PROMPT_SWAP": (
            bool(cfg.MONITOR.MODULE_EFFECT.INSTANCE_PROMPT_SWAP), False
        ),
        "MONITOR.MODULE_EFFECT.PROMPT_VALUE_ZERO": (
            bool(cfg.MONITOR.MODULE_EFFECT.PROMPT_VALUE_ZERO), False
        ),
    }

    for name, (actual, target) in expected.items():
        if actual != target:
            failures.append(f"{name}: expected {target!r}, got {actual!r}")

    if stage == "invalid":
        failures.append("prompt configuration is not one of A0/A1/A2/D")
    if (
        str(cfg.SOLVER.INIT_TRAINABLE_CHECKPOINT).strip()
        and str(cfg.MODEL.WEIGHT_PATH).strip()
    ):
        failures.append(
            "INIT_TRAINABLE_CHECKPOINT and MODEL.WEIGHT_PATH cannot both be set"
        )
    if stage == "B":
        dist_cfg = cfg.MODEL.PROMPT.DISTRIBUTOR
        residual_cfg = dist_cfg.DEEP_RESIDUAL
        if str(dist_cfg.SOURCE).lower() not in {
            "vit_cls_prepass",
            "vit_cls_prepass_constant",
        }:
            failures.append(
                "B-series direct-mean stage requires vit_cls_prepass or its constant control"
            )
        if int(dist_cfg.INSTANCE_TOKENS) != 16 or int(dist_cfg.DOMAIN_TOKENS) != 0:
            failures.append(
                "B-series direct-mean stage requires 16 instance delta Prompt slots and 0 domain slots"
            )
        if not bool(cfg.MONITOR.MODULE_EFFECT.DEEP_RESIDUAL_ZERO):
            failures.append("B-series direct-mean stage requires DEEP_RESIDUAL_ZERO=true")
        if not bool(cfg.MONITOR.MODULE_EFFECT.DEEP_RESIDUAL_SWAP):
            failures.append("B-series direct-mean stage requires DEEP_RESIDUAL_SWAP=true")
        if str(residual_cfg.CONTENT_MODE).lower() not in {"shared", "slot_low_rank"}:
            failures.append("B-series residual CONTENT_MODE must be shared or slot_low_rank")
        if int(residual_cfg.SLOT_RANK) <= 0:
            failures.append("B-series residual SLOT_RANK must be positive")
        if str(residual_cfg.SAMPLE_GATE_MODE).lower() not in {
            "none", "shared", "grouped", "layerwise"
        }:
            failures.append("B-series residual SAMPLE_GATE_MODE is invalid")
        if str(residual_cfg.SAMPLE_GATE_INPUT).lower() not in {
            "residual_source", "constant"
        }:
            failures.append("B-series residual SAMPLE_GATE_INPUT is invalid")
        if not 0.0 < float(residual_cfg.SAMPLE_GATE_INIT) < 1.0:
            failures.append("B-series residual SAMPLE_GATE_INIT must lie in (0, 1)")
    if int(cfg.MODEL.PROMPT.NUM_TOKENS) != 16:
        failures.append("MODEL.PROMPT.NUM_TOKENS must be 16 for the current A/B-series protocol")
    if str(cfg.DATA.XLSA.PROTOCOL_MODE).lower() != "final_gzsl":
        failures.append("DATA.XLSA.PROTOCOL_MODE must be 'final_gzsl' for the A/B-series")
    if int(cfg.MONITOR.TRAIN_EVAL.EVERY_N) <= 0:
        failures.append("MONITOR.TRAIN_EVAL.EVERY_N must be positive")
    transition_splits = [
        str(item).lower()
        for item in cfg.MONITOR.PREDICTION_TRANSITION_TRAJECTORY.SPLITS
    ]
    allowed_transition_splits = {"test_seen", "test_unseen"}
    if not transition_splits:
        failures.append(
            "MONITOR.PREDICTION_TRANSITION_TRAJECTORY.SPLITS must not be empty"
        )
    if len(set(transition_splits)) != len(transition_splits):
        failures.append(
            "MONITOR.PREDICTION_TRANSITION_TRAJECTORY.SPLITS must be unique"
        )
    invalid_transition_splits = sorted(
        set(transition_splits) - allowed_transition_splits
    )
    if invalid_transition_splits:
        failures.append(
            "PREDICTION_TRANSITION_TRAJECTORY only supports test_seen/test_unseen; got {}".format(
                ", ".join(invalid_transition_splits)
            )
        )
    milestone_cfg = cfg.MONITOR.MILESTONE_PROBE
    milestone_fractions = [float(item) for item in milestone_cfg.FRACTIONS]
    if not milestone_fractions or any(
        not 0.0 < value < 1.0 for value in milestone_fractions
    ):
        failures.append("MONITOR.MILESTONE_PROBE.FRACTIONS must lie in (0, 1)")
    if len(set(milestone_fractions)) != len(milestone_fractions):
        failures.append("MONITOR.MILESTONE_PROBE.FRACTIONS must be unique")
    if bool(milestone_cfg.ENABLE):
        if not bool(milestone_cfg.SAVE_CHECKPOINTS):
            failures.append(
                "enabled MILESTONE_PROBE requires SAVE_CHECKPOINTS=true"
            )
        if bool(milestone_cfg.RUN_FIXED_PROBE) and not bool(cfg.MONITOR.PROBE.ENABLE):
            failures.append(
                "MILESTONE_PROBE.RUN_FIXED_PROBE requires MONITOR.PROBE.ENABLE"
            )
    if cfg.SEED is None:
        failures.append("SEED must be set for the A/B-series reproducibility protocol")
    if int(cfg.NUM_GPUS) != 1:
        failures.append("NUM_GPUS must be 1 for the current A/B-series single-GPU contract")
    if int(cfg.NUM_SHARDS) != 1:
        failures.append("NUM_SHARDS must be 1 for the current A/B-series single-GPU contract")
    if int(cfg.DATA.NUM_WORKERS) != 4:
        failures.append("DATA.NUM_WORKERS must be 4 for the current A/B-series deterministic multi-worker loader")
    if int(cfg.MONITOR.PROBE.NUM_WORKERS) < 0:
        failures.append("MONITOR.PROBE.NUM_WORKERS must be non-negative")
    if bool(cfg.CUDNN_BENCHMARK):
        failures.append("CUDNN_BENCHMARK must be false for the current A/B-series single-GPU contract")
    probe_selection_seeds = [int(cfg.MONITOR.PROBE.SELECTION_SEED)] + [
        int(item) for item in cfg.MONITOR.PROBE.ROBUSTNESS_SELECTION_SEEDS
    ]
    if any(seed < 0 for seed in probe_selection_seeds):
        failures.append("fixed-probe selection seeds must be non-negative")
    if len(set(probe_selection_seeds)) != len(probe_selection_seeds):
        failures.append("fixed-probe primary and robustness selection seeds must be unique")
    if (
        bool(cfg.MONITOR.PROBE.ENABLE)
        and str(cfg.MONITOR.PROBE.EXECUTION_PROFILE).lower() == "final_full"
        and len(probe_selection_seeds) != 3
    ):
        failures.append(
            "formal final_full fixed-Probe evidence requires exactly three "
            "selection seeds (one primary plus two robustness seeds)"
        )
    if int(cfg.MONITOR.PROBE.TARGET_RELEVANCE.BATCH_SIZE) <= 0:
        failures.append("MONITOR.PROBE.TARGET_RELEVANCE.BATCH_SIZE must be positive")
    if bool(cfg.MONITOR.PROBE.EXPLANATION_VALIDITY.ENABLE):
        if not bool(cfg.MONITOR.PROBE.TARGET_RELEVANCE.ENABLE):
            failures.append(
                "EXPLANATION_VALIDITY requires TARGET_RELEVANCE.ENABLE"
            )
        explanation_layers = [
            int(item) for item in cfg.MONITOR.PROBE.EXPLANATION_VALIDITY.LAYERS
        ]
        probe_layers = [int(item) for item in cfg.MONITOR.PROBE.LAYERS]
        if not explanation_layers or not set(explanation_layers).issubset(
            probe_layers
        ):
            failures.append(
                "EXPLANATION_VALIDITY.LAYERS must be a non-empty subset of PROBE.LAYERS"
            )
        fractions = [
            float(item)
            for item in cfg.MONITOR.PROBE.EXPLANATION_VALIDITY.K_FRACTIONS
        ]
        if not fractions or any(not 0.0 < value < 1.0 for value in fractions):
            failures.append(
                "EXPLANATION_VALIDITY.K_FRACTIONS must lie in (0, 1)"
            )
        explanation_conditions = {
            str(item).lower()
            for item in cfg.MONITOR.PROBE.EXPLANATION_VALIDITY.CONDITIONS
        }
        if not {"positive", "negative", "random"}.issubset(
            explanation_conditions
        ):
            failures.append(
                "EXPLANATION_VALIDITY.CONDITIONS must include positive, negative, and random"
            )
    if bool(cfg.MONITOR.PROBE.PROMPT_ANALYSIS.SEMANTIC_GRANULARITY_ENABLE):
        local_attributes = {
            int(item)
            for item in cfg.MONITOR.PROBE.PROMPT_ANALYSIS.LOCAL_ATTRIBUTE_INDICES
        }
        global_attributes = {
            int(item)
            for item in cfg.MONITOR.PROBE.PROMPT_ANALYSIS.GLOBAL_ATTRIBUTE_INDICES
        }
        if not local_attributes or not global_attributes:
            failures.append(
                "semantic granularity requires non-empty local/global attribute groups"
            )
        if local_attributes.intersection(global_attributes):
            failures.append(
                "semantic granularity local/global attribute groups must be disjoint"
            )
    if float(cfg.MONITOR.PROBE.PATCH_SEMANTIC_TRANSPORT.TEMPERATURE) <= 0.0:
        failures.append(
            "MONITOR.PROBE.PATCH_SEMANTIC_TRANSPORT.TEMPERATURE must be positive"
        )
    object_cfg = cfg.MONITOR.PROBE.BAYESIAN_OBJECT_SELECTION
    if str(object_cfg.PERTURBATION_MODE).lower() != "normalized_direction":
        failures.append(
            "BAYESIAN_OBJECT_SELECTION.PERTURBATION_MODE must be normalized_direction"
        )
    object_scales = [float(item) for item in object_cfg.PERTURBATION_SCALES]
    if not object_scales or any(value <= 0.0 for value in object_scales):
        failures.append(
            "BAYESIAN_OBJECT_SELECTION.PERTURBATION_SCALES must be positive"
        )
    if int(object_cfg.DIRECTION_COUNT) <= 0:
        failures.append(
            "BAYESIAN_OBJECT_SELECTION.DIRECTION_COUNT must be positive"
        )
    if str(object_cfg.HIERARCHY_VARIANT_SOURCE).lower() != "controlled_perturbation":
        failures.append(
            "BAYESIAN_OBJECT_SELECTION cannot use posterior_sample before object selection"
        )
    if str(object_cfg.HIERARCHY_REFERENCE).lower() != "unperturbed_same_checkpoint":
        failures.append(
            "BAYESIAN_OBJECT_SELECTION reference must be unperturbed_same_checkpoint"
        )
    if not bool(object_cfg.HIERARCHY_TRACE_ENABLE):
        failures.append(
            "BAYESIAN_OBJECT_SELECTION v1 requires HIERARCHY_TRACE_ENABLE=true"
        )
    if bool(object_cfg.EXPORT_SAMPLE_VECTORS):
        failures.append(
            "A/B-series common config must not export Bayesian object sample vectors"
        )
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

    freeze_classifier = bool(
        cfg.MODEL.PROMPT.DISTRIBUTOR.DEEP_RESIDUAL.FREEZE_CLASSIFIER
    )
    freeze_static_prompt = bool(
        cfg.MODEL.PROMPT.DISTRIBUTOR.DEEP_RESIDUAL.FREEZE_STATIC_PROMPT
    )
    if not freeze_classifier and not any(
        name.startswith("r_similarity_head.prototype_proj") for name in trainable
    ):
        failures.append("R-similarity prototype projection is not trainable")
    if freeze_classifier and any(
        name.startswith("r_similarity_head.") for name in trainable
    ):
        failures.append("freeze-classifier config left classifier parameters trainable")
    if any("semantic_token" in name for name in trainable):
        failures.append("semantic-token parameters remain trainable")
    if any("attention_mediation" in name for name in trainable):
        failures.append("attention-mediation parameters remain trainable")
    if stage != "B" and any("prompt_init_provider" in name for name in trainable):
        failures.append("prompt-distributor parameters remain trainable")

    prompt_names = [name for name in trainable if "prompt_embeddings" in name]
    deep_names = [name for name in trainable if "deep_prompt_embeddings" in name]
    if stage == "A0" and prompt_names:
        failures.append("A0 unexpectedly has trainable prompt parameters")
    if stage == "A1" and (not prompt_names or deep_names):
        failures.append("A1 must train input prompt embeddings only")
    if stage == "A2" and (not prompt_names or not deep_names):
        failures.append("A2 must train both input and deep prompt embeddings")
    if stage == "B" and not freeze_static_prompt and (not prompt_names or not deep_names):
        failures.append("B-series must retain both static input and deep prompt embeddings")
    if stage == "B" and freeze_static_prompt and (prompt_names or deep_names):
        failures.append("freeze-static config left static Prompt parameters trainable")
    if stage == "B" and not any(
        "prompt_init_provider.stats_head" in name for name in trainable
    ):
        failures.append("B-series must train the existing Prompt Distributor stats MLP")
    if stage == "B" and not any(
        name.endswith("deep_prompt_residual.layer_gate") for name in trainable
    ):
        failures.append("B-series must train one direct-mean residual gate per layer")
    allowed_prefixes = {
        "A0": ("r_similarity_head.prototype_proj.",),
        "A1": ("enc.transformer.prompt_embeddings", "r_similarity_head.prototype_proj."),
        "A2": (
            "enc.transformer.prompt_embeddings",
            "enc.transformer.deep_prompt_embeddings",
            "r_similarity_head.prototype_proj.",
        ),
        "B": (
            "enc.transformer.prompt_embeddings",
            "enc.transformer.deep_prompt_embeddings",
            "enc.transformer.prompt_init_provider.stats_head",
            "enc.transformer.deep_prompt_residual.",
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
    """Check the A/B-series random streams that must stay aligned across stages."""
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
                "{} must match across compared A/B-series configs: {}".format(
                    stream_name,
                    ", ".join(
                        "{}={}".format(Path(record["path"]).name, record["streams"][stream_name])
                        for record in records
                    ),
                )
            )
    prompt_records = [
        record for record in records if record["stage"] in {"A1", "A2", "B"}
    ]
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
    print("PASS: configuration satisfies the A/B-series CE baseline gate")


if __name__ == "__main__":
    main()
