#!/usr/bin/env python3
"""Validate multi-loss gradient-audit artifacts or run synthetic contracts."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import weakref

import torch
import torch.nn as nn

from ..monitoring.multi_loss_gradient import MultiLossGradientAuditor
from ..configs.config import get_cfg
from ..engine.evaluator import Evaluator
from ..engine.trainer import Trainer
from ..solver.losses import LossTerm, build_loss
from ..utils.train_utils import AverageMeter


class _MemoryMonitor:
    def __init__(self) -> None:
        self.monitor_groups = {
            "multi_loss_gradient_audit": {"source_active": True}
        }
        self.evidence = {}
        self.records = []
        self.events = []

    def write_evidence(self, filename, payload):
        self.evidence[str(filename)] = dict(payload)

    def set_context(self, **kwargs):
        del kwargs

    def update_runtime_source_state(self, namespace, *, source_active):
        self.monitor_groups[str(namespace)]["source_active"] = bool(source_active)

    def append_evidence_jsonl(self, filename, payload):
        self.records.append((str(filename), dict(payload)))

    def record_event(self, event, payload):
        self.events.append((str(event), dict(payload)))


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stats_head = nn.Linear(1, 1, bias=False)
        self.deep_prompt_residual = nn.Linear(1, 1, bias=False)
        self.prompt_embeddings = nn.Parameter(torch.ones(1))

    def forward(self, value):
        return self.deep_prompt_residual(self.stats_head(value))


class _TinyDdpModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stats_head = nn.Linear(1, 1, bias=False)
        self.deep_prompt_residual = nn.Linear(1, 1, bias=False)

    def forward(self, value):
        return self.deep_prompt_residual(self.stats_head(value))


class _TinyTrainerModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stats_head = nn.Linear(4, 4)
        self.deep_prompt_residual = nn.Linear(4, 2)
        self.logvar_head = nn.Linear(4, 4)
        self.r_similarity_head = None
        self.enc = nn.Module()
        self.enc.transformer = nn.Module()
        self._runtime_stats = None

    def forward(self, inputs, semantics=None, class_ids=None, runtime_targets=None):
        del semantics, class_ids, runtime_targets
        mu = self.stats_head(inputs)
        logvar = self.logvar_head(inputs)
        self._runtime_stats = {"mu": mu, "logvar": logvar}
        return self.deep_prompt_residual(mu)

    def get_runtime_prompt_distribution_stats(self):
        return self._runtime_stats


class _TinyDataset(torch.utils.data.Dataset):
    name = "tiny"
    split_name = "train"

    def __init__(self) -> None:
        self.local_classes = [0, 1]
        self.eval_local_classes = [0, 1]
        self.global_to_local = torch.tensor([0, 1], dtype=torch.long)
        self.eval_global_to_local = torch.tensor([0, 1], dtype=torch.long)
        self.class_attributes = torch.zeros(2, 4)
        self.seen_classes = [0, 1]
        self.unseen_classes = []

    def __len__(self):
        return 4

    def __getitem__(self, index):
        return {
            "image": torch.tensor(
                [float(index), 1.0, -0.5, 0.25], dtype=torch.float32
            ),
            "label": torch.tensor(index % 2, dtype=torch.long),
            "attribute": torch.zeros(4, dtype=torch.float32),
            "sample_id": f"tiny-{index}",
        }

    @staticmethod
    def get_class_weights(_kind):
        return [1.0, 1.0]


def _cfg(*, weight: float = 1.0, alignment: bool = True):
    audit = SimpleNamespace(
        ENABLE=True,
        EPOCHS=[1],
        BATCHES_PER_EPOCH=1,
        INCLUDE_PAIRWISE_AUX_COSINE=True,
        INCLUDE_OPTIMIZER_ALIGNMENT=alignment,
        ALLOW_SINGLE_LOSS_SMOKE=False,
        NORM_EPS=1.0e-12,
        PARAMETER_BLOCK_OVERRIDES=[],
    )
    return SimpleNamespace(
        SEED=7,
        MONITOR=SimpleNamespace(
            ENABLE=True,
            MULTI_LOSS_GRADIENT_AUDIT=audit,
        ),
        SOLVER=SimpleNamespace(TOTAL_EPOCH=1),
        TEST_AUX_WEIGHT=float(weight),
    )


def _run_case(
    *, relation: str, weight: float = 1.0, include_second_auxiliary: bool = False
):
    torch.manual_seed(13)
    model = _TinyModel()
    with torch.no_grad():
        model.stats_head.weight.fill_(1.0)
        model.deep_prompt_residual.weight.fill_(1.0)
    criterion = nn.Module()
    criterion.aux_losses = (SimpleNamespace(weight=float(weight)),)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    manager = _MemoryMonitor()
    auditor = MultiLossGradientAuditor(
        _cfg(weight=weight),
        (("model", model), ("cls_criterion", criterion)),
        optimizer,
        manager,
        torch.device("cpu"),
    )
    x = torch.ones(4, 1)
    source = model.stats_head(x)
    carrier = model.deep_prompt_residual(source)
    primary = (carrier - 0.0).pow(2).mean()
    if relation == "same":
        auxiliary = 0.5 * carrier.pow(2).mean()
    elif relation == "opposite":
        auxiliary = (carrier - 2.0).pow(2).mean()
    else:
        raise ValueError(relation)
    terms = [
        LossTerm("ce_loss", primary, 1.0, "primary"),
        LossTerm("aux_loss", auxiliary, float(weight), "transfer"),
    ]
    second_auxiliary = 0.25 * carrier.pow(2).mean()
    if include_second_auxiliary:
        terms.append(
            LossTerm("aux_loss_2", second_auxiliary, 0.2, "regularization")
        )
    auditor.prepare(
        tuple(terms),
        targets=torch.zeros(4, dtype=torch.long),
        sample_ids=[f"sample-{index}" for index in range(4)],
        epoch=1,
        global_step=1,
    )
    optimizer.zero_grad()
    total = primary + float(weight) * auxiliary
    if include_second_auxiliary:
        total = total + 0.2 * second_auxiliary
    total.backward()
    optimizer.step()
    auditor.finalize_optimizer_step()
    auditor.finalize(status="completed")
    return manager.records[0][1], manager.evidence


def _find(rows, **identity):
    for row in rows:
        if all(row.get(key) == value for key, value in identity.items()):
            return row
    raise AssertionError(f"missing row: {identity}")


def run_synthetic_contracts() -> dict:
    same, same_evidence = _run_case(relation="same", weight=0.25)
    opposite, _ = _run_case(relation="opposite", weight=1.0)
    scaled_low, _ = _run_case(relation="same", weight=0.25)
    scaled_high, _ = _run_case(relation="same", weight=0.5)
    pairwise_case, _ = _run_case(
        relation="same", weight=0.25, include_second_auxiliary=True
    )

    same_row = _find(
        same["primary_auxiliary_comparisons"],
        right_loss="aux_loss",
        parameter_block="stats_source",
    )
    opposite_row = _find(
        opposite["primary_auxiliary_comparisons"],
        right_loss="aux_loss",
        parameter_block="stats_source",
    )
    low_row = _find(
        scaled_low["block_metrics"],
        loss_name="aux_loss",
        parameter_block="stats_source",
    )
    high_row = _find(
        scaled_high["block_metrics"],
        loss_name="aux_loss",
        parameter_block="stats_source",
    )
    disconnected = _find(
        same["block_metrics"],
        loss_name="aux_loss",
        parameter_block="static_prompt",
    )
    cancellation = _find(
        opposite["combined_block_metrics"],
        parameter_block="stats_source",
    )
    alignment = _find(
        same["optimizer_alignments"],
        loss_name="ce_loss",
        parameter_block="stats_source",
    )
    pairwise_row = _find(
        pairwise_case["auxiliary_pairwise_comparisons"],
        left_loss="aux_loss",
        right_loss="aux_loss_2",
        parameter_block="stats_source",
    )

    checks = {
        "same_direction_cosine": bool(float(same_row["grad_cosine"]) > 0.999),
        "opposite_direction_cosine": bool(
            float(opposite_row["grad_cosine"]) < -0.999
        ),
        "opposite_cancellation_detected": bool(
            float(cancellation["gradient_cancellation_ratio"]) < 0.1
        ),
        "disconnected_is_not_applicable": bool(
            disconnected["status"] == "not_applicable"
            and disconnected["grad_norm_raw"] is None
        ),
        "raw_gradient_weight_invariant": bool(
            math.isclose(
                float(low_row["grad_norm_raw"]),
                float(high_row["grad_norm_raw"]),
                rel_tol=1.0e-6,
                abs_tol=1.0e-8,
            )
        ),
        "weighted_gradient_scales_linearly": bool(
            math.isclose(
                float(high_row["grad_norm_weighted"]),
                2.0 * float(low_row["grad_norm_weighted"]),
                rel_tol=1.0e-6,
                abs_tol=1.0e-8,
            )
        ),
        "sgd_descent_alignment_positive": bool(
            float(alignment["optimizer_descent_alignment"]) > 0.0
        ),
        "auxiliary_pairwise_cosine": bool(
            float(pairwise_row["grad_cosine"]) > 0.999
        ),
        "summary_written": "multi_loss_gradient_audit_summary.json" in same_evidence,
        "validation_written": "multi_loss_gradient_audit_validation.json" in same_evidence,
    }
    cfg = get_cfg()
    cfg.SOLVER.RSIM.ALIGN_MODE = "none"
    cfg.SOLVER.RSIM.ALIGN_WEIGHT = 0.0
    cfg.SOLVER.LOSS_SEM_MED_WEIGHT = 0.0
    cfg.SOLVER.LOSS_SPV_WEIGHT = 0.0
    cfg.SOLVER.LOSS_ATTR_WEIGHT = 0.0
    cfg.SOLVER.LOSS_PROMPT_KL_WEIGHT = 0.0
    cfg.MODEL.GRAPH_PROB_PRIOR.ENABLE = False
    cfg.SOLVER.B3_CLASS_CONSISTENCY.ENABLE = False
    checks["default_audit_is_disabled"] = bool(
        not cfg.MONITOR.MULTI_LOSS_GRADIENT_AUDIT.ENABLE
    )
    criterion = build_loss(cfg)
    targets = torch.tensor([0, 1], dtype=torch.long)
    weights = [1.0, 1.0]
    logits_a = torch.tensor([[1.0, -0.5], [-0.2, 0.7]], requires_grad=True)
    logits_b = logits_a.detach().clone().requires_grad_(True)
    loss_a = criterion(
        logits_a, targets, weights, kwargs={"capture_loss_terms": False}
    )
    loss_a.backward()
    gradient_a = logits_a.grad.detach().clone()
    loss_b = criterion(
        logits_b, targets, weights, kwargs={"capture_loss_terms": True}
    )
    captured_count = len(criterion.get_last_loss_terms())
    loss_b.backward()
    gradient_b = logits_b.grad.detach().clone()
    criterion.clear_last_loss_terms()
    checks["loss_term_capture_preserves_loss"] = bool(
        torch.equal(loss_a.detach(), loss_b.detach())
    )
    checks["loss_term_capture_preserves_gradient"] = bool(
        torch.equal(gradient_a, gradient_b)
    )
    checks["captured_tensor_references_are_released"] = bool(
        captured_count == 1 and len(criterion.get_last_loss_terms()) == 0
    )
    logits_c = torch.tensor([[0.4, -0.1], [-0.3, 0.8]], requires_grad=True)
    loss_c = criterion(
        logits_c, targets, weights, kwargs={"capture_loss_terms": True}
    )
    captured_terms = criterion.get_last_loss_terms()
    captured_reference = weakref.ref(captured_terms[0].raw_tensor)
    criterion.clear_last_loss_terms()
    del captured_terms, loss_c, logits_c
    gc.collect()
    checks["captured_graph_tensor_is_collectable"] = captured_reference() is None

    def contains_tensor(value):
        if torch.is_tensor(value):
            return True
        if isinstance(value, dict):
            return any(contains_tensor(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return any(contains_tensor(item) for item in value)
        return False

    checks["serialized_records_hold_no_tensor"] = not contains_tensor(same)
    return {
        "format": "multi_loss_gradient_audit_synthetic_validation_v1",
        "checks": checks,
        "overall_pass": all(checks.values()),
        "ddp_smoke": {
            "status": "not_run",
            "reason": "run the formal training smoke under the project DDP launcher",
        },
    }


def _ddp_worker(rank: int, world_size: int, init_method: str, output_dir: str) -> None:
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=int(rank),
        world_size=int(world_size),
    )
    try:
        torch.manual_seed(23)
        model = _TinyDdpModel()
        ddp = torch.nn.parallel.DistributedDataParallel(model)
        optimizer = torch.optim.SGD(ddp.parameters(), lr=0.05)
        manager = _MemoryMonitor()
        criterion = nn.Module()
        criterion.aux_losses = (SimpleNamespace(weight=0.25),)
        auditor = MultiLossGradientAuditor(
            _cfg(weight=0.25),
            (("model", ddp.module), ("cls_criterion", criterion)),
            optimizer,
            manager,
            torch.device("cpu"),
        )
        x = torch.ones(4, 1) * float(rank + 1)
        carrier = ddp(x)
        primary = carrier.pow(2).mean()
        auxiliary = 0.5 * carrier.pow(2).mean()
        auditor.prepare(
            (
                LossTerm("ce_loss", primary, 1.0, "primary"),
                LossTerm("aux_loss", auxiliary, 0.25, "transfer"),
            ),
            targets=torch.zeros(4, dtype=torch.long),
            sample_ids=[f"rank{rank}-sample{index}" for index in range(4)],
            epoch=1,
            global_step=1,
        )
        optimizer.zero_grad()
        (primary + 0.25 * auxiliary).backward()
        optimizer.step()
        auditor.finalize_optimizer_step()
        second = ddp(x + 0.5)
        second_loss = second.pow(2).mean()
        optimizer.zero_grad()
        second_loss.backward()
        optimizer.step()
        Path(output_dir, f"rank_{rank}.json").write_text(
            json.dumps(manager.records[0][1], sort_keys=True),
            encoding="utf-8",
        )
    finally:
        torch.distributed.destroy_process_group()


def run_ddp_synthetic_contract() -> dict:
    if not torch.distributed.is_available():
        return {
            "status": "not_run",
            "reason": "torch.distributed_is_unavailable",
            "overall_pass": False,
        }
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary).resolve()
        rendezvous = root / "ddp_init"
        init_method = "file:///{}".format(rendezvous.as_posix().lstrip("/"))
        torch.multiprocessing.spawn(
            _ddp_worker,
            args=(2, init_method, str(root)),
            nprocs=2,
            join=True,
        )
        records = [
            json.loads((root / f"rank_{rank}.json").read_text(encoding="utf-8"))
            for rank in range(2)
        ]
        comparable_keys = (
            "block_metrics",
            "primary_auxiliary_comparisons",
            "combined_block_metrics",
            "optimizer_alignments",
        )
        scalar_aggregation_equal = all(
            records[0][key] == records[1][key] for key in comparable_keys
        )
        return {
            "status": "passed" if scalar_aggregation_equal else "failed",
            "world_size": 2,
            "backend": "gloo",
            "scalar_aggregation_equal_across_ranks": scalar_aggregation_equal,
            "overall_pass": scalar_aggregation_equal,
        }


def run_trainer_epoch_smoke() -> dict:
    with tempfile.TemporaryDirectory() as temporary:
        cfg = get_cfg()
        cfg.defrost()
        cfg.OUTPUT_DIR = str(Path(temporary, "run").resolve())
        cfg.SEED = 31
        cfg.NUM_GPUS = 1
        cfg.SOLVER.TOTAL_EPOCH = 1
        cfg.SOLVER.RSIM.ALIGN_MODE = "none"
        cfg.SOLVER.RSIM.ALIGN_WEIGHT = 0.0
        cfg.SOLVER.LOSS_PROMPT_KL_WEIGHT = 0.1
        cfg.SOLVER.PROGRESS.ENABLE = False
        cfg.MONITOR.OUTPUT_POLICY = "error_if_exists"
        cfg.MONITOR.STEP_EVERY_N = 100
        cfg.MONITOR.MULTI_LOSS_GRADIENT_AUDIT.ENABLE = True
        cfg.MONITOR.MULTI_LOSS_GRADIENT_AUDIT.EPOCHS = [1]
        cfg.MONITOR.MULTI_LOSS_GRADIENT_AUDIT.BATCHES_PER_EPOCH = 1
        cfg.MONITOR.MULTI_LOSS_GRADIENT_AUDIT.PARAMETER_BLOCK_OVERRIDES = [
            "model.logvar_head=stats_source"
        ]
        cfg.MONITOR.PROBE.ENABLE = False
        cfg.MONITOR.DIAGNOSTICS.ENABLE = False
        cfg.MODEL.PROMPT.ENABLE = False
        cfg.MODEL.SEMANTIC_TOKENS.ENABLE = False
        cfg.freeze()
        model = _TinyTrainerModel()
        trainer = Trainer(
            cfg,
            model,
            Evaluator(task_type="zsl"),
            torch.device("cpu"),
        )
        dataset = _TinyDataset()
        loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)
        trainer.cls_weights = dataset.get_class_weights("none")
        trainer._run_train_epoch(
            0,
            1,
            len(loader),
            loader,
            100,
            AverageMeter("Loss", ":.4e"),
            AverageMeter("Time", ":6.3f"),
            AverageMeter("Data", ":6.3f"),
        )
        trainer.multi_loss_gradient_auditor.finalize(status="completed")
        trainer.diagnostic_manager.finalize(status="completed")
        trainer.monitor_manager.finalize(status="completed")
        validation = validate_run_directory(Path(cfg.OUTPUT_DIR))
        return {
            "format": "multi_loss_gradient_audit_trainer_epoch_smoke_v1",
            "record_count": int(validation["record_count"]),
            "artifact_validation": validation,
            "overall_pass": bool(validation["overall_pass"]),
        }
def validate_run_directory(run_dir: Path) -> dict:
    required = (
        "multi_loss_gradient_audit.jsonl",
        "multi_loss_gradient_audit_summary.json",
        "multi_loss_parameter_group_manifest.json",
        "multi_loss_gradient_audit_validation.json",
    )
    missing = [name for name in required if not (run_dir / name).is_file()]
    records = []
    jsonl_path = run_dir / "multi_loss_gradient_audit.jsonl"
    if jsonl_path.is_file():
        for line_number, line in enumerate(
            jsonl_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid JSONL at {jsonl_path}:{line_number}"
                    ) from exc
    record_failures = []
    manifest = None
    manifest_path = run_dir / "multi_loss_parameter_group_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    def nonfinite_paths(value, prefix=""):
        failures = []
        if isinstance(value, dict):
            for key, item in value.items():
                failures.extend(nonfinite_paths(item, f"{prefix}.{key}" if prefix else str(key)))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                failures.extend(nonfinite_paths(item, f"{prefix}[{index}]"))
        elif isinstance(value, float) and not math.isfinite(value):
            failures.append(prefix)
        return failures

    required_record_keys = {
        "epoch",
        "global_step",
        "calibration_batch_manifest_hash",
        "sample_identity_status",
        "training_seed",
        "world_size",
        "initialization_checkpoint",
        "parameter_group_manifest_sha256",
        "loss_terms",
        "block_metrics",
        "combined_block_metrics",
        "validation_status",
    }
    for index, record in enumerate(records):
        absent = sorted(required_record_keys.difference(record))
        if absent:
            record_failures.append(f"record_{index}_missing:{','.join(absent)}")
        if record.get("validation_status") != "valid":
            record_failures.append(f"record_{index}_not_valid")
        nonfinite = nonfinite_paths(record)
        if nonfinite:
            record_failures.append(
                f"record_{index}_nonfinite:{','.join(nonfinite[:8])}"
            )
        loss_names = {
            str(term.get("name"))
            for term in record.get("loss_terms", [])
            if isinstance(term, dict) and term.get("name")
        }
        observed_pairs = [
            (str(row.get("loss_name")), str(row.get("parameter_block")))
            for row in record.get("block_metrics", [])
            if isinstance(row, dict)
        ]
        expected_pairs = {
            (loss_name, block)
            for loss_name in loss_names
            for block in (
                "stats_source",
                "residual_carrier_gate",
                "static_prompt",
                "semantic_classifier",
                "graph_prior",
                "other_trainable",
            )
        }
        if set(observed_pairs) != expected_pairs or len(observed_pairs) != len(
            set(observed_pairs)
        ):
            record_failures.append(f"record_{index}_loss_block_coverage_mismatch")
        if manifest is not None and record.get(
            "parameter_group_manifest_sha256"
        ) != manifest.get("manifest_sha256"):
            record_failures.append(f"record_{index}_parameter_manifest_hash_mismatch")
    validation_payload = None
    validation_path = run_dir / "multi_loss_gradient_audit_validation.json"
    if validation_path.is_file():
        validation_payload = json.loads(validation_path.read_text(encoding="utf-8"))
    summary_payload = None
    summary_path = run_dir / "multi_loss_gradient_audit_summary.json"
    if summary_path.is_file():
        summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    runtime_payload = None
    runtime_path = run_dir / "monitor_runtime_summary.json"
    if runtime_path.is_file():
        runtime_payload = json.loads(runtime_path.read_text(encoding="utf-8"))
    failures = [f"missing_artifact:{name}" for name in missing] + record_failures
    if validation_payload is not None and not bool(
        validation_payload.get("overall_pass", False)
    ):
        failures.append("producer_validation_did_not_pass")
    if validation_payload is not None and int(
        validation_payload.get("observed_record_count", -1)
    ) != len(records):
        failures.append("producer_validation_record_count_mismatch")
    if summary_payload is not None and int(
        summary_payload.get("record_count", -1)
    ) != len(records):
        failures.append("summary_record_count_mismatch")
    if runtime_payload is None:
        failures.append("monitor_runtime_summary_missing")
    else:
        runtime_group = runtime_payload.get("groups", {}).get(
            "multi_loss_gradient_audit", {}
        )
        if not bool(runtime_group.get("observed", False)):
            failures.append("runtime_namespace_not_observed")
    return {
        "format": "multi_loss_gradient_audit_external_validation_v1",
        "run_dir": str(run_dir.resolve()),
        "record_count": int(len(records)),
        "failure_reasons": failures,
        "overall_pass": not failures,
        "report_complete_claim_allowed": not failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--ddp-synthetic", action="store_true")
    parser.add_argument("--trainer-smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (
        not args.synthetic
        and not args.ddp_synthetic
        and not args.trainer_smoke
        and args.run_dir is None
    ):
        parser.error("use --synthetic, --ddp-synthetic, --trainer-smoke, or --run-dir")
    if args.trainer_smoke:
        payload = run_trainer_epoch_smoke()
    elif args.ddp_synthetic:
        payload = run_ddp_synthetic_contract()
    elif args.synthetic:
        payload = run_synthetic_contracts()
    else:
        payload = validate_run_directory(args.run_dir)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if bool(payload["overall_pass"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
