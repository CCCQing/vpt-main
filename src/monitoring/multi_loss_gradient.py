"""Sparse per-loss gradient and optimizer-direction auditing.

The audit is intentionally dormant on ordinary steps.  On predeclared
milestones it consumes graph-retaining ``LossTerm`` objects, uses
``torch.autograd.grad`` without touching ``parameter.grad``, and releases all
temporary tensors immediately after the normal optimizer step.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch


PARAMETER_BLOCKS = (
    "stats_source",
    "residual_carrier_gate",
    "static_prompt",
    "semantic_classifier",
    "graph_prior",
    "other_trainable",
)


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parameter_block(
    name: str,
    owner: str,
    overrides: Sequence[Tuple[str, str]] = (),
) -> Tuple[str, str]:
    lowered = str(name).lower()
    owner_lower = str(owner).lower()
    for pattern, block in overrides:
        if str(pattern).lower() in lowered:
            return str(block), "explicit_config_override:{}".format(pattern)
    if owner_lower == "cls_criterion" and (
        "graph_prob_prior" in lowered
        or ".computer." in lowered
        or "graphprobprior" in lowered
    ):
        return "graph_prior", "criterion_graph_prior_ownership"
    if (
        "prompt_init_provider.stats_head" in lowered
        or ".stats_head." in lowered
        or lowered.endswith(".stats_head")
    ):
        return "stats_source", "stats_source_module_ownership"
    if any(
        token in lowered
        for token in (
            "prompt_init_provider.domain_prompt",
            "prompt_init_provider.slot_embed",
        )
    ):
        return "static_prompt", "prompt_provider_static_parameter_ownership"
    if any(
        token in lowered
        for token in (
            "deep_prompt_residual",
            "residual_decoder",
            "slot_coefficients",
            "slot_basis",
            "sample_gate_head",
            "layer_gate",
            "residual_gate",
            "attention_mediation",
        )
    ):
        return "residual_carrier_gate", "residual_carrier_module_ownership"
    if (
        lowered.endswith("prompt_embeddings")
        or lowered.endswith("deep_prompt_embeddings")
        or ".prompt_embeddings." in lowered
        or ".deep_prompt_embeddings." in lowered
    ):
        return "static_prompt", "static_prompt_parameter_ownership"
    if any(
        token in lowered
        for token in (
            "r_similarity_head",
            "prototype_proj",
            "semantic_classifier",
            "compatibility_head",
            "logit_scale",
        )
    ):
        return "semantic_classifier", "semantic_classifier_module_ownership"
    if any(
        token in lowered
        for token in (
            "graph_prob_prior",
            "graph_prior",
            "prior_mean",
            "prior_logvar",
        )
    ):
        return "graph_prior", "graph_prior_module_ownership"
    return "other_trainable", "explicit_fallback_requires_review"


class ParameterBlockRegistry:
    """Resolve optimizer parameters to explicit research-facing blocks."""

    def __init__(
        self,
        modules: Sequence[Tuple[str, torch.nn.Module]],
        optimizer: torch.optim.Optimizer,
        overrides: Sequence[str] = (),
    ) -> None:
        self.optimizer = optimizer
        parsed_overrides = []
        for raw in overrides:
            text = str(raw)
            if "=" not in text:
                raise ValueError(
                    "parameter block override must use '<name substring>=<block>'"
                )
            pattern, block = (part.strip() for part in text.split("=", 1))
            if not pattern or block not in PARAMETER_BLOCKS:
                raise ValueError(
                    "invalid parameter block override '{}'; block must be one of {}".format(
                        text, PARAMETER_BLOCKS
                    )
                )
            parsed_overrides.append((pattern, block))
        optimizer_meta: Dict[int, Dict[str, Any]] = {}
        duplicate_optimizer_ids = set()
        for group_index, group in enumerate(optimizer.param_groups):
            for parameter in group.get("params", []):
                parameter_id = id(parameter)
                if parameter_id in optimizer_meta:
                    duplicate_optimizer_ids.add(parameter_id)
                optimizer_meta[parameter_id] = {
                    "optimizer_group_index": int(group_index),
                    "lr": float(group.get("lr", 0.0)),
                    "weight_decay": float(group.get("weight_decay", 0.0)),
                }

        entries = []
        seen_ids: Dict[int, int] = {}
        for owner, module in modules:
            for local_name, parameter in module.named_parameters():
                full_name = f"{owner}.{local_name}" if owner else str(local_name)
                parameter_id = id(parameter)
                if parameter_id in seen_ids:
                    entries[seen_ids[parameter_id]]["aliases"].append(full_name)
                    continue
                block, basis = _parameter_block(
                    full_name, owner, parsed_overrides
                )
                meta = optimizer_meta.get(parameter_id)
                row = {
                    "name": full_name,
                    "aliases": [],
                    "owner": str(owner),
                    "parameter_block": block,
                    "classification_basis": basis,
                    "shape": [int(value) for value in parameter.shape],
                    "numel": int(parameter.numel()),
                    "requires_grad": bool(parameter.requires_grad),
                    "optimizer_member": meta is not None,
                    **(meta or {}),
                    "parameter": parameter,
                }
                seen_ids[parameter_id] = len(entries)
                entries.append(row)

        self.entries = entries
        self.audit_entries = [
            row
            for row in entries
            if row["requires_grad"] and row["optimizer_member"]
        ]
        self.parameters = [row["parameter"] for row in self.audit_entries]
        self.parameter_blocks = [row["parameter_block"] for row in self.audit_entries]
        self.parameter_names = [row["name"] for row in self.audit_entries]
        trainable_missing = [
            row["name"]
            for row in entries
            if row["requires_grad"] and not row["optimizer_member"]
        ]
        frozen_in_optimizer = [
            row["name"]
            for row in entries
            if (not row["requires_grad"]) and row["optimizer_member"]
        ]
        other_trainable = [
            row["name"]
            for row in self.audit_entries
            if row["parameter_block"] == "other_trainable"
        ]
        block_summary: Dict[str, Dict[str, int]] = {}
        for block in PARAMETER_BLOCKS:
            rows = [row for row in entries if row["parameter_block"] == block]
            block_summary[block] = {
                "parameter_tensor_count": int(len(rows)),
                "parameter_count": int(sum(row["numel"] for row in rows)),
                "trainable_tensor_count": int(
                    sum(bool(row["requires_grad"]) for row in rows)
                ),
                "optimizer_tensor_count": int(
                    sum(bool(row["optimizer_member"]) for row in rows)
                ),
            }
        manifest_entries = [
            {key: value for key, value in row.items() if key != "parameter"}
            for row in entries
        ]
        manifest_core = {
            "format": "multi_loss_parameter_group_manifest_v1",
            "parameter_blocks": list(PARAMETER_BLOCKS),
            "parameter_block_overrides": [
                "{}={}".format(pattern, block)
                for pattern, block in parsed_overrides
            ],
            "block_summary": block_summary,
            "parameters": manifest_entries,
            "issues": {
                "trainable_missing_from_optimizer": trainable_missing,
                "frozen_in_optimizer": frozen_in_optimizer,
                "duplicate_optimizer_parameter_count": int(
                    len(duplicate_optimizer_ids)
                ),
                "other_trainable_requires_review": other_trainable,
            },
        }
        self.manifest = {
            **manifest_core,
            "manifest_sha256": _canonical_hash(manifest_core),
            "optimizer_membership_pass": bool(
                not trainable_missing
                and not frozen_in_optimizer
                and not duplicate_optimizer_ids
            ),
            "classification_review_required": bool(other_trainable),
        }


class MultiLossGradientAuditor:
    """Run sparse per-component gradient audits without changing normal grads."""

    def __init__(
        self,
        cfg: Any,
        modules: Sequence[Tuple[str, torch.nn.Module]],
        optimizer: torch.optim.Optimizer,
        monitor_manager: Any,
        device: torch.device,
    ) -> None:
        self.cfg = cfg
        self.monitor_manager = monitor_manager
        self.device = device
        audit_cfg = cfg.MONITOR.MULTI_LOSS_GRADIENT_AUDIT
        self.enabled = bool(cfg.MONITOR.ENABLE) and bool(audit_cfg.ENABLE)
        self.epochs = sorted({int(value) for value in audit_cfg.EPOCHS})
        if any(epoch <= 0 for epoch in self.epochs):
            raise ValueError(
                "MULTI_LOSS_GRADIENT_AUDIT.EPOCHS must contain positive 1-based epochs"
            )
        self.batches_per_epoch = max(1, int(audit_cfg.BATCHES_PER_EPOCH))
        self.include_pairwise = bool(audit_cfg.INCLUDE_PAIRWISE_AUX_COSINE)
        self.include_alignment = bool(audit_cfg.INCLUDE_OPTIMIZER_ALIGNMENT)
        self.allow_single_loss_smoke = bool(audit_cfg.ALLOW_SINGLE_LOSS_SMOKE)
        self.eps = float(audit_cfg.NORM_EPS)
        self.optimizer = optimizer
        self.registry = None
        self.parameter_group_manifest_sha256 = None
        self._epoch_counts: Dict[int, int] = defaultdict(int)
        self._records = []
        self._pending = None
        criterion = next(
            (module for owner, module in modules if str(owner) == "cls_criterion"),
            None,
        )
        active_auxiliary_count = 0
        if criterion is not None:
            main_loss = getattr(criterion, "main_loss", None)
            if (
                main_loss is not None
                and str(getattr(main_loss, "align_mode", "none")).lower() != "none"
                and float(getattr(main_loss, "align_weight", 0.0)) > 0.0
            ):
                active_auxiliary_count += 1
            for auxiliary in getattr(criterion, "aux_losses", ()):
                if float(getattr(auxiliary, "weight", 0.0)) > 0.0:
                    active_auxiliary_count += 1
        runtime_source_active = bool(
            active_auxiliary_count > 0 or self.allow_single_loss_smoke
        )
        self.monitor_manager.update_runtime_source_state(
            "multi_loss_gradient_audit",
            source_active=runtime_source_active,
        )
        if not self.enabled:
            return
        if not runtime_source_active:
            raise ValueError(
                "MULTI_LOSS_GRADIENT_AUDIT.ENABLE requires at least one active "
                "non-primary loss; checkpoint-only and CE-only runs must leave it disabled"
            )
        self.registry = ParameterBlockRegistry(
            modules,
            optimizer,
            overrides=list(audit_cfg.PARAMETER_BLOCK_OVERRIDES),
        )
        self.parameter_group_manifest_sha256 = self.registry.manifest[
            "manifest_sha256"
        ]
        self.monitor_manager.write_evidence(
            "multi_loss_parameter_group_manifest.json",
            self.registry.manifest,
        )

    def should_audit(self, *, epoch: int, batch_index: int, is_train: bool) -> bool:
        del batch_index
        return bool(
            self.enabled
            and is_train
            and int(epoch) in self.epochs
            and self._epoch_counts[int(epoch)] < self.batches_per_epoch
        )

    def _reduce(self, values: Iterable[float], *, op: str = "sum") -> list:
        result = torch.tensor(
            [float(value) for value in values],
            dtype=torch.float64,
            device=self.device,
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            reduce_op = (
                torch.distributed.ReduceOp.MAX
                if op == "max"
                else torch.distributed.ReduceOp.SUM
            )
            torch.distributed.all_reduce(result, op=reduce_op)
        return [float(value) for value in result.detach().cpu().tolist()]

    @staticmethod
    def _gradient_sq_and_dot(
        left: Sequence[Optional[torch.Tensor]],
        right: Sequence[Optional[torch.Tensor]],
        indices: Sequence[int],
    ) -> Tuple[float, float, float]:
        left_sq = 0.0
        right_sq = 0.0
        dot = 0.0
        for index in indices:
            left_value = left[index]
            right_value = right[index]
            if left_value is not None:
                left_sq += float(torch.sum(left_value * left_value).item())
            if right_value is not None:
                right_sq += float(torch.sum(right_value * right_value).item())
            if left_value is not None and right_value is not None:
                dot += float(torch.sum(left_value * right_value).item())
        return left_sq, right_sq, dot

    def _block_indices(self) -> Dict[str, list]:
        result: Dict[str, list] = {block: [] for block in PARAMETER_BLOCKS}
        for index, block in enumerate(self.registry.parameter_blocks):
            result[block].append(index)
        return result

    def prepare(
        self,
        loss_terms: Sequence[Any],
        *,
        targets: torch.Tensor,
        sample_ids: Optional[Sequence[str]],
        epoch: int,
        global_step: int,
    ) -> None:
        if not self.enabled:
            return
        if self._pending is not None:
            raise RuntimeError("multi-loss gradient audit already has a pending step")
        terms = [term for term in loss_terms if bool(getattr(term, "active", True))]
        if not terms:
            raise RuntimeError("audit milestone did not expose any active LossTerm")
        primary = [term for term in terms if str(term.role) == "primary"]
        if len(primary) != 1:
            raise RuntimeError(
                "multi-loss audit requires exactly one primary LossTerm, got {}".format(
                    [str(term.name) for term in primary]
                )
            )
        if len(terms) < 2 and not self.allow_single_loss_smoke:
            raise RuntimeError(
                "multi-loss gradient audit requires at least one auxiliary loss; "
                "set ALLOW_SINGLE_LOSS_SMOKE only for implementation validation"
            )
        names = [str(term.name) for term in terms]
        if len(names) != len(set(names)):
            raise RuntimeError("active LossTerm names are not unique: {}".format(names))

        gradients: Dict[str, Tuple[Optional[torch.Tensor], ...]] = {}
        term_meta = {}
        for term in terms:
            raw = term.raw_tensor
            if not torch.is_tensor(raw) or raw.numel() != 1:
                raise TypeError(
                    "LossTerm '{}' raw_tensor must be a scalar tensor".format(term.name)
                )
            if not bool(torch.isfinite(raw.detach()).all().item()):
                raise FloatingPointError(
                    "LossTerm '{}' is non-finite at audit milestone".format(term.name)
                )
            if not raw.requires_grad:
                raw_gradients = tuple(None for _ in self.registry.parameters)
            else:
                raw_gradients = torch.autograd.grad(
                    raw,
                    self.registry.parameters,
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=True,
                )
            copied = []
            for gradient in raw_gradients:
                copied.append(
                    None
                    if gradient is None
                    else gradient.detach().float().cpu().clone()
                )
            gradients[str(term.name)] = tuple(copied)
            term_meta[str(term.name)] = {
                "name": str(term.name),
                "role": str(term.role),
                "weight": float(term.weight),
                "raw_value": float(raw.detach().item()),
                "weighted_value": float(raw.detach().item()) * float(term.weight),
            }

        block_indices = self._block_indices()
        block_metrics = []
        invalid_reasons = []
        norms: Dict[Tuple[str, str], float] = {}
        block_statuses: Dict[Tuple[str, str], str] = {}
        for term in terms:
            name = str(term.name)
            values = gradients[name]
            for block, indices in block_indices.items():
                tensor_count = len(indices)
                present = sum(values[index] is not None for index in indices)
                nonzero = sum(
                    values[index] is not None
                    and float(values[index].norm().item()) > self.eps
                    for index in indices
                )
                finite = sum(
                    values[index] is not None
                    and bool(torch.isfinite(values[index]).all().item())
                    for index in indices
                )
                local_sq = sum(
                    float(torch.sum(values[index] * values[index]).item())
                    for index in indices
                    if values[index] is not None
                )
                global_sq, global_present, global_nonzero, global_finite = self._reduce(
                    [local_sq, present, nonzero, finite]
                )
                raw_norm = math.sqrt(max(global_sq, 0.0))
                norms[(name, block)] = raw_norm
                if global_present == 0:
                    status = "not_applicable"
                    reason = "loss_has_no_computational_path_to_parameter_block"
                elif global_finite < global_present:
                    status = "invalid"
                    reason = "nonfinite_gradient_tensor"
                    invalid_reasons.append(f"{name}:{block}:nonfinite")
                elif raw_norm <= self.eps:
                    status = "observed_zero_gradient"
                    reason = "computational_path_present_but_gradient_is_zero"
                else:
                    status = "observed"
                    reason = None
                block_statuses[(name, block)] = status
                block_metrics.append({
                    "loss_name": name,
                    "loss_role": str(term.role),
                    "parameter_block": block,
                    "status": status,
                    "not_applicable_reason": reason,
                    "grad_norm_raw": raw_norm if status != "not_applicable" else None,
                    "grad_norm_weighted": (
                        abs(float(term.weight)) * raw_norm
                        if status != "not_applicable"
                        else None
                    ),
                    "parameter_tensor_count": int(tensor_count),
                    "gradient_tensor_count_rank_sum": int(round(global_present)),
                    "nonzero_gradient_tensor_count_rank_sum": int(round(global_nonzero)),
                    "unused_parameter_tensor_count_rank_sum": int(
                        round(len(indices) * self._world_size() - global_present)
                    ),
                    "finite_ratio": (
                        float(global_finite) / float(global_present)
                        if global_present > 0
                        else None
                    ),
                })

        primary_name = str(primary[0].name)
        comparisons = []
        for term in terms:
            name = str(term.name)
            if name == primary_name:
                continue
            for block, indices in block_indices.items():
                left_sq, right_sq, dot = self._gradient_sq_and_dot(
                    gradients[primary_name], gradients[name], indices
                )
                left_sq, right_sq, dot = self._reduce([left_sq, right_sq, dot])
                left_norm = math.sqrt(max(left_sq, 0.0))
                right_norm = math.sqrt(max(right_sq, 0.0))
                left_status = block_statuses[(primary_name, block)]
                right_status = block_statuses[(name, block)]
                if left_status == "not_applicable" or left_norm <= self.eps:
                    status = "not_applicable"
                    reason = "primary_gradient_norm_is_zero_or_missing"
                    cosine = None
                    ratio = None
                    cosine_status = "not_applicable"
                    ratio_status = "not_applicable"
                elif right_status == "not_applicable":
                    status = "not_applicable"
                    reason = "auxiliary_has_no_computational_path_to_parameter_block"
                    cosine = None
                    ratio = None
                    cosine_status = "not_applicable"
                    ratio_status = "not_applicable"
                elif right_norm <= self.eps:
                    status = "partially_observed"
                    reason = "auxiliary_gradient_is_observed_zero"
                    cosine = None
                    ratio = 0.0
                    cosine_status = "not_applicable"
                    ratio_status = "observed"
                else:
                    status = "observed"
                    reason = None
                    cosine = dot / max(left_norm * right_norm, self.eps)
                    ratio = abs(float(term.weight)) * right_norm / max(
                        left_norm, self.eps
                    )
                    cosine_status = "observed"
                    ratio_status = "observed"
                comparisons.append({
                    "left_loss": primary_name,
                    "right_loss": name,
                    "parameter_block": block,
                    "status": status,
                    "not_applicable_reason": reason,
                    "grad_cosine": cosine,
                    "grad_cosine_status": cosine_status,
                    "weighted_grad_norm_ratio_vs_primary": ratio,
                    "weighted_grad_norm_ratio_status": ratio_status,
                })

        pairwise = []
        auxiliary = [term for term in terms if str(term.name) != primary_name]
        if self.include_pairwise:
            for left_index, left_term in enumerate(auxiliary):
                for right_term in auxiliary[left_index + 1:]:
                    left_name = str(left_term.name)
                    right_name = str(right_term.name)
                    for block, indices in block_indices.items():
                        left_sq, right_sq, dot = self._gradient_sq_and_dot(
                            gradients[left_name], gradients[right_name], indices
                        )
                        left_sq, right_sq, dot = self._reduce(
                            [left_sq, right_sq, dot]
                        )
                        denominator = math.sqrt(max(left_sq * right_sq, 0.0))
                        pairwise.append({
                            "left_loss": left_name,
                            "right_loss": right_name,
                            "parameter_block": block,
                            "status": (
                                "observed" if denominator > self.eps else "not_applicable"
                            ),
                            "not_applicable_reason": (
                                None
                                if denominator > self.eps
                                else "one_or_both_gradient_norms_are_zero_or_missing"
                            ),
                            "grad_cosine": (
                                dot / denominator if denominator > self.eps else None
                            ),
                        })

        cancellation = []
        for block, indices in block_indices.items():
            local_combined_sq = 0.0
            for index in indices:
                combined = None
                for term in terms:
                    gradient = gradients[str(term.name)][index]
                    if gradient is None:
                        continue
                    contribution = gradient * float(term.weight)
                    combined = contribution if combined is None else combined + contribution
                if combined is not None:
                    local_combined_sq += float(torch.sum(combined * combined).item())
            combined_sq = self._reduce([local_combined_sq])[0]
            numerator = math.sqrt(max(combined_sq, 0.0))
            denominator = sum(
                abs(float(term.weight)) * norms[(str(term.name), block)]
                for term in terms
            )
            cancellation.append({
                "parameter_block": block,
                "status": "observed" if denominator > self.eps else "not_applicable",
                "not_applicable_reason": (
                    None if denominator > self.eps else "all_component_gradients_are_zero_or_missing"
                ),
                "combined_gradient_norm": numerator if denominator > self.eps else None,
                "sum_component_gradient_norms": denominator if denominator > self.eps else None,
                "gradient_cancellation_ratio": (
                    numerator / max(denominator, self.eps)
                    if denominator > self.eps
                    else None
                ),
            })

        target_list = [int(value) for value in targets.detach().cpu().tolist()]
        sample_list = (
            [str(value) for value in sample_ids]
            if sample_ids is not None
            else []
        )
        if not sample_list:
            invalid_reasons.append("calibration_batch_sample_ids_missing")
        elif len(sample_list) != len(target_list):
            invalid_reasons.append("calibration_batch_sample_id_count_mismatch")
        identity_payload = {
            "sample_ids": sample_list,
            "targets": target_list,
        }
        initialization_path_text = str(
            getattr(self.cfg.SOLVER, "INIT_TRAINABLE_CHECKPOINT", "")
        ).strip()
        initialization_path = (
            Path(initialization_path_text).expanduser().resolve()
            if initialization_path_text
            else None
        )
        record = {
            "format": "multi_loss_gradient_audit_record_v1",
            "training_performed": True,
            "optimizer_created": True,
            "epoch": int(epoch),
            "global_step": int(global_step),
            "calibration_batch_index_in_epoch": int(self._epoch_counts[int(epoch)]),
            "calibration_batch_manifest_hash": _canonical_hash(identity_payload),
            "sample_identity_status": "observed" if sample_list else "targets_only",
            "sample_count": int(len(target_list)),
            "class_support": int(len(set(target_list))),
            "training_seed": int(self.cfg.SEED) if self.cfg.SEED is not None else None,
            "data_order_seed": int(self.cfg.SEED) if self.cfg.SEED is not None else None,
            "run_id": getattr(self.monitor_manager, "run_id", None),
            "session_id": getattr(self.monitor_manager, "session_id", None),
            "stage2_checkpoint_cell_id": getattr(self.monitor_manager, "cell_id", None),
            "git_commit": (
                (getattr(self.monitor_manager, "git_identity", None) or {}).get("commit")
            ),
            "initialization_checkpoint": {
                "configured": bool(initialization_path_text),
                "path": initialization_path_text or None,
                "sha256": (
                    _file_sha256(initialization_path)
                    if initialization_path is not None and initialization_path.is_file()
                    else None
                ),
            },
            "world_size": int(self._world_size()),
            "gradient_aggregation": "rank_concatenated_scalar_sufficient_statistics",
            "parameter_group_manifest_sha256": self.parameter_group_manifest_sha256,
            "loss_terms": [term_meta[str(term.name)] for term in terms],
            "primary_loss_name": primary_name,
            "block_metrics": block_metrics,
            "primary_auxiliary_comparisons": comparisons,
            "auxiliary_pairwise_comparisons": pairwise,
            "combined_block_metrics": cancellation,
            "optimizer_alignment_status": (
                "pending" if self.include_alignment else "not_requested"
            ),
            "validation_status": "invalid" if invalid_reasons else "pending_optimizer_step",
            "validation_failure_reasons": invalid_reasons,
        }
        before = None
        if self.include_alignment:
            before = tuple(
                parameter.detach().float().cpu().clone()
                for parameter in self.registry.parameters
            )
        self._pending = {
            "record": record,
            "gradients": gradients,
            "before": before,
            "block_indices": block_indices,
            "terms": terms,
        }
        self._epoch_counts[int(epoch)] += 1

    def finalize_optimizer_step(self) -> None:
        if self._pending is None:
            return
        pending = self._pending
        record = pending["record"]
        if self.include_alignment:
            alignments = []
            before = pending["before"]
            for term in pending["terms"]:
                name = str(term.name)
                gradients = pending["gradients"][name]
                for block, indices in pending["block_indices"].items():
                    local_update_sq = 0.0
                    local_gradient_sq = 0.0
                    local_dot = 0.0
                    for index in indices:
                        gradient = gradients[index]
                        if gradient is None:
                            continue
                        update = (
                            self.registry.parameters[index].detach().float().cpu()
                            - before[index]
                        )
                        local_update_sq += float(torch.sum(update * update).item())
                        local_gradient_sq += float(torch.sum(gradient * gradient).item())
                        local_dot += float(torch.sum(update * (-gradient)).item())
                    update_sq, gradient_sq, dot = self._reduce(
                        [local_update_sq, local_gradient_sq, local_dot]
                    )
                    denominator = math.sqrt(max(update_sq * gradient_sq, 0.0))
                    alignments.append({
                        "loss_name": name,
                        "parameter_block": block,
                        "status": (
                            "observed" if denominator > self.eps else "not_applicable"
                        ),
                        "not_applicable_reason": (
                            None
                            if denominator > self.eps
                            else "parameter_update_or_loss_gradient_is_zero_or_missing"
                        ),
                        "optimizer_descent_alignment": (
                            dot / denominator if denominator > self.eps else None
                        ),
                        "parameter_update_norm": (
                            math.sqrt(max(update_sq, 0.0))
                            if denominator > self.eps
                            else None
                        ),
                    })
            record["optimizer_alignments"] = alignments
            record["optimizer_alignment_status"] = "observed"
        if record["validation_status"] != "invalid":
            record["validation_status"] = "valid"
        self._records.append(record)
        self.monitor_manager.set_context(
            stage="train",
            epoch=int(record["epoch"]),
            global_step=int(record["global_step"]),
        )
        self.monitor_manager.append_evidence_jsonl(
            "multi_loss_gradient_audit.jsonl", record
        )
        self.monitor_manager.record_event(
            "multi_loss_gradient_audit",
            {
                "epoch": int(record["epoch"]),
                "global_step": int(record["global_step"]),
                "loss_count": int(len(record["loss_terms"])),
                "parameter_block_count": int(len(PARAMETER_BLOCKS)),
                "validation_status": str(record["validation_status"]),
            },
        )
        self._pending = None

    def abort_pending(self, reason: str) -> None:
        if self._pending is None:
            return
        record = self._pending["record"]
        record["validation_status"] = "invalid"
        record.setdefault("validation_failure_reasons", []).append(str(reason))
        self._records.append(record)
        self.monitor_manager.append_evidence_jsonl(
            "multi_loss_gradient_audit.jsonl", record
        )
        self._pending = None

    def clear_loss_terms(self, criterion: Any) -> None:
        if hasattr(criterion, "clear_last_loss_terms"):
            criterion.clear_last_loss_terms()

    def finalize(self, *, status: str) -> None:
        if not self.enabled:
            return
        if self._pending is not None:
            self.abort_pending("training_ended_before_optimizer_alignment")
        numeric: Dict[str, list] = defaultdict(list)
        status_counts: Dict[str, int] = defaultdict(int)
        milestone_summaries = []
        for record in self._records:
            milestone_values = {}
            for row in record.get("block_metrics", []):
                status_counts[
                    "block_metrics.{}".format(row.get("status", "missing"))
                ] += 1
                for field in ("grad_norm_raw", "grad_norm_weighted"):
                    value = row.get(field)
                    if value is not None and math.isfinite(float(value)):
                        key = "{}.{}.{}".format(
                            row["loss_name"], row["parameter_block"], field
                        )
                        numeric[key].append(float(value))
                        milestone_values[key] = float(value)
            for row in record.get("primary_auxiliary_comparisons", []):
                status_counts[
                    "primary_auxiliary.{}".format(row.get("status", "missing"))
                ] += 1
                for field in ("grad_cosine", "weighted_grad_norm_ratio_vs_primary"):
                    value = row.get(field)
                    if value is not None and math.isfinite(float(value)):
                        key = "{}.vs_{}.{}.{}".format(
                            row["right_loss"],
                            row["left_loss"],
                            row["parameter_block"],
                            field,
                        )
                        numeric[key].append(float(value))
                        milestone_values[key] = float(value)
            for row in record.get("auxiliary_pairwise_comparisons", []):
                status_counts[
                    "auxiliary_pairwise.{}".format(row.get("status", "missing"))
                ] += 1
                value = row.get("grad_cosine")
                if value is not None and math.isfinite(float(value)):
                    key = "{}.vs_{}.{}.grad_cosine".format(
                        row["left_loss"],
                        row["right_loss"],
                        row["parameter_block"],
                    )
                    numeric[key].append(float(value))
                    milestone_values[key] = float(value)
            for row in record.get("combined_block_metrics", []):
                status_counts[
                    "combined.{}".format(row.get("status", "missing"))
                ] += 1
                value = row.get("gradient_cancellation_ratio")
                if value is not None and math.isfinite(float(value)):
                    key = "combined.{}.gradient_cancellation_ratio".format(
                        row["parameter_block"]
                    )
                    numeric[key].append(float(value))
                    milestone_values[key] = float(value)
            for row in record.get("optimizer_alignments", []):
                status_counts[
                    "optimizer_alignment.{}".format(row.get("status", "missing"))
                ] += 1
                value = row.get("optimizer_descent_alignment")
                if value is not None and math.isfinite(float(value)):
                    key = "{}.{}.optimizer_descent_alignment".format(
                        row["loss_name"], row["parameter_block"]
                    )
                    numeric[key].append(float(value))
                    milestone_values[key] = float(value)
            milestone_summaries.append({
                "epoch": int(record["epoch"]),
                "global_step": int(record["global_step"]),
                "calibration_batch_manifest_hash": record[
                    "calibration_batch_manifest_hash"
                ],
                "validation_status": str(record["validation_status"]),
                "metrics": dict(sorted(milestone_values.items())),
            })
        aggregates = {
            key: {
                "mean": float(sum(values) / len(values)),
                "min": float(min(values)),
                "max": float(max(values)),
                "n": int(len(values)),
            }
            for key, values in sorted(numeric.items())
            if values
        }
        expected_epochs = [
            epoch
            for epoch in self.epochs
            if epoch <= int(self.cfg.SOLVER.TOTAL_EPOCH)
        ]
        expected_count = int(len(expected_epochs) * self.batches_per_epoch)
        observed_pairs = [
            [int(record["epoch"]), int(record["calibration_batch_index_in_epoch"])]
            for record in self._records
        ]
        failures = []
        if str(status) != "completed":
            failures.append("training_status_is_not_completed")
        if len(self._records) != expected_count:
            failures.append("predeclared_milestone_record_count_mismatch")
        if any(record.get("validation_status") != "valid" for record in self._records):
            failures.append("one_or_more_audit_records_are_invalid")
        if not bool(self.registry.manifest["optimizer_membership_pass"]):
            failures.append("parameter_optimizer_membership_contract_failed")
        if bool(self.registry.manifest["classification_review_required"]):
            failures.append("other_trainable_parameter_block_requires_review")
        validation = {
            "format": "multi_loss_gradient_audit_validation_v1",
            "requested": True,
            "source_active": True,
            "training_performed": True,
            "optimizer_created": True,
            "expected_epochs": expected_epochs,
            "batches_per_epoch": int(self.batches_per_epoch),
            "expected_record_count": expected_count,
            "observed_record_count": int(len(self._records)),
            "observed_epoch_batch_pairs": observed_pairs,
            "parameter_group_manifest_sha256": self.parameter_group_manifest_sha256,
            "failure_reasons": sorted(set(failures)),
            "overall_pass": not failures,
            "report_complete_claim_allowed": not failures,
        }
        summary = {
            "format": "multi_loss_gradient_audit_summary_v1",
            "status": str(status),
            "training_seed": int(self.cfg.SEED) if self.cfg.SEED is not None else None,
            "configured_epochs": list(self.epochs),
            "batches_per_epoch": int(self.batches_per_epoch),
            "record_count": int(len(self._records)),
            "parameter_group_manifest_sha256": self.parameter_group_manifest_sha256,
            "milestones": milestone_summaries,
            "aggregates": aggregates,
            "status_counts": dict(sorted(status_counts.items())),
            "validation_overall_pass": bool(validation["overall_pass"]),
        }
        self.monitor_manager.write_evidence(
            "multi_loss_gradient_audit_summary.json", summary
        )
        self.monitor_manager.write_evidence(
            "multi_loss_gradient_audit_validation.json", validation
        )

    @staticmethod
    def _world_size() -> int:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return int(torch.distributed.get_world_size())
        return 1
