from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch


def _component(name: str) -> str:
    lowered = str(name).lower()
    if "prompt" in lowered:
        return "prompt"
    if "semantic_token" in lowered or "semantic_project" in lowered:
        return "semantic_token"
    if "attention_mediation" in lowered:
        return "attention_mediation"
    if "graph_prob_prior" in lowered or lowered.startswith("cls_criterion."):
        return "loss_or_graph_prior"
    if "r_similarity_head" in lowered or "prototype_proj" in lowered or ".head" in lowered:
        return "classifier_head"
    return "backbone_or_other"


def _named_parameters(modules: Sequence[Tuple[str, torch.nn.Module]]) -> Dict[str, torch.nn.Parameter]:
    result: Dict[str, torch.nn.Parameter] = {}
    for prefix, module in modules:
        for name, parameter in module.named_parameters():
            full_name = f"{prefix}.{name}" if prefix else str(name)
            if full_name in result:
                raise ValueError(f"duplicate parameter name in monitoring guard: {full_name}")
            result[full_name] = parameter
    return result


def _finite_tensor(tensor: Optional[torch.Tensor]) -> bool:
    return tensor is None or bool(torch.isfinite(tensor.detach()).all().item())


def _fingerprint(parameter: torch.Tensor) -> Dict[str, float]:
    value = parameter.detach().float()
    return {
        "norm": float(value.norm().item()),
        "abs_sum": float(value.abs().sum().item()),
        "sum": float(value.sum().item()),
    }


class NumericalGuard:
    def __init__(self, modules: Sequence[Tuple[str, torch.nn.Module]]) -> None:
        self.parameters = _named_parameters(modules)
        self.first_success_recorded = False

    def check_forward(self, loss: Any, logits: Any) -> Optional[Dict[str, Any]]:
        failures = []
        if not torch.is_tensor(loss) or not _finite_tensor(loss):
            failures.append({"kind": "loss", "name": "loss"})
        if torch.is_tensor(logits) and not _finite_tensor(logits):
            bad = (~torch.isfinite(logits.detach())).nonzero(as_tuple=False)
            failures.append({
                "kind": "logits",
                "name": "logits",
                "nonfinite_count": int(bad.shape[0]),
                "first_indices": bad[:8].cpu().tolist(),
            })
        return {"phase": "forward", "failures": failures} if failures else None

    def check_gradients(self) -> Optional[Dict[str, Any]]:
        failures = []
        for name, parameter in self.parameters.items():
            if not parameter.requires_grad or parameter.grad is None:
                continue
            if not _finite_tensor(parameter.grad):
                failures.append({"kind": "gradient", "name": name, "component": _component(name)})
        return {"phase": "backward", "failures": failures} if failures else None

    def check_parameters(self) -> Optional[Dict[str, Any]]:
        failures = []
        for name, parameter in self.parameters.items():
            if not parameter.requires_grad:
                continue
            if not _finite_tensor(parameter):
                failures.append({"kind": "parameter", "name": name, "component": _component(name)})
        return {"phase": "optimizer_step", "failures": failures} if failures else None


class OptimizerSanity:
    def __init__(
        self,
        modules: Sequence[Tuple[str, torch.nn.Module]],
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self.parameters = _named_parameters(modules)
        self.optimizer = optimizer
        self._before = {
            name: _fingerprint(parameter)
            for name, parameter in self.parameters.items()
            if parameter.requires_grad
        }
        self._frozen_before = {
            name: _fingerprint(parameter)
            for name, parameter in self.parameters.items()
            if not parameter.requires_grad
        }

    def initialization_report(self) -> Dict[str, Any]:
        id_to_names: Dict[int, list] = defaultdict(list)
        for name, parameter in self.parameters.items():
            id_to_names[id(parameter)].append(name)
        optimizer_ids = []
        group_rows = []
        for group_index, group in enumerate(self.optimizer.param_groups):
            component_counts: Dict[str, int] = defaultdict(int)
            for parameter in group.get("params", []):
                parameter_id = id(parameter)
                optimizer_ids.append(parameter_id)
                matched = id_to_names.get(parameter_id, ["<unknown>"])
                for name in matched:
                    component_counts[_component(name)] += 1
            group_rows.append({
                "group_index": int(group_index),
                "lr": float(group.get("lr", 0.0)),
                "weight_decay": float(group.get("weight_decay", 0.0)),
                "parameter_count": int(len(group.get("params", []))),
                "component_parameter_counts": dict(sorted(component_counts.items())),
            })
        optimizer_id_set = set(optimizer_ids)
        trainable = {name for name, parameter in self.parameters.items() if parameter.requires_grad}
        missing = sorted(name for name in trainable if id(self.parameters[name]) not in optimizer_id_set)
        frozen_in_optimizer = sorted(
            name for name, parameter in self.parameters.items()
            if (not parameter.requires_grad) and id(parameter) in optimizer_id_set
        )
        duplicate_ids = sorted({parameter_id for parameter_id in optimizer_ids if optimizer_ids.count(parameter_id) > 1})
        duplicate_names = sorted({name for parameter_id in duplicate_ids for name in id_to_names.get(parameter_id, [])})
        component_counts: Dict[str, int] = defaultdict(int)
        for name in sorted(trainable):
            component_counts[_component(name)] += 1
        issue_names = {
            "missing_from_optimizer": missing,
            "frozen_in_optimizer": frozen_in_optimizer,
            "duplicate_optimizer_parameters": duplicate_names,
        }
        return {
            "phase": "initialization",
            "passed": not missing and not frozen_in_optimizer and not duplicate_names,
            "trainable_component_tensor_counts": dict(sorted(component_counts.items())),
            "optimizer_group_count": int(len(group_rows)),
            "optimizer_groups": group_rows,
            "issue_counts": {
                name: int(len(values))
                for name, values in issue_names.items()
            },
            "issues": {
                name: values
                for name, values in issue_names.items()
                if values
            },
        }

    def first_step_report(self) -> Dict[str, Any]:
        component_grad_sq: Dict[str, float] = defaultdict(float)
        component_nonzero: Dict[str, int] = defaultdict(int)
        component_total: Dict[str, int] = defaultdict(int)
        missing_gradients = []
        nonfinite_gradients = []
        unchanged_trainable = []
        changed_trainable = []
        component_changed: Dict[str, int] = defaultdict(int)
        for name, parameter in self.parameters.items():
            if not parameter.requires_grad:
                continue
            component = _component(name)
            component_total[component] += 1
            gradient = parameter.grad
            if gradient is None:
                missing_gradients.append(name)
            elif not _finite_tensor(gradient):
                nonfinite_gradients.append(name)
            else:
                norm = float(gradient.detach().float().norm().item())
                component_grad_sq[component] += norm * norm
                if norm > 0.0:
                    component_nonzero[component] += 1
            before = self._before[name]
            after = _fingerprint(parameter)
            delta = max(abs(after[key] - before[key]) for key in before)
            if delta > 1e-12:
                changed_trainable.append(name)
                component_changed[component] += 1
            else:
                unchanged_trainable.append(name)

        changed_frozen = []
        for name, before in self._frozen_before.items():
            after = _fingerprint(self.parameters[name])
            if max(abs(after[key] - before[key]) for key in before) > 1e-12:
                changed_frozen.append(name)
        component_rows = {
            component: {
                "gradient_norm": float(component_grad_sq[component] ** 0.5),
                "nonzero_gradient_tensor_count": int(component_nonzero[component]),
                "trainable_tensor_count": int(component_total[component]),
                "changed_parameter_count": int(component_changed[component]),
            }
            for component in sorted(component_total)
        }
        checks = {
            "all_gradients_present": not missing_gradients,
            "all_gradients_finite": not nonfinite_gradients,
            "all_trainable_components_have_nonzero_gradient": all(
                component_nonzero[component] > 0
                for component in component_total
            ),
            "any_trainable_parameter_updated": bool(changed_trainable),
            "all_frozen_parameters_unchanged": not changed_frozen,
        }
        issue_names = {
            "missing_gradients": sorted(missing_gradients),
            "nonfinite_gradients": sorted(nonfinite_gradients),
            "unchanged_trainable_parameters": sorted(unchanged_trainable),
            "changed_frozen_parameters": sorted(changed_frozen),
        }
        return {
            "phase": "first_backward_optimizer_step",
            "passed": all(checks.values()),
            "checks": checks,
            "changed_trainable_parameter_count": int(len(changed_trainable)),
            "components": component_rows,
            "issue_counts": {
                name: int(len(values))
                for name, values in issue_names.items()
            },
            "issues": {
                name: values
                for name, values in issue_names.items()
                if values
            },
        }

    @staticmethod
    def event_summary(report: Mapping[str, Any]) -> Dict[str, Any]:
        issue_counts = report.get("issue_counts", {})
        result = {
            "phase": str(report.get("phase", "unknown")),
            "passed": bool(report.get("passed", False)),
            "issue_count": int(sum(int(value) for value in issue_counts.values())),
        }
        checks = report.get("checks")
        if isinstance(checks, Mapping):
            result["checks"] = {str(name): bool(value) for name, value in checks.items()}
        return result


class PromptParameterTracker:
    def __init__(self, model: torch.nn.Module) -> None:
        self.parameters = {
            name: parameter
            for name, parameter in model.named_parameters()
            if "prompt" in name.lower() and parameter.requires_grad
        }
        self.initial = {
            name: parameter.detach().float().cpu().clone()
            for name, parameter in self.parameters.items()
        }

    @property
    def active(self) -> bool:
        return bool(self.parameters)

    def metrics(self) -> Dict[str, float]:
        if not self.parameters:
            return {}
        current_flat = []
        initial_flat = []
        grad_sq = 0.0
        for name, parameter in self.parameters.items():
            value = parameter.detach().float().cpu()
            current_flat.append(value.reshape(-1))
            initial_flat.append(self.initial[name].reshape(-1))
            if parameter.grad is not None and torch.isfinite(parameter.grad).all():
                grad_sq += float(parameter.grad.detach().float().norm().item()) ** 2
        current = torch.cat(current_flat)
        initial = torch.cat(initial_flat)
        delta = current - initial
        result = {
            "prompt_param_norm": float(current.norm().item()),
            "prompt_grad_norm": float(grad_sq ** 0.5),
            "prompt_relative_update": float(delta.norm().item() / max(float(initial.norm().item()), 1e-12)),
            "distance_from_initialization": float(delta.norm().item()),
            "prompt_parameter_tensor_count": float(len(self.parameters)),
        }
        token_matrices = []
        for parameter in self.parameters.values():
            value = parameter.detach().float()
            if value.dim() >= 2:
                token_matrices.append(value.reshape(-1, value.shape[-1]))
        if token_matrices:
            tokens = torch.cat(token_matrices, dim=0)
            normalized = torch.nn.functional.normalize(tokens, dim=-1)
            cosine = normalized @ normalized.t()
            mask = ~torch.eye(cosine.shape[0], dtype=torch.bool, device=cosine.device)
            offdiag = cosine[mask]
            centered = tokens - tokens.mean(dim=0, keepdim=True)
            singular = (
                torch.linalg.svdvals(centered)
                if hasattr(torch.linalg, "svdvals")
                else torch.svd(centered, some=False).S
            )
            prob = singular / singular.sum().clamp_min(1e-12)
            effective_rank = torch.exp(-(prob * prob.clamp_min(1e-12).log()).sum())
            result.update({
                "token_pair_cosine_mean": float(offdiag.mean().item()) if offdiag.numel() else 0.0,
                "token_pair_cosine_max": float(offdiag.max().item()) if offdiag.numel() else 0.0,
                "prompt_effective_rank": float(effective_rank.item()),
            })
        return result
