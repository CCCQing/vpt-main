from __future__ import annotations

import hashlib
from typing import Any, Dict, Sequence

import numpy as np


class EpochPredictionTransitionTracker:
    def __init__(self) -> None:
        self._states: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _identity(sample_ids: Sequence[str]) -> str:
        digest = hashlib.sha256()
        for sample_id in sample_ids:
            digest.update(str(sample_id).encode("utf-8"))
            digest.update(b"\0")
        return digest.hexdigest()

    @staticmethod
    def _prepare(
        sample_ids: Sequence[str],
        predictions: Any,
        targets_local: Any,
    ):
        ids = np.asarray([str(item) for item in sample_ids], dtype=object)
        prediction = np.asarray(predictions, dtype=np.int64).reshape(-1)
        target = np.asarray(targets_local, dtype=np.int64).reshape(-1)
        if ids.size != prediction.size or ids.size != target.size:
            raise ValueError("sample ids, predictions, and targets must have identical lengths")
        if len(set(ids.tolist())) != int(ids.size):
            raise ValueError("epoch prediction transition requires unique sample ids")
        order = np.argsort(ids.astype(str), kind="stable")
        return ids[order], prediction[order], target[order]

    def update(
        self,
        *,
        epoch: int,
        split: str,
        sample_ids: Sequence[str],
        predictions: Any,
        targets_local: Any,
        candidate_global_ids: Sequence[int],
    ) -> Dict[str, Any]:
        split = str(split)
        epoch = int(epoch)
        ids, prediction, target = self._prepare(
            sample_ids, predictions, targets_local
        )
        candidates = np.asarray(list(candidate_global_ids), dtype=np.int64)
        if target.size and (target.min() < 0 or target.max() >= candidates.size):
            raise ValueError("targets fall outside the candidate class space")
        correct = prediction == target
        identity = self._identity(ids.tolist())
        previous = self._states.get(split)
        invalid_reason = None
        if previous is None:
            invalid_reason = "previous_epoch_state_unavailable"
        elif epoch != int(previous["epoch"]) + 1:
            invalid_reason = "nonconsecutive_epoch"
        elif identity != str(previous["sample_identity_sha256"]):
            invalid_reason = "sample_identity_mismatch"
        elif not np.array_equal(target, previous["targets_local"]):
            invalid_reason = "target_identity_mismatch"
        elif not np.array_equal(candidates, previous["candidate_global_ids"]):
            invalid_reason = "candidate_identity_mismatch"

        if invalid_reason is not None:
            ever_correct = correct.copy()
            payload = {
                "valid": False,
                "status": invalid_reason,
                "epoch": epoch,
                "previous_epoch": None if previous is None else int(previous["epoch"]),
                "split": split,
                "sample_count": int(target.size),
                "sample_identity_sha256": identity,
                "candidate_global_ids": candidates,
                "summary": {
                    "transition_available": 0.0,
                    "sample_count": float(target.size),
                },
                "arrays": {},
            }
        else:
            previous_prediction = previous["predictions"]
            previous_correct = previous["correct"]
            previous_ever_correct = previous["ever_correct"]
            flip = prediction != previous_prediction
            correction = (~previous_correct) & correct
            regression = previous_correct & (~correct)
            persistent_correct = previous_correct & correct
            persistent_wrong = (~previous_correct) & (~correct)
            ever_correct_then_wrong = previous_ever_correct & (~correct)
            ever_correct = previous_ever_correct | correct
            count = max(1, int(target.size))
            summary = {
                "transition_available": 1.0,
                "sample_count": float(target.size),
                "prediction_flip_rate": float(flip.sum() / count),
                "correction_rate": float(correction.sum() / count),
                "regression_rate": float(regression.sum() / count),
                "net_correction_rate": float((correction.sum() - regression.sum()) / count),
                "persistent_correct_rate": float(persistent_correct.sum() / count),
                "persistent_wrong_rate": float(persistent_wrong.sum() / count),
                "ever_correct_then_wrong_rate": float(ever_correct_then_wrong.sum() / count),
            }
            class_count = int(candidates.size)
            support = np.bincount(target, minlength=class_count).astype(np.int64)
            arrays = {
                "candidate_global_ids": candidates,
                "support": support,
                "prediction_flip_count": np.bincount(target[flip], minlength=class_count).astype(np.int64),
                "correction_count": np.bincount(target[correction], minlength=class_count).astype(np.int64),
                "regression_count": np.bincount(target[regression], minlength=class_count).astype(np.int64),
                "persistent_correct_count": np.bincount(target[persistent_correct], minlength=class_count).astype(np.int64),
                "persistent_wrong_count": np.bincount(target[persistent_wrong], minlength=class_count).astype(np.int64),
                "ever_correct_then_wrong_count": np.bincount(
                    target[ever_correct_then_wrong], minlength=class_count
                ).astype(np.int64),
            }
            payload = {
                "valid": True,
                "status": "paired_consecutive_epochs",
                "epoch": epoch,
                "previous_epoch": int(previous["epoch"]),
                "split": split,
                "sample_count": int(target.size),
                "sample_identity_sha256": identity,
                "candidate_global_ids": candidates,
                "summary": summary,
                "arrays": arrays,
            }

        self._states[split] = {
            "epoch": epoch,
            "sample_identity_sha256": identity,
            "targets_local": target.copy(),
            "predictions": prediction.copy(),
            "correct": correct.copy(),
            "ever_correct": ever_correct.copy(),
            "candidate_global_ids": candidates.copy(),
        }
        return payload

    def clear(self) -> None:
        self._states.clear()
