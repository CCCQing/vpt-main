from collections import defaultdict
from typing import Dict, Mapping, Sequence


class PredictionTransitionAccumulator:
    def __init__(self, reference_method: str, target_method: str) -> None:
        self.reference_method = str(reference_method)
        self.target_method = str(target_method)
        self.counts: Dict[str, int] = defaultdict(int)

    def update(
        self,
        reference_predictions: Sequence[int],
        target_predictions: Sequence[int],
        targets: Sequence[int],
    ) -> None:
        reference_predictions = list(reference_predictions)
        target_predictions = list(target_predictions)
        targets = list(targets)
        if not (
            len(reference_predictions) == len(target_predictions) == len(targets)
        ):
            raise ValueError("paired prediction arrays must have equal lengths")
        for reference_prediction, target_prediction, target in zip(
            reference_predictions, target_predictions, targets
        ):
            reference_correct = int(reference_prediction) == int(target)
            target_correct = int(target_prediction) == int(target)
            self.counts["sample_count"] += 1
            self.counts["reference_correct_count"] += int(reference_correct)
            self.counts["target_correct_count"] += int(target_correct)
            self.counts["both_correct_count"] += int(
                reference_correct and target_correct
            )
            self.counts["corrected_count"] += int(
                not reference_correct and target_correct
            )
            self.counts["regressed_count"] += int(
                reference_correct and not target_correct
            )
            both_wrong = not reference_correct and not target_correct
            self.counts["both_wrong_count"] += int(both_wrong)
            same_prediction = int(reference_prediction) == int(target_prediction)
            self.counts["prediction_agreement_count"] += int(same_prediction)
            self.counts["both_wrong_same_prediction_count"] += int(
                both_wrong and same_prediction
            )

    def finalize(self) -> Dict[str, object]:
        sample_count = int(self.counts.get("sample_count", 0))
        result: Dict[str, object] = {
            "reference_method": self.reference_method,
            "target_method": self.target_method,
            **{name: int(value) for name, value in sorted(self.counts.items())},
        }
        if sample_count <= 0:
            result["valid"] = False
            return result
        result["valid"] = True
        for name in (
            "both_correct_count",
            "corrected_count",
            "regressed_count",
            "both_wrong_count",
            "prediction_agreement_count",
            "both_wrong_same_prediction_count",
        ):
            result[name.replace("_count", "_rate")] = float(
                self.counts.get(name, 0) / sample_count
            )
        reference_accuracy = float(
            self.counts.get("reference_correct_count", 0) / sample_count
        )
        target_accuracy = float(
            self.counts.get("target_correct_count", 0) / sample_count
        )
        result.update({
            "reference_accuracy": reference_accuracy,
            "target_accuracy": target_accuracy,
            "accuracy_delta": target_accuracy - reference_accuracy,
            "net_correction_count": int(
                self.counts.get("corrected_count", 0)
                - self.counts.get("regressed_count", 0)
            ),
            "net_correction_rate": float(
                (
                    self.counts.get("corrected_count", 0)
                    - self.counts.get("regressed_count", 0)
                )
                / sample_count
            ),
            "both_wrong_same_prediction_conditional_rate": float(
                self.counts.get("both_wrong_same_prediction_count", 0)
                / max(1, self.counts.get("both_wrong_count", 0))
            ),
        })
        return result


def correctness_pattern_counts(
    predictions_by_method: Mapping[str, Sequence[int]],
    targets: Sequence[int],
) -> Dict[str, object]:
    methods = sorted(str(name) for name in predictions_by_method)
    target_values = list(targets)
    predictions = {
        name: list(predictions_by_method[name]) for name in methods
    }
    if any(len(values) != len(target_values) for values in predictions.values()):
        raise ValueError("all prediction arrays must match the target array length")
    counts: Dict[str, int] = defaultdict(int)
    for sample_index, target in enumerate(target_values):
        pattern = "|".join(
            f"{name}={int(int(predictions[name][sample_index]) == int(target))}"
            for name in methods
        )
        counts[pattern] += 1
    sample_count = len(target_values)
    return {
        "methods": methods,
        "sample_count": sample_count,
        "counts": dict(sorted(counts.items())),
        "rates": {
            name: float(value / sample_count) if sample_count else None
            for name, value in sorted(counts.items())
        },
    }
