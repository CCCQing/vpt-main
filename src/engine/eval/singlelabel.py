#!/usr/bin/env python3

"""Single-label evaluation metrics in local label space."""

import numpy as np
from sklearn.metrics import accuracy_score


def accuracy(y_probs, y_true):
    y_preds = np.argmax(y_probs, axis=1)
    acc = accuracy_score(y_true, y_preds)
    err = 1.0 - acc
    return acc, err


def compute_top1(y_probs, y_true_ids):
    top1, _ = accuracy(y_probs, y_true_ids)
    return {"top1": top1}


def compute_per_class_top1(scores: np.ndarray, targets: np.ndarray) -> float:
    scores = np.asarray(scores)
    targets = np.asarray(targets).astype(np.int64)
    preds = scores.argmax(axis=1)
    class_ids = np.unique(targets)
    accs = []
    for cid in class_ids:
        mask = targets == cid
        if mask.sum() == 0:
            continue
        accs.append(float((preds[mask] == targets[mask]).mean()))
    if not accs:
        return 0.0
    return float(np.mean(accs))


def harmonic_mean(a: float, b: float) -> float:
    if (a + b) <= 0:
        return 0.0
    return float(2.0 * a * b / (a + b + 1e-8))
