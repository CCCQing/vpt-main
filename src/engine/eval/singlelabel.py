#!/usr/bin/env python3

"""Functions for computing metrics. all metrics has range of 0-1"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score, roc_auc_score
)


def accuracy(y_probs, y_true):
    # y_prob: (num_images, num_classes)
    y_preds = np.argmax(y_probs, axis=1)
    accuracy = accuracy_score(y_true, y_preds)
    error = 1.0 - accuracy
    return accuracy, error


def top_n_accuracy(y_probs, truths, n=1):
    # y_prob: (num_images, num_classes)
    # truth: (num_images, num_classes) multi/one-hot encoding
    best_n = np.argsort(y_probs, axis=1)[:, -n:]
    if isinstance(truths, np.ndarray) and truths.shape == y_probs.shape:
        ts = np.argmax(truths, axis=1)
    else:
        # a list of GT class idx
        ts = truths

    num_input = y_probs.shape[0]
    successes = 0
    for i in range(num_input):
        if ts[i] in best_n[i, :]:
            successes += 1
    return float(successes) / num_input


def compute_acc_auc(y_probs, y_true_ids):
    onehot_tgts = np.zeros_like(y_probs)
    for idx, t in enumerate(y_true_ids):
        onehot_tgts[idx, t] = 1.

    num_classes = y_probs.shape[1]
    if num_classes == 2:
        top1, _ = accuracy(y_probs, y_true_ids)
        # so precision can set all to 2
        try:
            auc = roc_auc_score(onehot_tgts, y_probs, average='macro')
        except ValueError as e:
            print(f"value error encountered {e}, set auc sccore to -1.")
            auc = -1
        return {"top1": top1, "rocauc": auc}

    top1, _ = accuracy(y_probs, y_true_ids)
    k = min([5, num_classes])  # if number of labels < 5, use the total class
    top5 = top_n_accuracy(y_probs, y_true_ids, k)
    return {"top1": top1, f"top{k}": top5}

def _per_class_accuracy(preds: np.ndarray, targets: np.ndarray, class_ids: np.ndarray) -> float:
    """
    Compute mean per-class accuracy over the provided class id set.
    """
    accs = []
    for cid in class_ids:
        mask = targets == cid
        if mask.sum() == 0:
            # 若该类在当前 targets 中没有出现，则跳过，保持与 ZSL 文献一致（只对出现的类求均值）
            continue
        accs.append(float((preds[mask] == targets[mask]).mean()))
    if not accs:
        return 0.0
    return float(np.mean(accs))


def compute_zsl_gzsl_metrics(
    scores: np.ndarray,
    targets: np.ndarray,
    seen_classes: np.ndarray,
    unseen_classes: np.ndarray,
) -> dict:
    """
    Compute ZSL / GZSL metrics used in zero-shot literature.

    返回一个包含：
      - zsl_unseen : 仅在 unseen 类别上的 per-class top1（ZSL 场景）
      - gzsl_seen  : GZSL 场景下 seen 类的 per-class top1（S）
      - gzsl_unseen: GZSL 场景下 unseen 类的 per-class top1（U）
      - gzsl_h     : 调和平均 H = 2SU / (S+U+1e-8)
    """
    preds = np.asarray(scores).argmax(axis=1)
    targets = np.asarray(targets).astype(np.int64)
    seen_classes = np.asarray(seen_classes).astype(np.int64)
    unseen_classes = np.asarray(unseen_classes).astype(np.int64)

    zsl_unseen = _per_class_accuracy(preds, targets, unseen_classes)
    gzsl_seen = _per_class_accuracy(preds, targets, seen_classes)
    gzsl_unseen = _per_class_accuracy(preds, targets, unseen_classes)

    gzsl_h = 0.0
    if (gzsl_seen + gzsl_unseen) > 0:
        gzsl_h = 2 * gzsl_seen * gzsl_unseen / (gzsl_seen + gzsl_unseen + 1e-8)

    return {
        "zsl_unseen": zsl_unseen,
        "gzsl_seen": gzsl_seen,
        "gzsl_unseen": gzsl_unseen,
        "gzsl_h": gzsl_h,
    }


def topks_correct(preds, labels, ks):
    """Computes the number of top-k correct predictions for each k."""
    assert preds.size(0) == labels.size(
        0
    ), "Batch dim of predictions and labels must match"
    # Find the top max_k predictions for each sample
    _top_max_k_vals, top_max_k_inds = torch.topk(
        preds, max(ks), dim=1, largest=True, sorted=True
    )
    # (batch_size, max_k) -> (max_k, batch_size)
    top_max_k_inds = top_max_k_inds.t()
    # (batch_size, ) -> (max_k, batch_size)
    rep_max_k_labels = labels.view(1, -1).expand_as(top_max_k_inds)
    # (i, j) = 1 if top i-th prediction for the j-th sample is correct
    top_max_k_correct = top_max_k_inds.eq(rep_max_k_labels)
    # Compute the number of topk correct predictions for each k
    topks_correct = [
        top_max_k_correct[:k, :].reshape(-1).float().sum() for k in ks
    ]
    return topks_correct


def topk_errors(preds, labels, ks):
    """Computes the top-k error for each k."""
    if int(labels.min()) < 0:  # has ignore
        keep_ids = np.where(labels.cpu() >= 0)[0]
        preds = preds[keep_ids, :]
        labels = labels[keep_ids]

    num_topks_correct = topks_correct(preds, labels, ks)
    return [(1.0 - x / preds.size(0)) for x in num_topks_correct]


def topk_accuracies(preds, labels, ks):
    """Computes the top-k accuracy for each k."""
    num_topks_correct = topks_correct(preds, labels, ks)
    return [(x / preds.size(0)) for x in num_topks_correct]

