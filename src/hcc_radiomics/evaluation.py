from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize


def classification_metrics(
    y_true,
    y_pred,
    y_proba: np.ndarray | None,
    labels: list[Any],
) -> dict[str, Any]:
    average = "binary" if len(labels) == 2 else "macro"
    pos_label = labels[1] if len(labels) == 2 else None
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, average=average, pos_label=pos_label, zero_division=0)
        ),
        "recall_sensitivity": float(
            recall_score(y_true, y_pred, average=average, pos_label=pos_label, zero_division=0)
        ),
        "f1": float(f1_score(y_true, y_pred, average=average, pos_label=pos_label, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }

    cm = np.asarray(metrics["confusion_matrix"])
    if len(labels) == 2 and cm.shape == (2, 2):
        tn, fp, _fn, _tp = cm.ravel()
        metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) else 0.0
    else:
        metrics["specificity_macro"] = _macro_specificity(cm)

    if y_proba is not None:
        try:
            if len(labels) == 2:
                y_binary = (np.asarray(y_true) == labels[1]).astype(int)
                metrics["roc_auc"] = float(roc_auc_score(y_binary, y_proba[:, 1]))
                metrics["average_precision"] = float(
                    average_precision_score(y_binary, y_proba[:, 1])
                )
            else:
                y_bin = label_binarize(y_true, classes=labels)
                metrics["roc_auc_ovr_macro"] = float(
                    roc_auc_score(y_bin, y_proba, average="macro", multi_class="ovr")
                )
                metrics["average_precision_macro"] = float(
                    average_precision_score(y_bin, y_proba, average="macro")
                )
        except ValueError as exc:
            metrics["probability_metric_warning"] = str(exc)
    return metrics


def _macro_specificity(cm: np.ndarray) -> float:
    values = []
    total = cm.sum()
    for idx in range(cm.shape[0]):
        tp = cm[idx, idx]
        fp = cm[:, idx].sum() - tp
        fn = cm[idx, :].sum() - tp
        tn = total - tp - fp - fn
        denom = tn + fp
        values.append(float(tn / denom) if denom else 0.0)
    return float(np.mean(values)) if values else 0.0


def save_confusion_matrix(cm: list[list[int]], labels: list[Any], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    matrix = np.asarray(cm)
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set_xticks(range(len(labels)), labels=[str(label) for label in labels], rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels=[str(label) for label in labels])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_probability_curves(y_true, y_proba: np.ndarray, labels: list[Any], output_dir: Path) -> None:
    if len(labels) == 2:
        y_binary = (np.asarray(y_true) == labels[1]).astype(int)
        fpr, tpr, _ = roc_curve(y_binary, y_proba[:, 1])
        precision, recall, _ = precision_recall_curve(y_binary, y_proba[:, 1])
        _plot_curve(fpr, tpr, "False Positive Rate", "True Positive Rate", output_dir / "roc_curve.png")
        _plot_curve(recall, precision, "Recall", "Precision", output_dir / "precision_recall_curve.png")
        return

    y_bin = label_binarize(y_true, classes=labels)
    fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
    fig_pr, ax_pr = plt.subplots(figsize=(6, 5))
    for idx, label in enumerate(labels):
        fpr, tpr, _ = roc_curve(y_bin[:, idx], y_proba[:, idx])
        precision, recall, _ = precision_recall_curve(y_bin[:, idx], y_proba[:, idx])
        ax_roc.plot(fpr, tpr, label=str(label))
        ax_pr.plot(recall, precision, label=str(label))
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend()
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.legend()
    fig_roc.tight_layout()
    fig_pr.tight_layout()
    fig_roc.savefig(output_dir / "roc_curve.png", dpi=150)
    fig_pr.savefig(output_dir / "precision_recall_curve.png", dpi=150)
    plt.close(fig_roc)
    plt.close(fig_pr)


def _plot_curve(x, y, xlabel: str, ylabel: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(x, y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
