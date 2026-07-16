from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.datasets import make_classification

from hcc_radiomics.cli import run_experiment
from hcc_radiomics.evaluation import classification_metrics


def _write_synthetic_dataset(path: Path, *, classes: int = 2) -> None:
    X, y = make_classification(
        n_samples=60,
        n_features=12,
        n_informative=6,
        n_redundant=0,
        n_classes=classes,
        n_clusters_per_class=1,
        random_state=42,
    )
    df = pd.DataFrame(X, columns=[f"radiomics_{i}" for i in range(X.shape[1])])
    df["patient_id"] = [f"synthetic_{i:03d}" for i in range(len(df))]
    df["Stage"] = y
    df.to_csv(path, index=False)


def test_end_to_end_pipeline_creates_result_files(tmp_path: Path) -> None:
    data_path = tmp_path / "synthetic.csv"
    output_dir = tmp_path / "outputs"
    _write_synthetic_dataset(data_path)

    result = run_experiment(
        {
            "data_path": data_path,
            "sheet_name": None,
            "target_column": "Stage",
            "group_column": None,
            "metadata_column": ["patient_id"],
            "method": "svm_kbest_bpso",
            "output_dir": output_dir,
            "random_state": 42,
            "test_size": 0.2,
            "validation_size": 0.2,
            "k_best": 6,
            "bpso_particles": 4,
            "bpso_iterations": 2,
            "bpso_alpha": 0.9,
            "cv_folds": 2,
        }
    )

    assert result["selected_feature_count"] > 0
    for filename in [
        "config_used.json",
        "dataset_summary.json",
        "split_class_distributions.json",
        "metrics.json",
        "selected_features.txt",
        "test_predictions.csv",
        "confusion_matrix.png",
        "roc_curve.png",
        "precision_recall_curve.png",
        "model.joblib",
        "run.log",
    ]:
        assert (output_dir / filename).exists()

    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["dataset_summary"]["target_column"] == "Stage"
    assert metrics["test_metrics"]["accuracy"] >= 0.0


def test_multiclass_metrics_are_supported() -> None:
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 1, 0, 2, 2]
    y_proba = pd.DataFrame(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.4, 0.4],
            [0.7, 0.2, 0.1],
            [0.2, 0.3, 0.5],
            [0.1, 0.2, 0.7],
        ]
    ).to_numpy()

    metrics = classification_metrics(y_true, y_pred, y_proba, labels=[0, 1, 2])

    assert "specificity_macro" in metrics
    assert "roc_auc_ovr_macro" in metrics


def test_no_hard_coded_absolute_paths_in_active_code() -> None:
    root = Path(__file__).resolve().parents[1]
    active_paths = [root / "src", root / "scripts", root / "tests"]
    forbidden = [
        "/" + "Users" + "/",
        "C:" + "\\\\" + "Users" + "\\\\",
        "Documents" + "/" + "MASTER" + "/" + "THESIS",
    ]
    for base in active_paths:
        for path in base.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for needle in forbidden:
                assert needle not in text, f"{needle} found in {path}"
