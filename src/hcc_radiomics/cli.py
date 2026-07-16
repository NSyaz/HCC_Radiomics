from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yaml

from hcc_radiomics.data import (
    infer_metadata_columns,
    load_table,
    split_class_distributions,
    split_dataset,
    validate_dataset,
)
from hcc_radiomics.evaluation import (
    classification_metrics,
    save_confusion_matrix,
    save_probability_curves,
)
from hcc_radiomics.feature_selection import apply_bpso, safe_select_k_best
from hcc_radiomics.models import build_svm
from hcc_radiomics.preprocessing import fit_transform_preprocessor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run HCC radiomics classification")
    parser.add_argument("--config", type=Path, help="Optional YAML configuration file")
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--sheet-name")
    parser.add_argument("--target-column", default="Stage")
    parser.add_argument("--group-column")
    parser.add_argument("--metadata-column", action="append", default=[])
    parser.add_argument(
        "--method",
        choices=["svm", "svm_kbest", "svm_bpso", "svm_kbest_bpso"],
        default="svm_bpso",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/run_001"))
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--k-best", type=int, default=150)
    parser.add_argument("--bpso-particles", type=int, default=30)
    parser.add_argument("--bpso-iterations", type=int, default=10)
    parser.add_argument("--bpso-alpha", type=float, default=0.9)
    parser.add_argument("--cv-folds", type=int, default=3)
    return parser


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def merge_config(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    defaults = vars(build_parser().parse_args([]))
    merged = defaults | config
    explicit = {key: value for key, value in vars(args).items() if value != defaults.get(key)}
    merged.update(explicit)
    merged.pop("config", None)
    if merged.get("data_path") is None:
        raise ValueError("--data-path is required unless supplied by --config")
    return merged


def run_experiment(config: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(output_dir)
    logging.info("Starting HCC radiomics experiment")

    df = load_table(config["data_path"], config.get("sheet_name"))
    target_column = config["target_column"]
    metadata_columns = list(config.get("metadata_column") or config.get("metadata_columns") or [])
    inferred_metadata = infer_metadata_columns(df.columns, target_column)
    for column in inferred_metadata:
        if column not in metadata_columns:
            metadata_columns.append(column)

    group_column = config.get("group_column")
    if group_column and group_column not in metadata_columns:
        metadata_columns.append(group_column)

    X, y, feature_names, dataset_summary = validate_dataset(
        df, target_column, metadata_columns=metadata_columns
    )
    groups = df[group_column] if group_column else None
    split = split_dataset(
        y,
        test_size=float(config["test_size"]),
        validation_size=float(config["validation_size"]),
        random_state=int(config["random_state"]),
        groups=groups,
    )
    distributions = split_class_distributions(y, split)

    X_train = X.loc[split.train_idx]
    X_validation = X.loc[split.validation_idx]
    X_test = X.loc[split.test_idx]
    y_train = y.loc[split.train_idx]
    y_validation = y.loc[split.validation_idx]
    y_test = y.loc[split.test_idx]

    preprocessor, X_train_t, X_validation_t, X_test_t, transformed_names = fit_transform_preprocessor(
        X_train, X_validation, X_test
    )

    method = config["method"]
    selection_steps: list[dict[str, Any]] = []
    selected_names = transformed_names

    if method in {"svm_kbest", "svm_kbest_bpso"}:
        k_result = safe_select_k_best(
            X_train_t,
            y_train,
            X_validation_t,
            X_test_t,
            transformed_names,
            int(config["k_best"]),
        )
        X_train_t, X_validation_t, X_test_t = (
            k_result.X_train,
            k_result.X_validation,
            k_result.X_test,
        )
        selected_names = k_result.feature_names
        selection_steps.append(k_result.metadata)

    if method in {"svm_bpso", "svm_kbest_bpso"}:
        bpso_result = apply_bpso(
            X_train_t,
            y_train,
            X_validation_t,
            X_test_t,
            selected_names,
            n_particles=int(config["bpso_particles"]),
            iterations=int(config["bpso_iterations"]),
            alpha=float(config["bpso_alpha"]),
            cv_folds=int(config["cv_folds"]),
            random_state=int(config["random_state"]),
        )
        X_train_t, X_validation_t, X_test_t = (
            bpso_result.X_train,
            bpso_result.X_validation,
            bpso_result.X_test,
        )
        selected_names = bpso_result.feature_names
        selection_steps.append(bpso_result.metadata)

    model = build_svm(random_state=int(config["random_state"]))
    model.fit(X_train_t, y_train)

    labels = sorted(y.unique(), key=str)
    validation_result = _evaluate_split(model, X_validation_t, y_validation, labels)
    test_result = _evaluate_split(model, X_test_t, y_test, labels)

    config_to_save = _serialisable_config(config)
    artifacts = {
        "config": config_to_save,
        "dataset_summary": dataset_summary,
        "split_class_distributions": distributions,
        "method": method,
        "selection_steps": selection_steps,
        "selected_feature_count": len(selected_names),
        "selected_features": selected_names,
        "validation_metrics": validation_result["metrics"],
        "test_metrics": test_result["metrics"],
    }

    _write_json(output_dir / "config_used.json", config_to_save)
    _write_json(output_dir / "dataset_summary.json", dataset_summary)
    _write_json(output_dir / "split_class_distributions.json", distributions)
    _write_json(output_dir / "metrics.json", artifacts)
    (output_dir / "selected_features.txt").write_text("\n".join(selected_names) + "\n", encoding="utf-8")

    predictions = pd.DataFrame(
        {
            "row_index": y_test.index.astype(str),
            "y_true": y_test.astype(str).to_numpy(),
            "y_pred": pd.Series(test_result["y_pred"]).astype(str).to_numpy(),
        }
    )
    proba = test_result["y_proba"]
    if proba is not None:
        for idx, label in enumerate(labels):
            predictions[f"probability_{label}"] = proba[:, idx]
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)

    save_confusion_matrix(
        test_result["metrics"]["confusion_matrix"], labels, output_dir / "confusion_matrix.png"
    )
    if proba is not None:
        save_probability_curves(y_test, proba, labels, output_dir)

    joblib.dump({"preprocessor": preprocessor, "model": model, "features": selected_names}, output_dir / "model.joblib")
    logging.info("Experiment completed")
    return artifacts


def _evaluate_split(model, X, y, labels: list[Any]) -> dict[str, Any]:
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X) if hasattr(model, "predict_proba") else None
    return {
        "y_pred": y_pred,
        "y_proba": y_proba,
        "metrics": classification_metrics(y, y_pred, y_proba, labels),
    }


def _setup_logging(output_dir: Path) -> None:
    logging.basicConfig(
        filename=output_dir / "run.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )


def _serialisable_config(config: dict[str, Any]) -> dict[str, Any]:
    serialised = {}
    for key, value in config.items():
        serialised[key] = str(value) if isinstance(value, Path) else value
    return serialised


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = merge_config(args, load_config(args.config))
    run_experiment(config)


if __name__ == "__main__":
    main()
