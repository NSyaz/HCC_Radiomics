from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Iterable
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


STAGE_TARGET_MAPPING = {
    0: "HCC groups/stages 1-2",
    1: "HCC groups/stages 3-4",
}

DEFAULT_METADATA_NAMES = {
    "id",
    "patient_id",
    "patientid",
    "patient",
    "subject_id",
    "subjectid",
    "case_id",
    "caseid",
    "record_id",
    "recordid",
    "mrn",
    "accession",
    "name",
    "dob",
    "birth_date",
    "date_of_birth",
}


@dataclass(frozen=True)
class SplitResult:
    train_idx: pd.Index
    validation_idx: pd.Index
    test_idx: pd.Index


def load_table(path: str | Path, sheet_name: str | int | None = None) -> pd.DataFrame:
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    suffix = data_path.suffix.lower()
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        return pd.read_excel(data_path, sheet_name=sheet_name or 0)
    if suffix == ".csv":
        return pd.read_csv(data_path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(data_path, sep="\t")
    raise ValueError(f"Unsupported data format: {suffix}")


def normalise_column_name(name: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def infer_metadata_columns(columns: Iterable[str], target_column: str) -> list[str]:
    inferred: list[str] = []
    for column in columns:
        if column == target_column:
            continue
        normalised = normalise_column_name(column)
        if normalised in DEFAULT_METADATA_NAMES or normalised.endswith("_id"):
            inferred.append(column)
    return inferred


def validate_dataset(
    df: pd.DataFrame,
    target_column: str,
    metadata_columns: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[str], dict]:
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' is missing")

    if df[target_column].isna().any():
        missing = int(df[target_column].isna().sum())
        raise ValueError(f"Target column '{target_column}' has {missing} missing labels")

    y = df[target_column]
    class_count = y.nunique(dropna=False)
    if class_count < 2:
        raise ValueError("Classification requires at least two target classes")

    metadata = list(dict.fromkeys(metadata_columns or []))
    unknown_metadata = [col for col in metadata if col not in df.columns]
    if unknown_metadata:
        raise ValueError(f"Metadata columns not found: {unknown_metadata}")

    drop_columns = [target_column, *metadata]
    X = df.drop(columns=drop_columns)
    numeric_X = X.apply(pd.to_numeric, errors="coerce")
    non_numeric = [
        col for col in X.columns if numeric_X[col].isna().all() and X[col].notna().any()
    ]
    if non_numeric:
        numeric_X = numeric_X.drop(columns=non_numeric)

    numeric_X = numeric_X.replace([np.inf, -np.inf], np.nan)
    if numeric_X.shape[1] == 0:
        raise ValueError("No numeric radiomics features remain after excluding target/metadata")

    _warn_high_dimensional_shape(n_samples=numeric_X.shape[0], n_features=numeric_X.shape[1])

    summary = {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "target_column": target_column,
        "target_classes": [str(value) for value in sorted(y.unique(), key=str)],
        "target_distribution": {
            str(key): int(value) for key, value in y.value_counts(dropna=False).items()
        },
        "metadata_columns_excluded": metadata,
        "non_numeric_columns_excluded": non_numeric,
        "n_numeric_features": int(numeric_X.shape[1]),
        "feature_sample_ratio": float(numeric_X.shape[1] / numeric_X.shape[0]),
        "stage_target_mapping": {str(key): value for key, value in STAGE_TARGET_MAPPING.items()},
        "missing_feature_cells": int(numeric_X.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }
    return numeric_X, y, list(numeric_X.columns), summary


def _warn_high_dimensional_shape(n_samples: int, n_features: int) -> None:
    if n_features >= 10 * n_samples:
        warnings.warn(
            (
                f"Dataset has {n_features} features and {n_samples} samples "
                f"(feature/sample ratio {n_features / n_samples:.1f}). "
                "This is an extreme high-dimensional, low-sample-size setting with high overfitting risk."
            ),
            RuntimeWarning,
            stacklevel=2,
        )
    elif n_features > n_samples:
        warnings.warn(
            (
                f"Dataset has more features ({n_features}) than samples ({n_samples}). "
                "This high-dimensional setting has elevated overfitting risk."
            ),
            RuntimeWarning,
            stacklevel=2,
        )


def _class_distribution(y: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in y.value_counts(dropna=False).items()}


def split_dataset(
    y: pd.Series,
    *,
    test_size: float = 0.2,
    validation_size: float = 0.2,
    random_state: int = 42,
    groups: pd.Series | None = None,
) -> SplitResult:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    if not 0 < validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1")
    if test_size + validation_size >= 1:
        raise ValueError("test_size + validation_size must be less than 1")

    if groups is not None:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_val_pos, test_pos = next(splitter.split(np.zeros(len(y)), y, groups))
        train_val_idx = y.index[train_val_pos]
        test_idx = y.index[test_pos]

        remaining_groups = groups.loc[train_val_idx]
        relative_val_size = validation_size / (1.0 - test_size)
        val_splitter = GroupShuffleSplit(
            n_splits=1, test_size=relative_val_size, random_state=random_state + 1
        )
        train_pos, val_pos = next(
            val_splitter.split(np.zeros(len(train_val_idx)), y.loc[train_val_idx], remaining_groups)
        )
        train_idx = train_val_idx[train_pos]
        validation_idx = train_val_idx[val_pos]
    else:
        _validate_stratified_split_feasibility(y, test_size, validation_size)
        try:
            train_val_idx, test_idx = train_test_split(
                y.index,
                test_size=test_size,
                random_state=random_state,
                stratify=y,
            )
        except ValueError as exc:
            raise ValueError(
                "Unable to create a stratified test split. Reduce test_size, collect more "
                "samples per class, or review the target labels."
            ) from exc
        y_train_val = y.loc[train_val_idx]
        relative_val_size = validation_size / (1.0 - test_size)
        try:
            train_idx, validation_idx = train_test_split(
                train_val_idx,
                test_size=relative_val_size,
                random_state=random_state + 1,
                stratify=y_train_val,
            )
        except ValueError as exc:
            raise ValueError(
                "Unable to create a stratified validation split from the training pool. "
                "Reduce validation_size, collect more samples per class, or review the target labels."
            ) from exc

    result = SplitResult(
        train_idx=pd.Index(train_idx),
        validation_idx=pd.Index(validation_idx),
        test_idx=pd.Index(test_idx),
    )
    assert_no_overlap(result)
    return result


def assert_no_overlap(split: SplitResult) -> None:
    train = set(split.train_idx)
    validation = set(split.validation_idx)
    test = set(split.test_idx)
    if train & validation or train & test or validation & test:
        raise ValueError("Train, validation, and test splits overlap")


def _validate_stratified_split_feasibility(
    y: pd.Series,
    test_size: float,
    validation_size: float,
) -> None:
    counts = y.value_counts()
    if counts.min() < 3:
        raise ValueError(
            "Stratified train/validation/test splitting requires at least 3 samples "
            f"per class; observed class counts are {_class_distribution(y)}."
        )

    n_classes = counts.shape[0]
    n_samples = len(y)
    test_n = math.ceil(test_size * n_samples)
    train_val_n = n_samples - test_n
    relative_val_size = validation_size / (1.0 - test_size)
    validation_n = math.ceil(relative_val_size * train_val_n)
    train_n = train_val_n - validation_n
    if min(test_n, validation_n, train_n) < n_classes:
        raise ValueError(
            "Requested split sizes are too small to preserve every class in train, "
            "validation, and test splits. Increase split sizes or use more samples."
        )


def split_class_distributions(y: pd.Series, split: SplitResult) -> dict[str, dict[str, int]]:
    return {
        "train": _class_distribution(y.loc[split.train_idx]),
        "validation": _class_distribution(y.loc[split.validation_idx]),
        "test": _class_distribution(y.loc[split.test_idx]),
    }
