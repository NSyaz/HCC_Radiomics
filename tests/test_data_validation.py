from __future__ import annotations

import pandas as pd
import pytest

from hcc_radiomics.data import split_dataset, validate_dataset


def test_target_column_is_excluded_from_features() -> None:
    df = pd.DataFrame(
        {
            "patient_id": ["p1", "p2", "p3", "p4"],
            "Stage": [0, 1, 0, 1],
            "feature_a": [1.0, 2.0, 3.0, 4.0],
            "feature_b": [4.0, 3.0, 2.0, 1.0],
        }
    )

    X, y, feature_names, summary = validate_dataset(
        df, "Stage", metadata_columns=["patient_id"]
    )

    assert "Stage" not in X.columns
    assert "patient_id" not in X.columns
    assert feature_names == ["feature_a", "feature_b"]
    assert y.tolist() == [0, 1, 0, 1]
    assert summary["metadata_columns_excluded"] == ["patient_id"]


def test_missing_or_single_class_target_is_rejected() -> None:
    df = pd.DataFrame({"Stage": [1, 1, 1], "feature": [0.1, 0.2, 0.3]})

    with pytest.raises(ValueError, match="at least two"):
        validate_dataset(df, "Stage")


def test_split_indices_do_not_overlap() -> None:
    y = pd.Series([0, 1] * 20)

    split = split_dataset(y, test_size=0.2, validation_size=0.2, random_state=42)

    assert set(split.train_idx).isdisjoint(split.validation_idx)
    assert set(split.train_idx).isdisjoint(split.test_idx)
    assert set(split.validation_idx).isdisjoint(split.test_idx)
