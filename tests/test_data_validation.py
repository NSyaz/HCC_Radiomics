from __future__ import annotations

import pandas as pd
import pytest

from hcc_radiomics.data import STAGE_TARGET_MAPPING, split_dataset, validate_dataset


def test_target_column_is_excluded_from_features() -> None:
    df = pd.DataFrame(
        {
            "row_label": ["s1", "s2", "s3", "s4"],
            "Stage": [0, 1, 0, 1],
            "feature_a": [1.0, 2.0, 3.0, 4.0],
            "feature_b": [4.0, 3.0, 2.0, 1.0],
        }
    )

    X, y, feature_names, summary = validate_dataset(
        df, "Stage", metadata_columns=["row_label"]
    )

    assert "Stage" not in X.columns
    assert "row_label" not in X.columns
    assert feature_names == ["feature_a", "feature_b"]
    assert y.tolist() == [0, 1, 0, 1]
    assert summary["metadata_columns_excluded"] == ["row_label"]
    assert summary["stage_target_mapping"] == {
        "0": "HCC groups/stages 1-2",
        "1": "HCC groups/stages 3-4",
    }


def test_target_mapping_constant_remains_binary() -> None:
    assert STAGE_TARGET_MAPPING == {
        0: "HCC groups/stages 1-2",
        1: "HCC groups/stages 3-4",
    }


def test_missing_or_single_class_target_is_rejected() -> None:
    df = pd.DataFrame({"Stage": [1, 1, 1], "feature": [0.1, 0.2, 0.3]})

    with pytest.raises(ValueError, match="Stage must contain both labels"):
        validate_dataset(df, "Stage")


def test_stage_accepts_valid_integer_labels() -> None:
    df = pd.DataFrame({"Stage": [0, 1, 0, 1], "feature": [0.1, 0.2, 0.3, 0.4]})

    _X, y, _features, _summary = validate_dataset(df, "Stage")

    assert y.tolist() == [0, 1, 0, 1]


def test_stage_accepts_lossless_float_labels() -> None:
    df = pd.DataFrame({"Stage": [0.0, 1.0, 0.0, 1.0], "feature": [0.1, 0.2, 0.3, 0.4]})

    _X, y, _features, _summary = validate_dataset(df, "Stage")

    assert y.tolist() == [0, 1, 0, 1]
    assert str(y.dtype).startswith("int")


@pytest.mark.parametrize(
    "labels",
    [
        [1, 2, 1, 2],
        [0, 1, 2, 1],
        ["early", "late", "early", "late"],
        [False, True, False, True],
    ],
)
def test_stage_rejects_unsupported_label_sets(labels: list[object]) -> None:
    df = pd.DataFrame({"Stage": labels, "feature": [0.1, 0.2, 0.3, 0.4]})

    with pytest.raises(ValueError, match="Stage"):
        validate_dataset(df, "Stage")


def test_stage_rejects_missing_labels() -> None:
    df = pd.DataFrame({"Stage": [0, 1, None, 1], "feature": [0.1, 0.2, 0.3, 0.4]})

    with pytest.raises(ValueError, match="missing labels"):
        validate_dataset(df, "Stage")


def test_split_indices_do_not_overlap() -> None:
    y = pd.Series([0, 1] * 20)

    split = split_dataset(y, test_size=0.2, validation_size=0.2, random_state=42)

    assert set(split.train_idx).isdisjoint(split.validation_idx)
    assert set(split.train_idx).isdisjoint(split.test_idx)
    assert set(split.validation_idx).isdisjoint(split.test_idx)


def test_high_dimensional_warning_is_emitted() -> None:
    df = pd.DataFrame({f"f{i}": range(8) for i in range(10)})
    df["Stage"] = [0, 1] * 4

    with pytest.warns(RuntimeWarning, match="more features"):
        validate_dataset(df, "Stage")


def test_extreme_feature_sample_warning_is_emitted() -> None:
    df = pd.DataFrame({f"f{i}": range(8) for i in range(80)})
    df["Stage"] = [0, 1] * 4

    with pytest.warns(RuntimeWarning, match="extreme high-dimensional"):
        validate_dataset(df, "Stage")


def test_impossible_stratified_split_raises_clear_error() -> None:
    y = pd.Series([0, 0, 1])

    with pytest.raises(ValueError, match="at least 3 samples per class"):
        split_dataset(y, test_size=0.2, validation_size=0.2, random_state=42)
