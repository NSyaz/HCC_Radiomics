from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification

from hcc_radiomics.feature_selection import (
    BinaryPSOFeatureSelector,
    bpso_mask_cost,
    resolve_cv_folds,
    safe_select_k_best,
)


def test_select_k_best_clamps_to_available_features() -> None:
    X, y = make_classification(
        n_samples=30,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        random_state=7,
    )

    with pytest.warns(RuntimeWarning, match="exceeds"):
        result = safe_select_k_best(X, y, X, X, [f"f{i}" for i in range(5)], k=99)

    assert result.X_train.shape[1] == 5
    assert result.metadata["selected_k"] == 5


def test_bpso_mask_has_expected_length() -> None:
    X, y = make_classification(
        n_samples=36,
        n_features=8,
        n_informative=4,
        n_redundant=0,
        random_state=3,
    )

    selector = BinaryPSOFeatureSelector(
        n_particles=4,
        iterations=2,
        cv_folds=2,
        random_state=3,
    ).fit(X, y, [f"feature_{i}" for i in range(8)])

    assert selector.mask_.shape == (8,)
    assert len(selector.feature_names_) == int(selector.mask_.sum())
    assert selector.transform(X).shape[1] == int(selector.mask_.sum())


def test_all_zero_bpso_mask_gets_penalty() -> None:
    X, y = make_classification(
        n_samples=24,
        n_features=4,
        n_informative=2,
        n_redundant=0,
        random_state=11,
    )

    cost = bpso_mask_cost(np.zeros(4, dtype=int), X, y, zero_penalty=1.23)

    assert cost == 1.23


def test_impossible_cv_folds_raise_clear_error() -> None:
    with pytest.raises(ValueError, match="at least 2 training samples"):
        resolve_cv_folds(np.array([0, 0, 1]), requested_folds=3)


def test_cv_folds_are_reduced_with_warning() -> None:
    y = np.array([0, 0, 0, 1, 1, 1])

    with pytest.warns(RuntimeWarning, match="using cv_folds=3"):
        assert resolve_cv_folds(y, requested_folds=5) == 3
