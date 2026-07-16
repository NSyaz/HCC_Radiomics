from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from hcc_radiomics.cli import run_experiment
from hcc_radiomics.experiment import ExperimentBundle, load_experiment


def _synthetic_dataframe() -> pd.DataFrame:
    X, y = make_classification(
        n_samples=60,
        n_features=12,
        n_informative=6,
        n_redundant=0,
        random_state=123,
    )
    df = pd.DataFrame(X, columns=[f"radiomics_{i}" for i in range(X.shape[1])])
    df["row_label"] = [f"synthetic_{i:03d}" for i in range(len(df))]
    df["Stage"] = y
    return df


@pytest.mark.parametrize(
    "method",
    ["dummy", "svm", "svm_kbest", "svm_bpso", "svm_kbest_bpso"],
)
def test_experiment_bundle_round_trip_for_all_methods(tmp_path: Path, method: str) -> None:
    df = _synthetic_dataframe()
    data_path = tmp_path / "synthetic.csv"
    output_dir = tmp_path / method
    df.to_csv(data_path, index=False)

    result = run_experiment(
        {
            "data_path": data_path,
            "sheet_name": None,
            "target_column": "Stage",
            "group_column": None,
            "metadata_column": ["row_label"],
            "method": method,
            "output_dir": output_dir,
            "random_state": 42,
            "test_size": 0.2,
            "validation_size": 0.2,
            "k_best": 5,
            "bpso_particles": 3,
            "bpso_iterations": 1,
            "bpso_alpha": 0.9,
            "cv_folds": 2,
        }
    )

    bundle = result["bundle"]
    assert isinstance(bundle, ExperimentBundle)
    loaded = load_experiment(output_dir / "model.joblib")
    heldout = df.loc[[int(idx) for idx in result["test_indices"]]]

    expected_predictions = bundle.predict(heldout)
    loaded_predictions = loaded.predict(heldout)
    np.testing.assert_array_equal(loaded_predictions, expected_predictions)

    expected_probabilities = bundle.predict_proba(heldout)
    loaded_probabilities = loaded.predict_proba(heldout)
    np.testing.assert_allclose(loaded_probabilities, expected_probabilities)

    shuffled = heldout.loc[:, list(reversed(heldout.columns))]
    np.testing.assert_array_equal(loaded.predict(shuffled), expected_predictions)
    np.testing.assert_allclose(loaded.predict_proba(shuffled), expected_probabilities)

    missing_feature = loaded.input_feature_names[0]
    with pytest.raises(ValueError, match="missing required radiomics feature"):
        loaded.predict(heldout.drop(columns=[missing_feature]))


def test_bundle_preserves_selection_state(tmp_path: Path) -> None:
    df = _synthetic_dataframe()
    data_path = tmp_path / "synthetic.csv"
    output_dir = tmp_path / "selected"
    df.to_csv(data_path, index=False)

    result = run_experiment(
        {
            "data_path": data_path,
            "sheet_name": None,
            "target_column": "Stage",
            "group_column": None,
            "metadata_column": ["row_label"],
            "method": "svm_kbest_bpso",
            "output_dir": output_dir,
            "random_state": 42,
            "test_size": 0.2,
            "validation_size": 0.2,
            "k_best": 5,
            "bpso_particles": 3,
            "bpso_iterations": 1,
            "bpso_alpha": 0.9,
            "cv_folds": 2,
        }
    )

    loaded = load_experiment(output_dir / "model.joblib")

    assert loaded.select_k_best is not None
    assert loaded.bpso_mask is not None
    assert loaded.bpso_input_feature_names is not None
    assert len(loaded.bpso_mask) == len(loaded.bpso_input_feature_names)
    assert loaded.selected_feature_names == result["selected_features"]
