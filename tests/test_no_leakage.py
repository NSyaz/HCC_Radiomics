from __future__ import annotations

import numpy as np
import pandas as pd

from hcc_radiomics.preprocessing import fit_transform_preprocessor


def test_preprocessing_imputer_is_fitted_only_on_training_data() -> None:
    X_train = pd.DataFrame({"feature": [1.0, np.nan, 3.0], "constant": [1.0, 1.0, 1.0]})
    X_validation = pd.DataFrame({"feature": [1000.0], "constant": [1.0]})
    X_test = pd.DataFrame({"feature": [2000.0, np.nan], "constant": [1.0, 1.0]})

    preprocessor, X_train_t, X_validation_t, X_test_t, names = fit_transform_preprocessor(
        X_train, X_validation, X_test
    )

    assert preprocessor.named_steps["imputer"].statistics_[0] == 2.0
    assert names == ["feature"]
    assert X_train_t.shape == (3, 1)
    assert X_validation_t.shape == (1, 1)
    assert X_test_t.shape == (2, 1)
