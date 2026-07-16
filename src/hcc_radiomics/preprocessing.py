from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_preprocessor() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("variance", VarianceThreshold(threshold=0.0)),
            ("scaler", StandardScaler()),
        ]
    )


def fit_transform_preprocessor(
    X_train: pd.DataFrame,
    X_validation: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[Pipeline, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    X_validation_t = preprocessor.transform(X_validation)
    X_test_t = preprocessor.transform(X_test)
    names = transformed_feature_names(list(X_train.columns), preprocessor)
    return preprocessor, X_train_t, X_validation_t, X_test_t, names


def transformed_feature_names(input_features: Sequence[str], preprocessor: Pipeline) -> list[str]:
    variance = preprocessor.named_steps["variance"]
    support = variance.get_support()
    return [name for name, keep in zip(input_features, support, strict=True) if keep]
