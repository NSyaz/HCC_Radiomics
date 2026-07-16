from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


BUNDLE_FORMAT_VERSION = "1.0"


@dataclass
class ExperimentBundle:
    method: str
    target_column: str
    metadata_columns: list[str]
    input_feature_names: list[str]
    transformed_feature_names: list[str]
    selected_feature_names: list[str]
    class_labels: list[Any]
    preprocessor: Any
    classifier: Any
    config: dict[str, Any]
    select_k_best: Any | None = None
    bpso_mask: np.ndarray | None = None
    bpso_input_feature_names: list[str] | None = None
    model_format_version: str = BUNDLE_FORMAT_VERSION
    package_name: str = "hcc-radiomics"
    extra_metadata: dict[str, Any] = field(default_factory=dict)

    def predict(self, dataframe: pd.DataFrame) -> np.ndarray:
        X = self.transform_features(dataframe)
        return self.classifier.predict(X)

    def predict_proba(self, dataframe: pd.DataFrame) -> np.ndarray:
        if not hasattr(self.classifier, "predict_proba"):
            raise AttributeError("The fitted classifier does not provide predict_proba")
        X = self.transform_features(dataframe)
        return self.classifier.predict_proba(X)

    def transform_features(self, dataframe: pd.DataFrame) -> np.ndarray:
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("ExperimentBundle inference expects a pandas DataFrame")

        missing = [name for name in self.input_feature_names if name not in dataframe.columns]
        if missing:
            raise ValueError(
                "Input data are missing required radiomics feature columns: "
                + ", ".join(missing[:10])
                + ("..." if len(missing) > 10 else "")
            )

        X = dataframe.loc[:, self.input_feature_names]
        X_t = self.preprocessor.transform(X)

        if self.select_k_best is not None:
            X_t = self.select_k_best.transform(X_t)

        if self.bpso_mask is not None:
            mask = np.asarray(self.bpso_mask, dtype=bool)
            if X_t.shape[1] != mask.shape[0]:
                raise ValueError(
                    "BPSO mask length does not match transformed feature count in the saved bundle"
                )
            X_t = X_t[:, mask]

        return X_t

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)


def load_experiment(path: str | Path) -> ExperimentBundle:
    loaded = joblib.load(path)
    if isinstance(loaded, ExperimentBundle):
        return loaded
    raise TypeError(
        "Loaded object is not an ExperimentBundle. Re-run the experiment to create a current model.joblib bundle."
    )
