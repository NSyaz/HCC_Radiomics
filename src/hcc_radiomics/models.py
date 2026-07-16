from __future__ import annotations

from sklearn.svm import SVC


def build_svm(random_state: int = 42) -> SVC:
    return SVC(
        class_weight="balanced",
        probability=True,
        random_state=random_state,
        max_iter=10000,
    )
