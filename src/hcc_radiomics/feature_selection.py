from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
from sklearn.base import clone
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import balanced_accuracy_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC


@dataclass
class SelectionResult:
    X_train: np.ndarray
    X_validation: np.ndarray
    X_test: np.ndarray
    feature_names: list[str]
    mask: np.ndarray
    metadata: dict
    selector: object | None = None


def safe_select_k_best(
    X_train: np.ndarray,
    y_train,
    X_validation: np.ndarray,
    X_test: np.ndarray,
    feature_names: list[str],
    k: int,
) -> SelectionResult:
    available = X_train.shape[1]
    if available != len(feature_names):
        raise ValueError("feature_names length does not match X_train columns")
    if k <= 0:
        raise ValueError("k must be positive")
    selected_k = min(k, available)
    if selected_k < k:
        warnings.warn(
            f"Requested k={k} exceeds {available} available features; using k={selected_k}",
            RuntimeWarning,
            stacklevel=2,
        )

    selector = SelectKBest(score_func=f_classif, k=selected_k)
    X_train_k = selector.fit_transform(X_train, y_train)
    X_validation_k = selector.transform(X_validation)
    X_test_k = selector.transform(X_test)
    mask = selector.get_support()
    selected_names = [name for name, keep in zip(feature_names, mask, strict=True) if keep]
    return SelectionResult(
        X_train=X_train_k,
        X_validation=X_validation_k,
        X_test=X_test_k,
        feature_names=selected_names,
        mask=mask,
        metadata={"method": "select_k_best", "requested_k": k, "selected_k": selected_k},
        selector=selector,
    )


def bpso_mask_cost(
    mask: np.ndarray,
    X_train: np.ndarray,
    y_train,
    *,
    estimator=None,
    alpha: float = 0.9,
    cv_folds: int = 3,
    random_state: int = 42,
    zero_penalty: float = 1.0,
) -> float:
    selected = np.flatnonzero(mask.astype(bool))
    if selected.size == 0:
        return zero_penalty

    total_features = X_train.shape[1]
    X_subset = X_train[:, selected]
    estimator = estimator or SVC(class_weight="balanced", probability=True, random_state=random_state)
    scorer = make_scorer(balanced_accuracy_score)
    folds = resolve_cv_folds(y_train, cv_folds)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    scores = cross_val_score(clone(estimator), X_subset, y_train, cv=cv, scoring=scorer)
    feature_fraction = selected.size / total_features
    return float(alpha * (1.0 - scores.mean()) + (1.0 - alpha) * feature_fraction)


def resolve_cv_folds(y_train, requested_folds: int) -> int:
    if requested_folds < 2:
        raise ValueError("cv_folds must be at least 2")

    min_class_count = int(np.min(np.unique(y_train, return_counts=True)[1]))
    if min_class_count < 2:
        raise ValueError(
            "Cross-validation requires at least 2 training samples in every class; "
            f"the smallest class has {min_class_count}."
        )
    if requested_folds > min_class_count:
        warnings.warn(
            (
                f"Requested cv_folds={requested_folds}, but the smallest training class "
                f"has {min_class_count} samples; using cv_folds={min_class_count}."
            ),
            RuntimeWarning,
            stacklevel=2,
        )
        return min_class_count
    return requested_folds


@dataclass
class BinaryPSOFeatureSelector:
    n_particles: int = 30
    iterations: int = 10
    alpha: float = 0.9
    cv_folds: int = 3
    random_state: int = 42
    c1: float = 0.5
    c2: float = 0.5
    inertia: float = 0.9
    zero_penalty: float = 1.0

    def fit(self, X_train: np.ndarray, y_train, feature_names: list[str]) -> "BinaryPSOFeatureSelector":
        if X_train.shape[1] != len(feature_names):
            raise ValueError("feature_names length does not match X_train columns")
        if X_train.shape[1] == 0:
            raise ValueError("BPSO requires at least one feature")

        rng = np.random.default_rng(self.random_state)
        n_features = X_train.shape[1]
        positions = rng.integers(0, 2, size=(self.n_particles, n_features), dtype=int)
        velocities = rng.normal(0.0, 1.0, size=(self.n_particles, n_features))
        personal_best = positions.copy()
        personal_cost = np.full(self.n_particles, np.inf)
        global_best = positions[0].copy()
        global_cost = np.inf
        estimator = SVC(class_weight="balanced", probability=True, random_state=self.random_state)
        history: list[float] = []

        for _ in range(self.iterations):
            for particle_idx in range(self.n_particles):
                cost = bpso_mask_cost(
                    positions[particle_idx],
                    X_train,
                    y_train,
                    estimator=estimator,
                    alpha=self.alpha,
                    cv_folds=self.cv_folds,
                    random_state=self.random_state,
                    zero_penalty=self.zero_penalty,
                )
                if cost < personal_cost[particle_idx]:
                    personal_cost[particle_idx] = cost
                    personal_best[particle_idx] = positions[particle_idx].copy()
                if cost < global_cost:
                    global_cost = cost
                    global_best = positions[particle_idx].copy()

            history.append(float(global_cost))
            r1 = rng.random(size=(self.n_particles, n_features))
            r2 = rng.random(size=(self.n_particles, n_features))
            velocities = (
                self.inertia * velocities
                + self.c1 * r1 * (personal_best - positions)
                + self.c2 * r2 * (global_best - positions)
            )
            probabilities = 1.0 / (1.0 + np.exp(-velocities))
            positions = (rng.random(size=(self.n_particles, n_features)) < probabilities).astype(int)

        if not np.any(global_best):
            variances = np.nan_to_num(np.var(X_train, axis=0), nan=0.0)
            global_best[int(np.argmax(variances))] = 1
            global_cost = bpso_mask_cost(
                global_best,
                X_train,
                y_train,
                estimator=estimator,
                alpha=self.alpha,
                cv_folds=self.cv_folds,
                random_state=self.random_state,
                zero_penalty=self.zero_penalty,
            )

        self.mask_ = global_best.astype(bool)
        self.feature_names_ = [
            name for name, keep in zip(feature_names, self.mask_, strict=True) if keep
        ]
        self.cost_ = float(global_cost)
        self.history_ = history
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return X[:, self.mask_]

    def fit_transform(self, X_train: np.ndarray, y_train, feature_names: list[str]) -> np.ndarray:
        return self.fit(X_train, y_train, feature_names).transform(X_train)

    def _check_fitted(self) -> None:
        if not hasattr(self, "mask_"):
            raise RuntimeError("BinaryPSOFeatureSelector must be fitted before transform")


def apply_bpso(
    X_train: np.ndarray,
    y_train,
    X_validation: np.ndarray,
    X_test: np.ndarray,
    feature_names: list[str],
    *,
    n_particles: int = 30,
    iterations: int = 10,
    alpha: float = 0.9,
    cv_folds: int = 3,
    random_state: int = 42,
) -> SelectionResult:
    selector = BinaryPSOFeatureSelector(
        n_particles=n_particles,
        iterations=iterations,
        alpha=alpha,
        cv_folds=cv_folds,
        random_state=random_state,
    ).fit(X_train, y_train, feature_names)
    return SelectionResult(
        X_train=selector.transform(X_train),
        X_validation=selector.transform(X_validation),
        X_test=selector.transform(X_test),
        feature_names=selector.feature_names_,
        mask=selector.mask_,
        metadata={
            "method": "bpso",
            "selected_count": int(selector.mask_.sum()),
            "cost": selector.cost_,
            "history": selector.history_,
        },
        selector=selector,
    )
