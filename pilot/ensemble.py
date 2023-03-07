import numpy as np
import multiprocessing as mp
from sklearn.base import BaseEstimator
from pilot import Pilot
from functools import partial


class RandomForestPilot(BaseEstimator):
    def __init__(
        self,
        n_estimators: int = 10,
        max_depth: int = 12,
        split_criterion: str = "BIC",
        min_sample_split: int = 10,
        min_sample_leaf: int = 5,
        step_size: int = 1,
        random_state: int = 42,
        truncation_factor: int = 3,
        n_features: float | str = 1.0,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.split_criterion = split_criterion
        self.min_sample_split = min_sample_split
        self.min_sample_leaf = min_sample_leaf
        self.step_size = step_size
        self.random_state = random_state
        self.truncation_factor = truncation_factor
        self.n_features = n_features

    def fit(self, X, y, categorical_idx=np.array([-1]), n_workers: int = 1):
        self.estimators = [
            Pilot.PILOT(
                self.max_depth,
                self.split_criterion,
                self.min_sample_split,
                self.min_sample_leaf,
                self.step_size,
                truncation_factor=self.truncation_factor,
            )
            for _ in range(self.n_estimators)
        ]
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        n_features = (
            int(np.sqrt(X.shape[1]))
            if self.n_features == "sqrt"
            else int(X.shape[1] * self.n_features)
        )
        if n_workers == -1:
            n_workers = mp.cpu_count()
        if n_workers == 1:
            # avoid overhead of parallel processing
            self.estimators = [
                _fit_single_estimator(estimator, X, y, categorical_idx, n_features)
                for estimator in self.estimators
            ]
        else:
            with mp.Pool(processes=n_workers) as p:
                self.estimators = p.map(
                    partial(
                        _fit_single_estimator,
                        X=X,
                        y=y,
                        categorical_idx=categorical_idx,
                        n_features=n_features,
                    ),
                    self.estimators,
                )
        # filter failed estimators
        self.estimators = [e for e in self.estimators if e is not None]

    def predict(self, X) -> np.ndarray:
        X = np.array(X)
        return np.concatenate([e.predict(X).reshape(-1, 1) for e in self.estimators], axis=1).mean(
            axis=1
        )


def _fit_single_estimator(estimator, X, y, categorical_idx, n_features):
    bootstrap_idx = np.random.choice(np.arange(len(X)), size=len(X), replace=True)
    try:
        estimator.fit(
            X[bootstrap_idx, :],
            y[bootstrap_idx],
            categorical=categorical_idx,
            max_features_considered=n_features,
        )
        return estimator
    except Exception as e:
        print(e)
        return None
