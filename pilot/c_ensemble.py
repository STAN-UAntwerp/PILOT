import numpy as np
import pandas as pd
import multiprocessing as mp
from sklearn.base import BaseEstimator
from pilot import CPILOT, DEFAULT_DF_SETTINGS
from functools import partial


class CPILOTWrapper(CPILOT):
    def __init__(
        self,
        feature_idx: list[int] | np.ndarray,
        df_settings=None,
        min_sample_leaf=5,
        min_sample_alpha=5,
        min_sample_fit=5,
        max_depth=20,
        max_model_depth=100,
        max_features=-1,  # -1 means that it must be set
        rel_tolerance=0.01,
        precision_scale=1e-10,
    ):
        if max_features == -1:
            raise ValueError("max_features must be set")
        super().__init__(
            df_settings,
            min_sample_leaf,
            min_sample_alpha,
            min_sample_fit,
            max_depth,
            max_model_depth,
            max_features,
            rel_tolerance,
            precision_scale,
        )
        self.feature_idx = feature_idx

    def predict(self, X):
        return super().predict(X[:, self.feature_idx])

    def tree_summary(self, feature_names: list | None = None) -> pd.DataFrame:
        df = pd.DataFrame(
            self.print(),
            columns=[
                "depth",
                "model_depth",
                "node_id",
                "node_type",
                "feature_index",
                "split_value",
                "intercept_left",
                "slope_left",
                "intercept_right",
                "slope_right",
            ],
        )

        df["node_type"] = df["node_type"].map(
            {0: "con", 1: "lin", 2: "pcon", 3: "blin", 4: "plin", 5: "pconc"}
        )

        if feature_names is not None:
            feature_names = np.array(feature_names)[self.feature_idx]
            df["feature_name"] = df["feature_index"].map(dict(enumerate(feature_names)))

        return df


class RandomForestCPilot(BaseEstimator):
    def __init__(
        self,
        n_estimators: int = 10,
        max_depth: int = 12,
        max_model_depth: int = 100,
        min_sample_fit: int = 10,
        min_sample_alpha: int = 5,
        min_sample_leaf: int = 5,
        random_state: int = 42,
        n_features_tree: float | str = 1.0,
        n_features_node: float | str = 1.0,
        df_settings: dict[str, int] | None = None,
        rel_tolerance: float = 0.01,
        precision_scale: float = 1e-10,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_model_depth = max_model_depth
        self.min_sample_fit = min_sample_fit
        self.min_sample_alpha = min_sample_alpha
        self.min_sample_leaf = min_sample_leaf
        self.random_state = random_state
        self.n_features_tree = n_features_tree
        self.n_features_node = n_features_node
        self.df_settings = (
            list(df_settings.values())
            if df_settings is not None
            else list(DEFAULT_DF_SETTINGS.values())
        )
        self.rel_tolerance = rel_tolerance
        self.precision_scale = precision_scale

    def fit(self, X, y, categorical_idx=None, n_workers: int = 1):

        categorical = np.zeros(X.shape[1], dtype=int)
        if categorical_idx is not None and not (categorical_idx == -1).any():
            categorical[categorical_idx] = 1

        X = np.array(X)
        y = np.array(y).flatten()
        n_features_tree = (
            int(np.sqrt(X.shape[1]))
            if self.n_features_tree == "sqrt"
            else int(X.shape[1] * self.n_features_tree)
        )
        n_features_node = min(
            n_features_tree,  # n_feature_node cannot be larger than n_features_tree
            (
                int(np.sqrt(X.shape[1]))
                if self.n_features_node == "sqrt"
                else int(X.shape[1] * self.n_features_node)
            ),
        )
        self.estimators = [
            CPILOTWrapper(
                feature_idx=np.random.choice(
                    np.arange(X.shape[1]), size=n_features_tree, replace=False
                ),
                df_settings=self.df_settings,
                min_sample_leaf=self.min_sample_leaf,
                min_sample_alpha=self.min_sample_alpha,
                min_sample_fit=self.min_sample_fit,
                max_depth=self.max_depth,
                max_model_depth=self.max_model_depth,
                max_features=n_features_node,
                rel_tolerance=self.rel_tolerance,
                precision_scale=self.precision_scale,
            )
            for _ in range(self.n_estimators)
        ]

        if n_workers == -1:
            n_workers = mp.cpu_count()
        if n_workers == 1:
            # avoid overhead of parallel processing
            self.estimators = [
                _fit_single_estimator(estimator, X, y, categorical) for estimator in self.estimators
            ]
        else:
            raise NotImplementedError("Parallel processing not available for CPILOT")
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


def _fit_single_estimator(estimator, X: np.ndarray, y: np.ndarray, categorical_idx: np.ndarray):
    bootstrap_idx = np.random.choice(np.arange(len(X)), size=len(X), replace=True)
    feature_idx = estimator.feature_idx
    categorical_idx = categorical_idx[feature_idx].astype(int)
    X_bootstrap = X[np.ix_(bootstrap_idx, feature_idx)]

    try:
        estimator.train(X_bootstrap, y[bootstrap_idx], categorical_idx)
        return estimator
    except ValueError as e:
        print(e)
        return None
