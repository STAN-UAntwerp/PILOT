from __future__ import annotations
import time
import pandas as pd
import numpy as np
import xgboost as xgb
from typing import Any
from retry import retry
from dataclasses import dataclass, asdict, field
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from ucimlrepo import fetch_ucirepo

from pilot.Pilot import PILOT
from pilot.ensemble import RandomForestPilot


@dataclass
class Dataset:
    id: int
    name: str
    X: pd.DataFrame
    X_oh_encoded: pd.DataFrame
    X_label_encoded: pd.DataFrame
    y: pd.Series
    cat_ids: list[int]
    cat_names: list[str]
    oh_encoder: OneHotEncoder
    label_encoders: dict[str, LabelEncoder]
    rows_removed: int
    cols_removed: int

    def subset(self, idx: list[int]) -> Dataset:
        return Dataset(
            self.id,
            self.name,
            self.X.iloc[idx, :],
            self.X_oh_encoded.iloc[idx, :],
            self.X_label_encoded.iloc[idx, :],
            self.y.iloc[idx],
            self.cat_ids,
            self.cat_names,
            self.oh_encoder,
            self.label_encoders,
            self.rows_removed,
            self.cols_removed,
        )

    @property
    def categorical(self) -> np.ndarray:
        return np.array(self.cat_ids) if len(self.cat_ids) > 0 else np.array([-1])

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    @property
    def n_samples(self) -> int:
        return len(self.X)

    def summary(
        self,
        include_fields=["id", "name", "n_samples", "n_features", "rows_removed", "cols_removed"],
    ):
        return {field: getattr(self, field) for field in include_fields}


@dataclass
class FitResult:
    r2: float
    fit_duration: float
    predict_duration: float
    kwargs: dict[str, Any] = field(default_factory=dict)

    def asdict(self):
        d = asdict(self)
        d.pop("kwargs")
        d.update(self.kwargs)
        return d


@retry(ConnectionError, tries=3, delay=10)
def load_data(repo_id: int) -> Dataset:
    data = fetch_ucirepo(id=repo_id)
    X = data.data.features
    X = X.replace("?", np.nan)
    y = data.data.targets.iloc[:, 0].astype(np.float64)
    pd.options.mode.use_inf_as_na = True
    rows_removed = 0
    cols_removed = 0
    if X.isna().any().any():
        n_before = len(X)
        cols_to_remove = X.columns[X.isna().mean() > 0.5]
        X = X.drop(columns=cols_to_remove)
        X = X.dropna(how="any")
        y = y.loc[X.index]
        rows_removed = n_before - len(X)
        cols_removed = len(cols_to_remove)
        print(
            f"Removed {rows_removed} rows and {cols_removed} columns with missing values. "
            f"{len(X)} rows  and {X.shape[1]} columns remaining."
        )
    pd.options.mode.use_inf_as_na = False

    variables = data.variables.set_index("name")
    cat_ids = [
        i
        for i, c in enumerate(X.columns)
        if (variables.loc[c, "type"] not in ["Continuous", "Integer"]) or (X[c].nunique() < 5)
    ]
    cat_names = X.columns[cat_ids]

    oh_encoder = OneHotEncoder(sparse_output=False).fit(X[cat_names])
    X_oh_encoded = pd.concat(
        [
            X.drop(columns=cat_names),
            pd.DataFrame(
                oh_encoder.transform(X[cat_names]),
                columns=oh_encoder.get_feature_names_out(),
                index=X.index,
            ),
        ],
        axis=1,
    ).astype(np.float64)

    label_encoders = {col: LabelEncoder().fit(X[col]) for col in cat_names}
    X_label_encoded = X.copy()
    for col, le in label_encoders.items():
        X_label_encoded.loc[:, col] = le.transform(X[col])
    X_label_encoded = X_label_encoded.astype(np.float64)

    return Dataset(
        id=repo_id,
        name=data.metadata.name,
        X=X,
        X_oh_encoded=X_oh_encoded,
        X_label_encoded=X_label_encoded,
        y=y,
        cat_ids=cat_ids,
        cat_names=cat_names,
        oh_encoder=oh_encoder,
        label_encoders=label_encoders,
        rows_removed=rows_removed,
        cols_removed=cols_removed,
    )


def fit_cart(train_dataset: Dataset, test_dataset: Dataset) -> FitResult:
    t1 = time.time()
    model = DecisionTreeRegressor()
    model.fit(train_dataset.X_oh_encoded, train_dataset.y)
    t2 = time.time()
    y_pred = model.predict(test_dataset.X_oh_encoded)
    t3 = time.time()
    r2 = float(r2_score(test_dataset.y, y_pred))
    return FitResult(r2=r2, fit_duration=t2 - t1, predict_duration=t3 - t2)


def fit_pilot(train_dataset: Dataset, test_dataset: Dataset) -> FitResult:
    t1 = time.time()
    model = PILOT(max_depth=20, max_model_depth=100, rel_tolerance=1e-3)
    model.fit(
        train_dataset.X_label_encoded.values,
        train_dataset.y.values,
        categorical=train_dataset.categorical,
    )
    t2 = time.time()
    y_pred = model.predict(test_dataset.X_label_encoded.values)
    t3 = time.time()
    r2 = float(r2_score(test_dataset.y, y_pred))
    return FitResult(r2=r2, fit_duration=t2 - t1, predict_duration=t3 - t2)


def fit_random_forest(train_dataset: Dataset, test_dataset: Dataset) -> FitResult:
    t1 = time.time()
    model = RandomForestRegressor()
    model.fit(train_dataset.X_oh_encoded, train_dataset.y)
    t2 = time.time()
    y_pred = model.predict(test_dataset.X_oh_encoded)
    t3 = time.time()
    r2 = float(r2_score(test_dataset.y, y_pred))
    return FitResult(r2=r2, fit_duration=t2 - t1, predict_duration=t3 - t2)


def fit_pilot_forest(train_dataset: Dataset, test_dataset: Dataset) -> FitResult:
    t1 = time.time()
    model = RandomForestPilot(max_depth=20, max_model_depth=50, rel_tolerance=1e-2)
    model.fit(
        train_dataset.X_label_encoded.values,
        train_dataset.y.values,
        categorical_idx=train_dataset.categorical,
        n_workers=1,
    )
    t2 = time.time()
    y_pred = model.predict(test_dataset.X_label_encoded.values)
    t3 = time.time()
    r2 = float(r2_score(test_dataset.y, y_pred))
    return FitResult(r2=r2, fit_duration=t2 - t1, predict_duration=t3 - t2)


def fit_xgboost(train_dataset: Dataset, test_dataset: Dataset) -> FitResult:
    t1 = time.time()
    model = xgb.XGBRegressor()
    model.fit(
        train_dataset.X_oh_encoded.values,
        train_dataset.y.values,
    )
    t2 = time.time()
    y_pred = model.predict(test_dataset.X_oh_encoded.values)
    t3 = time.time()
    r2 = float(r2_score(test_dataset.y, y_pred))
    return FitResult(r2=r2, fit_duration=t2 - t1, predict_duration=t3 - t2)
