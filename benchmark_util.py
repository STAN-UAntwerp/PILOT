from __future__ import annotations
import time
import pathlib
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from typing import Any, Literal
from retry import retry
from dataclasses import dataclass, asdict, field
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error
from sklearn.preprocessing import PowerTransformer
from ucimlrepo import fetch_ucirepo
from pmlb import fetch_data

from pilot import PILOT, CPILOT
from pilot.ensemble import RandomForestPilot
from pilot.c_ensemble import RandomForestCPilot


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
            self.X.iloc[idx, :].copy(),
            self.X_oh_encoded.iloc[idx, :].copy(),
            self.X_label_encoded.iloc[idx, :].copy(),
            self.y.iloc[idx].copy(),
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
        include_fields=[
            "id",
            "name",
            "n_samples",
            "n_features",
            "rows_removed",
            "cols_removed",
        ],
    ):
        return {field: getattr(self, field) for field in include_fields}

    def apply_transformer(self, feature_name: str, transformer: PowerTransformer):
        self.X.loc[:, feature_name] = np.nan_to_num(
            transformer.transform(self.X.loc[:, [feature_name]]).flatten()
        )
        self.X_oh_encoded.loc[:, feature_name] = np.nan_to_num(
            transformer.transform(self.X_oh_encoded.loc[:, [feature_name]]).flatten()
        )
        self.X_label_encoded.loc[:, feature_name] = np.nan_to_num(
            transformer.transform(self.X_label_encoded.loc[:, [feature_name]]).flatten()
        )
        
        


@dataclass
class FitResult:
    r2: float
    mse: float
    mae: float
    fit_duration: float
    predict_duration: float
    kwargs: dict[str, Any] = field(default_factory=dict)

    def asdict(self):
        d = asdict(self)
        d.pop("kwargs")
        d.update(self.kwargs)
        return d


def _load_uci_data(
    repo_id: int,
    ignore_feat: list[str] | None = None,
    logtransform_target: bool = False,
) -> Dataset:
    data = fetch_ucirepo(id=repo_id)
    variables = data.variables.set_index("name")
    X = data.data.features
    date_cols = [c for c in X.columns if (variables.loc[c, "type"] == "Date")]
    ignore_feat = ignore_feat + date_cols if ignore_feat is not None else date_cols
    if len(ignore_feat) > 0:
        print(f"Dropping features: {ignore_feat}")
        X = X.drop(columns=ignore_feat)
    X = X.replace("?", np.nan)
    y = data.data.targets.iloc[:, 0].astype(np.float64)
    pd.options.mode.use_inf_as_na = True
    rows_removed = 0
    cols_removed = 0
    if X.isna().any().any() or y.isna().any():
        cols_to_remove = X.columns[X.isna().mean() > 0.5]
        X = X.drop(columns=cols_to_remove)
        rows_to_remove = X.index[X.isna().any(axis=1) | y.isna()]
        X = X.drop(index=rows_to_remove)
        y = y.loc[X.index]
        rows_removed = len(rows_to_remove)
        cols_removed = len(cols_to_remove)
        print(
            f"Removed {rows_removed} rows and {cols_removed} columns with missing values. "
            f"{len(X)} rows  and {X.shape[1]} columns remaining."
        )
    pd.options.mode.use_inf_as_na = False

    if logtransform_target:
        y = np.log1p(y)

    cat_ids = [
        i
        for i, c in enumerate(X.columns)
        if (variables.loc[c, "type"] not in ["Continuous", "Integer"])
        or (X[c].nunique() < 5)
    ]
    cat_names = X.columns[cat_ids]

    oh_encoder = OneHotEncoder(sparse_output=False).fit(X[cat_names])

    X_oh_encoded = pd.concat(
        [
            X.drop(columns=cat_names),
            pd.DataFrame(
                oh_encoder.transform(X.loc[:, cat_names]),
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
        id=f"uci_{repo_id}",
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


def _get_date_columns(df):
    date_columns = []
    for col in df.columns:
        if df[col].dtype.kind in "biufc":
            continue
        try:
            pd.to_datetime(df[col])
            date_columns.append(col)
        except (ValueError, TypeError):
            continue
    return date_columns


def _load_pmlb_data(
    repo_id: str,
    ignore_feat: list[str] | None = None,
    logtransform_target: bool = False,
) -> Dataset:
    data = fetch_data(repo_id)
    X = data.drop(columns=["target"])
    date_cols = _get_date_columns(X)
    ignore_feat = ignore_feat + date_cols if ignore_feat is not None else date_cols
    if len(ignore_feat) > 0:
        print(f"Dropping features: {ignore_feat}")
        X = X.drop(columns=ignore_feat)
    X = X.replace("?", np.nan)
    y = data["target"].astype(np.float64)
    pd.options.mode.use_inf_as_na = True
    rows_removed = 0
    cols_removed = 0
    if X.isna().any().any() or y.isna().any():
        cols_to_remove = X.columns[X.isna().mean() > 0.5]
        X = X.drop(columns=cols_to_remove)
        rows_to_remove = X.index[X.isna().any(axis=1) | y.isna()]
        X = X.drop(index=rows_to_remove)
        y = y.loc[X.index]
        rows_removed = len(rows_to_remove)
        cols_removed = len(cols_to_remove)
        print(
            f"Removed {rows_removed} rows and {cols_removed} columns with missing values. "
            f"{len(X)} rows  and {X.shape[1]} columns remaining."
        )
    pd.options.mode.use_inf_as_na = False

    if logtransform_target:
        y = np.log1p(y)

    cat_ids = [
        i
        for i, c in enumerate(X.columns)
        if (X[c].nunique() < 5) or (X.dtypes[c] == "O")
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
        id=f"pmlb_{repo_id.split('_')[0]}",
        name="_".join(repo_id.split("_")[1:]),
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


@retry(ConnectionError, tries=5, delay=10)
def load_data(
    repo_id: int | str,
    ignore_feat: list[str] | None = None,
    use_download: bool = True,
    logtransform_target: bool = False,
    kind: Literal["uci", "pmlb"] = "uci",
) -> Dataset:
    if use_download:
        path = (
            pathlib.Path(__file__).parent.resolve() / "Data" / f"{kind}_{repo_id}.pkl"
        )
        if path.exists():
            print(f"Loading data from {path}")
            dataset = joblib.load(path)
            return dataset
        else:
            print(
                f"use_dowload was True, but path {path} does not exist. Trying to download."
            )

    if kind == "uci":
        return _load_uci_data(repo_id, ignore_feat, logtransform_target)
    elif kind == "pmlb":
        return _load_pmlb_data(repo_id, ignore_feat, logtransform_target)
    else:
        raise ValueError(f"kind must be one of 'uci' or 'pmlb' but received {kind}")


def fit_cart(train_dataset: Dataset, test_dataset: Dataset) -> FitResult:
    t1 = time.time()
    model = DecisionTreeRegressor()
    model.fit(train_dataset.X_oh_encoded, train_dataset.y)
    t2 = time.time()
    y_pred = model.predict(test_dataset.X_oh_encoded)
    t3 = time.time()
    r2 = float(r2_score(test_dataset.y, y_pred))
    mse = float(mean_squared_error(test_dataset.y, y_pred))
    mae = float(median_absolute_error(test_dataset.y, y_pred))
    return FitResult(
        r2=r2, mse=mse, mae=mae, fit_duration=t2 - t1, predict_duration=t3 - t2
    )


def fit_pilot(
    train_dataset: Dataset, test_dataset: Dataset, **init_kwargs
) -> FitResult:
    t1 = time.time()
    model = PILOT(**init_kwargs)
    model.fit(
        train_dataset.X_label_encoded.values,
        train_dataset.y.values,
        categorical=train_dataset.categorical,
    )
    t2 = time.time()
    y_pred = model.predict(test_dataset.X_label_encoded.values)
    t3 = time.time()
    r2 = float(r2_score(test_dataset.y, y_pred))
    mse = float(mean_squared_error(test_dataset.y, y_pred))
    mae = float(median_absolute_error(test_dataset.y, y_pred))
    return FitResult(
        r2=r2, mse=mse, mae=mae, fit_duration=t2 - t1, predict_duration=t3 - t2
    )


def fit_cpilot(
    train_dataset: Dataset,
    test_dataset: Dataset,
    dfs=[1, 2, 5, 5, 7, 5],
    min_sample_leaf=5,
    min_sample_alpha=5,
    min_sample_fit=5,
    max_depth=20,
    max_model_depth=100,
    max_node_features=1,
    rel_tolerance=0.01,
    precision_scale=1e-10,
    max_pivot: float | None = None,
) -> FitResult:
    t1 = time.time()
    max_features = (
        int(np.sqrt(train_dataset.n_features))
        if max_node_features == "sqrt"
        else int(max_node_features * train_dataset.n_features)
    )
    model = CPILOT(
        dfs,
        min_sample_leaf,
        min_sample_alpha,
        min_sample_fit,
        max_depth,
        max_model_depth,
        max_features,
        0 if max_pivot is None else max_pivot,
        rel_tolerance,
        precision_scale,
    )
    catids = np.zeros(train_dataset.n_features, dtype=int)
    if train_dataset.cat_ids:
        catids[train_dataset.categorical] = 1
    X = np.array(train_dataset.X_label_encoded.values, dtype=np.float64)
    y = np.array(train_dataset.y.values, dtype=np.float64)
    model.train(X, y, catids)
    t2 = time.time()
    y_pred = model.predict(test_dataset.X_label_encoded.values)
    t3 = time.time()
    r2 = float(r2_score(test_dataset.y, y_pred))
    mse = float(mean_squared_error(test_dataset.y, y_pred))
    mae = float(median_absolute_error(test_dataset.y, y_pred))
    return FitResult(
        r2=r2, mse=mse, mae=mae, fit_duration=t2 - t1, predict_duration=t3 - t2
    )


def fit_random_forest(
    train_dataset: Dataset, test_dataset: Dataset, **init_kwargs
) -> FitResult:
    t1 = time.time()
    model = RandomForestRegressor(**init_kwargs)
    model.fit(train_dataset.X_oh_encoded, train_dataset.y)
    t2 = time.time()
    y_pred = model.predict(test_dataset.X_oh_encoded)
    t3 = time.time()
    r2 = float(r2_score(test_dataset.y, y_pred))
    mse = float(mean_squared_error(test_dataset.y, y_pred))
    mae = float(median_absolute_error(test_dataset.y, y_pred))
    return FitResult(
        r2=r2, mse=mse, mae=mae, fit_duration=t2 - t1, predict_duration=t3 - t2
    )


def fit_ridge(
    train_dataset: Dataset, test_dataset: Dataset, **init_kwargs
) -> FitResult:
    t1 = time.time()
    model = Ridge(**init_kwargs)
    model.fit(train_dataset.X_oh_encoded, train_dataset.y)
    t2 = time.time()
    y_pred = model.predict(test_dataset.X_oh_encoded)
    t3 = time.time()
    r2 = float(r2_score(test_dataset.y, y_pred))
    mse = float(mean_squared_error(test_dataset.y, y_pred))
    mae = float(median_absolute_error(test_dataset.y, y_pred))
    return FitResult(
        r2=r2, mse=mse, mae=mae, fit_duration=t2 - t1, predict_duration=t3 - t2
    )


def fit_lasso(
    train_dataset: Dataset, test_dataset: Dataset, **init_kwargs
) -> FitResult:
    t1 = time.time()
    model = Lasso(**init_kwargs)
    model.fit(train_dataset.X_oh_encoded, train_dataset.y)
    t2 = time.time()
    y_pred = model.predict(test_dataset.X_oh_encoded)
    t3 = time.time()
    r2 = float(r2_score(test_dataset.y, y_pred))
    mse = float(mean_squared_error(test_dataset.y, y_pred))
    mae = float(median_absolute_error(test_dataset.y, y_pred))
    return FitResult(
        r2=r2, mse=mse, mae=mae, fit_duration=t2 - t1, predict_duration=t3 - t2
    )


def fit_pilot_forest(
    train_dataset: Dataset, test_dataset: Dataset, **init_kwargs
) -> FitResult:
    t1 = time.time()
    model = RandomForestPilot(**init_kwargs)
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
    mse = float(mean_squared_error(test_dataset.y, y_pred))
    mae = float(median_absolute_error(test_dataset.y, y_pred))
    return FitResult(
        r2=r2, mse=mse, mae=mae, fit_duration=t2 - t1, predict_duration=t3 - t2
    )


def fit_cpilot_forest(
    train_dataset: Dataset,
    test_dataset: Dataset,
    n_estimators: int = 10,
    max_depth: int = 12,
    max_model_depth: int = 100,
    min_sample_fit: int = 10,
    min_sample_alpha: int = 5,
    min_sample_leaf: int = 5,
    random_state: int = 42,
    n_features_tree: float | str = 1,
    n_features_node: float | str = 1,
    df_settings: dict[str, int] | None = None,
    rel_tolerance: float = 1e-2,
    precision_scale: float = 1e-10,
    max_pivot: int | float = None,
) -> FitResult:
    t1 = time.time()
    model = RandomForestCPilot(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_model_depth=max_model_depth,
        min_sample_fit=min_sample_fit,
        min_sample_alpha=min_sample_alpha,
        min_sample_leaf=min_sample_leaf,
        n_features_tree=n_features_tree,
        n_features_node=n_features_node,
        df_settings=df_settings,
        rel_tolerance=rel_tolerance,
        precision_scale=precision_scale,
        random_state=random_state,
        max_pivot=max_pivot,
    )
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
    mse = float(mean_squared_error(test_dataset.y, y_pred))
    mae = float(median_absolute_error(test_dataset.y, y_pred))
    # mean tree depth
    depths = pd.concat(
        [
            e.tree_summary()[["depth", "model_depth"]].max().to_frame().T
            for e in model.estimators
        ]
    ).agg(["mean", "max"])

    return FitResult(
        r2=r2,
        mse=mse,
        mae=mae,
        fit_duration=t2 - t1,
        predict_duration=t3 - t2,
        kwargs=dict(
            mean_tree_depth=depths.loc["mean", "depth"],
            max_tree_depth=depths.loc["max", "depth"],
            mean_tree_model_depth=depths.loc["mean", "model_depth"],
            max_tree_model_depth=depths.loc["max", "model_depth"],
        ),
    )


def fit_xgboost(
    train_dataset: Dataset,
    test_dataset: Dataset,
    max_depth: int = 6,
    max_node_features: float = 1,
    n_estimators: int = 100,
) -> FitResult:
    t1 = time.time()
    model = xgb.XGBRegressor(
        max_depth=max_depth,
        colsample_bynode=max_node_features,
        n_estimators=n_estimators,
    )
    model.fit(
        train_dataset.X_oh_encoded.values,
        train_dataset.y.values,
    )
    t2 = time.time()
    y_pred = model.predict(test_dataset.X_oh_encoded.values)
    t3 = time.time()
    r2 = float(r2_score(test_dataset.y, y_pred))
    mse = float(mean_squared_error(test_dataset.y, y_pred))
    mae = float(median_absolute_error(test_dataset.y, y_pred))
    return FitResult(
        r2=r2, mse=mse, mae=mae, fit_duration=t2 - t1, predict_duration=t3 - t2
    )
