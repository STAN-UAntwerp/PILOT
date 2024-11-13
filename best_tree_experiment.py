import pathlib

from sklearn import model_selection
import click
import numpy as np
import pandas as pd
from typing import Any

from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

from pilot import DEFAULT_DF_SETTINGS
from benchmark_config import LOGTRANSFORM_TARGET, UCI_DATASET_IDS, IGNORE_COLUMNS
from benchmark_util import *

OUTPUTFOLDER = pathlib.Path(__file__).parent / "Output"

df_setting_alpha01 = dict(
    zip(
        DEFAULT_DF_SETTINGS.keys(),
        1 + 0.01 * (np.array(list(DEFAULT_DF_SETTINGS.values())) - 1),
    )
)

df_setting_alpha5 = dict(
    zip(
        DEFAULT_DF_SETTINGS.keys(),
        1 + 0.5 * (np.array(list(DEFAULT_DF_SETTINGS.values())) - 1),
    )
)

df_setting_alpha01_no_blin = df_setting_alpha01.copy()
df_setting_alpha01_no_blin["blin"] = -1

df_setting_alpha5_no_blin = df_setting_alpha5.copy()
df_setting_alpha5_no_blin["blin"] = -1

df_setting_no_blin = DEFAULT_DF_SETTINGS.copy()
df_setting_no_blin["blin"] = -1


@click.command()
@click.option("--experiment_name", "-e", required=True, help="Name of the experiment")
def run_benchmark(experiment_name):
    experiment_folder = OUTPUTFOLDER / experiment_name
    experiment_folder.mkdir(exist_ok=True)
    experiment_file = experiment_folder / "results.csv"
    print(f"Results will be stored in {experiment_file}")
    np.random.seed(42)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    if experiment_file.exists():
        results = pd.read_csv(experiment_file)
        processed_repo_ids = results["id"].unique()
        results = results.to_dict("records")
    else:
        results = []
        processed_repo_ids = []

    repo_ids_to_process = [
        repo_id for repo_id in UCI_DATASET_IDS if repo_id not in processed_repo_ids
    ]
    for repo_id in repo_ids_to_process:
        print(repo_id)
        dataset = load_data(
            repo_id,
            ignore_feat=IGNORE_COLUMNS.get(repo_id),
            logtransform_target=(repo_id in LOGTRANSFORM_TARGET),
        )
        for i, (train, test) in enumerate(cv.split(dataset.X, dataset.y), start=1):
            print(f"\tFold {i} / 5")
            fold_result: dict[str, Any] = dict(
                id=repo_id,
                name=dataset.name,
                n_samples=dataset.n_samples,
                n_features=dataset.n_features,
                fold=i,
            )

            train_dataset = dataset.subset(train)
            test_dataset = dataset.subset(test)

            cpf = RandomForestCPilot(
                n_estimators=100,
                max_depth=20,
                max_model_depth=100,
                min_sample_fit=2,
                min_sample_alpha=1,
                min_sample_leaf=1,
                random_state=42,
                n_features_tree=1.0,
                n_features_node=1.0,
                df_settings=df_setting_alpha5_no_blin,
                rel_tolerance=0.01,
                precision_scale=1e-10,
            )
            cpf.fit(train_dataset.X_label_encoded, train_dataset.y, train_dataset.categorical)
            test_tree_pred = cpf.predict(test_dataset.X_label_encoded, individual=True)
            cpf_pred = test_tree_pred.mean(axis=1)
            cpf_r2 = r2_score(test_dataset.y, cpf_pred)
            test_tree_pred = pd.DataFrame(test_tree_pred, columns=range(100))
            tree_match_scores = test_tree_pred.apply(lambda c: r2_score(cpf_pred, c), axis=0)
            best_tree_idx = tree_match_scores.idxmax()
            best_tree_match = tree_match_scores.max()
            best_tree_r2 = r2_score(test_dataset.y, test_tree_pred[best_tree_idx])

            results.append(
                dict(
                    **fold_result,
                    model="CPF",
                    r2_forest=cpf_r2,
                    tree_match_score=best_tree_match,
                    best_tree_r2=best_tree_r2,
                )
            )

            rf = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42)
            rf.fit(train_dataset.X_oh_encoded, train_dataset.y)
            rf_pred = rf.predict(test_dataset.X_oh_encoded)
            rf_r2 = r2_score(test_dataset.y, rf_pred)
            rf_tree_pred = pd.DataFrame(
                np.array(
                    [tree.predict(test_dataset.X_oh_encoded.values) for tree in rf.estimators_]
                ).T,
                columns=range(100),
            )
            tree_match_scores = rf_tree_pred.apply(lambda c: r2_score(rf_pred, c), axis=0)
            best_tree_idx = tree_match_scores.idxmax()
            best_tree_match = tree_match_scores.max()
            best_tree_r2 = r2_score(test_dataset.y, rf_tree_pred[best_tree_idx])

            results.append(
                dict(
                    **fold_result,
                    model="RF",
                    r2_forest=rf_r2,
                    tree_match_score=best_tree_match,
                    best_tree_r2=best_tree_r2,
                )
            )
        pd.DataFrame(results).to_csv(experiment_file, index=False)


if __name__ == "__main__":
    run_benchmark()
