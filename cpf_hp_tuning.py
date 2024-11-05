import pathlib
import click
import numpy as np
import pandas as pd
import gc

from itertools import product
from sklearn.model_selection import KFold

from pilot import DEFAULT_DF_SETTINGS
from benchmark_config import UCI_DATASET_IDS, IGNORE_COLUMNS
from benchmark_util import *

OUTPUTFOLDER = pathlib.Path(__file__).parent / "Output"


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

    # repo_subset = [87, 186, 291, 381, 597]
    repo_subset = [492]
    repo_ids_to_process = [repo_id for repo_id in repo_subset if repo_id not in processed_repo_ids]

    min_sample_leaf = [1, 5, 10]
    min_sample_alpha = [2]
    min_sample_fit = [2]
    max_depth = [10, 20, 50]
    max_model_depth = [100]
    max_node_features = [1, "sqrt", 0.7]
    max_tree_features = [1]
    alpha = [0, 0.001, 0.01, 0.1, 1]

    for repo_id in repo_ids_to_process:
        print(repo_id)
        dataset = load_data(repo_id, ignore_feat=IGNORE_COLUMNS.get(repo_id))
        for i, (train, test) in enumerate(cv.split(dataset.X, dataset.y), start=1):
            print(f"\tFold {i} / 5")
            train_dataset = dataset.subset(train)
            test_dataset = dataset.subset(test)

            # RF
            r = fit_random_forest(
                train_dataset=train_dataset, test_dataset=test_dataset, n_estimators=100
            )
            results.append(dict(**dataset.summary(), fold=i, model="RF", **r.asdict()))

            # XGB
            r = fit_xgboost(train_dataset=train_dataset, test_dataset=test_dataset)
            results.append(dict(**dataset.summary(), fold=i, model="XGB", **r.asdict()))

            # CPF
            for hp, (msl, msa, msf, md, mmd, mnf, mtf, a) in enumerate(
                product(
                    min_sample_leaf,
                    min_sample_alpha,
                    min_sample_fit,
                    max_depth,
                    max_model_depth,
                    max_node_features,
                    max_tree_features,
                    alpha,
                )
            ):
                print(f"\t\tCPF {hp}")
                dfset = dict(
                    zip(
                        DEFAULT_DF_SETTINGS.keys(),
                        1 + a * (np.array(list(DEFAULT_DF_SETTINGS.values())) - 1),
                    )
                )

                r = fit_cpilot_forest(
                    train_dataset=train_dataset,
                    test_dataset=test_dataset,
                    min_sample_leaf=msl,
                    min_sample_alpha=msa,
                    min_sample_fit=msf,
                    max_depth=md,
                    max_model_depth=mmd,
                    n_features_node=mnf,
                    n_features_tree=mtf,
                    df_settings=dfset,
                )
                results.append(
                    dict(
                        **dataset.summary(),
                        fold=i,
                        model=f"CPF {hp}",
                        **r.asdict(),
                        min_sample_leaf=msl,
                        min_sample_alpha=msa,
                        min_sample_fit=msf,
                        max_depth=md,
                        max_model_depth=mmd,
                        n_features_node=mnf,
                        m_features_tree=mtf,
                        alpha=a,
                    )
                )

                # Clear unused variables and call garbage collector
                del r
                gc.collect()

        # Save results after processing each repo_id to avoid memory issues
        pd.DataFrame(results).to_csv(experiment_file, index=False)

        # Clear dataset and call garbage collector
        del dataset
        gc.collect()


if __name__ == "__main__":
    run_benchmark()
