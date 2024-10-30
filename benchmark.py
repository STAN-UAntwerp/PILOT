import pathlib
import click
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold

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

    repo_ids_to_process = [
        repo_id for repo_id in UCI_DATASET_IDS if repo_id not in processed_repo_ids
    ]
    for repo_id in repo_ids_to_process:
        print(repo_id)
        dataset = load_data(repo_id, ignore_feat=IGNORE_COLUMNS.get(repo_id))
        for i, (train, test) in enumerate(cv.split(dataset.X, dataset.y), start=1):
            print(f"\tFold {i} / 5")
            train_dataset = dataset.subset(train)
            test_dataset = dataset.subset(test)

            # CART
            r = fit_cart(train_dataset=train_dataset, test_dataset=test_dataset)
            results.append(dict(**dataset.summary(), fold=i, model="CART", **r.asdict()))

            # # PILOT
            # r = fit_pilot(
            #     train_dataset=train_dataset,
            #     test_dataset=test_dataset,
            #     max_depth=20,
            #     truncation_factor=1,
            #     rel_tolerance=0.01,
            # )
            # results.append(dict(**dataset.summary(), fold=i, model="PILOT", **r.asdict()))

            # r = fit_pilot(
            #     train_dataset=train_dataset,
            #     test_dataset=test_dataset,
            #     max_depth=20,
            #     truncation_factor=1,
            #     rel_tolerance=0.01,
            #     regression_nodes=["con", "lin", "pcon", "plin"],
            # )
            # results.append(dict(**dataset.summary(), fold=i, model="PILOT - no blin", **r.asdict()))

            r = fit_cpilot(
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                max_depth=20,
            )
            results.append(dict(**dataset.summary(), fold=i, model="CPILOT", **r.asdict()))

            r = fit_cpilot(
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                max_depth=20,
                dfs=[1, 2, 5, -1, 7, 5],
            )
            results.append(
                dict(**dataset.summary(), fold=i, model="CPILOT - no blin", **r.asdict())
            )

            # RF
            r = fit_random_forest(
                train_dataset=train_dataset, test_dataset=test_dataset, n_estimators=100
            )
            results.append(dict(**dataset.summary(), fold=i, model="RF", **r.asdict()))

            # # PF
            # r = fit_pilot_forest(
            #     train_dataset=train_dataset,
            #     test_dataset=test_dataset,
            #     max_depth=20,
            #     n_estimators=100,
            #     truncation_factor=1,
            #     rel_tolerance=0.01,
            # )
            # results.append(dict(**dataset.summary(), fold=i, model="PF", **r.asdict()))

            # r = fit_pilot_forest(
            #     train_dataset=train_dataset,
            #     test_dataset=test_dataset,
            #     max_depth=20,
            #     n_estimators=100,
            #     truncation_factor=1,
            #     rel_tolerance=0.01,
            #     regression_nodes=["con", "lin", "pcon", "plin"],
            # )
            # results.append(dict(**dataset.summary(), fold=i, model="PF - no blin", **r.asdict()))

            r = fit_cpilot_forest(
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                max_depth=20,
                n_estimators=100,
            )
            results.append(dict(**dataset.summary(), fold=i, model="CPF", **r.asdict()))

            r = fit_cpilot_forest(
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                max_depth=20,
                n_estimators=100,
                df_settings={"con": 1, "lin": 2, "pcon": 5, "blin": -1, "plin": 7, "pconc": 5},
            )
            results.append(dict(**dataset.summary(), fold=i, model="CPF - no blin", **r.asdict()))

            # XGB
            r = fit_xgboost(train_dataset=train_dataset, test_dataset=test_dataset)
            results.append(dict(**dataset.summary(), fold=i, model="XGB", **r.asdict()))

        pd.DataFrame(results).to_csv(experiment_file, index=False)


if __name__ == "__main__":
    run_benchmark()
