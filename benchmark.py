import pathlib
import click
import psutil
import itertools
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold

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

df_setting_alpha01_no_blin = df_setting_alpha01.copy()
df_setting_alpha01_no_blin["blin"] = -1

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
            print('\tRAM Used (GB):', psutil.virtual_memory()[3]/1000000000)
            train_dataset = dataset.subset(train)
            test_dataset = dataset.subset(test)

            # CART
            print("\t\tCART")
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

            print("\t\tCPILOT")
            r = fit_cpilot(
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                max_depth=20,
            )
            results.append(dict(**dataset.summary(), fold=i, model="CPILOT", **r.asdict()))

            # RF
            print("\t\tRF")
            r = fit_random_forest(
                train_dataset=train_dataset, test_dataset=test_dataset, n_estimators=100
            )
            results.append(dict(**dataset.summary(), fold=i, model="RF", **r.asdict()))

            print("\t\tRF - max_depth = 6")
            r = fit_random_forest(
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                n_estimators=100,
                max_depth=6,
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
            for j, ((df_name, df_setting), max_depth, max_features) in enumerate(
                itertools.product(
                    [
                        ("default df", DEFAULT_DF_SETTINGS),
                        ("df alpha = 0.01", df_setting_alpha01),
                        ("df alpha = 0.01, no blin", df_setting_alpha01_no_blin),
                        ("df no blin", df_setting_no_blin),
                    ],
                    [6, 20],
                    [0.7, 1],
                )
            ):
                model_name = f"CPF - {df_name} - max_depth = {max_depth} - max_node_features = {max_features}"
                print(f"\t\t{model_name}")
                r = fit_cpilot_forest(
                    train_dataset=train_dataset,
                    test_dataset=test_dataset,
                    n_estimators=100,
                    min_sample_leaf=1,
                    min_sample_alpha=2,
                    min_sample_fit=2,
                    max_depth=max_depth,
                    n_features_node=max_features,
                    df_settings=df_setting,
                )
                results.append(
                    dict(
                        **dataset.summary(),
                        fold=i,
                        model=model_name,
                        **r.asdict(),
                        max_depth=max_depth,
                        df_setting=df_setting,
                    )
                )
                print('\t\t\tRAM Used (GB):', psutil.virtual_memory()[3]/1000000000)

            # XGB
            print("\t\tXGB")
            r = fit_xgboost(train_dataset=train_dataset, test_dataset=test_dataset)
            results.append(dict(**dataset.summary(), fold=i, model="XGB", **r.asdict()))

            # XGB
            print("\t\tXGB - max_depth = 20")
            r = fit_xgboost(train_dataset=train_dataset, test_dataset=test_dataset, max_depth=20)
            results.append(
                dict(**dataset.summary(), fold=i, model="XGB - max_depth = 20", **r.asdict())
            )

        pd.DataFrame(results).to_csv(experiment_file, index=False)


if __name__ == "__main__":
    run_benchmark()
