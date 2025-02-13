import pathlib
import click
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor

from pilot.c_ensemble import RandomForestCPilot
from pilot import DEFAULT_DF_SETTINGS

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
    results = []

    for i in range(5):
        print(f"Iteration {i}")
        for noise in [0, 0.1, 0.5, 1]:
            print(f"\tNoise = {noise}")
            X, y = make_regression(
                n_samples=8000,
                n_features=20,
                n_informative=20,
                effective_rank=16,
                noise=noise,
                random_state=42 + i,
            )
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=2000, random_state=42
            )
            for n_train in np.arange(10, len(X_train) + 11, step=200):
                print(f"\t\tn_train = {n_train}")
                X_train_subset = X_train[: min(n_train, len(X_train)), :]
                y_train_subset = y_train[: min(n_train, len(y_train))]

                print("RaFFLE")
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
                cpf.fit(X_train_subset, y_train_subset, np.array([-1]))
                pred = cpf.predict(X_test)
                results.append(
                    {
                        "iter": i,
                        "noise": noise,
                        "n_samples": n_train,
                        "model": "CPF",
                        "r2": r2_score(y_test, pred),
                    }
                )

                print("RF")
                rf = RandomForestRegressor(n_estimators=100)
                rf.fit(X_train_subset, y_train_subset)
                pred = rf.predict(X_test)
                results.append(
                    {
                        "iter": i,
                        "noise": noise,
                        "n_samples": n_train,
                        "model": "RF",
                        "r2": r2_score(y_test, pred),
                    }
                )

                print("LR")
                lr = LinearRegression()
                lr.fit(X_train_subset, y_train_subset)
                pred = lr.predict(X_test)
                results.append(
                    {
                        "iter": i,
                        "noise": noise,
                        "n_samples": n_train,
                        "model": "LR",
                        "r2": r2_score(y_test, pred),
                    }
                )

                print("XGB")
                xgbr = xgb.XGBRegressor()
                xgbr.fit(X_train_subset, y_train_subset)
                pred = lr.predict(X_test)
                results.append(
                    {
                        "iter": i,
                        "noise": noise,
                        "n_samples": n_train,
                        "model": "LR",
                        "r2": r2_score(y_test, pred),
                    }
                )

        pd.DataFrame(results).to_csv(experiment_file, index=False)


if __name__ == "__main__":
    run_benchmark()
