import pathlib
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold

from benchmark_config import UCI_DATASET_IDS
from benchmark_util import *

OUTPUTFOLDER = pathlib.Path(__file__).parent / "Output"
experiment_file = OUTPUTFOLDER / "benchmark_results.csv"
print(f"Results will be stored in {experiment_file}")
np.random.seed(42)
cv = KFold(n_splits=5, shuffle=True, random_state=42)

if experiment_file.exists():
    results = pd.read_csv(experiment_file)
    processed_repo_ids = results["id"].unique()
    results = results.to_dict("records")
else:
    results = []

repo_ids_to_process = [repo_id for repo_id in UCI_DATASET_IDS if repo_id not in processed_repo_ids]
for repo_id in repo_ids_to_process:
    print(repo_id)
    dataset = load_data(repo_id)
    for i, (train, test) in enumerate(cv.split(dataset.X, dataset.y), start=1):
        print(f"\tFold {i} / 5")
        train_dataset = dataset.subset(train)
        test_dataset = dataset.subset(test)

        # CART
        r = fit_cart(train_dataset=train_dataset, test_dataset=test_dataset)
        results.append(dict(**dataset.summary(), fold=i, model="CART", **r.asdict()))

        # PILOT
        r = fit_pilot(train_dataset=train_dataset, test_dataset=test_dataset)
        results.append(dict(**dataset.summary(), fold=i, model="PILOT", **r.asdict()))

        # RF
        r = fit_random_forest(train_dataset=train_dataset, test_dataset=test_dataset)
        results.append(dict(**dataset.summary(), fold=i, model="RF", **r.asdict()))

        # PF
        r = fit_pilot_forest(train_dataset=train_dataset, test_dataset=test_dataset)
        results.append(dict(**dataset.summary(), fold=i, model="PF", **r.asdict()))

        # XGB
        r = fit_xgboost(train_dataset=train_dataset, test_dataset=test_dataset)
        results.append(dict(**dataset.summary(), fold=i, model="XGB", **r.asdict()))

    pd.DataFrame(results).to_csv(experiment_file, index=False)
