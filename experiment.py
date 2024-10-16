import time
import pandas as pd
import numpy as np

from sklearn.metrics import r2_score, mean_absolute_error
from pilot import Pilot

datasets = [
    "Data/airfoil.csv",
    "Data/bodyfat_preprocessed.csv",
    "Data/communities.csv",
    "Data/concrete.csv",
    "Data/diabetes.csv",
    "Data/electricity.csv",
    "Data/housing.csv",
    "Data/residential.csv",
    "Data/ribo_preprocessed.csv",
    "Data/skills.csv",
    "Data/superconductor.csv",
]

np.random.seed(123)
results = []
for dataset in datasets:
    print(dataset)
    df = pd.read_csv(dataset)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    start = time.time()
    model = Pilot.PILOT(
        max_depth=50,
        max_model_depth=100,
        min_sample_split=2,
        min_sample_leaf=2,
        df_settings={"con": 1, "lin": 2, "blin": 5, "pcon": 5, "plin": 7, "pconc": 5},
    )

    model.fit(X, y)

    predictions = model.predict(X=X)

    end = time.time()
    r2 = r2_score(y, predictions)
    mae = mean_absolute_error(y, predictions)

    results.append({"dataset": dataset, "time": end - start, "R2": r2, "MAE": mae})
    print(results[-1])

pd.DataFrame(results).to_csv("Output/pilot_duration_benchmark.csv", index=False)
