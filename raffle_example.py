import pandas as pd
import numpy as np
from pilot.c_ensemble import RandomForestCPilot
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

housing = fetch_california_housing(as_frame=True)
cat_idx = np.array(
    [
        i
        for i in range(housing.data.shape[1])
        if (
            (not pd.api.types.is_numeric_dtype(housing.data.iloc[:, i]))
            or (housing.data.iloc[:, i].nunique() < 5)
        )
    ]
).astype(int)
X_train, X_test, y_train, y_test = train_test_split(housing.data, housing.target)

raffle = RandomForestCPilot(n_estimators=10, max_depth=10)

print("Fitting RaFFLE")
raffle.fit(X_train, y_train, categorical_idx=cat_idx)

print("Predicting on test set")
predictions = raffle.predict(X_test)

r2 = r2_score(y_test, predictions)

print(f"$R^2$ on test set = {r2:.2f}")
