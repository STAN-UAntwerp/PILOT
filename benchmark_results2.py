import pathlib
import click
import pandas as pd
import numpy as np
import dataframe_image as dfi

output_folder = pathlib.Path(__file__).parent.resolve() / "Output"


def highlight_max(s, c1="green", c2="lightgreen"):
    is_max = s == s.max()
    is_close = s > (0.95 * s.max())
    return np.where(
        is_max,
        f"background-color: {c1}",
        np.where(is_close, f"background-color: {c2}; color: grey", ""),
    ).tolist()


experiment_name = "cpilot_benchmark"
other_experiment_name = "benchmark_v2"

COLUMN_ORDER = ["CART", "PILOT", "CPILOT"]
experiment_folder = output_folder / experiment_name
experiment_folder2 = output_folder / other_experiment_name
results = pd.concat(
    [
        pd.read_csv(experiment_folder / "results.csv"),
        pd.read_csv(experiment_folder2 / "results.csv"),
    ],
    axis=0,
)
scores = results.groupby(["id", "name", "model"])["r2"].mean().unstack()
column_order = [c for c in COLUMN_ORDER if c in scores.columns]
if len(column_order) != len(COLUMN_ORDER):
    print(
        f"WARNING: ignoring columns as they are not in the results: "
        f"{set(COLUMN_ORDER) - set(column_order)}"
    )
scores = scores.loc[:, column_order]

scores = scores.style.apply(highlight_max, axis=1).format("{:.2f}")

scores.to_html(experiment_folder / "r2_scores.html")
dfi.export(scores, experiment_folder / "r2_scores.png", table_conversion="matplotlib")

times = (
    results.groupby(["id", "name", "n_samples", "n_features", "model"])["fit_duration"]
    .mean()
    .unstack()
    .loc[:, column_order]
)
times = times.style.apply(lambda s: highlight_max(s, c1="red", c2="lightred"), axis=1).format(
    "{:.2f}"
)
times.to_html(experiment_folder / "fit_duration.html")
dfi.export(times, experiment_folder / "fit_duration.png", table_conversion="matplotlib")
