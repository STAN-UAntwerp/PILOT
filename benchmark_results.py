import pathlib
import pandas as pd
import numpy as np

filename = "benchmark_results_v2"
output_folder = pathlib.Path(__file__).parent.resolve() / "Output"
results = pd.read_csv(output_folder / f"{filename}.csv")


def highlight_max(s):
    is_max = s == s.max()
    is_close = s > (0.95 * s.max())
    return np.where(
        is_max,
        "background-color: green",
        np.where(is_close, "background-color: lightgreen; color: grey", ""),
    ).tolist()


scores = results.groupby(["id", "name", "model"])["r2"].mean().unstack()

scores.style.apply(highlight_max, axis=1).format("{:.2f}").to_html(
    output_folder / f"{filename}.html"
)
