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


def draw_borders(df, columns=["CART", "RF", "XGB"]):
    style = pd.DataFrame("", index=df.index, columns=df.columns)
    for c in columns:
        style[c] = "border-left: 3px solid black;"

    return style


@click.command()
@click.option(
    "--experiment_name",
    "-e",
    required=True,
    help="Name of the experiment (assuming the results are stored in a folder with the same name)",
)
def store_results(experiment_name):
    COLUMN_ORDER = ["CART", "PILOT", "PILOT - no blin", "RF", "PF", "PF - no blin", "XGB"]
    experiment_folder = output_folder / experiment_name
    results = pd.read_csv(experiment_folder / "results.csv")
    scores = results.groupby(["id", "name", "model"])["r2"].mean().unstack()
    column_order = [c for c in COLUMN_ORDER if c in scores.columns]
    if len(column_order) != len(COLUMN_ORDER):
        print(
            f"WARNING: ignoring columns as they are not in the results: "
            f"{set(COLUMN_ORDER) - set(column_order)}"
        )
    scores = scores.loc[:, column_order]

    scores = (
        scores.style.apply(highlight_max, axis=1).apply(draw_borders, axis=None).format("{:.2f}")
    )

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


if __name__ == "__main__":
    store_results()
