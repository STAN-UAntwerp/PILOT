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
    COLUMN_ORDER = [
        "CART",
        "PILOT",
        "CPILOT",
        "RF",
        "PF",
        "CPF",
        "XGB",
    ]
    experiment_folder = output_folder / experiment_name
    results = pd.read_csv(experiment_folder / "results.csv")
    scores = results.groupby(["id", "name", "model"])["r2"].mean().unstack()
    column_order = [
        column for c in COLUMN_ORDER for column in scores.columns if column.startswith(c)
    ]
    scores = scores.loc[:, column_order]

    # scores.to_html(experiment_folder / "r2_scores.html")
    dfi.export(
        scores.style.apply(highlight_max, axis=1).format("{:.2f}"),
        experiment_folder / "r2_scores.png",
        table_conversion="matplotlib",
        max_cols=50,
    )

    times = (
        results.groupby(["id", "name", "n_samples", "n_features", "model"])["fit_duration"]
        .mean()
        .unstack()
        .loc[:, column_order]
    )
    times = times.style.apply(lambda s: highlight_max(s, c1="red", c2="lightcoral"), axis=1).format(
        "{:.2f}"
    )
    # times.to_html(experiment_folder / "fit_duration.html")
    dfi.export(
        times, experiment_folder / "fit_duration.png", table_conversion="matplotlib", max_cols=50
    )

    aggregated_scores = pd.concat(
        [
            scores[[c for c in scores.columns if c.startswith(m)]].max(axis=1).rename(m)
            for m in COLUMN_ORDER
            if any([c.startswith(m) for c in scores.columns])
        ],
        axis=1,
    )
    dfi.export(
        aggregated_scores.style.apply(highlight_max, axis=1).format("{:.2f}"),
        experiment_folder / "agg_r2_scores.png",
        table_conversion="matplotlib",
        max_cols=50,
    )


if __name__ == "__main__":
    store_results()
