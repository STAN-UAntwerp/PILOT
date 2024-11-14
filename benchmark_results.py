import pathlib
import click
import re
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
    results = results.assign(mainmodel=results["model"].str.split("-").str[0])
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
        max_rows=150,
    )

    times = (
        results.groupby(["id", "name", "n_samples", "n_features", "model"])["fit_duration"]
        .mean()
        .unstack()
        .loc[:, column_order]
    )
    # times.to_html(experiment_folder / "fit_duration.html")
    dfi.export(
        times.style.apply(lambda s: highlight_max(s, c1="red", c2="lightcoral"), axis=1).format(
            "{:.2f}"
        ),
        experiment_folder / "fit_duration.png",
        table_conversion="matplotlib",
        max_cols=50,
        max_rows=150
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
        max_rows=150
    )

    cpf_scores = scores[[c for c in scores.columns if c.startswith("CPF")]].T
    cpf_scores = (
        cpf_scores.assign(
            excl_blin=cpf_scores.index.map(lambda x: "no blin" in x),
            alpha=cpf_scores.index.map(
                lambda x: re.search(r"alpha = ([\d.]+)", x).group(1) if "alpha" in x else 1
            ),
            max_depth=cpf_scores.index.map(lambda x: re.search(r"max_depth = (\d+)", x).group(1)),
            max_node_features=cpf_scores.index.map(
                lambda x: re.search(r"max_node_features = ([\d.]+)", x).group(1)
            ),
            n_estimators=cpf_scores.index.map(
                lambda x: (
                    re.search(r"n_estimators = (\d+)", x).group(1) if "n_estimators" in x else None
                )
            ),
        )
        .reset_index(drop=True)
        .set_index(["excl_blin", "alpha", "max_depth", "max_node_features", "n_estimators"])
        .T
    )
    dfi.export(
        cpf_scores.style.apply(highlight_max, axis=1).format("{:.2f}"),
        experiment_folder / "cpf_r2_scores.png",
        table_conversion="matplotlib",
        max_cols=50,
        max_rows=150
    )
    cpf_ranks = pd.DataFrame(
        {
            "average_R2_ratio": (
                np.nanmean(
                    np.where(cpf_scores < 0, np.NaN, cpf_scores)
                    / cpf_scores.max(axis=1).values.reshape(-1, 1),
                    axis=0,
                )
            )
        },
        index=cpf_scores.columns,
    )

    dfi.export(
        cpf_ranks.style,
        experiment_folder / "cpf_avg_rank.png",
        table_conversion="matplotlib",
        max_cols=50,
        max_rows=150
    )

    cpf_ranks = cpf_ranks.reset_index()
    for c in ["excl_blin", "alpha", "max_depth", "max_node_features", "n_estimators"]:
        dfi.export(
            cpf_ranks.groupby(c)["average_R2_ratio"].mean().to_frame().style,
            experiment_folder / f"cpf_avg_rank_{c}.png",
            table_conversion="matplotlib",
        )

    cpf_times = times[[c for c in times.columns if c.startswith("CPF")]].T
    cpf_times = (
        cpf_times.assign(
            excl_blin=cpf_times.index.map(lambda x: "no blin" in x),
            alpha=cpf_times.index.map(
                lambda x: re.search(r"alpha = ([\d.]+)", x).group(1) if "alpha" in x else 1
            ),
            max_depth=cpf_times.index.map(lambda x: re.search(r"max_depth = (\d+)", x).group(1)),
            max_node_features=cpf_times.index.map(
                lambda x: re.search(r"max_node_features = ([\d.]+)", x).group(1)
            ),
            n_estimators=cpf_times.index.map(
                lambda x: (
                    re.search(r"n_estimators = (\d+)", x).group(1) if "n_estimators" in x else None
                )
            ),
        )
        .reset_index(drop=True)
        .set_index(["excl_blin", "alpha", "max_depth", "max_node_features", "n_estimators"])
        .T
    )
    dfi.export(
        cpf_times.style.apply(lambda s: highlight_max(s, c1="red", c2="lightcoral"), axis=1).format(
            "{:.2f}"
        ),
        experiment_folder / "cpf_times.png",
        table_conversion="matplotlib",
        max_cols=50,
        max_rows=150,
    )


if __name__ == "__main__":
    store_results()
