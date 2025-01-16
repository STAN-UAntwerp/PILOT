import click
import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

outputfolder = pathlib.Path(__file__).parent.resolve() / "Output"
figurefolder = outputfolder / "paperplots"

MODELORDER = ["CART", "PILOT", "RF", "RaFFLE", "dRaFFLE", "XGB", "Lasso", "Ridge"]


def load_basetable():
    MODELMAP = {
        "CPILOT": "PILOT",
        "CPF": "RaFFLE",
    }
    DRAFFLE = "CPF - df alpha = 0.5, no blin - max_depth = 20 - max_node_features = 1.0 - n_estimators = 100"

    basetable = pd.concat(
        [
            pd.read_csv(outputfolder / "cpilot_forest_benchmark_v11" / "results.csv"),
            pd.read_csv(
                outputfolder
                / "cpilot_forest_benchmark_v11_linear_models3"
                / "results.csv"
            ),
        ]
    )

    basetable = basetable.groupby(["id", "model"])["r2"].mean().reset_index()
    basetable = basetable.assign(
        basemodel=basetable["model"].str.split("-").str[0].str.strip()
    )

    besttable = (
        basetable.groupby(["id", "basemodel"])["r2"]
        .max()
        .reset_index()
        .rename(columns={"basemodel": "model"})
        .assign(model=lambda df: df["model"].map(lambda m: MODELMAP.get(m, m)))
    )
    draffletable = (
        basetable.loc[basetable["model"] == DRAFFLE]
        .assign(model="dRaFFLE")
        .drop(columns="basemodel")
    )

    return pd.concat([besttable, draffletable])


def get_relative_table(basetable):
    reltable = basetable.pivot(index="id", columns="model", values="r2")
    lintype = pd.Series(
        index=reltable.index,
        data=np.where(
            reltable[["Lasso", "Ridge"]].max(axis=1) > reltable["CART"],
            "Linear",
            "Non-linear",
        ).flatten(),
        name="Type",
    )
    reltable = reltable.clip(0, 1) / reltable.clip(0, 1).max(axis=1).values.reshape(
        -1, 1
    )
    return (
        reltable.reset_index()
        .melt(id_vars="id", var_name="model", value_name="r2")
        .set_index("id")
        .join(lintype)
        .reset_index()
    )


def plot_overall_boxplot(reltable):
    fig, ax = plt.subplots(1, 1, figsize=(20, 12))
    sns.boxplot(data=reltable, x="model", y="r2", ax=ax, order=MODELORDER)
    ax.set_ylabel(r"Relative $R^2$", fontsize=30)
    ax.set_xlabel("Model", fontsize=30)
    ax.tick_params(axis="y", which="major", labelsize=22)
    ax.tick_params(axis="x", which="major", labelsize=26)
    fig.tight_layout()
    fig.savefig(outputfolder / "paperplots" / "boxplots_overall_relative.png", dpi=300)
    fig.savefig(outputfolder / "paperplots" / "boxplots_overall_relative.pdf", dpi=300)


def plot_lin_vs_nonlin_boxplot(reltable):
    fig, ax = plt.subplots(1, 1, figsize=(20, 12))
    sns.boxplot(data=reltable, x="model", y="r2", hue="Type", ax=ax, order=MODELORDER)
    ax.set_ylabel(r"Relative $R^2$", fontsize=30)
    ax.set_xlabel("Model", fontsize=30)
    ax.tick_params(axis="y", which="major", labelsize=22)
    ax.tick_params(axis="x", which="major", labelsize=26)
    _ = ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=2, fontsize=26)
    fig.tight_layout()
    fig.savefig(
        outputfolder / "paperplots" / "boxplots_lin_vs_nonlin_relative_all.png", dpi=300
    )
    fig.savefig(
        outputfolder / "paperplots" / "boxplots_lin_vs_nonlin_relative_all.pdf", dpi=300
    )


@click.command()
@click.option(
    "--overall_boxplot",
    is_flag=True,
    help="Plot overall boxplot with relative R2 values",
)
@click.option(
    "--typed_boxplot",
    is_flag=True,
    help="Plot linear vs non-linear boxplot with relative R2 values",
)
@click.option("--all", is_flag=True, help="Create all plots")
@click.pass_context
def main(ctx, overall_boxplot, typed_boxplot, all):
    if overall_boxplot or all:
        plot_overall_boxplot(ctx.obj["reltable"])
    if typed_boxplot or all:
        plot_lin_vs_nonlin_boxplot(ctx.obj["reltable"])


if __name__ == "__main__":
    figurefolder.mkdir(exist_ok=True)
    basetable = load_basetable()
    reltable = get_relative_table(basetable)
    context = {"basetable": basetable, "reltable": reltable}
    main(obj=context)
