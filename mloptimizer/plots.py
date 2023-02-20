import seaborn as sns
import pandas as pd
import numpy as np
import itertools as it
from mloptimizer.genoptimizer import BaseOptimizer


def logbook_to_pandas(logbook):
    data = logbook.chapters["parameters"]
    df = pd.DataFrame(data)
    return df


def plot_search_space(optimizer, height=2, s=25, features: list = None):
    """
    Parameters
    ----------
    optimizer: optimizer object
        A fitted optimizer from :class:`~mloptimizer.genoptimizer.BaseOptimizer`
    height: float, default=2
        Height of each facet
    s: float, default=5
        Size of the markers in scatter plot
    features: list, default=None
        Subset of features to plot, if ``None`` it plots all the features by default
    Returns
    -------
    Pair plot of the used hyperparameters during the search
    """

    if not isinstance(optimizer, BaseOptimizer):
        raise TypeError(
            "optimizer must be a BaseOptimizer instance"
        )

    sns.set_style("white")

    param_names = list(optimizer.get_params().keys())
    df = pd.DataFrame(pd.DataFrame(optimizer.populations).unstack().reset_index(drop=True),
                      columns=["Pop"])
    df2 = pd.DataFrame(df.Pop.tolist(), index=df.index, columns=["a", "fitness"])
    df3 = pd.DataFrame(df2.a.tolist(), columns=param_names)
    df3["fitness"] = df2["fitness"].apply(lambda x: x.values[0])

    g = sns.PairGrid(df3, diag_sharey=False, height=height)
    g = g.map_upper(sns.scatterplot, s=s, color="r", alpha=0.2)
    g = g.map_lower(
        sns.kdeplot,
        shade=True,
        cmap=sns.color_palette("ch:s=.25,rot=-.25", as_cmap=True)
    )
    g = g.map_diag(sns.kdeplot, shade=True, palette="crest", alpha=0.2, color="red")
    return g