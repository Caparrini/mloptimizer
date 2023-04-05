import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def logbook_to_pandas(logbook):
    data = logbook.chapters["parameters"]
    df = pd.DataFrame(data)
    return df


def plot_logbook(logbook):
    df = pd.DataFrame(logbook)
    g = sns.lineplot(df.drop(columns=['gen', 'nevals']))
    return g.get_figure()


def plot_search_space(populations_df: pd.DataFrame, height=2, s=25, features: list = None):
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
    sns.set_style("white")
    g = sns.PairGrid(populations_df, diag_sharey=False, height=height)
    g = g.map_upper(sns.scatterplot, s=s, color="r", alpha=0.2)
    g = g.map_lower(
        sns.kdeplot,
        shade=True,
        cmap=sns.color_palette("ch:s=.25,rot=-.25", as_cmap=True)
    )
    g = g.map_diag(sns.kdeplot, shade=True, palette="crest", alpha=0.2, color="red")
    return g
