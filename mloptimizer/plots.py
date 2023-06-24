import seaborn as sns
import pandas as pd


def logbook_to_pandas(logbook):
    """
    :param logbook: A logbook from :class:`~mloptimizer.genoptimizer.BaseOptimizer`
    :return: A pandas dataframe with the logbook
    """
    data = logbook.chapters["parameters"]
    df = pd.DataFrame(data)
    return df


def plot_logbook(logbook):
    """
    :param logbook: A logbook from :class:`~mloptimizer.genoptimizer.BaseOptimizer`
    :return: A line plot of the logbook
    """
    df = pd.DataFrame(logbook)
    g = sns.lineplot(df.drop(columns=['gen', 'nevals']))
    return g.get_figure()


def plot_search_space(populations_df: pd.DataFrame, height=2, s=25, features: list = None):
    """
    :param pd.DataFrame populations_df: A dataframe with the population
    :param int height: The height of the plot
    :param int s: The size of the points
    :param list features: The features to plot
    :return: A pairplot of the search space
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
