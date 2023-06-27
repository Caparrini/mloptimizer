import seaborn as sns
import pandas as pd
import numpy as np
import plotly.graph_objects as go


def logbook_to_pandas(logbook):
    """
    :param logbook: A logbook from :class:`~mloptimizer.genoptimizer.BaseOptimizer`
    :return: A pandas dataframe with the logbook
    """
    data = logbook.chapters["parameters"]
    df = pd.DataFrame(data)
    return df


def plotly_logbook(logbook, population):
    """
    :param logbook: A logbook from :class:`~mloptimizer.genoptimizer.BaseOptimizer`
    :return: A plotly line plot of the logbook
    """
    df = pd.DataFrame(logbook)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['gen'], y=df['avg'],
                             line_color='rgb(0,0,255)',
                             mode='lines+markers',
                             name='Avg'))
    fig.add_trace(go.Scatter(x=df['gen'], y=df['min'],
                             line_color='rgb(255,0,0)',
                             mode='lines+markers',
                             name='Min'))
    fig.add_trace(go.Scatter(x=df['gen'], y=df['max'],
                             line_color='rgb(0,100,80)',
                             mode='lines+markers',
                             name='Max'))
    fig.add_trace(go.Scatter(
        x=df['gen'],
        y=df['avg'],
        fill='tonexty',
        fillcolor='rgba(0,100,80,0.2)',
        line_color='rgba(255,255,255,0)',
        showlegend=False,
        name='Avg',
    ))
    fig.add_trace(go.Scatter(
        x=df['gen'],
        y=df['min'],
        fill='tonexty',
        fillcolor='rgba(255,0,00,0.2)',
        line_color='rgba(255,255,255,0)',
        showlegend=False,
        name='Min',
    ))
    fig.add_trace(go.Scatter(
        x=df['gen'],
        y=df['max'].max() * np.ones(len(df['gen'])),
        line=dict(color='rgba(0,0,0,1)'),
        mode='lines',
        showlegend=False,
        name='Max value',
    ))
    fig.add_trace(go.Scatter(
        x=df['gen'],
        y=df['min'].min() * np.ones(len(df['gen'])),
        line=dict(color='rgba(0,0,0,1)'),
        mode='lines',
        showlegend=False,
        name='Min value',
    ))

    fig.add_trace(go.Scatter(
        x=population['population'], y=population['fitness'],
        name='Populations',
        mode='markers',
        marker_color='rgba(0, 0, 0, .1)'
    ))

    fig.update_layout(
        title="Fitness evolution",
        xaxis_title="Generation",
        yaxis_title="Fitness",
        # yaxis_range=[df['min'].min(), 1],
        legend_title="Fitness agg",
        font=dict(
            family="Arial",
            size=18,
            color="Black"
        )
    )

    return fig


def plot_logbook(logbook):
    """
    :param logbook: A logbook from :class:`~mloptimizer.genoptimizer.BaseOptimizer`
    :return: A line plot of the logbook
    """
    df = pd.DataFrame(logbook)
    g = sns.lineplot(df.drop(columns=['gen', 'nevals']))
    return g.get_figure()


def plotly_search_space(populations_df: pd.DataFrame, features: list = None):
    """
    :param pd.DataFrame populations_df: A dataframe with the population
    :param list features: The features to plot
    :return: A pairplot of the search space
    """
    if features is None:
        features = populations_df.columns
    fig = go.Figure(data=go.Splom(
        dimensions=[dict(label=feat, values=populations_df[feat]) for feat in features],
        text=populations_df['fitness'],
        marker=dict(color=populations_df['fitness'],
                    colorscale='Blues', showscale=True, opacity=0.5)
    ))

    fig.update_layout(
        title='Search space',
        dragmode='select',
        width=1000,
        height=1000,
        hovermode='closest',
    )

    return fig


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
