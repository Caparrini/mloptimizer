import seaborn as sns
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy import stats


def logbook_to_pandas(logbook):
    """
    Function to transform a deap logbook to a pandas dataframe.

    Parameters
    ----------
    logbook : deap.tools.Logbook
        The logbook to transform

    Returns
    -------
    df : pd.DataFrame
        The logbook as a pandas dataframe
    """
    data = logbook.chapters["parameters"]
    df = pd.DataFrame(data)
    return df



def plotly_logbook(logbook, population):
    """
    Generate a plot from the logbook with no additional points for "Max per Gen" and "Min per Gen".
    The population points grow based on the number of overlapping individuals.

    Parameters
    ----------
    logbook : deap.tools.Logbook
        The logbook to plot
    population : pd.DataFrame
        The population to plot

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The figure
    """
    # --- Prep ---------------------------------------------------------------
    #df = pd.DataFrame(logbook).rename(columns=str.lower).sort_values("gen")
    df = pd.DataFrame(logbook)
    xcat = df["gen"].astype(str)


    pop = population.copy()
    pop["population"] = pd.to_numeric(pop["population"], errors="coerce")
    pop["fitness"] = pd.to_numeric(pop["fitness"], errors="coerce")
    q = pop.groupby("population")["fitness"].quantile([0.25, 0.75]).unstack()
    q = q.rename(columns={0.25: "q1", 0.75: "q3"}).reindex(df["gen"]).interpolate()
    lower = q["q1"].clip(0, 1)
    upper = q["q3"].clip(0, 1)
    band_name = "Avg (IQR)"

    # y-range for readability
    data_min = float(min(df["min"].min(), lower.min()))
    data_max = float(max(df["max"].max(), upper.max()))
    y_min = data_min - (data_max - data_min)*0.1
    y_max = data_max + (data_max - data_min)*0.1

    # palette
    c_avg = "rgb(56,128,255)"  # blue
    c_best_gen = "rgb(147,112,219)"  # purple
    c_best_all = "rgb(255,140,0)"  # orange
    c_band = "rgba(56,128,255,0.10)"  # lighter so it doesn’t hide violins/boxes

    fig = go.Figure()

    # --- draw BAND FIRST (in the back) -------------------------------------
    fig.add_trace(go.Scatter(x=xcat, y=upper, mode="lines",
                             line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xcat, y=lower, mode="lines", name=band_name,
                             fill="tonexty", fillcolor=c_band, line=dict(width=0),
                             hoverinfo="skip"))

    # --- Individuals as DISTRIBUTIONS --------------------------------------
    tall = population.copy()
    tall["gen"] = pd.to_numeric(tall["population"], errors="coerce").astype("Int64")
    tall["fitness"] = pd.to_numeric(tall["fitness"], errors="coerce")
    tall = tall.dropna(subset=["gen", "fitness"])
    tall["gen"] = tall["gen"].astype(str)

    # Choose violin when n>=10, else box (violins look bad with tiny samples)
    counts = tall.groupby("gen")["fitness"].size()
    use_violin = (counts.min() >= 10)

    if use_violin:
        fig.add_trace(go.Violin(
            x=tall["gen"], y=tall["fitness"],
            name="Individuals",
            points=False,
            box_visible=True,
            meanline_visible=True,
            opacity=0.7,
            line=dict(width=1.2, color="rgba(30,30,30,0.7)"),
            fillcolor="rgba(90,160,255,0.45)",
            width=0.7,
            spanmode="hard",
            scalemode="width",
        ))
        fig.update_layout(violinmode="overlay")
    else:
        fig.add_trace(go.Box(
            x=tall["gen"], y=tall["fitness"],
            name="Individuals",
            boxpoints=False,  # keep it clean
            opacity=0.8,
            line=dict(width=1.2, color="rgba(30,30,30,0.7)"),
            fillcolor="rgba(90,160,255,0.35)",
        ))

    # --- Lines on top -------------------------------------------------------
    fig.add_trace(go.Scatter(
        x=xcat, y=df["avg"], name="Average",
        mode="lines+markers",
        line=dict(color=c_avg, width=4),
        marker=dict(size=6, color=c_avg)
    ))

    fig.add_trace(go.Scatter(
        x=xcat, y=df["max"], name="Best per generation",
        mode="markers",
        line=dict(color=c_best_gen, width=2.5)
    ))

    # --- Layout -------------------------------------------------------------
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Generation",
        yaxis_title="Fitness",
        xaxis=dict(type="category",
                   categoryorder="array",
                   categoryarray=[str(g) for g in df["gen"]]),
        yaxis=dict(range=[y_min, y_max]),
        legend=dict(orientation="h", y=1.02, yanchor="bottom",
                    x=0.5, xanchor="center", font=dict(size=14)),
        #margin=dict(l=60, r=40, t=50, b=60),
        width=950, height=560,
        font=dict(family="Arial", size=15)
    )

    return fig


def plot_logbook(logbook):
    """
    Generate sns figure from logbook. Evolution of fitness and population.

    Parameters
    ----------
    logbook : deap.tools.Logbook
        The logbook to plot

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The figure
    """
    df = pd.DataFrame(logbook)
    g = sns.lineplot(data=df.drop(columns=['gen', 'nevals']), palette="colorblind")
    return g.get_figure()


def plotly_search_space(populations_df: pd.DataFrame, features: list = None, s=25,
                        marker_color: str = "blue",
                        colorscale: str = "Blues",
                        kde_line_color: str = "rgba(31, 119, 180, 1.0)",
                        kde_fillcolor: str = "rgba(31, 119, 180, 0.2)",
                        ):
    """
    Generate plotly figure from populations dataframe and features. Search space.

    Parameters
    ----------
    populations_df : pd.DataFrame
        The dataframe with the population data
    features : list
        The features to plot (column names of the dataframe)
    s : int
        The size of the points
    marker_color : str
        The color of the scatter plot markers
    colorscale : str
        The colorscale for the KDE contour plots
    kde_line_color : str
        The line color for the KDE distribution plots
    kde_fillcolor : str
        The fill color for the KDE distribution plots
    Returns
    -------
    fig : plotly.graph_objects.Figure
        The figure
    """
    # Use specified features or default to all numeric columns
    if features is None:
        features = populations_df.select_dtypes(include="number").columns.tolist()
    else:
        missing_features = [f for f in features if f not in populations_df.columns]
        if missing_features:
            raise ValueError(f"Features not found in dataframe: {missing_features}")
        features = list(features)

    n_features = len(features)

    # Smart scaling based on number of features
    if n_features <= 3:
        fig_size = 700
        font_size = 11
        title_size = 12
        marker_size = s / 5
    elif n_features <= 5:
        fig_size = 900
        font_size = 9
        title_size = 10
        marker_size = s / 6
    elif n_features <= 8:
        fig_size = 1000
        font_size = 7
        title_size = 8
        marker_size = s / 8
    else:
        fig_size = 1100
        font_size = 6
        title_size = 7
        marker_size = s / 10

    # Create subplots with tighter spacing for many features
    spacing = max(0.01, 0.03 - (n_features * 0.002))

    fig = make_subplots(
        rows=n_features,
        cols=n_features,
        vertical_spacing=spacing,
        horizontal_spacing=spacing
    )

    for i, row_feature in enumerate(features):
        for j, col_feature in enumerate(features):
            row_idx = i + 1
            col_idx = j + 1

            if i < j:  # Upper triangle - scatter plots
                fig.add_trace(
                    go.Scattergl(
                        x=populations_df[col_feature],
                        y=populations_df[row_feature],
                        mode='markers',
                        marker=dict(
                            size=marker_size,
                            color=marker_color,
                            opacity=0.2
                        ),
                        showlegend=False,
                        hovertemplate=f'{col_feature}: %{{x}}<br>{row_feature}: %{{y}}<extra></extra>'
                    ),
                    row=row_idx,
                    col=col_idx
                )

            elif i > j:  # Lower triangle - KDE contour plots
                x_data = populations_df[col_feature].dropna()
                y_data = populations_df[row_feature].dropna()

                try:
                    xmin, xmax = x_data.min(), x_data.max()
                    ymin, ymax = y_data.min(), y_data.max()

                    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
                    positions = np.vstack([xx.ravel(), yy.ravel()])
                    values = np.vstack([x_data, y_data])
                    kernel = stats.gaussian_kde(values)
                    f = np.reshape(kernel(positions).T, xx.shape)

                    fig.add_trace(
                        go.Contour(
                            x=xx[:, 0],
                            y=yy[0, :],
                            z=f.T,
                            colorscale=colorscale,
                            showscale=False,
                            contours=dict(
                                coloring='fill',
                                showlines=False
                            ),
                            opacity=0.6,
                            hovertemplate=f'{col_feature}: %{{x}}<br>{row_feature}: %{{y}}<extra></extra>'
                        ),
                        row=row_idx,
                        col=col_idx
                    )
                except:
                    pass

            else:  # Diagonal - KDE distribution plots
                data = populations_df[row_feature].dropna()

                try:
                    kde = stats.gaussian_kde(data)
                    x_range = np.linspace(data.min(), data.max(), 200)
                    y_kde = kde(x_range)

                    fig.add_trace(
                        go.Scatter(
                            x=x_range,
                            y=y_kde,
                            fill='tozeroy',
                            mode='lines',
                            line=dict(color=kde_line_color, width=1),
                            fillcolor=kde_fillcolor,
                            showlegend=False,
                            hovertemplate=f'{row_feature}: %{{x}}<br>Density: %{{y}}<extra></extra>'
                        ),
                        row=row_idx,
                        col=col_idx
                    )
                except:
                    pass

            # Update axis labels only on edges
            if i == n_features - 1:  # Bottom row
                fig.update_xaxes(
                    title_text=col_feature,
                    title_font=dict(size=title_size),
                    tickfont=dict(size=font_size),
                    row=row_idx,
                    col=col_idx
                )
            if j == 0:  # Left column
                fig.update_yaxes(
                    title_text=row_feature,
                    title_font=dict(size=title_size),
                    tickfont=dict(size=font_size),
                    row=row_idx,
                    col=col_idx
                )

    # Fixed size with smart margins
    margin_size = max(40, 80 - n_features * 3)

    fig.update_layout(
        height=fig_size,
        width=fig_size,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=margin_size, r=20, t=20, b=margin_size)
    )

    # Update all axes
    fig.update_xaxes(showgrid=False, zeroline=False, tickfont=dict(size=font_size))
    fig.update_yaxes(showgrid=False, zeroline=False, tickfont=dict(size=font_size))

    return fig


def plot_search_space(populations_df: pd.DataFrame, height=2, s=25):
    """
    Generate sns figure from populations dataframe and features. Search space.

    Parameters
    ----------
    populations_df : pd.DataFrame
        The dataframe with the population
    height : int
        The height of the figure
    s : int
        The size of the points

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The figure
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
