import numpy as np
import pandas as pd
from phdu import savefig
import plotly.graph_objects as go
import plotly.express as px
from phdu.plots.plotly_utils import plotly_default_colors, get_figure, fig_base_layout, mod_simple_axes

from ..preprocessing import load

@savefig
def dataset_split(split=[80, 10, 10], **kwargs):
    data, *_ = load.load_data(**kwargs)
    t_min = data.iloc[data.time_idx.argmin()].t
    t_bins = 365 * int(24/6)
    min_t_idx = int(t_bins*t_min/366)
    t_idx = data.time_idx + min_t_idx
    split_endpoints = np.cumsum(split)
    training_cutoff, val_cutoff, _ = np.percentile(t_idx, split_endpoints).astype(np.int32) # test_cutoff = time_idx.max()

    def year_to_idx(year):
        return (t_bins*year).astype(int)
    years = np.arange(data.YEAR.min(), data.YEAR.max()+1, dtype=int)
    years_idx = year_to_idx(years-years.min())
    years_text = [str(year) for year in years]

    H, edges = np.histogram(t_idx, bins=100, density=True)
    dx = edges[1] - edges[0]
    p = H*dx
    mid = (edges[1:] + edges[:-1])/2

    colors = plotly_default_colors()
    splits = dict(Train=(mid <= training_cutoff),
                  Validation=(mid > training_cutoff) & (mid <= val_cutoff),
                  Test=(mid > val_cutoff))

    fig = get_figure(xaxis_title="Time", yaxis_title="Probability")
    for color, (label, idx) in zip(colors, splits.items()):
        percentage = int(idx.mean() * 100)
        fig.add_trace(go.Bar(x=mid[idx], y=p[idx], marker_color=color, name=label))
        fig.add_annotation(x=np.percentile(mid[idx], 30), y=0.8,
                     xref="x", yref="paper",
                     text=f"<b>{percentage}%</b>",
                     font=dict(family="sans serif", size=32, color=color),
                     showarrow=False)
    fig.update_layout(xaxis=dict(tickvals=years_idx, ticktext=years_text,
                                 range=[-100, mid.max()+200]
                                 ),
                      legend=dict(yanchor="top", orientation="h", y=1.1, xanchor="center", x=0.5),
                      )
    return fig

@savefig
def dataset_split_single_trajectory(title=False, training_cutoff=68, max_encoder_length=28, max_prediction_length=7):
    val_cutoff = training_cutoff + max_prediction_length
    test_cutoff = val_cutoff + max_prediction_length

    data = [
        {"Task": "Train", "Start": 0, "Finish": training_cutoff},
        {"Task": "Validation Input", "Start": training_cutoff - max_encoder_length, "Finish": val_cutoff - max_prediction_length},
        {"Task": "Validation Evaluation", "Start": val_cutoff - max_prediction_length, "Finish": val_cutoff},
        {"Task": "Test Input", "Start": test_cutoff - max_encoder_length, "Finish": test_cutoff - max_prediction_length},
        {"Task": "Test Evaluation", "Start": test_cutoff - max_prediction_length, "Finish": test_cutoff}
    ]
# Convert data to DataFrame
    df = pd.DataFrame(data)

    jan_1 = pd.Timestamp("2021-01-01")
    df_dates = df.set_index('Task').applymap(lambda x: jan_1 + pd.Timedelta(days=x))
    df_dates = df_dates.reset_index()
    df_dates['phase'] = df.Task.str.split().str[0].apply(lambda x: x + " ")
    df_dates['type'] = df.Task.str.split().str[1]
    df_dates['color'] = df_dates['type'].fillna(df_dates['phase'])
    colormap = {'Train ': 'dimgray', 'Input': 'teal', 'Evaluation': 'orange'}

# Create the Gantt chart
    fig = px.timeline(df_dates, x_start="Start", x_end="Finish", y="phase", color="color",
                      color_discrete_map=colormap,
                      labels={"color": ""})
    fig.update_layout(**fig_base_layout())
    fig.update_layout(**mod_simple_axes(fig), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', margin_l=0)
    if title:
        fig.update_layout(title=dict(text='Single trajectory split', x=0.5), margin_t=90)
    else:
        # horizontal legend
        fig.update_layout(legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5))

# Update layout for better readability
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="",
        # hide xticks
        xaxis=dict(tickmode="array", tickvals=[]),
        yaxis=dict(autorange="reversed", tickfont_size=44),
        bargap=0.2,
        width=1300,
        height=800,
        legend_font_size=45,
    )
    return fig

@savefig
def feature_multicollinearity(method='average', corr_method='spearman', alpha=0.05, **loading_kwargs):
    """
    Computes the correlation between the features and plots the heatmap.
    """
    from scipy.cluster.hierarchy import dendrogram, linkage
    c, p = load.feature_corr(ns_to_nan=False, method=corr_method, alpha=alpha, **loading_kwargs)
    corr_linkage = linkage(c, method=method)
    order = dendrogram(corr_linkage, no_plot=True)['leaves']
    c[p > 0.05] = np.nan
    df_ordered = c.iloc[order, order]
    fig = px.imshow(df_ordered, zmin=-1, zmax=1)
    fig = fig.update_layout(**fig_base_layout())
    fig.update_layout(height=1000, width=1300, xaxis_tickfont_size=15, yaxis_tickfont_size=15,
                      xaxis_tickangle=-90,
                      margin=dict(l=100, r=60, t=0, b=100),
                      coloraxis_colorbar=dict(len=0.5, tickfont_size=20, title=dict(text='{} correlation'.format(corr_method.capitalize()), font=dict(size=26))),
                      )
    return fig
