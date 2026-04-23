import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from plotly.subplots import make_subplots
import plotly.express as px

st.set_page_config(
    page_title="The dataset",
    layout="centered",
)

st.title("The dataset")

# Data
# ====

data = pd.read_csv(
    "LoRaWAN Path Loss Measurement Campaign No Outliers V2.csv",
    index_col=0,
)
data = data.drop_duplicates().replace([np.inf, -np.inf], np.nan)

qoi = ["timestamp", "device_id", "distance", "frequency", "frame_length", "rssi", "snr", "toa", "sf"]
df = data.sample(10_000, random_state=42).loc[:, qoi]

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
GROUP_COL = "device_id"
order_by = "alphabetical"

# Sidebar — ONLY continuous tuning controls
# =========================================
with st.sidebar:
    st.header("Plot tuning")

    value_col = st.selectbox(
    "Feature",
    numeric_cols,
    index=numeric_cols.index("rssi") if "rssi" in numeric_cols else 0,
    )
    
    max_points = st.slider(
        "Max points per device (for KDE + rain)",
        50, 1500, 500, 50,
    )

    x_spread = st.slider(
        "Rain X-axis spread",
        0.01, 0.20, 0.05, 0.01,
        help="Horizontal dispersion of sampled points (kept non-overlapping)",
    )

    fig_height = st.slider(
        "Figure height",
        500, 1100, 720, 20,
    )

# Utilities
# =========

def to_rgba(color: str, alpha: float) -> str:
    if color.startswith("rgb"):
        return color.replace("rgb", "rgba").replace(")", f", {alpha})")
    c = color.lstrip("#")
    r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def downsample(x: np.ndarray, max_n: int, rng: np.random.Generator) -> np.ndarray:
    if len(x) <= max_n:
        return x
    return x[rng.choice(len(x), size=max_n, replace=False)]

def build_color_map(df, color_col):
    palette = px.colors.qualitative.Plotly
    uniq = sorted(df[color_col].dropna().unique())
    return {u: palette[i % len(palette)] for i, u in enumerate(uniq)}

# Plotting functions
# ==================

def raincloud_plot(df: pd.DataFrame) -> go.Figure:

    plot_df = df[[GROUP_COL, value_col]].dropna()
    grouped = plot_df.groupby(GROUP_COL)[value_col]

    if order_by == "median":
        devices = grouped.median().sort_values().index.tolist()
    elif order_by == "mean":
        devices = grouped.mean().sort_values().index.tolist()
    else:
        devices = sorted(plot_df[GROUP_COL].astype(str).unique())

    xs = np.arange(len(devices), dtype=float)
    x_map = dict(zip(devices, xs))

    palette = px.colors.qualitative.Safe
    colors = {d: palette[i % len(palette)] for i, d in enumerate(devices)}

    fig = go.Figure()
    rng = np.random.default_rng(123)

    # semantic order: data → box → KDE
    DATA_OFFSET = -0.20
    BOX_OFFSET  =  0.00
    KDE_OFFSET  = +0.15

    for d in devices:
        y = plot_df.loc[plot_df[GROUP_COL] == d, value_col].to_numpy()
        if len(y) == 0:
            continue

        y_ds = downsample(y, max_points, rng)
        x0 = x_map[d]
        col = colors[d]

        # raw data
        fig.add_trace(
            go.Scatter(
                x=x0 + DATA_OFFSET + rng.uniform(-x_spread, x_spread, size=len(y_ds)),
                y=y_ds,
                mode="markers",
                marker=dict(
                    size=4,
                    color=to_rgba(col, 0.55),
                    line=dict(width=0.3, color=to_rgba(col, 0.85)),
                ),
                showlegend=False,
            )
        )

        # box plot (narrow, reference)
        fig.add_trace(
            go.Box(
                x=np.full(len(y_ds), x0 + BOX_OFFSET),
                y=y_ds,
                width=0.15,
                boxpoints=False,
                quartilemethod="linear",
                line=dict(color=col, width=1.4),
                fillcolor=to_rgba(col, 0.18),
                marker_color=col,
                showlegend=False,
            )
        )

        # KDE (from downsample only)
        fig.add_trace(
            go.Violin(
                x=np.full(len(y_ds), x0 + KDE_OFFSET),
                y=y_ds,
                side="positive",
                width=0.55,
                scalegroup="all",
                points=False,
                spanmode="hard",
                line=dict(color=col, width=1.2),
                fillcolor=to_rgba(col, 0.35),
                showlegend=False,
            )
        )

    fig.update_layout(
        height=fig_height,
        violingap=0,
        violinmode="overlay",
        boxmode="overlay",
        # title=dict(
        #     text=f"{value_col} by device_id",
        #     x=0.01,
        #     xanchor="left",
        # ),
        xaxis=dict(
            title="device_id",
            tickmode="array",
            tickvals=xs.tolist(),
            ticktext=[str(d) for d in devices],
            range=[-0.9, len(devices) - 0.1],
            showgrid=False,
        ),
        yaxis=dict(
            title=value_col,
            showgrid=True,
        ),
    )

    return fig



def marginal_cdf_plot(df, feature, group_col):
    cmap = build_color_map(df, group_col)
    fig = go.Figure()

    for g, sub in df.groupby(group_col):
        vals = np.sort(sub[feature].dropna().to_numpy())
        if len(vals) == 0:
            continue

        cdf = np.arange(1, len(vals) + 1) / len(vals)

        fig.add_trace(
            go.Scatter(
                x=vals,
                y=cdf,
                mode="lines",
                line_shape="hv",
                line_width=2,
                name=str(g),
                marker_color=cmap[g],
                hovertemplate=(
                    f"{group_col}: {g}<br>"
                    f"{feature}: %{{x:.2f}}<br>"
                    f"CDF: %{{y:.3f}}"
                ),
            )
        )

    fig.update_xaxes(showline=True, linewidth=2, zeroline=False)
    fig.update_yaxes(showline=True, linewidth=2, zeroline=False)

    fig.update_layout(
        height=420,
        margin=dict(l=60, r=30, t=60, b=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5,
        ),
    )

    return fig

def feature_vs_distance_plot(df, feature, group_col, dist_col="distance"):
    cmap = build_color_map(df, group_col)
    fig = go.Figure()

    for g, sub in df.groupby(group_col):
        fig.add_trace(
            go.Scatter(
                x=sub[dist_col],
                y=sub[feature],
                mode="markers",
                marker=dict(size=5, opacity=0.7, color=cmap[g]),
                name=str(g),
                hovertemplate=(
                    f"{group_col}: {g}<br>"
                    f"{dist_col}: %{{x:.1f}}<br>"
                    f"{feature}: %{{y:.2f}}"
                ),
            )
        )

    fig.update_xaxes(
        title="distance [m]",
        showline=True,
        linewidth=2,
        zeroline=False,
    )
    fig.update_yaxes(
        title=feature,
        showline=True,
        linewidth=2,
        zeroline=False,
    )

    fig.update_layout(
        height=420,
        margin=dict(l=60, r=30, t=60, b=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5,
        ),
    )

    return fig

def feature_vs_time_plot(df, feature, group_col, time_col="timestamp"):
    # ensure datetime conversion (non-destructive)
    t = pd.to_datetime(df[time_col], errors="coerce")
    dff = df.copy()
    dff[time_col] = t

    cmap = build_color_map(dff, group_col)
    fig = go.Figure()

    for g, sub in dff.groupby(group_col):
        fig.add_trace(
            go.Scatter(
                x=sub[time_col],
                y=sub[feature],
                mode="markers",
                marker=dict(
                    size=4,
                    opacity=0.6,
                    color=cmap[g],
                ),
                name=str(g),
                hovertemplate=(
                    f"{group_col}: {g}<br>"
                    f"time: %{{x|%Y-%m-%d %H:%M:%S}}<br>"
                    f"{feature}: %{{y:.2f}}"
                ),
            )
        )

    fig.update_xaxes(
        title="time",
        showline=True,
        linewidth=2,
        zeroline=False,
    )
    fig.update_yaxes(
        title=feature,
        showline=True,
        linewidth=2,
        zeroline=False,
    )

    fig.update_layout(
        height=420,
        margin=dict(l=60, r=30, t=60, b=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5,
        ),
    )

    return fig

st.markdown(r'''
As described in the previous section, We define the training set as the tuple

$$x = (t, \mathrm{id}, d, f, l_{\text{frame}}, \mathrm{RSSI}, \mathrm{SNR}, \mathrm{ToA}, \mathrm{SF})$$

Here, one can explore every selected feature, its per-device distribution, and dependence on two key varaibles -- relative distance and time.

''')

st.markdown(r'''
### Raincloud plots
One of the best way to decribe 
''')


fig = raincloud_plot(df)
st.plotly_chart(fig, use_container_width=True)

st.markdown(r'''
Let $\mathbf{X}=(X_1,\dots,X_d)$ be a random vector. The *marginal* CDF of component $X_i$ is

$$
F_{X_i}(x) \, = \, \mathrm{P}(X_i \le x).
$$

In some sense it is the CDF of $X_i$ alone, i.e. the joint distribution with all other coordinates “ignored”.
''')

st.plotly_chart(
    marginal_cdf_plot(df, value_col, GROUP_COL),
    use_container_width=True,
)


st.plotly_chart(
    feature_vs_distance_plot(df, value_col, GROUP_COL),
    use_container_width=True,

)


st.plotly_chart(
    feature_vs_time_plot(df, value_col, GROUP_COL),
    use_container_width=True,
)
