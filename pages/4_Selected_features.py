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
import scipy.stats as stats
from scipy.stats import ks_2samp

st.set_page_config(
    page_title="The dataset",
    layout="centered",
)

st.title("The dataset")

# Data
# ====

data = pd.read_csv("LoRaWAN Path Loss Measurement Campaign No Outliers V2.csv.gz", index_col=0)

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
        margin=dict(l=60, r=30, t=30, b=50),
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

def ks_heatmap(df: pd.DataFrame, feature: str, group_col: str, max_points: int) -> go.Figure:
    devices = sorted(df[group_col].dropna().unique())
    n_dev = len(devices)
    ks_mat = np.zeros((n_dev, n_dev))
    rng = np.random.default_rng(123)

    device_samples = {}
    for dev in devices:
        vals = df[df[group_col] == dev][feature].dropna().to_numpy()
        if len(vals) == 0:
            device_samples[dev] = np.array([])
        elif len(vals) > max_points:
            device_samples[dev] = rng.choice(vals, size=max_points, replace=False)
        else:
            device_samples[dev] = vals

    for i, dev1 in enumerate(devices):
        for j, dev2 in enumerate(devices):
            if i == j:
                ks_mat[i, j] = 0.0
            else:
                s1 = device_samples[dev1]
                s2 = device_samples[dev2]
                if len(s1) == 0 or len(s2) == 0:
                    ks_mat[i, j] = np.nan
                else:
                    ks_mat[i, j] = ks_2samp(s1, s2).statistic

    text = np.round(ks_mat, decimals=2)
    fig = go.Figure(data=go.Heatmap(
        z=ks_mat,
        x=devices,
        y=devices,
        colorscale='Viridis',
        zmin=0, zmax=1,
        colorbar=dict(title="KS statistic", thickness=15),
        xgap=6,
        ygap=6,
        text=text,
        texttemplate="%{text:.2f}",
        hovertemplate="Device %{x} vs %{y}<br>KS = %{z:.3f}<extra></extra>"
    ))
    fig.update_layout(
        # height=420,
        # title=f"Pairwise KS statistic - {feature}",
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        xaxis_title="device",
        yaxis_title="device",
        margin=dict(l=60, r=30, t=60, b=50)
    )
    return fig

st.markdown(r'''
As described in the previous section, we define the training set as the tuple

$$x = (t, \mathrm{id}, d, f, l_{\text{frame}}, \mathrm{RSSI}, \mathrm{SNR}, \mathrm{ToA}, \mathrm{SF})$$

**Here, one can explore every selected feature, its per-device distribution, and dependence on two key variables -- relative distance and time.** You can select the feature and control its visualisation with widgets on the side panel.

- ```NOTE``` Here, we exclude TX and RX gains and losses. They will be important in later stages, and are part of the training set. However, we will test the generative model on data, as they are give. Once we will proceed to predictive maintenance, gains and losses will be important in the estimation, or analysis, of the path loss.
- Here, we mention above quantities as they are important to understand certain anomalies in the data.
- One of the key quantities that can be used in predictive maintenance is consistency of features, mainly link quality measures, with established/historical local radio environment.
- While time dependence might be crucial in detection of anomalous behaviour, below we can see that the nature of this dataset will not allow us to use it as basis for anomaly detection.

''')

st.markdown(r'''
### Raincloud plots
One of the best ways to visualise distributions, esepcailly conditional on a group label, is to use so-called raincloud plots. Such plot combines a probability density estimate, location and spread (box-plot), and the individual observations in a single coherent figure. More about this can be found in later sections, below visualisations.
''')

fig = raincloud_plot(df)
st.plotly_chart(fig, use_container_width=True)

st.markdown(r'''
### Marginal distributions

Distribution of each feature could also be represented by marginal cumulative distribution function --- CDF. As a reminder, consider a random vector $\mathbf{X}=(X_1,\dots,X_d)$. Then the empirical CDF of the component $X_i$ is given by

$$
F_{X_i}(x) \, = \, \mathrm{P}(X_i \le x).
$$

In some sense, marginal CDF is a distribution of a single feature taken from a sample of a multivariate vector.
''')

st.plotly_chart(
    marginal_cdf_plot(df, value_col, GROUP_COL),
    use_container_width=True,
)

st.markdown(r"""
### Distributional similarity across devices (KS heatmap)

The Kolmogorov-Smirnov statistic measures the maximum distance between two empirical CDFs.  
""")
fig_ks = ks_heatmap(df, value_col, GROUP_COL, max_points)
st.plotly_chart(fig_ks, use_container_width=True)

st.markdown(r'''
### Dependence on distance and time

**Relative distance**

''')


st.plotly_chart(
    feature_vs_distance_plot(df, value_col, GROUP_COL),
    use_container_width=True,

)

st.markdown(r'''
- ```NOTE``` In the case of EN3 features related to propagation like RSSI and SNR seem anomalous. Analysis of all qunatities (see The dataset panel) reveals that this device has unusually large cabel losses.

''')



st.markdown(r'''
**Dependance on time**
''')


st.plotly_chart(
    feature_vs_time_plot(df, value_col, GROUP_COL),
    use_container_width=True,
)

st.markdown(r'''

### What exactly is rainclooud plot?

Formally, let $$\{x_i\}_{i=1}^n$$ be a sample of a univariate random variable $$X$$. A raincloud plot consists of three aligned components:

1.  **Density (cloud)**  
    A smoothed estimate of the probability density function
    $$
    \hat{f}(x) = \frac{1}{n h} \sum_{i=1}^{n} K\!\left(\frac{x - x_i}{h}\right),
    $$
    where $$K$$ is a kernel (typically Gaussian) and $$h$$ is the bandwidth.

2.  **Summary statistics (box or bar)**  
    Typically the median and interquartile range, providing a robust summary of location and spread.

3.  **Raw observations (rain)**  
    Individual data points displayed with slight jitter to reveal sample size, clustering, and multimodality.

By combining these elements, the raincloud plot enables simultaneous assessment of **distributional structure**, **between-group differences**, and **within-group variability**, making it particularly well suited for exploratory data analysis and groupwise comparisons.

''')
