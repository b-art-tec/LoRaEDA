import streamlit as st

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np


import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, RobustScaler

st.markdown(r'''
### The data set

In our studies we are using [Medellin](https://www.mdpi.com/2306-5729/8/1/4) in Chile -- published at [GitHub](https://github.com/magonzalezudem/MDPI_LoRaWAN_Dataset_With_Environmental_Variables). As with any machine-learning system, model quality is limited by the quality, coverage, and structure of the underlying data. Although the Medellín dataset contains measurements from only four devices, this is precisely what makes it a useful cybersecurity case study. It captures a small, well-characterised LoRaWAN deployment over four months in a real urban setting, together with an unusually rich set of environmental, link-budget, and channel variables. This provides detailed visibility into a realistic maintenance scenario, where some latent factors, such as device configuration, placement, and local shadowing, are only weakly randomised, while other factors, including environmental conditions, are well sampled over time. In modelling terms, this creates a setting in which irreducible variability in the observations coexists with uncertainty caused by limited coverage of devices and configurations, that is, a combination of aleatoric and epistemic uncertainty.

''')

st.markdown(r'''

### Overview
Here, we show a snapshot of the complete dataset - as **table** a **pair plot**, and a **correlation heatmap**. Description of all features can be found in a foldable section under the table. Feature selection and engineerig will be presented in another section. 

''')


data = pd.read_csv("LoRaWAN Path Loss Measurement Campaign No Outliers V2.csv", index_col=0)
data = data.drop_duplicates().replace([np.inf, -np.inf], np.nan)


st.dataframe(data.head(100))
st.caption("A snapshot of Medellin dataset.")

data_description = r"""
This dataset contains **990,750 LoRaWAN transmission records** collected from multiple End-Nodes (EN) communicating with a Gateway (GW). Each row corresponds to a single transmission with physical, environmental, and link-quality measurements.

---

### Row Identification
- **row_number** -- Sequential index from 1 to 990,750.  
- **timestamp** -- Date and time of the transmission event.  
- **device_id** -- Identifier of the End-Node (EN) producing the measurement.

### Physical / Geometric Conditions
- **distance** -- Distance between EN and GW (m).  
- **ht** -- EN antenna height above ground (m).  
- **hr** -- GW antenna height above ground (m).

### Link Budget Parameters
- **ptx** -- Transmit power of the EN (dBm).  
- **ltx** -- Transmitter cable & connector losses (dB).  
- **gtx** -- EN antenna gain (dBi), characterised with a VNA.  
- **lrx** -- Receiver cable & connector losses at the GW (dB).  
- **grx** -- GW antenna gain (dBi), characterised with a VNA.  
- **frequency** -- Carrier frequency (Hz).  
- **frame_length** -- Payload size of the LoRaWAN frame (bytes).

### Environmental Variables
- **temperature** -- Air temperature (°C).  
- **rh** -- Relative humidity (%).  
- **bp** -- Barometric pressure (hPa).  
- **pm2_5** -- Particulate matter concentration (µg/m³).

### Received Signal and Quality Indicators
- **rssi** -- Received Signal Strength Indicator at the GW (dBm).  
- **snr** -- Signal-to-Noise Ratio (dB).  
- **toa** -- Time-on-air of the transmission (s).

### Derived Metrics
- **experimental_pl** -- Experimental path loss:  
  $$
  \text{experimental\_pl} = \text{ptx} + \text{gtx} - \text{ltx} + \text{grx} - \text{lrx} - \text{rssi}
  $$
- **energy** -- Estimated energy consumed during the transmission (J).

---

**Source repository:**  
https://github.com/magonzalezudem/MDPI_LoRaWAN_Dataset_With_Environmental_Variables
"""

with st.expander("Show dataset description (expand/collapse)"):
    st.markdown(data_description)

st.markdown(r'''

To optimise the user experience, we display only 1000 random pairs that can be resampled. We also limit the pool of values to random sub-sample of 10 thousands.

''')

log_columns = ["distance", "frequency", "ht", "hr"]

cw_log10 = ColumnTransformer(
    transformers=[
        ("log", FunctionTransformer(func=np.log10, inverse_func=lambda x: 10**x, validate=False, feature_names_out="one-to-one"), log_columns),
    ],
    remainder="passthrough",
    verbose_feature_names_out=False
).set_output(transform="pandas")

# data = cw_log10.fit_transform(data)

encoder = LabelEncoder()
data["label"] = encoder.fit_transform(data["device_id"])

original_df = data.sample(10_000)

# --------- CONFIG ---------
COLOR_BY = "device_id"
# --------------------------

# --- 1. Initialise State ---
if "original" not in st.session_state:
    st.session_state.original = original_df.copy()
    st.session_state.df = st.session_state.original.sample(1000)

if st.button("Resample 1000"):
    st.session_state.df = st.session_state.original.sample(1000)

df = st.session_state.df

# --- 2. Controls ---
num_cols = df.select_dtypes(include=["number", "bool"]).columns.tolist()
if len(num_cols) < 2:
    st.error("Need at least two numeric columns.")
    st.stop()

c1, c2 = st.columns(2)
x_feat = c1.selectbox("Feature X", num_cols, index=15)
y_feat = c2.selectbox("Feature Y", [c for c in num_cols if c != x_feat], index=15)

# --- 3. Helpers ---
def build_color_map(df, color_col):
    palette = px.colors.qualitative.Plotly
    uniq = sorted(df[color_col].dropna().unique())
    return {u: palette[i % len(palette)] for i, u in enumerate(uniq)}


def ecdf_traces(df, feature, color_col, cmap, legend):
    out = []
    for g, sub in df.groupby(color_col):
        vals = np.sort(sub[feature].dropna().to_numpy())
        if len(vals) == 0:
            continue
        cdf = np.arange(1, len(vals) + 1) / len(vals)
        out.append(go.Scatter(
            x=vals,
            y=cdf,
            mode="lines",
            line_shape="hv",
            line_width=2,
            name=str(g),
            marker=dict(color=cmap[g]),
            showlegend=legend,
            hovertemplate=(
                f"{feature}: %{{x}}<br>"
                f"CDF: %{{y:.3f}}<br>"
                f"{color_col}: {g}<extra></extra>"
            ),
        ))
    return out


def scatter_traces(df, x, y, color_col, cmap, legend):
    out = []
    for g, sub in df.groupby(color_col):
        out.append(go.Scatter(
            x=sub[x],
            y=sub[y],
            mode="markers",
            marker=dict(size=5, opacity=0.7, color=cmap[g]),
            name=str(g),
            showlegend=legend,
            hovertemplate=(
                f"{x}: %{{x}}<br>"
                f"{y}: %{{y}}<br>"
                f"{color_col}: {g}<extra></extra>"
            ),
        ))
    return out


# --- 4. Build Pairplot-style Subplots ---
fig = make_subplots(
    rows=2, cols=2,
    horizontal_spacing=0.12,
    vertical_spacing=0.12,
    subplot_titles=[
        f"CDF: {x_feat}",
        f"CDF: {y_feat}",
        f"{y_feat} vs {x_feat}",
        f"{x_feat} vs {y_feat}"
    ]
)

cmap = build_color_map(df, COLOR_BY)

# --- Add traces ---
# Column 1: X feature
for tr in ecdf_traces(df, x_feat, COLOR_BY, cmap, legend=True):
    fig.add_trace(tr, row=1, col=1)

for tr in scatter_traces(df, x_feat, y_feat, COLOR_BY, cmap, legend=False):
    fig.add_trace(tr, row=2, col=1)

# Column 2: Y feature
for tr in ecdf_traces(df, y_feat, COLOR_BY, cmap, legend=False):
    fig.add_trace(tr, row=1, col=2)

for tr in scatter_traces(df, y_feat, x_feat, COLOR_BY, cmap, legend=False):
    fig.add_trace(tr, row=2, col=2)


# --- 5. Axis Linking (correct) ---
# Column 1: CDF(X) <-> scatter(Y vs X)
fig.update_layout(xaxis3=dict(matches="x"))

# Column 2: CDF(Y) <-> scatter(X vs Y)
fig.update_layout(xaxis4=dict(matches="x2"))

# Styling
fig.update_xaxes(showline=True, linewidth=2, zeroline=False)
fig.update_yaxes(showline=True, linewidth=2, zeroline=False)

fig.update_layout(
    height=820,
    margin=dict(l=50, r=30, t=70, b=50),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.12,
        xanchor="center",
        x=0.5
    )
)

st.plotly_chart(fig, use_container_width=True)
st.caption("Pairs and marginals")

corr = data.select_dtypes(include="number").corr()

fig_corr = go.Figure()

fig_corr.add_trace(
    go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale="RdBu",
        zmid=0,
        colorbar=dict(
            title="corr",
            thickness=12
        ),
        hovertemplate=(
            "X: %{x}<br>"
            "Y: %{y}<br>"
            "Corr: %{z:.3f}<extra></extra>"
        ),
    )
)

fig_corr.update_layout(
    width=650,
    height=600,
    margin=dict(l=40, r=40, t=60, b=40),
)

fig_corr.update_xaxes(
    side="top",
    tickangle=45,
    showline=True,
    linewidth=2,
    mirror=True
)
fig_corr.update_yaxes(
    autorange="reversed",
    showline=True,
    linewidth=2,
    mirror=True
)

st.plotly_chart(fig_corr, use_container_width=True)
st.caption("Correlations. Features lacking correlations have only one unique value.")

st.markdown(r'''

After short exploration of different pairs we can recognise three main chellenges in modelling of this data:
- despite large number od samples, many features have limited cardinality (low number of unique values),
- we have mixed data types: discrete and continuous - at least in principle,
- there are meaningfull correlation of various degree.

''')
