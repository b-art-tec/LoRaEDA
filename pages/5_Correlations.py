import streamlit as st
import numpy as np
import pandas as pd

import plotly.graph_objects as go
import plotly.express as px

from sklearn.covariance import LedoitWolf

# =========================
# Page config (consistent)
# =========================
st.set_page_config(
    page_title="Correlations",
    layout="centered",
)

st.title("Correlations & partial correlations")

st.markdown(r"""
#### Why correlations are not enough

A correlation matrix answers: *“Which variables move together?”*  
A **partial correlation** matrix answers: *“Which variables move together after controlling for the others?”*  

""")

# =========================
# Data loading
# (consistent with other pages)
# =========================
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    data = pd.read_csv(path, index_col=0)
    data = data.drop_duplicates().replace([np.inf, -np.inf], np.nan)
    return data


DATA_PATH = "LoRaWAN Path Loss Measurement Campaign No Outliers V2.csv.gz"
data = load_data(DATA_PATH)

# QoI features from your feature-selection page
DEFAULT_QOI = ["distance", "frequency", "frame_length", "rssi", "snr", "toa", "sf"]
AUX_LINK_BUDGET = ["gtx", "ltx", "grx", "lrx"]

# Numeric columns
numeric_cols_all = data.select_dtypes(include=[np.number]).columns.tolist()

# =========================
# Sidebar controls
# =========================
with st.sidebar:
    st.header("Analysis settings")

    # Scope
    devices = sorted(data["device_id"].dropna().unique())

    selected_device = st.selectbox(
        label="Device",
        options=["All devices"] + devices,
        index=0,  # default = All devices
        help="Choose a single device, or 'All devices' to pool everything.",
        placeholder="All devices",
        width="stretch",
    )


    max_n = st.slider(
        "Sample size",
        min_value=200,
        max_value=10000,
        value=1000,
        step=200,
        help="Sampling keeps the page responsive on large datasets."
    )

    seed = st.number_input("Random seed", value=42, step=1)


    st.divider()
    st.subheader("Variables")

    include_aux = st.checkbox(
        "Include TX/RX gains & losses (gtx, ltx, grx, lrx)",
        value=False,
        help="When enabled, we augment QoI with detailed link-budget components."
    )

    # Derive the feature set deterministically (no multiselect)
    selected_vars = [c for c in DEFAULT_QOI if c in numeric_cols_all]
    if include_aux:
        selected_vars += [c for c in AUX_LINK_BUDGET if c in numeric_cols_all]

    log_distance = st.checkbox("Use log10(distance)", value=True)
    log_frequency = st.checkbox("Use log10(frequency)", value=False)

    st.divider()

    # Method settings
    st.subheader("Method")
    corr_kind = st.radio("Correlation type", ["Pearson", "Spearman"], index=0)

    partial_mode = st.radio(
        "Partial correlation computation",
        ["Precision matrix (Ledoit-Wolf shrinkage)", "Off"],
        index=0,
        help="Precision-matrix partial correlations are fast and stable with shrinkage."
    )

    reorder = st.checkbox(
        "Reorder variables by |correlation| (simple heuristic)",
        value=True,
        help="Reorders rows/columns by average absolute correlation for readability (not clustering)."
    )

    abs_scale = st.checkbox("Show absolute values (|r|)", value=False)

    edge_k = st.slider(
        "Top relationships to list",
        min_value=1,
        max_value=20,
        value=4,
        step=1
    )

# =========================
# Preprocessing
# =========================
if len(selected_vars) < 2:
    st.warning("Select at least two numeric variables.")
    st.stop()

df = data.copy()

# Optional device filter (consistent with other pages using device_id as grouping)
if selected_device != "All devices":
    df = df[df["device_id"] == selected_device]

# Sample
if len(df) > max_n:
    df = df.sample(max_n, random_state=int(seed))

# Take selected variables only
X = df[selected_vars].copy()

# Optional log transforms
if log_distance and "distance" in X.columns:
    # Guard against non-positive values
    X.loc[X["distance"] <= 0, "distance"] = np.nan
    X["distance"] = np.log10(X["distance"])

if log_frequency and "frequency" in X.columns:
    X.loc[X["frequency"] <= 0, "frequency"] = np.nan
    X["frequency"] = np.log10(X["frequency"])

# Drop columns with <2 unique values (they create NaNs in correlations)
nunique = X.nunique(dropna=True)
constant_cols = nunique[nunique < 2].index.tolist()
if constant_cols:
    st.info(f"Dropping constant/near-constant columns: {constant_cols}")
    X = X.drop(columns=constant_cols)
    selected_vars = [c for c in selected_vars if c not in constant_cols]

if len(selected_vars) < 2:
    st.warning("After dropping constant columns, fewer than 2 variables remain.")
    st.stop()

# Drop rows with NaNs in selected features (simple and explicit)
X = X.dropna(axis=0, how="any")

if len(X) < 10:
    st.warning("Not enough complete rows after filtering/transforms (need at least ~10).")
    st.stop()

st.caption(f"Using {len(X):,} rows and {len(selected_vars)} variables.")


# =========================
# Helpers
# =========================
def simple_reorder(mat: pd.DataFrame) -> pd.DataFrame:
    """Reorder by average absolute association (readability heuristic)."""
    score = mat.abs().mean(axis=1).sort_values(ascending=False)
    return mat.loc[score.index, score.index]


def corr_matrix(X: pd.DataFrame, kind: str) -> pd.DataFrame:
    if kind.lower() == "spearman":
        return X.corr(method="spearman")
    return X.corr(method="pearson")


def partial_corr_precision(X: pd.DataFrame, kind: str) -> pd.DataFrame:
    """
    Partial correlation via precision matrix:
      P = inv(Cov)
      pcorr_ij = -P_ij / sqrt(P_ii * P_jj)

    For Spearman partial correlation we rank-transform first (Pearson on ranks).
    """
    Z = X.copy()
    if kind.lower() == "spearman":
        Z = Z.rank(axis=0, method="average")

    # Standardize
    Z = (Z - Z.mean(axis=0)) / (Z.std(axis=0, ddof=0) + 1e-12)
    Z = Z.to_numpy()

    lw = LedoitWolf().fit(Z)  # shrinkage covariance estimate
    cov = lw.covariance_
    prec = np.linalg.inv(cov)

    d = np.sqrt(np.diag(prec))
    pcorr = -prec / np.outer(d, d)
    np.fill_diagonal(pcorr, 1.0)

    return pd.DataFrame(pcorr, index=X.columns, columns=X.columns)


def heatmap_fig(mat: pd.DataFrame, title: str, abs_values: bool = False) -> go.Figure:
    vals = mat.values.copy()
    if abs_values:
        vals = np.abs(vals)

    text = np.round(vals, decimals=2)
    
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=vals,
            x=mat.columns,
            y=mat.index,
            colorscale="RdBu" if not abs_values else "Viridis",
            zmid=0 if not abs_values else None,
            colorbar=dict(title="|r|" if abs_values else "r", thickness=15),
            hovertemplate="X: %{x}<br>Y: %{y}<br>Value: %{z:.3f}<extra></extra>",
            text=text,
            texttemplate="%{text:.2f}",
            xgap=4,
            ygap=4
        )
    )
    fig.update_layout(
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        height=650,
        margin=dict(l=40, r=40, t=60, b=40),
        # title=title,
    )
    fig.update_xaxes(side="top", tickangle=45) # , showline=True, linewidth=2, mirror=True)
    fig.update_yaxes(autorange="reversed") # , showline=True, linewidth=2, mirror=True)
    return fig


def top_edges(mat: pd.DataFrame, k: int) -> pd.DataFrame:
    """Return top-k absolute relationships excluding diagonal (no in-place writes)."""
    arr = mat.to_numpy(copy=False)  # copy not needed because we won't modify it
    cols = list(mat.columns)

    # upper triangle indices without diagonal
    iu = np.triu_indices_from(arr, k=1)

    vals = arr[iu]
    out = pd.DataFrame(
        {
            "var_1": [cols[i] for i in iu[0]],
            "var_2": [cols[j] for j in iu[1]],
            "value": vals,
        }
    )

    # Drop NaNs (can occur if a column is constant/invalid after filtering)
    out = out.dropna(subset=["value"])

    # Sort by absolute value and return top-k
    out["abs_value"] = out["value"].abs()
    out = out.sort_values("abs_value", ascending=False).head(k).drop(columns=["abs_value"])

    return out.reset_index(drop=True)


# =========================
# Compute matrices
# =========================
corr = corr_matrix(X, corr_kind)

if reorder:
    corr = simple_reorder(corr)
    X = X[corr.columns]  # keep X in same order

if partial_mode.startswith("Precision"):
    pcorr = partial_corr_precision(X, corr_kind)
    if reorder:
        pcorr = pcorr.loc[corr.index, corr.columns]
else:
    pcorr = None


# =========================
# Visualisations
# =========================
st.markdown("### Correlation matrix")
st.plotly_chart(
    heatmap_fig(corr, f"{corr_kind} correlation", abs_values=abs_scale),
    use_container_width=True,
)

st.markdown("### Strongest correlations")
st.dataframe(top_edges(corr, edge_k), use_container_width=True)

if pcorr is not None:
    st.markdown("### Partial correlation matrix")
    st.plotly_chart(
        heatmap_fig(pcorr, f"{corr_kind} partial correlation (precision matrix)", abs_values=abs_scale),
        use_container_width=True,
    )

    st.markdown("### Strongest partial correlations")
    st.dataframe(top_edges(pcorr, edge_k), use_container_width=True)

st.markdown(r"""
#### Interpretation tips
- If a pair has **high correlation** but **low partial correlation**, the relationship may be **indirect** (mediated by other variables).
- If a pair remains **high in partial correlation**, it suggests a more **direct** (conditional) linear association.
- Discrete/low-cardinality variables (e.g., parameters with few unique values) can create blocky patterns; interpret with care.
""")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def pair_matrix_2x2_fig(
    df: pd.DataFrame,
    x: str,
    y: str,
    color_col: str | None = "device_id",
    marginal: str = "ecdf",          # "ecdf" or "hist"
    opacity: float = 0.65,
    marker_size: int = 5,
    height: int = 820,
    title: str | None = None,
    sync_ranges: bool = True,        # <- key: range synchronization without axis sharing
    pad_frac: float = 0.03,          # small padding around ranges
) -> go.Figure:
    """
    2×2 symmetric mini pair-plot WITHOUT shared axes.
    Layout:
      (1,1) marginal(x)   | (1,2) scatter(y vs x)
      (2,1) scatter(x vs y) | (2,2) marginal(y)
    """

    cols = [x, y]
    if color_col and color_col in df.columns:
        cols.append(color_col)

    dff = df[cols].dropna()

    # If no grouping column, create a single group
    if not (color_col and color_col in dff.columns):
        dff = dff.copy()
        dff["_group"] = "all"
        color_col = "_group"

    palette = px.colors.qualitative.Plotly
    uniq = sorted(dff[color_col].dropna().unique())
    cmap = {u: palette[i % len(palette)] for i, u in enumerate(uniq)}

    def _range(a: pd.Series):
        v = a.to_numpy()
        v = v[np.isfinite(v)]
        if v.size == 0:
            return None
        lo, hi = float(v.min()), float(v.max())
        if lo == hi:
            # avoid degenerate range
            lo -= 1.0
            hi += 1.0
        pad = (hi - lo) * pad_frac
        return [lo - pad, hi + pad]

    xr = _range(dff[x]) if sync_ranges else None
    yr = _range(dff[y]) if sync_ranges else None

    def ecdf_trace(vals: np.ndarray, name: str, color: str, showlegend: bool):
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            return None
        vals = np.sort(vals)
        cdf = np.arange(1, len(vals) + 1) / len(vals)
        return go.Scatter(
            x=vals, y=cdf,
            mode="lines",
            line_shape="hv",
            line_width=2,
            name=name,
            marker=dict(color=color),
            showlegend=showlegend,
            hovertemplate="Value: %{x}<br>CDF: %{y:.3f}<extra></extra>",
        )

    def hist_trace(vals: np.ndarray, name: str, color: str, showlegend: bool):
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            return None
        return go.Histogram(
            x=vals,
            name=name,
            marker=dict(color=color),
            opacity=0.55,
            showlegend=showlegend,
            nbinsx=40,
            hovertemplate="Value: %{x}<br>Count: %{y}<extra></extra>",
        )

    def scatter_trace(xv, yv, name: str, color: str, showlegend: bool, hx: str, hy: str):
        return go.Scatter(
            x=xv, y=yv,
            mode="markers",
            name=name,
            marker=dict(size=marker_size, opacity=opacity, color=color),
            showlegend=showlegend,
            hovertemplate=f"{hx}: %{{x}}<br>{hy}: %{{y}}<extra></extra>",
        )

    fig = make_subplots(
        rows=2, cols=2,
        horizontal_spacing=0.12,
        vertical_spacing=0.12,
        subplot_titles=[
            f"{marginal.upper()}({x})",
            f"{y} vs {x}",
            f"{x} vs {y}",
            f"{marginal.upper()}({y})",
        ],
    )

    # Traces
    for i, (g, sub) in enumerate(dff.groupby(color_col)):
        col = cmap[g]
        showleg = (i == 0)

        vx = sub[x].to_numpy()
        vy = sub[y].to_numpy()

        # Diagonals
        tr1 = hist_trace(vx, str(g), col, showlegend=showleg) if marginal == "hist" else ecdf_trace(vx, str(g), col, showlegend=showleg)
        tr2 = hist_trace(vy, str(g), col, showlegend=False) if marginal == "hist" else ecdf_trace(vy, str(g), col, showlegend=False)
        if tr1 is not None:
            fig.add_trace(tr1, row=1, col=1)
        if tr2 is not None:
            fig.add_trace(tr2, row=2, col=2)

        # Off-diagonal scatters
        fig.add_trace(scatter_trace(sub[x], sub[y], str(g), col, showlegend=False, hx=x, hy=y), row=1, col=2)
        fig.add_trace(scatter_trace(sub[y], sub[x], str(g), col, showlegend=False, hx=y, hy=x), row=2, col=1)

    # Styling
    fig.update_xaxes(showline=True, linewidth=2, zeroline=False)
    fig.update_yaxes(showline=True, linewidth=2, zeroline=False)

    fig.update_layout(
        height=height,
        margin=dict(l=50, r=30, t=70, b=50),
        title=title,
        legend=dict(orientation="h", yanchor="bottom", y=-0.12, xanchor="center", x=0.5),
    )

    # Histogram overlay if chosen
    if marginal == "hist":
        fig.update_layout(barmode="overlay")

    # ---- Range synchronization (NO shared axes) ----
    if sync_ranges and xr is not None and yr is not None:
        # (1,1) marginal(x): x-range = xr
        fig.update_xaxes(range=xr, row=1, col=1)

        # (1,2) scatter(y vs x): x-range = xr, y-range = yr
        fig.update_xaxes(range=xr, row=1, col=2)
        fig.update_yaxes(range=yr, row=1, col=2)

        # (2,1) scatter(x vs y): x-range = yr, y-range = xr
        fig.update_xaxes(range=yr, row=2, col=1)
        fig.update_yaxes(range=xr, row=2, col=1)

        # (2,2) marginal(y): x-range = yr
        fig.update_xaxes(range=yr, row=2, col=2)

    return fig

st.markdown('''
### Pair inspector

Here, one can inspect pairs that seem the most interesting.

''')

c1, c2 = st.columns(2)
x_feat = c1.selectbox("Feature X", selected_vars, index=0, key="inspect_x")
y_feat = c2.selectbox("Feature Y", [c for c in selected_vars if c != x_feat], index=0, key="inspect_y")

# Optional: show correlation value from the already computed matrix
if x_feat in corr.columns and y_feat in corr.columns:
    r = float(corr.loc[x_feat, y_feat])
    title = f"{corr_kind} r({x_feat}, {y_feat}) = {r:.3f}"
else:
    title = None


fig = pair_matrix_2x2_fig(df, x_feat, y_feat, color_col="device_id", title=title)
# or:
# fig = pair_matrix_2x2_fig(base_df, x_feat, y_feat, color_col="device_id", marginal="ecdf", title=title)

st.plotly_chart(fig, use_container_width=True)
