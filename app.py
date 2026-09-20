"""
app.py — Interactive OCPI hedging demo (Streamlit).

Run from the project folder (same place as hedge_sim.py and ocpi_pull.py):
    python -m pip install streamlit plotly
    python -m streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from hedge_sim import HOURS_PER_MONTH, calibrate, optimal_hedge_ratio, simulate

CSV_PATH = Path("data/ocpi_history.csv")
BLUE, ORANGE, GRAY = "#2E6FBA", "#E8833A", "#8A8F98"

st.set_page_config(page_title="OCPI Hedging Simulator", page_icon="📉", layout="wide")


# ----------------------------------------------------------------------------
# Data helpers (cached so sliders stay fast)
# ----------------------------------------------------------------------------
@st.cache_data
def load_history(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["settle_date"] = pd.to_datetime(df["settle_date"])
    return df


@st.cache_data
def cached_calibration(path: str, gpu: str) -> dict:
    return calibrate(Path(path), gpu)


@st.cache_data
def cached_sim(start_price, vol, rho, n_gpus, months, n_sims, drift, buyer_vol, hedge_ratio):
    return simulate(start_price, vol, rho, n_gpus, months, n_sims, drift,
                    buyer_vol, hedge_ratio)


@st.cache_data
def cached_sweep(start_price, vol, n_gpus, months, drift, buyer_vol, n_sims=4000):
    """Hedge effectiveness across correlations (fewer sims to stay responsive)."""
    rows = []
    for rho in np.round(np.arange(1.0, -0.001, -0.05), 2):
        base = dict(start_price=start_price, vol=vol, rho=float(rho), n_gpus=n_gpus,
                    months=months, n_sims=n_sims, drift=drift, buyer_vol=buyer_vol)
        full = simulate(**base, hedge_ratio=1.0)
        opt = simulate(**base, hedge_ratio=optimal_hedge_ratio(rho, vol, buyer_vol))
        var_u = full["unhedged"].var()
        rows.append({"rho": rho,
                     "full": 1 - full["hedged"].var() / var_u,
                     "optimal": 1 - opt["hedged"].var() / var_u})
    return pd.DataFrame(rows)


def refresh_data() -> str:
    """Pull the latest OCPI history from Ornn's free API into the local CSV."""
    from ocpi_pull import OrnnClient, history_to_frame, merge_into_store
    client = OrnnClient()
    frames = [history_to_frame(client.index_history(g), g) for g in client.gpu_types()]
    store = merge_into_store(pd.concat(frames, ignore_index=True), CSV_PATH)
    return f"Updated: {len(store)} rows through {store['settle_date'].max()}"


def money(x: float) -> str:
    return f"${x / 1e6:,.2f}M"


# ----------------------------------------------------------------------------
# Sidebar: scenario inputs
# ----------------------------------------------------------------------------
st.sidebar.header("Scenario")

if not CSV_PATH.exists():
    st.error(f"Couldn't find {CSV_PATH}. Run `python ocpi_pull.py` first, "
             "or click the refresh button in the sidebar.")

if st.sidebar.button("Refresh OCPI data from Ornn"):
    try:
        with st.spinner("Pulling from Ornn's API..."):
            msg = refresh_data()
        st.cache_data.clear()
        st.sidebar.success(msg)
    except Exception as exc:  # noqa: BLE001
        st.sidebar.error(f"Refresh failed: {exc}")

if not CSV_PATH.exists():
    st.stop()

history = load_history(str(CSV_PATH))
gpus = sorted(history["gpu"].unique())
gpu = st.sidebar.selectbox("GPU", gpus, index=gpus.index("H100 SXM") if "H100 SXM" in gpus else 0)
cal = cached_calibration(str(CSV_PATH), gpu)

n_gpus = st.sidebar.slider("Number of GPUs", 50, 5000, 500, step=50)
months = st.sidebar.slider("Contract length (months)", 3, 24, 12)

st.sidebar.subheader("Basis risk")
rho = st.sidebar.slider("Correlation of buyer's price with OCPI (ρ)", 0.0, 1.0, 0.80, 0.05,
                        help="1.0 = the buyer pays exactly the index. Lower = their "
                             "regional/provider price drifts away from OCPI.")

st.sidebar.subheader("Hedge")
hedge_mode = st.sidebar.radio("Hedge ratio", ["Optimal (h* = ρ·σ_buyer/σ_index)", "Custom"])

st.sidebar.subheader("Market assumptions")
vol_default = float(round(cal["vol_weekly_ann"], 2))
vol = st.sidebar.slider("Index volatility (annualized)", 0.05, 1.20, vol_default, 0.01,
                        help=f"Default is calibrated from weekly OCPI returns ({vol_default:.0%}).")
buyer_vol_mult = st.sidebar.slider("Buyer price volatility vs index", 0.5, 2.0, 1.0, 0.05)
drift = st.sidebar.slider("Expected annual price drift", -0.5, 0.5, 0.0, 0.05,
                          help="0 = no view on direction. Hedge locks in today's price.")
n_sims = st.sidebar.select_slider("Simulated futures", [2000, 5000, 10000, 20000], 10000)

buyer_vol = vol * buyer_vol_mult
h_opt = optimal_hedge_ratio(rho, vol, buyer_vol)
if hedge_mode == "Custom":
    hedge_ratio = st.sidebar.slider("Share of usage hedged", 0.0, 1.5, 1.0, 0.05)
else:
    hedge_ratio = h_opt
    st.sidebar.caption(f"Optimal ratio at these settings: **{h_opt:.2f}**")

# ----------------------------------------------------------------------------
# Run the simulation
# ----------------------------------------------------------------------------
res = cached_sim(cal["start_price"], vol, rho, n_gpus, months, n_sims, drift,
                 buyer_vol, hedge_ratio)
u, h = res["unhedged"], res["hedged"]
var_cut = 1 - h.var() / u.var()
gpu_hours = n_gpus * HOURS_PER_MONTH * months
budget = gpu_hours * cal["start_price"]

# ----------------------------------------------------------------------------
# Header + headline numbers
# ----------------------------------------------------------------------------
st.title("Hedging GPU compute costs with the Ornn Compute Price Index")
st.write(
    f"A buyer needs **{n_gpus:,} {gpu}s for {months} months** "
    f"({gpu_hours / 1e6:.2f}M GPU-hours). At today's OCPI of "
    f"**${cal['start_price']:.2f}/hr**, that's a **{money(budget)}** budget. "
    f"How much of the price risk can an OCPI hedge remove?"
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Bad case, unhedged (95th pct)", money(np.percentile(u, 95)),
          f"{money(np.percentile(u, 95) - budget)} over budget", delta_color="inverse")
c2.metric("Bad case, hedged (95th pct)", money(np.percentile(h, 95)),
          f"{money(np.percentile(h, 95) - budget)} over budget", delta_color="inverse")
c3.metric("Budget variance removed", f"{var_cut:.0%}",
          f"theory ρ² = {rho**2:.0%}" if hedge_mode != "Custom" else None,
          delta_color="off")
c4.metric("Hedge ratio used", f"{hedge_ratio:.2f}",
          f"optimal = {h_opt:.2f}", delta_color="off")

if hedge_mode == "Custom" and var_cut < 0:
    st.warning("At this correlation, the chosen hedge **adds** risk: its payoffs "
               "don't line up with the buyer's actual costs. Try the optimal ratio.")

# ----------------------------------------------------------------------------
# Chart 1: cost distributions
# ----------------------------------------------------------------------------
left, right = st.columns(2)

with left:
    st.subheader("Total cost across simulated futures")
    lo, hi = np.percentile(np.concatenate([u, h]), [0.2, 99.8]) / 1e6
    bins = dict(start=lo, end=hi, size=(hi - lo) / 70)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=u / 1e6, xbins=bins, name="Unhedged",
                               marker_color=BLUE, opacity=0.55))
    fig.add_trace(go.Histogram(x=h / 1e6, xbins=bins, name="Hedged with OCPI",
                               marker_color=ORANGE, opacity=0.6))
    fig.add_vline(x=budget / 1e6, line_dash="dot", line_color=GRAY,
                  annotation_text="Budget at today's price")
    fig.update_layout(barmode="overlay", xaxis_title="Total cost ($M)",
                      yaxis_title="Number of simulated futures", height=400,
                      margin=dict(t=10, b=10), legend=dict(x=0.65, y=0.95))
    st.plotly_chart(fig)
    st.caption("The hedge narrows the range: it gives up cheap years in exchange "
               "for protection against expensive ones. The average barely moves.")

# ----------------------------------------------------------------------------
# Chart 2: effectiveness vs correlation
# ----------------------------------------------------------------------------
with right:
    st.subheader("Why regional correlation matters")
    sweep = cached_sweep(cal["start_price"], vol, n_gpus, months, drift, buyer_vol)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sweep["rho"], y=sweep["optimal"] * 100, mode="lines+markers",
                             name="Optimal hedge ratio", line_color=BLUE))
    fig.add_trace(go.Scatter(x=sweep["rho"], y=sweep["full"] * 100, mode="lines+markers",
                             name="Full hedge (ratio = 1)", line=dict(color=ORANGE, dash="dash")))
    fig.add_hline(y=0, line_color=GRAY, line_width=1)
    fig.add_vline(x=rho, line_dash="dot", line_color=GRAY, annotation_text=f"ρ = {rho:.2f}")
    fig.update_layout(xaxis=dict(title="Correlation of buyer's price with OCPI (ρ)",
                                 autorange="reversed"),
                      yaxis_title="Budget variance removed (%)", height=400,
                      margin=dict(t=10, b=10), legend=dict(x=0.02, y=0.05))
    st.plotly_chart(fig)
    st.caption("A naive full hedge adds risk once ρ falls below about 0.5. "
               "The optimal ratio shrinks the hedge as correlation weakens.")

# ----------------------------------------------------------------------------
# Chart 3 + 4: history and simulated paths
# ----------------------------------------------------------------------------
left, right = st.columns(2)

with left:
    st.subheader(f"OCPI history: {gpu}")
    s = history[history["gpu"] == gpu].sort_values("settle_date")
    fig = go.Figure(go.Scatter(x=s["settle_date"], y=s["index_value"],
                               mode="lines", line_color=BLUE, name=gpu))
    fig.update_layout(yaxis_title="$ per GPU-hour", height=340, margin=dict(t=10, b=10))
    st.plotly_chart(fig)

with right:
    st.subheader("Simulated index paths")
    idx = res["index"]
    x = np.arange(0, months + 1)
    start_col = np.full((idx.shape[0], 1), cal["start_price"])
    paths = np.hstack([start_col, idx])
    p5, p25, p50, p75, p95 = np.percentile(paths, [5, 25, 50, 75, 95], axis=0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=p95, line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=p5, fill="tonexty", line=dict(width=0),
                             fillcolor="rgba(46,111,186,0.15)", name="5th–95th pct"))
    fig.add_trace(go.Scatter(x=x, y=p75, line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=p25, fill="tonexty", line=dict(width=0),
                             fillcolor="rgba(46,111,186,0.30)", name="25th–75th pct"))
    fig.add_trace(go.Scatter(x=x, y=p50, line_color=BLUE, name="Median"))
    for i in range(5):  # a few individual futures, for intuition
        fig.add_trace(go.Scatter(x=x, y=paths[i], line=dict(color=GRAY, width=1),
                                 opacity=0.6, showlegend=(i == 0), name="Sample paths"))
    fig.update_layout(xaxis_title="Months from today", yaxis_title="$ per GPU-hour",
                      height=340, margin=dict(t=10, b=10))
    st.plotly_chart(fig)

# ----------------------------------------------------------------------------
# Calibration details + method
# ----------------------------------------------------------------------------
with st.expander("Calibration from OCPI data"):
    a, b, c, d = st.columns(4)
    a.metric("Days of history", cal["n_days"])
    b.metric("Vol from daily returns", f"{cal['vol_daily_ann']:.0%}")
    c.metric("Vol from weekly returns", f"{cal['vol_weekly_ann']:.0%}")
    d.metric("Variance ratio", f"{cal['variance_ratio']:.2f}")
    st.write(
        "Daily moves partly reverse within a week (variance ratio below 1), so daily "
        "returns overstate longer-horizon risk. The simulation defaults to the weekly "
        "estimate, which is closer to the horizon a budget hedger cares about."
    )

with st.expander("How the simulation works, and its simplifications"):
    st.markdown(
        """
Each simulated future is one path of monthly prices. The index follows a random walk
in log price at the chosen volatility. The buyer's price gets a shock that is ρ times
the index's shock plus independent noise, so the two move together only partly.

The hedge is a strip of monthly swaps locked at today's OCPI: when the index settles
above the locked price, the hedge pays the buyer the difference; when it settles below,
the buyer pays. Hedged cost = compute bill − hedge payoff.

Simplifications worth knowing:
- The buyer's price starts at the index level (no provider premium or discount).
- Swaps lock at today's spot, not a published forward curve.
- Each month settles on a single index value rather than the monthly average.
- Volatility comes from a short sample during a rising market.
        """
    )

st.caption(f"Data: Ornn Compute Price Index via Ornn's free public API, "
           f"{cal['first']} to {cal['last']}. Demo only, not investment advice.")
