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
import plotly.io as pio
import streamlit as st
from streamlit import config

from hedge_sim import (DAYS_PER_YEAR, HOURS_PER_MONTH, calibrate,
                       optimal_hedge_ratio, simulate)

CSV_PATH = Path("data/ocpi_history.csv")

st.set_page_config(page_title="OCPI Hedging Simulator", page_icon="📉", layout="wide")


def palette(mode: str) -> dict:
    """Every colour the app uses, for one theme, from ornn.com's tokens.

    Bronze is constant across both modes — it's the brand accent. The neutrals
    have to flip, because a light grey that reads on #141414 disappears on
    #F3F3F3.
    """
    if mode == "light":
        # neutral is #3D4045, not the obvious #6F7681: that grey has almost the
        # same luminance as the bronze (contrast 1.03), so the two chart series
        # would be near-indistinguishable. #3D4045 gives 2.33 against the accent
        # and 9.38 against the page.
        return dict(accent="#9A6F35", accent_rgb="154,111,53", neutral="#3D4045",
                    muted="#6F7681", grid="#DCDCDC", border="#C2C2C2",
                    ink="#141414", surface="#E8E8E8", page="#F3F3F3")
    return dict(accent="#AD8147", accent_rgb="173,129,71", neutral="#B2B8C0",
                muted="#6F7681", grid="#2B2B2B", border="#3D4045",
                ink="#F3F3F3", surface="#212121", page="#141414")


# --- theme toggle ----------------------------------------------------------
# config.toml holds one [theme] table so the app always opens dark. Flipping the
# toggle rewrites those options and reruns: the theme is serialised into the
# NewSession message at the START of a run, so a change only lands on the next
# one. _theme_applied is seeded to "dark" so the first load doesn't rerun.
# Note this mutates process-global config — fine for a local demo, but in a
# multi-user deployment one viewer's toggle would move everyone's theme.
st.session_state.setdefault("_theme_applied", "dark")
THEME = "dark" if st.session_state.get("dark_mode", True) else "light"

if st.session_state["_theme_applied"] != THEME:
    _p = palette(THEME)
    for _key, _value in {
        "base": THEME,
        "primaryColor": _p["accent"],
        "backgroundColor": _p["page"],
        "secondaryBackgroundColor": _p["surface"],
        "textColor": _p["ink"],
        "borderColor": _p["border"],
    }.items():
        config.set_option(f"theme.{_key}", _value)
    st.session_state["_theme_applied"] = THEME
    st.rerun()

P = palette(THEME)
# ACCENT carries the hedge — the thing the demo is arguing for — while NEUTRAL
# carries the unhedged baseline, so bronze always marks the signal.
ACCENT, ACCENT_RGB = P["accent"], P["accent_rgb"]
NEUTRAL, MUTED = P["neutral"], P["muted"]
GRID, BORDER, INK, SURFACE = P["grid"], P["border"], P["ink"], P["surface"]

# Plotly defaults to a white canvas, which would sit as a bright slab on a black
# page. Registering a template once restyles every chart without touching each one.
pio.templates["ornn"] = go.layout.Template(layout=dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=INK, size=12),
    colorway=[ACCENT, NEUTRAL, MUTED],
    xaxis=dict(gridcolor=GRID, linecolor=BORDER, zerolinecolor=BORDER,
               tickcolor=BORDER, title=dict(font=dict(color=MUTED))),
    yaxis=dict(gridcolor=GRID, linecolor=BORDER, zerolinecolor=BORDER,
               tickcolor=BORDER, title=dict(font=dict(color=MUTED))),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=INK)),
    hoverlabel=dict(bgcolor=SURFACE, bordercolor=BORDER, font=dict(color=INK)),
))
pio.templates.default = "ornn"

# Streamlit reserves a lot of room above the first element in both columns: a ~6rem
# top padding in the main block, and a separate stSidebarHeader div holding the
# collapse arrow. Trim both so the page opens on content rather than empty space.
# Test IDs verified present in the installed Streamlit build before being used here.
st.markdown(
    f"""
    <style>
      [data-testid="stMainBlockContainer"], .block-container {{
          padding-top: 2.2rem; padding-bottom: 2rem;
      }}
      /* The sidebar's own header reserves space for the collapse arrow; keep just
         enough for the button itself. */
      [data-testid="stSidebarHeader"] {{
          padding-top: 0.5rem; padding-bottom: 0rem; min-height: 2.25rem; height: auto;
      }}
      [data-testid="stSidebarUserContent"] {{ padding-top: 0.25rem; }}
      /* st.sidebar.header renders an h2 that carries its own top padding; zero it
         for the first one only. Subheaders are h3 and keep their spacing. */
      [data-testid="stSidebarUserContent"] h2 {{ padding-top: 0; margin-top: 0; }}
      [data-testid="stHeader"] {{ background: transparent; }}

      /* --- ornn.com styling: square corners --------------------------------
         theme.baseRadius = "none" squares most chrome; these catch the pieces
         that carry their own radius. Every test ID here was checked against
         the installed frontend bundle — see the note in CLAUDE.md about
         data-baseweb, which does NOT exist in this build. */
      [data-testid="stSlider"] [role="slider"],
      [data-testid="stSelectbox"] > div,
      [data-testid="stSelectboxVirtualDropdown"],
      [data-testid="stExpander"], [data-testid="stMetric"],
      [data-testid="stAlert"],
      .stPlotlyChart, button, input, textarea {{
          border-radius: 0 !important;
      }}
      /* Sidebar reads as a distinct panel against the page. */
      [data-testid="stSidebar"] {{ border-right: 1px solid {BORDER}; }}

      /* --- GPU picker -------------------------------------------------------
         It drives every number on the page, so the control gets a bronze border.
         Scoped to the sidebar's selectbox because there is exactly one; add a
         scope here if another is ever introduced. Inside stSelectbox the label
         is a <label>, so `> div` is the control wrapper — the box you click to
         open the dropdown. Bolding the label and value was tried and reverted;
         it read badly. */
      [data-testid="stSidebar"] [data-testid="stSelectbox"] > div {{
          border: 1px solid {ACCENT} !important;
      }}
      [data-testid="stSidebar"] [data-testid="stSelectbox"] label,
      [data-testid="stSidebar"] [data-testid="stSelectbox"] label p {{
          color: {INK} !important;
      }}

      /* st.metric renders a trend arrow next to any delta. Here the delta is a
         description ("95% CI: 60-64%"), not a movement, so the arrow is noise —
         but hiding it leaves the text flush against the left edge, hence the
         padding that stands in for the space the icon used to occupy. */
      [data-testid="stMetricDeltaIcon-Up"],
      [data-testid="stMetricDeltaIcon-Down"] {{ display: none !important; }}
      [data-testid="stMetricDelta"] {{ padding-left: 0.4rem !important; }}
    </style>
    """,
    unsafe_allow_html=True,
)


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


@st.cache_data
def cached_vol_ci(path: str, gpu: str, n_boot: int = 2000,
                  block_len: int = 7, seed: int = 42) -> dict:
    """95% CI on annualized volatility, via a stationary block bootstrap.

    Cached on the GPU alone: this depends only on that GPU's price history,
    not on rho, GPU count, hedge ratio or any other scenario input.

    A plain iid bootstrap would be wrong here. OCPI shows short-horizon
    reversal — the variance ratio of weekly to daily returns is about 0.36,
    well under the 1.0 of a random walk — so daily moves are not independent.
    Resampling individual days would shuffle that structure away, rebuild a
    series that really is a random walk, and understate how uncertain the
    volatility estimate is. Resampling *blocks* of roughly a week keeps the
    reversal intact. Block lengths are geometric with mean `block_len` and wrap
    around circularly (Politis & Romano's stationary bootstrap), which avoids
    the fixed-block version's dependence on where the block boundaries land.
    """
    s = (load_history(path)
         .pipe(lambda d: d[d["gpu"] == gpu])
         .sort_values("settle_date")["index_value"])
    log_p = np.log(s.to_numpy(dtype=float))
    rets = np.diff(log_p)
    n_ret = len(rets)
    ann = np.sqrt(DAYS_PER_YEAR / 7)

    def weekly_vol(paths: np.ndarray) -> np.ndarray:
        """Re-estimate vol exactly as calibrate() does: overlapping 7-day returns."""
        weekly = paths[:, 7:] - paths[:, :-7]
        return weekly.std(axis=1, ddof=1) * ann

    point = float(weekly_vol(log_p[None, :])[0])
    if n_ret < 20:                       # too short to bootstrap meaningfully
        return {"point": point, "lo": np.nan, "hi": np.nan,
                "n_boot": 0, "block_len": block_len}

    rng = np.random.default_rng(seed)    # fixed seed: reproducible, per convention
    # Stationary bootstrap indices: with probability 1/block_len start a fresh
    # block at a random day, otherwise continue the current block by one day.
    starts = rng.integers(0, n_ret, size=(n_boot, n_ret))
    carry_on = rng.random((n_boot, n_ret)) >= (1.0 / block_len)
    idx = np.empty((n_boot, n_ret), dtype=np.int64)
    idx[:, 0] = starts[:, 0]
    for t in range(1, n_ret):
        idx[:, t] = np.where(carry_on[:, t], (idx[:, t - 1] + 1) % n_ret, starts[:, t])

    # Rebuild each resampled log-price path from day 0, then re-estimate vol.
    resampled = np.concatenate(
        [np.zeros((n_boot, 1)), np.cumsum(rets[idx], axis=1)], axis=1)
    vols = weekly_vol(resampled)
    lo, hi = np.percentile(vols, [2.5, 97.5])
    return {"point": point, "lo": float(lo), "hi": float(hi),
            "n_boot": n_boot, "block_len": block_len}


@st.cache_data
def cached_ci_sims(start_price, vols, rho, n_gpus, months, n_sims, drift,
                   buyer_vol_mult, hedge_mode, custom_ratio):
    """Run the full simulation at just three volatilities: CI low, point, CI high.

    Deliberately outside the bootstrap loop — 2,000 resamples x 10,000 paths
    would hang the app. Three runs bracket the same uncertainty at 3/2000ths of
    the cost. Cached on the whole parameter set, like cached_sim, because the
    resulting cost range depends on all of it.
    """
    out = []
    for v in vols:
        bv = v * buyer_vol_mult
        hr = custom_ratio if hedge_mode == "Custom" else optimal_hedge_ratio(rho, v, bv)
        r = simulate(start_price, v, rho, n_gpus, months, n_sims, drift, bv, hr)
        out.append({
            "vol": v,
            "p95_unhedged": float(np.percentile(r["unhedged"], 95)),
            "p95_hedged": float(np.percentile(r["hedged"], 95)),
            "var_cut": float(1 - r["hedged"].var() / r["unhedged"].var()),
        })
    return out


@st.cache_data
def cached_backtest(path: str, gpu: str, n_gpus: int) -> dict:
    """Replay the real OCPI history: lock a swap on day one, then buy at spot.

    The buyer needs the GPUs continuously, so every settled day is bought at
    that day's index. The hedged cost is the locked price for every one of
    those days. No transaction costs, no margin, no discounting.
    """
    s = (load_history(path)
         .pipe(lambda d: d[d["gpu"] == gpu])
         .sort_values("settle_date")
         .reset_index(drop=True))
    hours_per_day = n_gpus * 24
    locked = float(s["index_value"].iloc[0])
    daily_unhedged = hours_per_day * s["index_value"]
    unhedged = float(daily_unhedged.sum())
    hedged = float(hours_per_day * locked * len(s))
    return {
        "dates": s["settle_date"],
        "index": s["index_value"],
        "cum_unhedged": daily_unhedged.cumsum(),
        "cum_hedged": pd.Series(hours_per_day * locked, index=s.index).cumsum(),
        "locked": locked,
        "final": float(s["index_value"].iloc[-1]),
        "unhedged": unhedged,
        "hedged": hedged,
        "diff": unhedged - hedged,          # positive = the hedge saved money
        "pct": (unhedged - hedged) / hedged,            # vs the locked cost
        "pct_unhedged": (unhedged - hedged) / unhedged,  # share of the bill saved
        "n_days": len(s),
        "first": s["settle_date"].iloc[0],
        "last": s["settle_date"].iloc[-1],
    }


def refresh_data() -> str:
    """Pull the latest OCPI history from Ornn's free API into the local CSV."""
    from ocpi_pull import OrnnClient, history_to_frame, merge_into_store
    client = OrnnClient()
    frames = [history_to_frame(client.index_history(g), g) for g in client.gpu_types()]
    store = merge_into_store(pd.concat(frames, ignore_index=True), CSV_PATH)
    return f"Updated: {len(store)} rows through {store['settle_date'].max()}"


def money(x: float) -> str:
    """Plain text. For st.metric value/delta, which are NOT markdown-rendered."""
    return f"${x / 1e6:,.2f}M"


def money_md(x: float) -> str:
    r"""Markdown-safe. Streamlit reads a bare $...$ as LaTeX, which swallows the
    text between two dollar amounts and renders it as math, so escape every $
    used in st.write / st.markdown / st.caption."""
    return money(x).replace("$", r"\$")


def price_md(x: float) -> str:
    """Markdown-safe per-GPU-hour price."""
    return rf"\${x:,.2f}"


# ----------------------------------------------------------------------------
# Sidebar: scenario inputs
# ----------------------------------------------------------------------------
st.sidebar.header("Scenario inputs")

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
gpu = st.sidebar.selectbox("GPU", gpus,
                           index=gpus.index("B200") if "B200" in gpus else 0)
cal = cached_calibration(str(CSV_PATH), gpu)

n_gpus = st.sidebar.slider("Number of GPUs", 50, 5000, 500, step=50)
months = st.sidebar.slider("Contract length (months)", 3, 24, 12)

# Model settings render ABOVE market assumptions, but the hedge-ratio control at the
# bottom of this block needs rho and the volatilities, which are set further down.
# A container reserves the slot so the widget lands in the right place on the page
# while still being created after its inputs exist.
model_box = st.sidebar.container()
model_box.subheader("Model settings")
n_sims = model_box.select_slider("Simulated futures (Monte Carlo sample size)",
                                 [2000, 5000, 10000, 20000], 10000)
hedge_mode = model_box.radio("Hedge ratio", ["Optimal (h* = ρ·σ_buyer/σ_index)", "Custom"])

st.sidebar.subheader("Market assumptions")
rho = st.sidebar.slider("Correlation of buyer's price with OCPI (ρ)", 0.0, 1.0, 0.80, 0.05,
                        help="Basis risk. 1.0 = the buyer pays exactly the index, so the "
                             "hedge tracks their bill perfectly. Lower = the price they "
                             "actually pay drifts away from OCPI, and the hedge can only "
                             "remove the part of the risk that moves with the index.")
vol_default = float(round(cal["vol_weekly_ann"], 2))
vol = st.sidebar.slider("Index volatility (annualized)", 0.05, 1.20, vol_default, 0.01,
                        help=f"Default is calibrated from weekly OCPI returns ({vol_default:.0%}).")
buyer_vol_mult = st.sidebar.slider("Buyer price volatility vs index", 0.5, 2.0, 1.0, 0.05)
drift = st.sidebar.slider("Expected annual price drift", -0.5, 0.5, 0.0, 0.05,
                          help="0 = no view on direction. Hedge locks in today's price.")

buyer_vol = vol * buyer_vol_mult
h_opt = optimal_hedge_ratio(rho, vol, buyer_vol)
if hedge_mode == "Custom":
    hedge_ratio = model_box.slider("Share of usage hedged", 0.0, 1.5, 1.0, 0.05)
else:
    hedge_ratio = h_opt
    model_box.caption(f"Optimal hedge ratio at these settings: **{h_opt:.2f}**")

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
# Headline numbers
# ----------------------------------------------------------------------------
_title_col, _toggle_col = st.columns([7, 1], vertical_alignment="center")
with _toggle_col:
    # Reads back into st.session_state["dark_mode"], which the theme block at the
    # top of the script picks up on the next run.
    st.toggle("Dark mode", value=True, key="dark_mode")
with _title_col:
    st.title("Dashboard for hedging GPU compute costs with the Ornn Compute Pricing Index")
st.write(
    f"**Scenario:** buyer needs **{n_gpus:,} {gpu}s for {months} months** "
    f"({gpu_hours / 1e6:.2f}M GPU-hours). At today's OCPI of "
    f"**{price_md(cal['start_price'])}/hr**, that's a **{money_md(budget)}** budget. "
    f"How much of the price risk can an OCPI hedge remove?"
)

# --- how much of this rests on a 3-month volatility estimate? ---------------
# The bootstrap is cheap; the three simulations behind it are not, so they are
# cached and run only at the CI bounds and the live slider volatility.
vci = cached_vol_ci(str(CSV_PATH), gpu)
have_ci = bool(np.isfinite(vci["lo"]))
if have_ci:
    # Run at the CI bounds plus the live slider volatility, so the interval always
    # brackets the numbers actually shown elsewhere on the page.
    ci_runs = cached_ci_sims(cal["start_price"], (vci["lo"], vol, vci["hi"]), rho,
                             n_gpus, months, n_sims, drift, buyer_vol_mult,
                             hedge_mode, hedge_ratio)
    span = lambda key: (min(r[key] for r in ci_runs), max(r[key] for r in ci_runs))
    u_lo, u_hi = span("p95_unhedged")
    h_lo, h_hi = span("p95_hedged")
    v_lo, v_hi = span("var_cut")

# The bad-case levels moved to the interval chart lower down; what stays up top is
# what the demo is actually arguing about — how much risk the hedge removes.
c1, c2, c3, c4 = st.columns(4)
c1.metric("Simulated budget variance removed", f"{var_cut:.0%}",
          f"95% CI: {v_lo * 100:.0f}–{v_hi:.0%}" if have_ci else None,
          delta_color="off")
c2.metric("Theoretical budget variance removed", f"{rho ** 2:.0%}",
          "ρ² at the optimal hedge ratio", delta_color="off")
c3.metric("Hedge ratio used", f"{hedge_ratio:.2f}",
          f"optimal = {h_opt:.2f}", delta_color="off")
c4.metric("Simulated futures", f"{n_sims:,}",
          "Monte Carlo paths", delta_color="off")

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
                               marker_color=NEUTRAL, opacity=0.55))
    fig.add_trace(go.Histogram(x=h / 1e6, xbins=bins, name="Hedged with OCPI",
                               marker_color=ACCENT, opacity=0.6))
    fig.add_vline(x=budget / 1e6, line_dash="dot", line_color=MUTED,
                  annotation_text="Budget at today's price")
    fig.update_layout(barmode="overlay", xaxis_title="Total cost ($M)",
                      yaxis_title="Number of simulated futures", height=400,
                      margin=dict(t=10, b=10), legend=dict(x=0.65, y=0.95))
    st.plotly_chart(fig)
    st.caption("The hedge narrows the range: it gives up cheap years in exchange "
               "for protection against expensive ones. The average barely moves.")

# ----------------------------------------------------------------------------
# Chart 2: simulated price paths
# ----------------------------------------------------------------------------
with right:
    st.subheader("Monte Carlo simulated index paths")
    idx = res["index"]
    x = np.arange(0, months + 1)
    start_col = np.full((idx.shape[0], 1), cal["start_price"])
    paths = np.hstack([start_col, idx])
    p5, p25, p50, p75, p95 = np.percentile(paths, [5, 25, 50, 75, 95], axis=0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=p95, line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=p5, fill="tonexty", line=dict(width=0),
                             fillcolor=f"rgba({ACCENT_RGB},0.15)", name="5th–95th pct"))
    fig.add_trace(go.Scatter(x=x, y=p75, line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=p25, fill="tonexty", line=dict(width=0),
                             fillcolor=f"rgba({ACCENT_RGB},0.30)", name="25th–75th pct"))
    fig.add_trace(go.Scatter(x=x, y=p50, line_color=ACCENT, name="Median"))
    for i in range(5):  # a few individual futures, for intuition
        fig.add_trace(go.Scatter(x=x, y=paths[i], line=dict(color=MUTED, width=1),
                                 opacity=0.6, showlegend=(i == 0), name="Sample paths"))
    fig.update_layout(xaxis_title="Months from today", yaxis_title="$ per GPU-hour",
                      height=400, margin=dict(t=10, b=10))
    st.plotly_chart(fig)
    st.caption(f"Bands cover the middle 50% and 90% of {n_sims:,} simulated price "
               f"paths; five individual paths are drawn in grey for intuition.")

# ----------------------------------------------------------------------------
# Chart 3: the bad case, with the uncertainty the volatility estimate carries
# ----------------------------------------------------------------------------
left, right = st.columns(2)

with left:
    st.subheader("Bad case: 95th-percentile cost")
    u_mid, h_mid = float(np.percentile(u, 95)), float(np.percentile(h, 95))
    fig = go.Figure()
    # One marker per case at the 95th percentile, with a whisker spanning the range
    # the volatility CI implies. Asymmetric error bars: the interval is not centred.
    rows = [("Unhedged", u_mid, u_lo if have_ci else u_mid, u_hi if have_ci else u_mid, NEUTRAL),
            ("Hedged", h_mid, h_lo if have_ci else h_mid, h_hi if have_ci else h_mid, ACCENT)]
    for label, mid, lo_, hi_, colour in rows:
        fig.add_trace(go.Scatter(
            x=[mid / 1e6], y=[label], mode="markers+text",
            marker=dict(size=15, color=colour, symbol="diamond"),
            text=[money(mid)], textposition="top center",
            textfont=dict(size=13, color=colour),
            cliponaxis=False,          # let labels spill past the axis, not get cut
            error_x=dict(type="data", symmetric=False,
                         array=[(hi_ - mid) / 1e6], arrayminus=[(mid - lo_) / 1e6],
                         color=colour, thickness=3, width=12),
            showlegend=False,
            hovertemplate=(f"<b>{label}</b><br>95th pct: {money(mid)}"
                           f"<br>95% CI: {money(lo_)} – {money(hi_)}<extra></extra>")))
        if have_ci:  # label the whisker ends, placed OUTWARD so they clear the bar
            fig.add_trace(go.Scatter(
                x=[lo_ / 1e6, hi_ / 1e6], y=[label, label], mode="text",
                text=[money(lo_), money(hi_)],
                textposition=["middle left", "middle right"],
                textfont=dict(size=10, color=MUTED),
                cliponaxis=False,
                showlegend=False, hoverinfo="skip"))
    fig.add_vline(x=budget / 1e6, line_dash="dot", line_color=MUTED,
                  annotation_text="Budget at today's price")
    # Pad the x-range so the outward end labels aren't clipped at the plot edge.
    span_lo = min(r[2] for r in rows) / 1e6
    span_hi = max(r[3] for r in rows) / 1e6
    pad = max((span_hi - span_lo) * 0.20, 0.6)
    # Height matches the chart beside it, but the y-range is widened past the default
    # [-0.5, 1.5] so the two rows stay close instead of spreading to fill the space.
    # The extra range also gives the top label room rather than clipping it.
    fig.update_layout(xaxis_title="Total cost ($M)", height=320,
                      margin=dict(t=50, b=55),
                      xaxis=dict(range=[min(span_lo, budget / 1e6) - pad, span_hi + pad]),
                      yaxis=dict(categoryorder="array",
                                 categoryarray=["Hedged", "Unhedged"],
                                 range=[-0.95, 1.95]))
    st.plotly_chart(fig)

    relief = u_mid - h_mid
    if have_ci:
        st.caption(f"Whiskers span the 95% volatility CI. The hedge cuts the bad case "
                   f"by {money_md(relief)} ({relief / u_mid:.0%}) and shortens the "
                   f"whisker: less exposure, and less doubt about it.")
    else:
        st.caption(f"The hedge cuts the bad case by {money_md(relief)} "
                   f"({relief / u_mid:.0%}). Not enough history yet to bootstrap "
                   f"confidence intervals for {gpu}.")

# ----------------------------------------------------------------------------
# Chart 4: effectiveness vs correlation
# ----------------------------------------------------------------------------
with right:
    st.subheader("Variance removed based on basis risk")
    sweep = cached_sweep(cal["start_price"], vol, n_gpus, months, drift, buyer_vol)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sweep["rho"], y=sweep["optimal"] * 100, mode="lines+markers",
                             name="Optimal hedge ratio", line_color=ACCENT))
    fig.add_trace(go.Scatter(x=sweep["rho"], y=sweep["full"] * 100, mode="lines+markers",
                             name="Full hedge (ratio = 1)", line=dict(color=NEUTRAL, dash="dash")))
    fig.add_hline(y=0, line_color=MUTED, line_width=1)
    fig.add_vline(x=rho, line_dash="dot", line_color=MUTED, annotation_text=f"ρ = {rho:.2f}")
    fig.update_layout(xaxis=dict(title="Correlation of buyer's price with OCPI (ρ)",
                                 autorange="reversed"),
                      yaxis_title="Budget variance removed (%)", height=320,
                      margin=dict(t=10, b=10), legend=dict(x=0.02, y=0.05))
    st.plotly_chart(fig)
    st.caption("A naive full hedge adds risk once ρ falls below about 0.5. "
               "The optimal ratio shrinks the hedge as correlation weakens.")

# ----------------------------------------------------------------------------
# Backtest on realized OCPI history
# ----------------------------------------------------------------------------
bt = cached_backtest(str(CSV_PATH), gpu, n_gpus)
saved = bt["diff"] > 0              # average index settled above the locked price
ended_above = bt["final"] >= bt["locked"]

st.header("Backtest example", divider="gray")
st.write(
    f"Replaying the real OCPI history. A buyer needing **{n_gpus:,} {gpu}s** "
    f"continuously locks a swap at the first settled index of the window — "
    f"**{price_md(bt['locked'])}/hr** on {bt['first']:%b %d, %Y} — then buys at the "
    f"daily index through {bt['last']:%b %d, %Y}, {bt['n_days']} days in all."
)

b1, b2, b3 = st.columns(3)
b1.metric("Cost without hedge", money(bt["unhedged"]))
b2.metric("Cost with hedge", money(bt["hedged"]),
          f"{(bt['hedged'] - bt['unhedged']) / 1e6:+,.2f}M vs unhedged",
          delta_color="inverse")          # cheaper is better, so invert the colour
b3.metric("Hedge saved", f"{bt['pct_unhedged']:+.1%}",
          help="Share of the unhedged bill the hedge saved. Negative means the hedge "
               "cost money over this window.")

left, right = st.columns(2)

with left:
    st.subheader("Cumulative cost")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bt["dates"], y=bt["cum_unhedged"] / 1e6, mode="lines",
                             line_color=NEUTRAL, name="Unhedged (buy at the daily index)"))
    fig.add_trace(go.Scatter(x=bt["dates"], y=bt["cum_hedged"] / 1e6, mode="lines",
                             line=dict(color=ACCENT, dash="dash"),
                             name=f"Hedged (locked at ${bt['locked']:.2f}/hr)"))
    fig.update_layout(yaxis_title="Cumulative cost ($M)", height=340,
                      margin=dict(t=10, b=10), legend=dict(x=0.02, y=0.98))
    st.plotly_chart(fig)

with right:
    st.subheader(f"OCPI vs the locked price: {gpu}")
    fig = go.Figure(go.Scatter(x=bt["dates"], y=bt["index"],
                               mode="lines", line_color=ACCENT, name=gpu))
    fig.add_hline(y=bt["locked"], line_dash="dash", line_color=MUTED,
                  annotation_text=f"Locked at ${bt['locked']:.2f}/hr",
                  annotation_position="bottom right")
    fig.update_layout(yaxis_title="$ per GPU-hour", height=340, margin=dict(t=10, b=10))
    st.plotly_chart(fig)
    st.caption("Days above the dashed line are days the hedge paid the buyer; "
               "days below are days the buyer paid the hedge.")

# Direction-aware: the hedge settles against the AVERAGE index over the window,
# so its P&L can disagree with where prices happened to finish.
if saved and ended_above:
    st.caption("Prices rose over this window, so the hedge paid off — the buyer "
               "locked in a price below what they'd have paid at spot.")
elif not saved and not ended_above:
    st.caption("Prices fell over this window, so the hedge cost money — the buyer "
               "locked in a price above spot. This is the other side of insurance: "
               "it doesn't pay out every time, and that's not a failure of the hedge.")
elif saved:
    st.caption("The index finished below the locked price but averaged above it, so "
               "the hedge still paid off — what matters is the average index over the "
               "window, not where prices ended.")
else:
    st.caption("The index finished above the locked price but averaged below it, so "
               "the hedge cost money — what matters is the average index over the "
               "window, not where prices ended. This is the other side of insurance: "
               "it doesn't pay out every time, and that's not a failure of the hedge.")

st.caption(f"This is one realized path, not a distribution. The outcome depends "
           f"entirely on which way prices happened to move, which is why the "
           f"simulation covers {n_sims:,} possible futures instead.")
st.caption("The hedge is fair in expectation at entry, so this result is neither "
           "alpha nor a mistake — it's what predictability costs or pays in one "
           "particular history.")

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
    if have_ci:
        st.caption(
            f"The confidence intervals on this page come from a stationary block "
            f"bootstrap with an expected block length of {vci['block_len']} days — "
            f"blocks rather than individual days, because the reversal above means "
            f"daily returns aren't independent, and resampling them one at a time "
            f"would erase that structure and make the estimate look more precise than "
            f"it is. For {gpu} that gives a 95% interval of "
            f"**{vci['lo']:.1%} – {vci['hi']:.1%}** around the {vci['point']:.1%} point "
            f"estimate, which is what the whiskers on the bad-case chart are drawn from."
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

Simplifications:
- The buyer's price starts at the index level (no provider premium or discount).
- Swaps lock at today's spot, not a published forward curve.
- Each month settles on a single index value rather than the monthly average.
- Volatility comes from a short sample during a rising market.
        """
    )

st.caption(f"Data: Ornn Compute Price Index via Ornn's free public API, "
           f"{cal['first']} to {cal['last']}. Demo only, not investment advice.")
