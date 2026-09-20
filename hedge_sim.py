"""
hedge_sim.py — Monte Carlo simulation of hedging GPU compute costs with OCPI.

The question: a startup needs N GPUs for the next year and pays its own
regional/provider price. How much budget risk does it carry, and how much does
a hedge settled on the Ornn Compute Price Index (OCPI) remove?

How it works (one simulated "future" = one year of monthly prices):
  1. Calibrate volatility from your OCPI history (data/ocpi_history.csv).
  2. Simulate the index month by month as a random walk in log price.
  3. Simulate the buyer's own price, correlated with the index at rho.
     (rho < 1 is basis risk: the buyer's price doesn't track OCPI exactly.)
  4. Cost unhedged = sum over months of (GPU-hours x buyer's price).
  5. Hedge = monthly swaps locked at today's index level: when the index rises
     above the locked price, the hedge pays the difference; when it falls, the
     buyer pays the difference. Hedged cost = unhedged cost - hedge payoff.
  6. Repeat thousands of times and compare the two cost distributions.

Usage:
    python hedge_sim.py                       # H100 SXM, 500 GPUs, rho sweep
    python hedge_sim.py --rho 0.8 --n-gpus 1000
    python hedge_sim.py --gpu B200 --vol 0.5  # override calibrated volatility
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HOURS_PER_MONTH = 730          # 8,760 hours / 12
DAYS_PER_YEAR = 365            # OCPI settles 7 days a week


# ----------------------------------------------------------------------------
# 1. Calibration
# ----------------------------------------------------------------------------
def calibrate(csv_path: Path, gpu: str) -> dict:
    """Estimate starting price and volatility from OCPI history."""
    df = pd.read_csv(csv_path)
    s = (df[df["gpu"] == gpu]
         .assign(settle_date=lambda d: pd.to_datetime(d["settle_date"]))
         .set_index("settle_date")["index_value"]
         .sort_index())
    if s.empty:
        raise SystemExit(f"No rows for '{gpu}'. Options: {sorted(df['gpu'].unique())}")

    log_p = np.log(s)
    daily = log_p.diff().dropna()
    weekly = (log_p - log_p.shift(7)).dropna()   # overlapping 7-day returns

    vol_daily_ann = daily.std() * np.sqrt(DAYS_PER_YEAR)
    vol_weekly_ann = weekly.std() * np.sqrt(DAYS_PER_YEAR / 7)
    lag1_autocorr = daily.autocorr(lag=1)
    # Variance ratio: 1.0 = random walk; < 1 = moves partly reverse within a week
    variance_ratio = weekly.var() / (7 * daily.var())

    return {
        "gpu": gpu,
        "start_price": float(s.iloc[-1]),
        "n_days": len(s),
        "first": s.index[0].date(),
        "last": s.index[-1].date(),
        "vol_daily_ann": vol_daily_ann,
        "vol_weekly_ann": vol_weekly_ann,
        "lag1_autocorr": lag1_autocorr,
        "variance_ratio": variance_ratio,
    }


# ----------------------------------------------------------------------------
# 2-5. Simulation
# ----------------------------------------------------------------------------
def simulate(start_price: float, vol: float, rho: float, n_gpus: int,
             months: int = 12, n_sims: int = 10_000, drift: float = 0.0,
             buyer_vol: float | None = None, hedge_ratio: float = 1.0,
             seed: int = 42) -> dict:
    """Simulate yearly compute cost with and without an OCPI hedge.

    vol, buyer_vol, drift are annualized. hedge_ratio = fraction of GPU-hours hedged.
    """
    rng = np.random.default_rng(seed)
    buyer_vol = vol if buyer_vol is None else buyer_vol
    dt = 1 / 12
    q = n_gpus * HOURS_PER_MONTH                     # GPU-hours per month

    # Correlated monthly shocks: buyer shock = rho * index shock + independent noise
    z_idx = rng.standard_normal((n_sims, months))
    z_own = rng.standard_normal((n_sims, months))
    z_buy = rho * z_idx + np.sqrt(1 - rho**2) * z_own

    # Log random walk; the -0.5*vol^2 term keeps the expected price equal to
    # start_price when drift = 0 (so today's price is a fair locked-in price).
    def path(z, sigma):
        steps = (drift - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
        return start_price * np.exp(np.cumsum(steps, axis=1))

    index = path(z_idx, vol)             # OCPI, month by month
    buyer = path(z_buy, buyer_vol)       # what the buyer actually pays

    locked = start_price                 # swap price locked in today
    cost_unhedged = (q * buyer).sum(axis=1)
    hedge_payoff = (hedge_ratio * q * (index - locked)).sum(axis=1)
    cost_hedged = cost_unhedged - hedge_payoff

    return {"unhedged": cost_unhedged, "hedged": cost_hedged,
            "index": index, "buyer": buyer}


def summarize(costs: np.ndarray) -> dict:
    return {
        "mean": costs.mean(),
        "std": costs.std(),
        "p5": np.percentile(costs, 5),
        "p95": np.percentile(costs, 95),
    }


def optimal_hedge_ratio(rho: float, vol: float, buyer_vol: float | None) -> float:
    """Minimum-variance hedge ratio h* = rho * sigma_buyer / sigma_index."""
    buyer_vol = vol if buyer_vol is None else buyer_vol
    return rho * buyer_vol / vol


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------
def fmt_m(x: float) -> str:
    return f"${x / 1e6:,.2f}M"


def rho_sweep(cal: dict, vol: float, args) -> pd.DataFrame:
    """Hedge effectiveness across correlations, full hedge vs optimal hedge."""
    rows = []
    for rho in [1.0, 0.95, 0.9, 0.8, 0.6, 0.4, 0.2, 0.0]:
        base = dict(start_price=cal["start_price"], vol=vol, rho=rho,
                    n_gpus=args.n_gpus, months=args.months, n_sims=args.sims,
                    drift=args.drift, buyer_vol=args.buyer_vol)
        h_opt = optimal_hedge_ratio(rho, vol, args.buyer_vol)
        full = simulate(**base, hedge_ratio=1.0)
        opt = simulate(**base, hedge_ratio=h_opt)
        var_u = full["unhedged"].var()
        rows.append({
            "rho": rho,
            "risk_cut_full_hedge": 1 - full["hedged"].var() / var_u,
            "optimal_ratio": h_opt,
            "risk_cut_optimal": 1 - opt["hedged"].var() / var_u,
            "theory_rho_sq": rho**2,
            "p95_unhedged": np.percentile(full["unhedged"], 95),
            "p95_optimal": np.percentile(opt["hedged"], 95),
        })
    return pd.DataFrame(rows)


def plot(result: dict, rho: float, sweep: pd.DataFrame, cal: dict,
         out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    bins = np.linspace(min(result["unhedged"].min(), result["hedged"].min()),
                       max(result["unhedged"].max(), result["hedged"].max()), 80) / 1e6
    ax.hist(result["unhedged"] / 1e6, bins=bins, alpha=0.55, label="Unhedged")
    ax.hist(result["hedged"] / 1e6, bins=bins, alpha=0.55, label="Hedged with OCPI")
    ax.set_title(f"{cal['gpu']}: 12-month compute cost (rho = {rho:.2f})")
    ax.set_xlabel("Total cost ($M)")
    ax.set_ylabel("Number of simulated futures")
    ax.legend()

    ax = axes[1]
    ax.plot(sweep["rho"], sweep["risk_cut_optimal"] * 100, "o-", label="Optimal hedge ratio")
    ax.plot(sweep["rho"], sweep["risk_cut_full_hedge"] * 100, "s--", label="Full hedge (ratio = 1)")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_title("How much budget risk the hedge removes")
    ax.set_xlabel("Correlation between buyer's price and OCPI (rho)")
    ax.set_ylabel("Variance reduction (%)")
    ax.invert_xaxis()
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"\nChart saved -> {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Monte Carlo OCPI hedging simulation")
    p.add_argument("--csv", default="data/ocpi_history.csv")
    p.add_argument("--gpu", default="H100 SXM")
    p.add_argument("--n-gpus", type=int, default=500)
    p.add_argument("--months", type=int, default=12)
    p.add_argument("--rho", type=float, default=0.8,
                   help="Correlation of buyer's price with OCPI (for the chart)")
    p.add_argument("--vol", type=float, default=None,
                   help="Annualized index vol (default: calibrated from weekly returns)")
    p.add_argument("--buyer-vol", type=float, default=None,
                   help="Annualized vol of buyer's price (default: same as index)")
    p.add_argument("--drift", type=float, default=0.0,
                   help="Annualized price drift (default 0: no view on direction)")
    p.add_argument("--sims", type=int, default=10_000)
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    cal = calibrate(Path(args.csv), args.gpu)
    vol = args.vol if args.vol is not None else cal["vol_weekly_ann"]

    print(f"=== Calibration: {cal['gpu']} ({cal['n_days']} days, {cal['first']} to {cal['last']}) ===")
    print(f"Starting price (latest OCPI):     ${cal['start_price']:.2f}/GPU-hr")
    print(f"Annualized vol, daily returns:    {cal['vol_daily_ann']:.1%}")
    print(f"Annualized vol, weekly returns:   {cal['vol_weekly_ann']:.1%}")
    print(f"Lag-1 autocorrelation of daily:   {cal['lag1_autocorr']:+.2f}"
          "  (negative = next-day reversal)")
    print(f"Variance ratio (weekly vs daily): {cal['variance_ratio']:.2f}"
          "  (1.0 = random walk; below 1 = moves partly reverse within a week)")
    print(f"Vol used in simulation:           {vol:.1%}"
          f"{'  (override)' if args.vol is not None else '  (weekly-based)'}")

    gpu_hours = args.n_gpus * HOURS_PER_MONTH * args.months
    print(f"\n=== Scenario: {args.n_gpus} GPUs x {args.months} months "
          f"= {gpu_hours / 1e6:.2f}M GPU-hours ===")
    print(f"Expected cost at today's price:   {fmt_m(gpu_hours * cal['start_price'])}")

    h_opt = optimal_hedge_ratio(args.rho, vol, args.buyer_vol)
    res = simulate(cal["start_price"], vol, args.rho, args.n_gpus, args.months,
                   args.sims, args.drift, args.buyer_vol, hedge_ratio=h_opt)
    u, h = summarize(res["unhedged"]), summarize(res["hedged"])
    print(f"\nAt rho = {args.rho:.2f}, optimal hedge ratio = {h_opt:.2f}:")
    print(f"{'':22}{'Unhedged':>14}{'Hedged':>14}")
    print(f"{'Average cost':22}{fmt_m(u['mean']):>14}{fmt_m(h['mean']):>14}")
    print(f"{'Std deviation':22}{fmt_m(u['std']):>14}{fmt_m(h['std']):>14}")
    print(f"{'Good case (5th pct)':22}{fmt_m(u['p5']):>14}{fmt_m(h['p5']):>14}")
    print(f"{'Bad case (95th pct)':22}{fmt_m(u['p95']):>14}{fmt_m(h['p95']):>14}")
    print(f"Variance reduction: {1 - h['std']**2 / u['std']**2:.1%}  (theory: rho^2 = {args.rho**2:.1%})")

    sweep = rho_sweep(cal, vol, args)
    print("\n=== Hedge effectiveness vs. correlation ===")
    show = sweep.copy()
    for c in ["risk_cut_full_hedge", "risk_cut_optimal", "theory_rho_sq"]:
        show[c] = (show[c] * 100).map("{:.0f}%".format)
    for c in ["p95_unhedged", "p95_optimal"]:
        show[c] = show[c].map(fmt_m)
    show["optimal_ratio"] = show["optimal_ratio"].map("{:.2f}".format)
    print(show.to_string(index=False))
    print("\nNote: a full hedge (ratio = 1) at low rho can ADD risk — negative variance reduction.")

    if not args.no_plot:
        try:
            plot(res, args.rho, sweep, cal, Path("hedge_results.png"))
        except ImportError:
            print("\n(Install matplotlib for the chart: python -m pip install matplotlib)")


if __name__ == "__main__":
    main()
