# CLAUDE.md — Ornn compute hedging demo

Context file for Claude Code. Read this first, then `ocpi_pull.py`, `hedge_sim.py`, `app.py`.

## What this project is

A technical demo built for an interview with **Ornn**, a startup that publishes the
**Ornn Compute Price Index (OCPI)**, a transaction-based benchmark for GPU rental prices.
The interview is for their **data science team**. The goal of the demo is to show data
judgment and commodities-market fluency, not to ship a product.

**The question the demo answers:** an AI startup needs N GPUs for the next year and pays
its own regional/provider price. How much budget risk does it carry, and how much of that
risk can a hedge settled on OCPI remove?

**The headline finding:** hedge effectiveness is governed by the correlation (rho) between
the buyer's actual price and the blended index. That is basis risk, and it makes regional
sub-indices commercially important — an argument to raise in the interview.

## Current state (working end to end)

| File | Purpose |
|---|---|
| `ocpi_pull.py` | Pulls OCPI daily history from Ornn's free public API into `data/ocpi_history.csv`; merges incrementally, saves raw JSON, runs data-quality checks, prints summary stats |
| `hedge_sim.py` | Calibrates from the CSV, runs the Monte Carlo hedging simulation, prints results, saves `hedge_results.png` |
| `app.py` | Streamlit UI over the same engine. Opens with a **backtest on realized OCPI history** (three metrics + cumulative-cost chart), then the simulation: sliders for rho, GPU count, volatility, hedge ratio; cost histograms, effectiveness-vs-rho curve, OCPI-vs-locked-price chart, simulated path fan chart |
| `ocpi_daily.yml` | GitHub Actions workflow (not yet installed) to run the pull every weekday and commit the CSV |

Run commands (Windows PowerShell, Python 3.14):

```
python ocpi_pull.py --stats
python hedge_sim.py
python -m streamlit run app.py
```

Installed packages: `requests pandas numpy matplotlib streamlit plotly`.

## Data sources

**Ornn's public API** (`https://api.ornnai.com`, docs at `https://dashboard.ornnai.com/docs`)
— no API key or account needed. Free tier: current value plus trailing 3 months of daily
settled history for 5 GPUs (A100 SXM4, B200, H100 SXM, H200, RTX 5090). Anonymous requests
are rate limited per IP. Endpoints used: `/api/gpu-types-free`, `/api/gpu/{name}`,
`/api/gpu/{name}/index-history`, `/api/daily-index/all`. An `ORNN_API_KEY` env var is
supported and would unlock full history, but is not required.

**Forward curve — gated (probed 2026-09-20).** `GET /api/forward` is a real, registered
route that returns `401 {"error":"Unauthorized","message":"API key required. Use
Authorization: Bearer YOUR_API_KEY"}`, and `{"message":"Invalid API key"}` when sent a
bogus key — so it validates credentials, which a non-existent route would not. Unmatched
paths return an Express **HTML** 404 instead, and `POST /api/forward` 404s while `GET`
401s, confirming a genuine single-method route. It is **not in the docs** (the raw docs
HTML contains zero occurrences of forward/tenor/curve/term/future/swap). Every other name
tried 404s: `forward-curve`, `forwards`, `forward-marks`, `term-structure`, `tenors`,
`futures`, `swaps`, `curve`, `marks`, plus per-GPU and `-free` variants. One other route
shows the same gated signature: `/api/index`. Tenors, fields and whether forwards sit
above or below spot are **unknown** — the gate returns nothing but the error.

**OTPI** — `/api/otpi` is public with 1 month of history and returns a token price index
per lab (`indexPerMtok`; e.g. anthropic 1.639, openai 0.470, google 0.591, deepseek 0.046
on 2026-09-19). A second published Ornn index; not used by the demo.

**Vast.ai public API** — marketplace listings with per-host prices and location. This is the
planned source for regional buyer prices (see Task 1). Not yet built.

**Public trackers for sanity checks only** (not in the pipeline): AIMultiple's weekly GPU
price index (July 2024 onward), GetDeploying's weekly index, Silicon Data (a competing daily
H100 index, mostly paywalled), SemiAnalysis (reported 1-year H100 contract prices rising from
about $1.70/hr in Oct 2025 to about $2.35/hr by Mar 2026 — a good stress scenario).

## What the data showed (92 days, 2026-06-20 to 2026-09-19)

- OCPI settles **every day including weekends** (weekend values change; they aren't carried
  forward). Annualize with 365, not 252.
- H100 SXM: $2.34 → $2.85 over the window (range $2.29–$3.17). B200 +74%, H200 +48%.
- **Daily volatility overstates longer-horizon risk.** H100 annualized vol is 66% from daily
  returns but 40% from weekly returns. Variance ratio (weekly vs daily) is 0.36, well below
  the 1.0 of a random walk, so moves partly reverse within a week. Lag-1 autocorrelation is
  only −0.04, so the reversal is not next-day. The simulation therefore calibrates from
  **weekly** returns.
- Daily log-return correlations **between GPUs** are near zero despite all trending up
  together — another symptom of daily noise.

## Model design

- Monthly steps over a 12-month horizon; log random walk with drift default 0 (a 3-month
  sample in a rising market cannot support a drift estimate).
- Buyer's price shock = `rho * index_shock + sqrt(1 - rho^2) * idiosyncratic_shock`.
- Hedge = strip of monthly swaps locked at today's index level. Payoff = `h * Q * (I_m - F)`.
- Minimum-variance hedge ratio `h* = rho * sigma_buyer / sigma_index`.
- Validation: simulated variance reduction at `h*` matches theory (`rho^2`) to within 0.3pp.
  **Keep this check** — matching a closed form is how we know the simulation is right.
- Known result to preserve in the UI: a naive full hedge (ratio 1.0) **adds** risk below
  about rho = 0.5.

Baseline numbers (500 H100s, 12 months, rho = 0.8, vol 39.9%): budget $12.48M; 95th
percentile $18.34M unhedged vs $15.84M hedged; variance cut 64.3%.

### Backtest on realized history (`app.py`, top section)

Lock a swap at the first settled index in the CSV, then buy at the daily index for the
rest of the window. Unhedged = `sum(N * 24 * daily index)`; hedged = `N * 24 * locked *
days`. Percentages are quoted **relative to the locked (hedged) cost**. No transaction
costs, no margin, no discounting.

Realized results, 500 GPUs, 93 days (2026-06-20 to 2026-09-20):

| GPU | locked | unhedged | hedged | hedge P&L |
|---|---|---|---|---|
| H100 SXM | $2.34 | $3.01M | $2.61M | **+$0.40M (+15.3%)** |
| B200 | $4.28 | $6.90M | $4.78M | **+$2.12M (+44.4%)** |
| H200 | $3.46 | $5.07M | $3.86M | +$1.20M (+31.2%) |
| A100 SXM4 | $1.05 | $1.15M | $1.17M | **−$0.02M (−1.5%)** |
| RTX 5090 | $0.57 | $0.61M | $0.64M | **−$0.03M (−4.7%)** |

**The hedge loses on two of five GPUs, and that is the point** — it is insurance, fair in
expectation at entry, not alpha. Captions must stay conditional on the realized outcome;
never hard-code "prices rose".

**Subtlety worth preserving:** hedge P&L is set by the **average** index over the window
versus the locked price, *not* by where prices finished. RTX 5090 rose start-to-end
($0.57 → $0.66) yet still **lost** 4.7%, because it averaged below the lock. The caption
logic therefore branches on both `saved` and `ended_above` (four cases), so a rising
chart never renders next to the words "prices fell".

## Task list, in priority order

### 1. Vast.ai scraper (`vast_pull.py`) — start immediately
Correlation needs a time series, so every day without data is lost. Requirements:
- Query Vast.ai's public offers API for H100 (and ideally H200/A100) on-demand offers.
- Store one row per offer per run: timestamp, gpu name, price per **single** GPU (normalize
  multi-GPU listings), location/region, verified flag, host id, plus any reliability score.
- Append to `data/vast_offers.csv`, same incremental pattern as `ocpi_pull.py`.
- Run every few hours; a local Windows Task Scheduler entry or a GitHub Actions cron is fine.
- Add an aggregation step producing daily median price by region.
- Then estimate rho between regional daily medians and OCPI, using **weekly** returns, and
  feed the measured value into the app as the default (keep the slider).

### 2. Automate the OCPI pull
Put the repo on GitHub, install `.github/workflows/ocpi_daily.yml`, confirm it commits daily.
Add a `.gitignore` for `__pycache__/`, `*.pyc`, and optionally `data/raw/`.

### 3. Realism upgrades to the simulation
- **Monthly-average settlement**: settle each month on the average index over the month
  rather than a single point. This matches how compute is consumed and resists manipulation;
  it is the Asian-option / energy-swap convention. Needs finer time steps (daily or weekly)
  inside each month.
- **Buyer premium/discount**: let the buyer's starting price differ from the index level.
- **Forward curve pricing**: ~~If forward marks are reachable on the free tier~~ —
  **probed 2026-09-20: blocked.** `GET /api/forward` exists but is key-gated and
  undocumented (see "Forward curve" under Data sources). Stays blocked until Ornn
  grants a key; zero drift and lock-at-spot remain the defensible defaults.

### 4. Writeup (`README.md`, one page)
Question, method, findings, limitations, and what would change with access to Ornn's real
transaction data. This matters more than another feature.

### 5. Interview questions list
Keep a short file of questions the work raised: Does OCPI publish regional sub-indices (the
API's `region` field is empty on free GPUs and `"global"` on the daily index)? Why 7-day
settlement? How do they handle the short-horizon noise visible in daily returns? How is the
winsorization threshold chosen?

Sharpened by the 2026-09-20 probe: *"`GET /api/forward` is registered but returns 401
without a key, and it isn't in the docs. Is that the forward marks endpoint? What tenors
does it publish, and is it reachable on any free or trial tier?"* Pair it with the
regional question — the free tier exposes global spot only (`region` is `""` on
`/api/gpu-types-free`, `"global"` on `/api/daily-index/all`), so neither regional
sub-indices nor the forward curve are reachable without a key.

## Conventions and standards

- Plain Python with `requests`/`pandas`/`numpy`; no heavy frameworks. Readable over clever —
  the author will be explaining this code line by line in an interview.
- Scripts are CLI-driven with `argparse` and sensible defaults; everything runs from the
  project root on Windows.
- Data writes are **incremental and additive**: merge and de-duplicate on a natural key,
  never overwrite history. Keep raw API responses for audit.
- Fixed random seed so results are reproducible.
- Every new estimate gets a data-quality check and a stated limitation.

## Honesty rules for this project (important)

The value of the demo is judgment, so never paper over limitations:
- Vast.ai listings are **asking prices**, not cleared trades; OCPI is transaction-based. Any
  measured basis partly reflects that gap.
- Vast.ai is a peer-to-peer marketplace skewed toward smaller hosts — a different segment
  from where large buyers transact.
- The sample is short and covers one rising-price regime.
- Keep uncertain inputs as sliders rather than hard-coding a false-precision estimate.
- Check Ornn's terms of use before publishing anything built on their data.
