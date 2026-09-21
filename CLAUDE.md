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
| `app.py` | Streamlit dashboard over the same engine. Its **"Refresh OCPI data from Ornn"** button mirrors `ocpi_pull.py`'s `main()` — same incremental merge, same raw JSON snapshot to `data/raw/`, same `validate()` checks, with warnings surfaced via `st.sidebar.warning`. Keep it that way: it writes to the tracked CSV, so it must not be a shortcut around the audit trail or the data-quality checks. Top: scenario line + four headline metrics (simulated variance removed with its 95% CI, theoretical ρ², hedge ratio, Monte Carlo sample size). Charts are cost histogram / simulated path fan chart on the first row, **bad-case interval chart** / variance-removed-vs-rho curve on the second. An **"About this dashboard"** expander at the foot carries the standing write-up: what the demo is, the data source, how the model works, its four stated limitations (short history, rho is an assumption not a measurement, no forward curve, point settlement) and next steps. Its simulation count is interpolated from the `n_sims` slider, not hardcoded — keep it that way. **Backtest on realized OCPI history sits last** (three metrics + cumulative-cost and OCPI-vs-locked charts). Six charts total. Default GPU is **H100 SXM** |
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

### Backtest on realized history (`app.py`, bottom section)

Lock a swap at the first settled index in the CSV, then buy at the daily index for the
rest of the window. Unhedged = `sum(N * 24 * daily index)`; hedged = `N * 24 * locked *
days`. No transaction costs, no margin, no discounting.

The "Hedge saved" metric is quoted as a **share of the unhedged bill**
(`pct_unhedged = diff / unhedged`), which reads as "the hedge saved X% of what you'd
otherwise have paid". The dollar amount is not repeated there — it is already the delta
under "Cost with hedge". `bt["pct"]` (vs the locked cost) is still computed and is what
the earlier 15.3% / 44.4% figures referred to; don't mix the two denominators.

Realized results, 500 GPUs, 93 days (2026-06-20 to 2026-09-20):

| GPU | locked | unhedged | hedged | hedge P&L | % of unhedged (shown in UI) |
|---|---|---|---|---|---|
| H100 SXM | $2.34 | $3.01M | $2.61M | +$0.40M | **+13.3%** |
| B200 | $4.28 | $6.90M | $4.78M | +$2.12M | **+30.7%** |
| H200 | $3.46 | $5.07M | $3.86M | +$1.20M | +23.8% |
| A100 SXM4 | $1.05 | $1.15M | $1.17M | −$0.02M | **−1.5%** |
| RTX 5090 | $0.57 | $0.61M | $0.64M | −$0.03M | **−4.9%** |

**The hedge loses on two of five GPUs, and that is the point** — it is insurance, fair in
expectation at entry, not alpha. Captions must stay conditional on the realized outcome;
never hard-code "prices rose".

**Subtlety worth preserving:** hedge P&L is set by the **average** index over the window
versus the locked price, *not* by where prices finished. RTX 5090 rose start-to-end
($0.57 → $0.66) yet still **lost** 4.7%, because it averaged below the lock. The caption
logic therefore branches on both `saved` and `ended_above` (four cases), so a rising
chart never renders next to the words "prices fell".

That caption is the backtest's **only** commentary and sits inside the left column,
under the Cumulative cost chart. It is built as a direction-aware opener plus one shared
tail ("The outcome depends entirely on which way prices happened to move, which is why
the simulation covers {n_sims:,} possible futures"), so the four branches only differ in
their first sentence. The two loss branches carry "insurance doesn't pay out every time"
— **keep that clause**, it is the only remaining statement of the fair-in-expectation
point after the separate full-width captions were merged away.

### Confidence intervals on the headline metrics (`app.py`)

The p95 and variance-reduction numbers rest on a volatility estimated from ~3 months of
data, so the app states how uncertain that estimate is — as captions under the existing
metrics, no extra charts.

- **Stationary block bootstrap** (Politis & Romano) of the daily log returns, geometric
  block lengths with expected length **7 days**, circular wrap, 2,000 resamples, seed 42.
- **Blocks, not iid resampling.** The variance ratio of ~0.36 says daily returns are not
  independent; resampling single days would erase the reversal, rebuild something that
  really is a random walk, and make the vol estimate look more precise than it is.
- Each resample is rebuilt into a log-price path and vol is re-estimated **exactly as
  `calibrate()` does** — overlapping 7-day returns, `std(ddof=1) * sqrt(365/7)`.
- CI = 2.5th/97.5th percentiles of the resampled vols.
- The **full simulation runs at only three vols** (CI low, point, CI high) via
  `cached_ci_sims`. Never run the simulation inside the bootstrap loop: 2,000 x 10,000
  paths would hang the app.
- **Cache keys:** `cached_vol_ci` is keyed on **GPU only** — it depends on that GPU's
  price history and nothing else. `cached_ci_sims` is keyed on the full parameter set,
  like `cached_sim`, since the cost range depends on all of it.

H100 SXM, 500 GPUs, rho 0.8: vol point estimate **39.9%**, 95% CI **32.9%–65.6%**
(93 days); variance removed holds at **63–65%** across that whole range — the headline
finding is robust to the volatility estimate. Measured cost: bootstrap ~20ms, cold
full-page run 1.9s, warm rerun 0.3s, so no deferred-fill fallback was needed.

The **bad-case interval chart** (bottom right, beside the fan chart) is where the
95th-percentile costs live. Two rows, unhedged and hedged: a diamond at the p95 and an
asymmetric whisker spanning what that p95 becomes across the volatility CI, with a
dotted reference line at the budget. It makes the hedge's two effects visible at once —
the diamond moves left (less exposure) and the whisker shortens (less uncertainty about
the exposure). Everything is computed from the selected GPU; nothing hardcoded.

Data labels: the p95 sits above each diamond in the series colour (`top center`); the CI
bounds sit at the whisker ends in small grey, positioned **outward** (`middle left` /
`middle right`) so they clear the bar — placing them inward puts the text straight on
top of the whisker. The x-range is padded by 20% of the span (floor 0.6M) and widened to
include the budget line so those outward labels aren't clipped; that leaves 11–16% of
the visible width free on the right across all five GPUs. Recheck the padding if label
text or font sizes change.

Sizing: this chart and the variance-removed curve beside it are both **320px** so the
row lines up. Because plotly spaces categories as a fraction of plot height, matching
heights would normally push the two bars apart, so the y-range is widened past the
default `[-0.5, 1.5]` to **`[-0.95, 1.95]`**. That yields a 74px gap between the rows
(down from 145px) with 70px of clear space above and below — balanced, and enough
headroom that the top data label isn't clipped. Every trace also sets
**`cliponaxis=False`**, which is the actual fix for labels being cut off at the plot
edge; the range widening alone would not guarantee it.

Layout note: top padding is trimmed via injected CSS right after `set_page_config`,
because Streamlit's default ~6rem gap made the page open on empty space. Three
selectors matter, all verified present in the installed Streamlit build first:
`stMainBlockContainer` (main column), `stSidebarHeader` (a separate div reserving room
for the collapse arrow — the one that's easy to miss), and the sidebar's `h2`, which
carries its own top padding. Sidebar subheaders are `h3` and keep their spacing.

### Visual theme (matched to ornn.com)

`.streamlit/config.toml` sets the theme; the palette is lifted from the design tokens
published in ornn.com's markup.

**`config.toml` deliberately holds ONE `[theme]` table — do not add `[theme.light]` /
`[theme.dark]` sub-tables.** When both are defined Streamlit treats the app as
supporting either and picks from the browser/OS preference, which silently overrode
`base = "dark"` and opened the app in light mode. With a single table the custom theme
always wins and the app opens dark.

Light mode is instead an **in-app `st.toggle`** (`key="dark_mode"`) in a narrow
right-hand column beside the title. Its **label is dynamic** — it names the mode the
switch takes you *to* ("Light mode" while in dark, "Dark mode" while in light) — while
the switch position still tracks dark on/off, which is the state the theme block reads.
A button variant was tried and reverted; the toggle is the wanted affordance.
(A `position: fixed` variant that floated the control into Streamlit's toolbar beside
Deploy was also tried and reverted — Streamlit has no API for adding widgets to its
toolbar, and the fixed-offset hack is fragile.)
The theme block at the top of `app.py` reads that session-state key,
rewrites the `theme.*` config options via `config.set_option` and calls `st.rerun()` —
necessary because the theme is serialised into the `NewSession` message at the *start*
of a run, so a change only takes effect on the next one. `_theme_applied` is seeded to
`"dark"` so first load doesn't rerun, and guards against a rerun loop. Caveat: this
mutates **process-global** config, so in a multi-user deployment one viewer's toggle
would move everyone's theme. Fine for a local demo; would need rethinking if hosted.

| | dark | light |
|---|---|---|
| accent (`primaryColor`) | `#AD8147` | `#9A6F35` |
| page | `#141414` | `#F3F3F3` |
| surfaces | `#212121` | `#E8E8E8` |
| text | `#F3F3F3` | `#141414` |
| borders | `#3D4045` | `#C2C2C2` |

ornn.com itself sits on pure `#000000`, but across a full dashboard that reads as harsh,
so dark mode uses `#141414`. The config comments carry darker and lighter alternatives.

`baseRadius = "none"` squares the corners, matching the site (its CSS declares
`border-radius: 0` and carries no numeric radii). A short CSS block squares the few
widgets that carry their own radius — BaseWeb selects, popovers, expanders, metrics,
alerts, plotly containers, buttons and inputs.

**Chart palette lives in `app.py`, not the config** — Streamlit's theme doesn't reach
inside plotly. `palette(mode)` is the single source of truth for both modes: it feeds
the `pio.templates["ornn"]` template, the injected CSS, and the `config.set_option`
calls that restyle Streamlit's own chrome. Add a colour there, not in three places.

Semantic rule to preserve: **`ACCENT` (bronze) carries the hedge and any single-series
chart; `NEUTRAL` carries the unhedged baseline.** Bronze always marks the signal.
`MUTED` is for reference lines and secondary labels.

**Don't set the light-mode neutral to `#6F7681`.** It's the obvious pick from the
palette, but its luminance is almost identical to the bronze — contrast ratio 1.03, so
the two chart series become near-indistinguishable. Light mode uses `#3D4045` instead
(2.33 against the accent, 9.38 against the page). Measured ratios against their own
page: dark accent 5.27, neutral 9.22, muted 4.02; light accent 4.03, neutral 9.38,
muted 4.13. Gridlines are decorative and intentionally below 3:1.

### CSS: verify every selector against the frontend bundle

**`data-baseweb` attributes do not exist in this Streamlit build.** Four rounds of
selectbox styling silently did nothing because every rule was scoped through
`[data-baseweb="select"]`, which matches zero elements. The bronze outline that seemed
to appear on click was Streamlit's own focus ring — bronze because `primaryColor` is.
`stThumbValue` and `stNotification` were dead for the same reason.

Before writing a selector, grep the bundle:

```
python -c "import streamlit,pathlib; p=pathlib.Path(streamlit.__file__).parent/'static'; \
print(sum(f.read_text(errors='ignore').count('stSelectbox') for f in p.rglob('*.js')))"
```

A zero means the selector will never match, and no amount of `!important` will save it.
Everything currently in `app.py`'s style block has been checked this way.

The **GPU picker** drives every number on the page, so its control carries a bronze
border and an `INK` label — `INK` rather than a literal white, so it flips to near-black
in light mode instead of disappearing. Rules are scoped to
`[data-testid="stSidebar"] [data-testid="stSelectbox"]`; there is exactly one selectbox,
so no per-widget key is needed. Bolding the label and the selected value was tried and
reverted — it read badly. Note that if anything ever does need to style the open
dropdown list, it renders in a portal *outside* the sidebar and needs an unscoped rule
on `stSelectboxVirtualDropdown`.

`st.metric` draws a **trend arrow** beside any delta. Every delta on this page is a
description ("95% CI: 60–64%", "Monte Carlo paths"), not a movement, so
`stMetricDeltaIcon-Up` / `-Down` are hidden — with `padding-left` on `stMetricDelta` to
replace the indent the icon used to provide, otherwise the text sits flush against the
tile's left edge.

Note the CSS block is an f-string: every literal CSS brace must be doubled.

Fonts are left alone. ornn.com uses Geist Mono / Fragment Mono throughout, so switching
the app to a monospace stack (via `theme.font` or `theme.fontFaces`) would move it
closer still.

**`cached_ci_sims` runs at `(lo, point_estimate, hi)` — never the volatility slider.**
An earlier version passed the slider value as the middle run, so dragging volatility
past a CI bound stretched the span to include it while the label still read "95% CI".
The interval is a claim about the data and must not move when a slider does.

That decoupling means the diamond (which tracks the slider, so it matches the histogram
and the headline metrics) can now legitimately fall **outside** its whisker. The chart
supports this because the interval is drawn as **its own line segment**, not as
`error_x` on the marker — error bars are offsets from the marker and cannot render a
marker outside them. When the slider sits beyond the CI the diamond visibly detaches,
the x-range expands to keep it in frame, and the caption says why. Verified at 5%, 33%,
40%, 66% and 120% volatility: whiskers pinned at $16.00M–$21.19M throughout, diamond
detaching and being flagged in both directions.

Only "Simulated budget variance removed" keeps a CI caption up top, and it likewise no
longer moves with the slider.

Relief and whisker width by GPU (500 GPUs, rho 0.8, 93 days):

| GPU | bad case cut by | whisker unhedged → hedged | variance removed |
|---|---|---|---|
| B200 | $14.77M (22%) | $23.02M → $12.87M | 61% |
| H100 SXM | $2.33M (14%) | $5.18M → $2.90M | 64% |
| RTX 5090 | $0.67M (15%) | $1.78M → $1.00M | 64% |
| A100 SXM4 | $0.55M (10%) | $1.70M → $0.94M | 65% |

**The CI's skew is not always to the right** — worth knowing even though the UI no
longer states it in words. The natural explanation (volatility is bounded at zero, so
the sampling distribution skews right) holds for only three of five GPUs. Percentage
points below/above the point estimate:

| GPU | below | above | skew | variance ratio |
|---|---|---|---|---|
| H100 SXM | 7 | 26 | right | 0.35 |
| A100 SXM4 | 5 | 25 | right | 0.24 |
| H200 | 19 | 29 | right | 0.64 |
| RTX 5090 | 25 | 21 | ~symmetric | 0.98 |
| **B200** | **36** | **12** | **LEFT** | **2.17** |

B200 — the default GPU — leans left: its point estimate is propped up by a sustained
run in a short sample, and resampled histories that break the run come out calmer. **If
any copy ever describes the interval's shape again, branch on the measured asymmetry
rather than assuming right-skew** — same rule as the backtest captions: never assert a
direction the data hasn't been checked for.

Sidebar order is **Scenario inputs → Model settings → Market assumptions**. Model
settings is built into an `st.sidebar.container()` reserved up front, because the
"Share of usage hedged" control belongs at the bottom of that block but needs `rho` and
the volatilities that are only set further down the page. Create the container early,
fill it late.

**Reconciliation note:** on the 92-day window this reproduces 31.6%–63.2% (seed 42; 20
seeds span 31.2–32.3% and 62.5–64.4%), matching the earlier 31.1%–62.4% to within
bootstrap Monte Carlo error. The CI widened on day 93 because H100 fell 7% in one day
($2.85 → $2.65). That same move rescaled every p95: budget went $12.48M → $11.61M, and
p95 unhedged $18.34M → $17.07M — exactly the ratio 2.65/2.85. **Absolute p95 levels
track the latest spot and will keep moving; the CI width and the 63–65% band are the
stable results.**

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
- **The rolling free-tier window has already started dropping days** (verified
  2026-09-21): the API returns from 2026-06-21, while the local CSV holds 2026-06-20.
  That day now exists *only* in the local file. The additive merge is what preserves it,
  so this is no longer a hypothetical benefit — and it means letting the CSV go stale
  permanently loses whatever rolls out of the window in the meantime. Checked at the
  same time: OCPI does not restate settled history (zero existing values revised by a
  refresh), which matters because the merge uses `keep="last"` and would accept
  restatements silently.
- Data writes are **incremental and additive**: merge and de-duplicate on a natural key,
  never overwrite history. Keep raw API responses for audit.
- Fixed random seed so results are reproducible.
- **Escape `$` as `\$` in every Streamlit markdown string** (`st.write`, `st.markdown`,
  `st.caption`). Streamlit parses a bare `$...$` as LaTeX, so two dollar amounts in one
  sentence swallow the text between them and render it as math — this is what produced
  the stray-asterisk and part-green/part-grey rendering bugs. Use the `money_md()` and
  `price_md()` helpers in `app.py` for markdown, and plain `money()` for `st.metric`
  values and deltas, which are **not** markdown-rendered and would show the backslash.
  Plotly titles, legend names and annotations are not markdown either — leave those bare.
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
