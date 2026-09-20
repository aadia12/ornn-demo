"""
ocpi_pull.py — Pull Ornn Compute Price Index (OCPI) data from Ornn's free public API.

Free tier (no key, no account): current price + trailing 3 months of daily settled
history for 5 GPUs. Anonymous requests are rate-limited per IP.
Docs: https://dashboard.ornnai.com/docs

Because the free window is a rolling 3 months, this script stores history
INCREMENTALLY: every run merges new rows into a local CSV and never deletes old
ones. Run it daily (cron / GitHub Actions) and your history grows past 3 months.

Usage:
    pip install requests pandas numpy
    python ocpi_pull.py                       # all free GPUs, trailing 3 months
    python ocpi_pull.py --gpus "H100 SXM" H200
    python ocpi_pull.py --stats               # also print summary stats
    ORNN_API_KEY=sk_... python ocpi_pull.py   # optional: key unlocks full history
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_URL = "https://api.ornnai.com"
FALLBACK_FREE_GPUS = ["A100 SXM4", "B200", "H100 SXM", "H200", "RTX 5090"]
SETTLEMENT_TZ = "America/New_York"  # OCPI settles once per trading day at 4:00 PM ET
TRADING_DAYS_PER_YEAR = 365  # OCPI settles 7 days a week
REQUEST_PAUSE_S = 0.5  # be polite to the anonymous rate limit

log = logging.getLogger("ocpi")


# ----------------------------------------------------------------------------
# HTTP client
# ----------------------------------------------------------------------------
class OrnnClient:
    """Thin wrapper around the Ornn API with retries and optional API key."""

    def __init__(self, api_key: str | None = None, timeout: float = 20.0):
        self.timeout = timeout
        self.session = requests.Session()
        retry = Retry(
            total=5,
            backoff_factor=1.5,  # 0s, 1.5s, 3s, 6s, 12s
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
            respect_retry_after_header=True,
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retry))
        self.session.headers["User-Agent"] = "ocpi-hedging-demo/0.1"
        if api_key:
            self.session.headers["Authorization"] = f"Bearer {api_key}"
        self.has_key = bool(api_key)

    def get(self, path: str, params: dict | None = None) -> dict:
        url = f"{BASE_URL}{path}"
        resp = self.session.get(url, params=params, timeout=self.timeout)
        if resp.status_code == 401:
            raise PermissionError(f"401 from {url} — this resource needs an API key.")
        resp.raise_for_status()
        payload = resp.json()
        if not payload.get("success", False):
            raise RuntimeError(f"API reported failure for {url}: {payload}")
        time.sleep(REQUEST_PAUSE_S)
        return payload

    # --- endpoints -----------------------------------------------------------
    def gpu_types(self) -> list[str]:
        """GPUs reachable with the current credentials."""
        path = "/api/gpu-types" if self.has_key else "/api/gpu-types-free"
        try:
            data = self.get(path)["data"]
            return sorted({row["gpu_name"] for row in data})
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not fetch GPU list (%s); using fallback list.", exc)
            return FALLBACK_FREE_GPUS

    def current_price(self, gpu: str) -> dict:
        """Live hourly index value (not the settled close)."""
        return self.get(f"/api/gpu/{quote(gpu)}")["data"]

    def daily_all(self) -> dict:
        """Most recent settled daily index for every accessible GPU."""
        return self.get("/api/daily-index/all")

    def index_history(self, gpu: str, start: str | None = None,
                      end: str | None = None) -> list[dict]:
        """Daily settled history. Free tier clamps to the trailing 3 months."""
        params = {}
        if start:
            params["startDate"] = start
        if end:
            params["endDate"] = end
        return self.get(f"/api/gpu/{quote(gpu)}/index-history", params)["data"]


# ----------------------------------------------------------------------------
# Transform / validate
# ----------------------------------------------------------------------------
def history_to_frame(rows: list[dict], gpu: str) -> pd.DataFrame:
    """Convert raw history rows into a tidy frame keyed by settlement date (ET)."""
    if not rows:
        return pd.DataFrame(columns=["settle_date", "gpu", "index_value", "timestamp_utc"])
    df = pd.DataFrame(rows)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp"], utc=True)
    # Settlement timestamps land at 20:00 or 21:00 UTC depending on daylight
    # saving; converting to ET before taking the date keeps one row per trading day.
    df["settle_date"] = df["timestamp_utc"].dt.tz_convert(SETTLEMENT_TZ).dt.date
    df["gpu"] = gpu
    df["index_value"] = pd.to_numeric(df["index_value"], errors="coerce")
    return df[["settle_date", "gpu", "index_value", "timestamp_utc"]]


def validate(df: pd.DataFrame) -> list[str]:
    """Basic data-quality checks. Returns a list of human-readable warnings."""
    issues = []
    for gpu, g in df.groupby("gpu"):
        g = g.sort_values("settle_date")
        if g["index_value"].isna().any():
            issues.append(f"{gpu}: {g['index_value'].isna().sum()} missing values")
        if (g["index_value"] <= 0).any():
            issues.append(f"{gpu}: non-positive index values present")
        dupes = g["settle_date"].duplicated().sum()
        if dupes:
            issues.append(f"{gpu}: {dupes} duplicate settlement dates")
        # Flag gaps longer than a long weekend (> 4 calendar days)
        dates = pd.to_datetime(g["settle_date"])
        gaps = dates.diff().dt.days
        big = gaps[gaps > 4]
        for idx, days in big.items():
            issues.append(f"{gpu}: {int(days)}-day gap ending {dates[idx].date()}")
        # Flag one-day moves > 25% as possible data errors worth eyeballing
        rets = np.log(g["index_value"]).diff().abs()
        jumps = rets[rets > np.log(1.25)]
        for idx in jumps.index:
            issues.append(f"{gpu}: >25% one-day move on {g.loc[idx, 'settle_date']}")
    return issues


def merge_into_store(new: pd.DataFrame, store_path: Path) -> pd.DataFrame:
    """Append new rows to the local CSV; newest pull wins on (gpu, settle_date)."""
    if store_path.exists():
        old = pd.read_csv(store_path, parse_dates=["timestamp_utc"])
        old["settle_date"] = pd.to_datetime(old["settle_date"]).dt.date
        combined = pd.concat([old, new], ignore_index=True)
    else:
        combined = new
    combined = (
        combined.drop_duplicates(subset=["gpu", "settle_date"], keep="last")
        .sort_values(["gpu", "settle_date"])
        .reset_index(drop=True)
    )
    store_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(store_path, index=False)
    return combined


# ----------------------------------------------------------------------------
# Summary statistics (the inputs the hedging simulation will need)
# ----------------------------------------------------------------------------
def summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    wide = df.pivot(index="settle_date", columns="gpu", values="index_value").sort_index()
    log_rets = np.log(wide).diff()
    out = pd.DataFrame({
        "obs": wide.count(),
        "first_date": wide.apply(lambda s: s.first_valid_index()),
        "last_date": wide.apply(lambda s: s.last_valid_index()),
        "last_value": wide.ffill().iloc[-1],
        "ann_vol": log_rets.std() * np.sqrt(TRADING_DAYS_PER_YEAR),
        "total_return": wide.ffill().iloc[-1] / wide.bfill().iloc[0] - 1,
    })
    # AR(1) on log price: x_{t+1} = a + b x_t + e -> half-life = ln(2) / -ln(b)
    half_lives = {}
    for gpu in wide.columns:
        x = np.log(wide[gpu].dropna())
        if len(x) < 20:
            half_lives[gpu] = np.nan
            continue
        b = np.polyfit(x.values[:-1], x.values[1:], 1)[0]
        half_lives[gpu] = np.log(2) / -np.log(b) if 0 < b < 1 else np.inf
    out["mr_half_life_days"] = pd.Series(half_lives)
    return out, log_rets.corr()


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Pull OCPI data from Ornn's API.")
    parser.add_argument("--gpus", nargs="*", help="GPU names (default: all accessible)")
    parser.add_argument("--start", help="startDate YYYY-MM-DD (free tier clamps to 3 months)")
    parser.add_argument("--end", help="endDate YYYY-MM-DD")
    parser.add_argument("--outdir", default="data", help="Output directory")
    parser.add_argument("--stats", action="store_true", help="Print summary stats")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    outdir = Path(args.outdir)
    raw_dir = outdir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    client = OrnnClient(api_key=os.environ.get("ORNN_API_KEY"))
    gpus = args.gpus or client.gpu_types()
    log.info("Pulling %d GPU(s): %s (key=%s)", len(gpus), gpus, client.has_key)

    frames, raw_snapshot = [], {}
    for gpu in gpus:
        try:
            rows = client.index_history(gpu, args.start, args.end)
        except Exception as exc:  # noqa: BLE001
            log.error("History failed for %s: %s", gpu, exc)
            continue
        raw_snapshot[gpu] = rows
        frames.append(history_to_frame(rows, gpu))
        log.info("  %-10s %4d daily rows", gpu, len(rows))

    # Also capture the latest settled values in one call, for a sanity check
    try:
        raw_snapshot["_daily_index_all"] = client.daily_all()
    except Exception as exc:  # noqa: BLE001
        log.warning("daily-index/all failed: %s", exc)

    # Keep raw JSON for every run: an audit trail if anything looks off later
    (raw_dir / f"ocpi_raw_{run_ts}.json").write_text(json.dumps(raw_snapshot, indent=2))

    if not frames:
        raise SystemExit("No data pulled — check connectivity or rate limits.")

    new = pd.concat(frames, ignore_index=True)
    store = merge_into_store(new, outdir / "ocpi_history.csv")
    log.info("Store now holds %d rows across %d GPU(s) -> %s",
             len(store), store["gpu"].nunique(), outdir / "ocpi_history.csv")

    for issue in validate(store):
        log.warning("DATA CHECK: %s", issue)

    if args.stats:
        stats, corr = summary_stats(store)
        pd.set_option("display.width", 120)
        print("\n=== OCPI summary (daily settled) ===")
        print(stats.round(4).to_string())
        print("\n=== Correlation of daily log returns ===")
        print(corr.round(3).to_string())
        print(f"\nNote: {len(wide)} days is a short sample; treat vol and "
              "half-life estimates as rough starting points.")


if __name__ == "__main__":
    main()
