"""Data acquisition and cleaning.

Raw downloads are cached in data/raw/ and recorded in data/raw/manifest.json (source URL, rows,
date range, SHA-256, download time). Analysis code only ever reads the cache, so a run is
reproducible given the same raw files, even if an upstream vendor later revises history.
"""
from __future__ import annotations

import hashlib
import io
import json
import tomllib
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
MANIFEST = RAW / "manifest.json"


def load_config(path: Path = ROOT / "config" / "study.toml") -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


# ---------------------------------------------------------------- download
def _record(path: Path, source: str, df: pd.DataFrame) -> None:
    man = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    man[path.name] = {
        "source": source,
        "rows": len(df),
        "first": str(df.index.min()),
        "last": str(df.index.max()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    MANIFEST.write_text(json.dumps(man, indent=2))


def download_yahoo(ticker: str, start: str, end: str, interval: str = "1d", period: str | None = None) -> pd.DataFrame:
    import yfinance as yf
    kw = dict(period=period) if period else dict(start=start, end=(pd.Timestamp(end) + pd.Timedelta(days=1)).date().isoformat())
    df = yf.download(ticker, interval=interval, auto_adjust=False, progress=False, **kw)
    if df.empty:
        raise RuntimeError(f"Yahoo returned no data for {ticker}")
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df.index.name = "date"
    return df


def download_fred(series: str, start: str, end: str) -> pd.DataFrame:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), na_values=["."], parse_dates=[0], index_col=0)
    df.index.name = "date"
    df.columns = ["Close"]
    return df.loc[start:end]


def download_binance_month(symbol: str, interval: str, month: str) -> pd.DataFrame:
    url = f"https://data.binance.vision/data/spot/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{month}.zip"
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        raw = pd.read_csv(z.open(z.namelist()[0]), header=None, usecols=range(6))
    raw.columns = ["open_time", "Open", "High", "Low", "Close", "Volume"]
    if not np.issubdtype(raw["open_time"].dtype, np.number):  # some files carry a header row
        raw = raw[pd.to_numeric(raw["open_time"], errors="coerce").notna()].astype(float)
    # Binance switched spot timestamps from milliseconds to microseconds on 2025-01-01.
    unit = "us" if raw["open_time"].iloc[0] > 1e14 else "ms"
    raw.index = pd.to_datetime(raw.pop("open_time").astype("int64"), unit=unit, utc=True)
    raw.index.name = "date"
    return raw


def fetch_all(cfg: dict, force: bool = False) -> None:
    """Download every configured series into data/raw (skips files already cached)."""
    RAW.mkdir(parents=True, exist_ok=True)
    s, e = cfg["study"]["start"], cfg["study"]["end"]
    for a in cfg["assets"]:
        path = RAW / f"{a['id']}.csv"
        if path.exists() and not force:
            continue
        df = download_yahoo(a["ticker"], s, e) if a["source"] == "yahoo" else download_fred(a["ticker"], s, e)
        df.to_csv(path)
        _record(path, f"{a['source']}:{a['ticker']}", df)
        print(f"  {a['id']:7s} {len(df):6d} rows  {df.index.min().date()} .. {df.index.max().date()}")

    ic = cfg["intraday"]
    path = RAW / f"{ic['binance_symbol']}_{ic['binance_interval']}.csv.gz"
    if force or not path.exists():
        months = pd.period_range(ic["binance_start"], ic["binance_end"], freq="M").strftime("%Y-%m")
        df = pd.concat([download_binance_month(ic["binance_symbol"], ic["binance_interval"], m) for m in months])
        df = df[~df.index.duplicated()].sort_index()
        df.to_csv(path)
        _record(path, f"binance:{ic['binance_symbol']}:{ic['binance_interval']}", df)
        print(f"  {path.name}: {len(df)} bars")

    # Yahoo intraday is a rolling window: this is a dated snapshot, not reproducible from source.
    for interval, period in [("5m", "60d"), ("60m", "730d")]:
        path = RAW / f"{ic['yahoo_ticker']}_{interval}.csv"
        if force or not path.exists():
            df = download_yahoo(ic["yahoo_ticker"], s, e, interval=interval, period=period)
            df.to_csv(path)
            _record(path, f"yahoo:{ic['yahoo_ticker']}:{interval}:{period}", df)
            print(f"  {path.name}: {len(df)} bars")


# ---------------------------------------------------------------- load + clean
def load_raw(name: str) -> pd.DataFrame:
    return pd.read_csv(RAW / name, index_col=0, parse_dates=[0])


def load_prices(asset: dict, cfg: dict) -> pd.Series:
    """Clean daily price (or yield) series for one configured asset.
    Missing / non-positive values are dropped, never forward-filled: a forward-fill would create
    artificial zero returns and bias the ACF of |r| downwards."""
    df = load_raw(f"{asset['id']}.csv")
    col = "Adj Close" if "Adj Close" in df.columns else "Close"
    p = df[col].astype(float).loc[cfg["study"]["start"]:cfg["study"]["end"]]
    p = p[np.isfinite(p)]
    if asset["kind"] == "price":
        p = p[p > 0]
    p.name = asset["id"]
    return p


def returns(prices: pd.Series, kind: str) -> pd.Series:
    """Log returns between consecutive available observations; for yields, first differences."""
    r = np.log(prices).diff() if kind == "price" else prices.diff()
    return r.dropna()


def load_returns(cfg: dict) -> dict[str, pd.Series]:
    return {a["id"]: returns(load_prices(a, cfg), a["kind"]) for a in cfg["assets"]}


def data_quality(r: pd.Series, big: float = 0.25) -> dict:
    """Summary used in the data section: sample, zero-return share (stale prices), extreme moves."""
    return {
        "start": r.index.min().date(), "end": r.index.max().date(), "n": len(r),
        "zero_share": float((r == 0).mean()),
        "n_abs_gt_25pct": int((r.abs() > big).sum()),
        "max_abs": float(r.abs().max()),
    }
