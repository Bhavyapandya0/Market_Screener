import os
import logging
import subprocess
import sys
from pathlib import Path

# Suppress Streamlit ScriptRunContext warnings from background threads
for _logger_name in [
    "streamlit.runtime.scriptrunner_utils.script_run_context",
    "streamlit.runtime.scriptrunner.script_run_context",
    "streamlit",
]:
    logging.getLogger(_logger_name).setLevel(logging.ERROR)

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import concurrent.futures
import time

# ── Stock Universe ──────────────────────────────────────────────────────────
NIFTY_500 = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR", "ITC",
    "SBIN", "BHARTIARTL", "BAJFINANCE", "KOTAKBANK", "LT", "HCLTECH",
    "AXISBANK", "ASIANPAINT", "MARUTI", "SUNPHARMA", "TITAN", "ULTRACEMCO",
    "BAJAJFINSV", "WIPRO", "ONGC", "NTPC", "JSWSTEEL", "POWERGRID",
    "M&M", "TATAMOTORS", "ADANIENT", "ADANIPORTS", "TATASTEEL",
    "NESTLEIND", "TECHM", "INDUSINDBK", "HDFCLIFE", "BAJAJ-AUTO",
    "GRASIM", "DIVISLAB", "CIPLA", "BRITANNIA", "DRREDDY", "EICHERMOT",
    "APOLLOHOSP", "COALINDIA", "SBILIFE", "BPCL", "TATACONSUM",
    "HINDALCO", "HEROMOTOCO", "DABUR", "HAVELLS", "PIDILITIND",
    "SIEMENS", "GODREJCP", "DLF", "BANKBARODA", "IOC", "AMBUJACEM",
    "SHREECEM", "TRENT", "ICICIPRULI", "ACC", "COLPAL", "TORNTPHARM",
    "ABB", "MARICO", "PNB", "BERGEPAINT", "NAUKRI", "MCDOWELL-N",
    "INDIGO", "VEDL", "JINDALSTEL", "GAIL", "LUPIN", "PIIND",
    "CANBK", "SAIL", "FEDERALBNK", "IDFCFIRSTB", "PERSISTENT",
    "LTIM", "MAXHEALTH", "TATAPOWER", "NHPC", "IDEA", "IRCTC",
    "HAL", "BEL", "RECLTD", "PFC", "CONCOR", "MOTHERSON",
    "SOLARINDS", "DMART", "ZOMATO", "PAYTM", "POLICYBZR",
    "ADANIGREEN", "ADANIPOWER", "ATGL", "AWL", "LODHA",
    "JIOFIN", "MANKIND", "TIINDIA", "CUMMINSIND", "VOLTAS",
    "AUROPHARMA", "MUTHOOTFIN", "CHOLAFIN", "TVSMOTOR", "PAGEIND",
    "ASTRAL", "BALKRISIND", "CROMPTON", "LICHSGFIN", "OBEROIRLTY",
    "PRESTIGE", "PHOENIXLTD", "COFORGE", "MPHASIS", "LTTS",
    "DIXON", "POLYCAB", "DEEPAKNTR", "ATUL", "NAVINFLUOR",
    "SRF", "UPL", "BIOCON", "ALKEM", "LALPATHLAB", "METROPOLIS",
    "IPCALAB", "ABCAPITAL", "MANAPPURAM", "L&TFH", "SBICARD",
    "AUBANK", "BANDHANBNK", "RBLBANK", "IDFC", "GMRINFRA",
    "IRFC", "SJVN", "NMDC", "HINDZINC", "NATIONALUM",
    "RAIN", "GNFC", "CHAMBLFERT", "COROMANDEL", "UBL",
    "OFSS", "MFSL", "BSE", "CDSL", "MCX",
    "KPITTECH", "HAPPSTMNDS", "ZYDUSLIFE", "TORNTPOWER", "CESC",
    "TATAELXSI", "SONACOMS", "JUBLFOOD", "DEVYANI", "SAPPHIRE",
    "ABFRL", "VBL", "BATAINDIA", "RELAXO", "RAJESHEXPO",
    "GODREJPROP", "BRIGADE", "SOBHA", "SUNTV", "PVRINOX",
    "ESCORTS", "ASHOKLEY", "EXIDEIND", "AMARAJABAT", "BOSCHLTD",
    "BHARATFORG", "SUNDARMFIN", "CANFINHOME", "AAVAS", "HOMEFIRST",
    "KAJARIACER", "CENTURYTEX", "JKCEMENT", "RAMCOCEM", "DALBHARAT",
    "STARCEMENT", "PETRONET", "GSPL", "IGL", "MGL",
    "HDFCAMC", "NIACL", "ICICIGI", "STARHEALTH",
    "KEI", "SUPREMEIND", "APLAPOLLO", "RATNAMANI",
    "CARBORUNIV", "GRINDWELL", "FINEORG", "CLEAN",
    "SUMICHEM", "TATACHEM", "AARTI", "FLUOROCHEM",
    "SYNGENE", "GLAND", "NATCOPHARM", "LAURUSLABS",
    "WHIRLPOOL", "BLUESTARCO", "ORIENTELEC",
    "KAYNES", "CYIENT", "MASTEK", "ROUTE", "TANLA",
    "JBCHEPHARM", "SUPRIYA", "LAXMIMACH", "CERA",
    "GRSE", "COCHINSHIP", "MAZDOCK", "GARDENREACH",
    "ZEEL", "NETWORK18", "TV18BRDCST", "HATHWAY",
    "ENGINERSIN", "RITES", "RAILTEL", "RVNL", "HUDCO",
    "JSWENERGY", "ADANITRANS", "THERMAX", "CGPOWER",
    "SCHAEFFLER", "TIMKEN", "SKFIND",
    "CENTRALBK", "UNIONBANK", "INDIANB", "MAHABANK",
    "UCOBANK", "BANKINDIA", "PSB", "IOB",
]

# Preferred external symbol file (latest Nifty 500 constituents).
# Keep this file in the repo so Streamlit Cloud can access it.
DEFAULT_NIFTY500_CSV = Path("ind_nifty500list.csv")
# Optional local sector mapping file for fast lookups.
SECTOR_CSV = Path("nifty500_sectors.csv")

# Map stale NSE symbols to current/working Yahoo symbols.
# Best-effort replacements for renamed/merged companies.
SYMBOL_ALIASES = {
    "ZOMATO": "ETERNAL",
    "MCDOWELL-N": "UNITDSPR",
    "L&TFH": "LTF",
    "IDFC": "IDFCFIRSTB",
    "GMRINFRA": "GMRP&UI",
    "AMARAJABAT": "ARE&M",
    "CENTURYTEX": "ABREL",
    "AARTI": "AARTIIND",
    "SKFIND": "SKFINDIA",
    "ADANITRANS": "ADANIENSOL",
    "GARDENREACH": "GRSE",
}

# Symbols intentionally excluded because no reliable 1:1 active NSE Yahoo ticker
# was found in this environment.
DISABLED_SYMBOLS = {
    "TV18BRDCST",
}


def _build_nse_symbols(symbols: list[str]) -> list[str]:
    """Normalize symbols, drop duplicates, and append NSE suffix."""
    normalized: list[str] = []
    seen: set[str] = set()
    for sym in symbols:
        if sym in DISABLED_SYMBOLS:
            continue
        mapped = SYMBOL_ALIASES.get(sym, sym)
        if mapped in seen:
            continue
        seen.add(mapped)
        normalized.append(mapped)
    return [f"{s}.NS" for s in normalized]


def _load_symbols_from_csv(path: Path) -> list[str]:
    """
    Load symbols from CSV with columns like:
    Company Name, Symbol, ISIN Code
    """
    if not path.exists():
        return []
    try:
        df = pd.read_csv(path)
    except Exception:
        return []

    col_map = {str(c).strip().lower(): c for c in df.columns}
    symbol_col = col_map.get("symbol")
    if symbol_col is None:
        return []

    symbols = []
    for raw in df[symbol_col].dropna().astype(str):
        s = raw.strip().upper()
        if not s:
            continue
        symbols.append(s)
    return symbols


def _load_sector_map_from_csv(path: Path) -> dict[str, str]:
    """Load sector/industry map from CSV when available."""
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}

    col_map = {str(c).strip().lower(): c for c in df.columns}
    symbol_col = col_map.get("symbol")
    sector_col = col_map.get("sector") or col_map.get("industry")
    if symbol_col is None or sector_col is None:
        return {}

    out: dict[str, str] = {}
    for _, row in df[[symbol_col, sector_col]].dropna().iterrows():
        raw_sym = str(row[symbol_col]).strip().upper()
        raw_sector = str(row[sector_col]).strip()
        if not raw_sym or not raw_sector:
            continue
        sym = SYMBOL_ALIASES.get(raw_sym, raw_sym)
        out[sym] = raw_sector
    return out


LOADED_SYMBOLS = _load_symbols_from_csv(DEFAULT_NIFTY500_CSV)
BASE_SYMBOLS = LOADED_SYMBOLS or NIFTY_500
NSE_SYMBOLS = _build_nse_symbols(BASE_SYMBOLS)
SECTOR_MAP_SOURCE = SECTOR_CSV if SECTOR_CSV.exists() else DEFAULT_NIFTY500_CSV
SYMBOL_SECTORS = _load_sector_map_from_csv(SECTOR_MAP_SOURCE)


# ── Scanner Logic ───────────────────────────────────────────────────────────
def fetch_stock_data(symbol: str, period: str, interval: str) -> pd.DataFrame | None:
    """Fetch OHLCV data for a single stock."""
    try:
        # Use per-symbol history instead of bulk-style download to avoid
        # occasional column-mixing issues in threaded scans.
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval, auto_adjust=False)
        if df.empty or len(df) < 15:
            return None
        # yfinance may still return duplicated labels in edge-cases.
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated()]

        df["Symbol"] = symbol
        return df
    except Exception:
        return None


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    """Return a 1D numeric series for a column name, resilient to odd shapes."""
    obj = df.get(name)
    if obj is None:
        raise KeyError(name)
    if isinstance(obj, pd.DataFrame):
        obj = obj.iloc[:, 0]
    return pd.to_numeric(obj, errors="coerce")


def close_near_high(df: pd.DataFrame, pct: float) -> bool:
    """Check if the latest close is within `pct`% of the candle's high."""
    dist_pct = close_to_high_pct(df)
    if dist_pct is None:
        return False
    return dist_pct <= pct


def close_near_low(df: pd.DataFrame, pct: float) -> bool:
    """Check if the latest close is within `pct`% of the candle's low."""
    dist_pct = close_to_low_pct(df)
    if dist_pct is None:
        return False
    return dist_pct <= pct


def close_to_high_pct(df: pd.DataFrame) -> float | None:
    """Return percentage distance of latest close from latest high."""
    high_s = _col(df, "High")
    close_s = _col(df, "Close")
    high = high_s.iloc[-1]
    close = close_s.iloc[-1]
    if pd.isna(high) or pd.isna(close):
        return None
    if high == 0:
        return None
    return ((high - close) / high) * 100


def close_to_low_pct(df: pd.DataFrame) -> float | None:
    """Return percentage distance of latest close from latest low."""
    low_s = _col(df, "Low")
    close_s = _col(df, "Close")
    low = low_s.iloc[-1]
    close = close_s.iloc[-1]
    if pd.isna(low) or pd.isna(close):
        return None
    if low == 0:
        return None
    return ((close - low) / low) * 100


def range_consolidation_break_signal(
    df: pd.DataFrame,
    lookback: int,
    max_range_pct: float,
    buffer_pct: float = 0.0,
    enforce_structure: bool = True,
    side: str = "BUY",
) -> tuple[bool, float | None]:
    """
    Consolidation window = previous `lookback` candles (excluding current).
    Conditions:
    1) Window range% <= max_range_pct
    2) No sustained new high/low inside window:
       high[i] > high[i-1] and high[i] > high[i-2]  -> reject
       low[i]  < low[i-1]  and low[i]  < low[i-2]   -> reject
    Breakout/Breakdown:
       BUY  -> current_high >= window_high * (1 + buffer_pct/100)
       SELL -> current_low  <= window_low  * (1 - buffer_pct/100)
    """
    if len(df) < lookback + 1:
        return False, None

    high_s = _col(df, "High")
    low_s = _col(df, "Low")
    close_s = _col(df, "Close")

    window_high = high_s.iloc[-(lookback + 1):-1]
    window_low = low_s.iloc[-(lookback + 1):-1]
    window_close = close_s.iloc[-(lookback + 1):-1]
    current_high = high_s.iloc[-1]
    current_low = low_s.iloc[-1]

    if (
        window_high.isna().any()
        or window_low.isna().any()
        or window_close.isna().any()
        or pd.isna(current_high)
        or pd.isna(current_low)
    ):
        return False, None

    highest = window_high.max()
    lowest = window_low.min()
    avg_close = window_close.mean()
    if pd.isna(highest) or pd.isna(lowest) or pd.isna(avg_close) or avg_close == 0:
        return False, None

    range_pct = ((highest - lowest) / avg_close) * 100
    if range_pct > max_range_pct:
        return False, float(range_pct)

    # Strict structure condition from spec.
    if enforce_structure:
        for i in range(2, len(window_high)):
            if window_high.iloc[i] > window_high.iloc[i - 1] and window_high.iloc[i] > window_high.iloc[i - 2]:
                return False, float(range_pct)
            if window_low.iloc[i] < window_low.iloc[i - 1] and window_low.iloc[i] < window_low.iloc[i - 2]:
                return False, float(range_pct)

    side = (side or "BUY").upper()
    if side == "SELL":
        breakdown_level = lowest * (1 - buffer_pct / 100.0)
        passed = bool(current_low <= breakdown_level)
    else:
        breakout_level = highest * (1 + buffer_pct / 100.0)
        passed = bool(current_high >= breakout_level)
    return passed, float(range_pct)


def volume_spike(df: pd.DataFrame, lookback: int) -> bool:
    """Current candle volume > average of previous `lookback` candles."""
    ratio = volume_spike_ratio(df, lookback)
    if ratio is None:
        return False
    return ratio > 1.0


def volume_spike_ratio(df: pd.DataFrame, lookback: int) -> float | None:
    """Return current volume / average previous lookback volume."""
    if len(df) < lookback + 1:
        return None
    vol_s = _col(df, "Volume")
    avg_vol = vol_s.iloc[-(lookback + 1):-1].mean()
    curr_vol = vol_s.iloc[-1]
    if pd.isna(avg_vol) or pd.isna(curr_vol):
        return None
    if avg_vol == 0:
        return None
    return float(curr_vol / avg_vol)


def rsi_value(df: pd.DataFrame, length: int = 14) -> float | None:
    """Return latest RSI value using Wilder smoothing."""
    close_s = _col(df, "Close")
    if len(close_s) < length + 1:
        return None
    delta = close_s.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    last = rsi.iloc[-1]
    if pd.isna(last):
        return None
    return float(last)


def is_consolidating(df: pd.DataFrame, lookback: int, max_range_pct: float = 5.0) -> bool:
    """
    Consolidation: the high-low range of the previous `lookback` candles
    is within `max_range_pct`% of their average close.
    """
    if len(df) < lookback + 1:
        return False
    high_s = _col(df, "High")
    low_s = _col(df, "Low")
    close_s = _col(df, "Close")
    highest = high_s.iloc[-(lookback + 1):-1].max()
    lowest = low_s.iloc[-(lookback + 1):-1].min()
    avg_close = close_s.iloc[-(lookback + 1):-1].mean()
    if pd.isna(highest) or pd.isna(lowest) or pd.isna(avg_close):
        return False
    if avg_close == 0:
        return False
    range_pct = ((highest - lowest) / avg_close) * 100
    return range_pct <= max_range_pct


def minervini_trend_template(df: pd.DataFrame) -> tuple[bool, dict[str, float | int | str]]:
    """
    Mark Minervini technical Trend Template (long-side quality setup):
    1) Price > 150DMA and 200DMA
    2) 150DMA > 200DMA
    3) 200DMA trending up vs ~1 month ago (20 sessions)
    4) 50DMA > 150DMA and 200DMA
    5) Price > 50DMA
    6) Price >= 30% above 52-week low
    7) Price within 25% of 52-week high
    """
    close_s = _col(df, "Close").dropna()
    if len(close_s) < 220:
        return False, {"MM Score": "0/7"}

    ma50 = close_s.rolling(50).mean()
    ma150 = close_s.rolling(150).mean()
    ma200 = close_s.rolling(200).mean()

    price = float(close_s.iloc[-1])
    ma50_last = float(ma50.iloc[-1])
    ma150_last = float(ma150.iloc[-1])
    ma200_last = float(ma200.iloc[-1])
    ma200_prev = ma200.iloc[-21] if len(ma200) >= 221 else np.nan

    lookback_52w = min(252, len(close_s))
    low_52w = float(close_s.iloc[-lookback_52w:].min())
    high_52w = float(close_s.iloc[-lookback_52w:].max())

    c1 = price > ma150_last and price > ma200_last
    c2 = ma150_last > ma200_last
    c3 = bool(not pd.isna(ma200_prev) and ma200_last > float(ma200_prev))
    c4 = ma50_last > ma150_last and ma50_last > ma200_last
    c5 = price > ma50_last
    c6 = low_52w > 0 and price >= (1.30 * low_52w)
    c7 = high_52w > 0 and price >= (0.75 * high_52w)

    checks = [c1, c2, c3, c4, c5, c6, c7]
    score = sum(bool(x) for x in checks)
    passed = score == 7

    stats: dict[str, float | int | str] = {
        "MM Score": f"{score}/7",
        "50DMA": round(ma50_last, 2),
        "150DMA": round(ma150_last, 2),
        "200DMA": round(ma200_last, 2),
        "52W Low": round(low_52w, 2),
        "52W High": round(high_52w, 2),
    }
    return passed, stats


def classify_market_stage(df: pd.DataFrame) -> tuple[str, float]:
    """
    Approximate Stage Analysis classification (Stage 1-4) using MA structure:
    - Stage 1: base/accumulation near flat 150DMA
    - Stage 2: advancing uptrend above rising 150DMA
    - Stage 3: topping/distribution with flattening trend
    - Stage 4: declining downtrend below falling 150DMA
    Returns (stage_label, fit_score_0_to_100).
    """
    close_s = _col(df, "Close").dropna()
    if len(close_s) < 220:
        return "Stage 1", 0.0

    ma50 = close_s.rolling(50).mean()
    ma150 = close_s.rolling(150).mean()
    ma200 = close_s.rolling(200).mean()

    price = float(close_s.iloc[-1])
    ma50_last = float(ma50.iloc[-1])
    ma150_last = float(ma150.iloc[-1])
    ma200_last = float(ma200.iloc[-1])
    ma150_prev = ma150.iloc[-21] if len(ma150) >= 221 else np.nan
    ma200_prev = ma200.iloc[-21] if len(ma200) >= 221 else np.nan

    ma150_slope_pct = 0.0
    if not pd.isna(ma150_prev) and float(ma150_prev) != 0:
        ma150_slope_pct = ((ma150_last - float(ma150_prev)) / float(ma150_prev)) * 100.0

    dist_150_pct = ((price - ma150_last) / ma150_last) * 100.0 if ma150_last != 0 else 0.0
    dist_50_pct = ((price - ma50_last) / ma50_last) * 100.0 if ma50_last != 0 else 0.0
    ma50_vs_150 = ((ma50_last - ma150_last) / ma150_last) * 100.0 if ma150_last != 0 else 0.0

    s2 = 0
    if price > ma150_last:
        s2 += 25
    if ma150_slope_pct > 0.5:
        s2 += 25
    if ma50_last > ma150_last and ma150_last > ma200_last:
        s2 += 25
    if price > ma50_last and (pd.isna(ma200_prev) or ma200_last >= float(ma200_prev)):
        s2 += 25

    s4 = 0
    if price < ma150_last:
        s4 += 25
    if ma150_slope_pct < -0.5:
        s4 += 25
    if ma50_last < ma150_last and ma150_last < ma200_last:
        s4 += 25
    if price < ma50_last and (pd.isna(ma200_prev) or ma200_last <= float(ma200_prev)):
        s4 += 25

    s1 = 0
    if abs(dist_150_pct) <= 8:
        s1 += 35
    if abs(ma150_slope_pct) <= 1.0:
        s1 += 35
    if abs(ma50_vs_150) <= 4 and abs(dist_50_pct) <= 8:
        s1 += 30

    s3 = 0
    if price >= ma150_last and price <= (ma150_last * 1.20):
        s3 += 30
    if abs(ma150_slope_pct) <= 1.5:
        s3 += 30
    if price < ma50_last and ma50_last >= ma150_last:
        s3 += 20
    if ma50_vs_150 <= 6:
        s3 += 20

    stage_scores = {
        "Stage 1": float(s1),
        "Stage 2": float(s2),
        "Stage 3": float(s3),
        "Stage 4": float(s4),
    }
    best_stage = max(stage_scores, key=stage_scores.get)
    return best_stage, stage_scores[best_stage]


def scan_stock(symbol, period, interval, filters):
    """Run all enabled filters on a single stock. Returns dict or None."""
    df = fetch_stock_data(symbol, period, interval)
    if df is None:
        return None

    close_last = _col(df, "Close").iloc[-1]
    high_last = _col(df, "High").iloc[-1]
    low_last = _col(df, "Low").iloc[-1]
    vol_last = _col(df, "Volume").iloc[-1]
    if pd.isna(close_last) or pd.isna(high_last) or pd.isna(low_last) or pd.isna(vol_last):
        return None

    side = filters.get("trade_side", "BUY").upper()
    base_symbol = symbol.replace(".NS", "")
    sector = SYMBOL_SECTORS.get(base_symbol)
    result = {
        "Symbol": base_symbol,
        "Sector": sector if sector else "",
        "Close": round(float(close_last), 2),
        "High": round(float(high_last), 2),
        "Low": round(float(low_last), 2),
        "Volume": int(float(vol_last)),
    }
    if side == "SELL":
        result["Dist from Low %"] = round(((float(close_last) - float(low_last)) / float(low_last)) * 100, 2) if float(low_last) != 0 else None
    else:
        result["Dist from High %"] = round(((float(high_last) - float(close_last)) / float(high_last)) * 100, 2) if float(high_last) != 0 else None
    rsi_last = rsi_value(df, filters.get("rsi_length", 14))
    result["RSI"] = round(rsi_last, 2) if rsi_last is not None else None
    flags = {}

    # Always compute range % so it appears in results dataframe.
    _, range_pct_all = range_consolidation_break_signal(
        df,
        lookback=filters["breakout_lookback"],
        max_range_pct=filters["breakout_max_range_pct"],
        buffer_pct=filters["breakout_buffer_pct"],
        enforce_structure=filters["breakout_enforce_structure"],
        side=side,
    )
    result["Range %"] = round(range_pct_all, 2) if range_pct_all is not None else None

    if filters.get("close_near_high_enabled"):
        dist_pct = close_to_low_pct(df) if side == "SELL" else close_to_high_pct(df)
        if dist_pct is None:
            return None
        passed = dist_pct <= filters["close_near_high_pct"]
        if not passed:
            return None

    if filters.get("breakout_enabled"):
        passed, range_pct = range_consolidation_break_signal(
            df,
            lookback=filters["breakout_lookback"],
            max_range_pct=filters["breakout_max_range_pct"],
            buffer_pct=filters["breakout_buffer_pct"],
            enforce_structure=filters["breakout_enforce_structure"],
            side=side,
        )
        if not passed:
            return None

    if filters.get("volume_spike_enabled"):
        ratio = volume_spike_ratio(df, filters["volume_lookback"])
        if ratio is None:
            return None
        passed = ratio > 1.0
        flags["Volume Spike"] = f"{ratio:.2f}x"
        if not passed:
            return None

    if filters.get("consolidation_enabled"):
        passed = is_consolidating(df, filters["consol_lookback"], filters.get("consol_range_pct", 5.0))
        flags["Consolidating"] = passed
        if not passed:
            return None

    if filters.get("rsi_enabled"):
        if rsi_last is None:
            return None
        threshold = filters.get("rsi_threshold", 70.0 if side == "SELL" else 30.0)
        passed = (rsi_last >= threshold) if side == "SELL" else (rsi_last <= threshold)
        if not passed:
            return None

    if filters.get("mm_enabled"):
        # Trend template is defined on daily data; fetch dedicated daily history.
        mm_df = fetch_stock_data(symbol, period="2y", interval="1d")
        if mm_df is None:
            return None
        passed, mm_stats = minervini_trend_template(mm_df)
        stage, stage_fit = classify_market_stage(mm_df)
        selected_stages = filters.get("mm_stages", ["Stage 1", "Stage 2", "Stage 3", "Stage 4"])
        if stage not in selected_stages:
            return None
        flags.update(mm_stats)
        flags["Stage"] = stage
        flags["Stage Fit %"] = round(stage_fit, 1)
        flags["MM Quality"] = "Pass" if passed else "Watch"

    result.update(flags)
    return result


def scan_sector_proximity(
    symbol: str,
    period: str,
    interval: str,
    near_high_pct: float,
    near_low_pct: float,
) -> dict | None:
    """
    Return sector proximity snapshot for latest selected timeframe candle:
    - close within X% of High
    - close within Y% of Low
    """
    df = fetch_stock_data(symbol, period, interval)
    if df is None:
        return None

    close_last = _col(df, "Close").iloc[-1]
    high_last = _col(df, "High").iloc[-1]
    low_last = _col(df, "Low").iloc[-1]
    if pd.isna(close_last) or pd.isna(high_last) or pd.isna(low_last):
        return None
    if float(high_last) == 0 or float(low_last) == 0:
        return None

    dist_high = ((float(high_last) - float(close_last)) / float(high_last)) * 100.0
    dist_low = ((float(close_last) - float(low_last)) / float(low_last)) * 100.0

    base_symbol = symbol.replace(".NS", "")
    sector = SYMBOL_SECTORS.get(base_symbol) or "Unknown"
    return {
        "Symbol": base_symbol,
        "Sector": sector,
        "Dist from High %": round(dist_high, 2),
        "Dist from Low %": round(dist_low, 2),
        "Near High": dist_high <= near_high_pct,
        "Near Low": dist_low <= near_low_pct,
    }


# ── Streamlit UI ────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="Stock Scanner", page_icon="📈", layout="wide")
    st.title("Stock Scanner")
    st.caption("Scan NSE stocks using customisable technical filters powered by yfinance")

    # ── Sidebar ─────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Scan Settings")
        using_csv = bool(LOADED_SYMBOLS)
        source = f"CSV: {DEFAULT_NIFTY500_CSV.name}" if using_csv else "Built-in list"
        st.caption(f"Universe: {source} ({len(NSE_SYMBOLS)} symbols)")
        if not using_csv:
            st.caption(
                f"Add `{DEFAULT_NIFTY500_CSV.name}` to the repo to load the full Nifty 500 universe on Streamlit Cloud."
            )

        timeframe = st.selectbox("Timeframe", ["Day", "Week", "Month"], index=0)
        trade_side = st.selectbox("Signal Side", ["Buy", "Sell"], index=0)
        tf_map = {"Day": ("6mo", "1d"), "Week": ("2y", "1wk"), "Month": ("5y", "1mo")}
        period, interval = tf_map[timeframe]

        st.markdown("---")
        st.subheader("Select & Configure Filters")
        st.caption("Only enabled filters will be applied. Stocks must pass ALL enabled filters.")
        side_upper = trade_side.upper()

        # Filter 1 – Close near High
        f1_title = "Close near Low" if side_upper == "SELL" else "Close near High"
        f1_enabled = st.checkbox(f1_title, value=False)
        f1_pct = 1.0
        if f1_enabled:
            dist_label = "Max % distance from Low" if side_upper == "SELL" else "Max % distance from High"
            f1_pct = st.slider(dist_label, 0.1, 5.0, 1.0, 0.1, key="f1")

        # Filter 2 – Range consolidation break signal
        f2_title = "Range Consolidation Breakdown" if side_upper == "SELL" else "Range Consolidation Breakout"
        f2_enabled = st.checkbox(f2_title, value=False)
        f2_lookback = 15
        f2_max_range = 5.0
        f2_buffer = 0.0
        if f2_enabled:
            f2_lookback = st.slider("Lookback (candles)", 5, 30, 15, 1, key="f2_lb")
            f2_max_range = st.slider("Max range %", 1.0, 10.0, 5.0, 0.5, key="f2_rng")
            buffer_label = "Breakdown buffer %" if side_upper == "SELL" else "Breakout buffer %"
            f2_buffer = st.slider(buffer_label, 0.0, 2.0, 0.0, 0.1, key="f2_buf")
            f2_structure = st.checkbox("No break of recent highs/lows", value=True, key="f2_struct")
        else:
            f2_structure = True

        # Filter 3 – Volume spike
        f3_enabled = st.checkbox("Volume Spike", value=False)
        f3_lookback = 5
        if f3_enabled:
            f3_lookback = st.slider("Avg Volume lookback (candles)", 2, 20, 5, 1, key="f3")

        # Filter 4 – Consolidation
        f4_enabled = st.checkbox("Prior Consolidation", value=False)
        f4_lookback = 10
        f4_range = 5.0
        if f4_enabled:
            f4_lookback = st.slider("Consolidation lookback (candles)", 5, 30, 10, 1, key="f4_lb")
            f4_range = st.slider("Max range % for consolidation", 1.0, 15.0, 5.0, 0.5, key="f4_rng")

        # Filter 5 – RSI (directional)
        f5_enabled = st.checkbox("RSI Filter", value=False)
        f5_len = 14
        f5_threshold = 30.0 if side_upper != "SELL" else 70.0
        if f5_enabled:
            f5_len = st.slider("RSI Length", 5, 30, 14, 1, key="f5_len")
            if side_upper == "SELL":
                f5_threshold = st.slider("RSI >= (Overbought)", 55.0, 90.0, 70.0, 1.0, key="f5_ob")
            else:
                f5_threshold = st.slider("RSI <= (Oversold)", 10.0, 45.0, 30.0, 1.0, key="f5_os")

        # Filter 6 – MM Stocks (Minervini technical quality setup)
        f6_enabled = st.checkbox("MM Stocks", value=False)
        f6_stages = ["Stage 1", "Stage 2", "Stage 3", "Stage 4"]
        if f6_enabled:
            st.caption("Applies Minervini 7-point Trend Template (technical setup).")
            f6_stages = st.multiselect(
                "Stage Filter",
                ["Stage 1", "Stage 2", "Stage 3", "Stage 4"],
                default=["Stage 1", "Stage 2", "Stage 3", "Stage 4"],
                key="f6_stages",
            )

        st.markdown("---")
        st.subheader("Sector Summary")
        ss_enabled = st.checkbox("Enable Sector Summary", value=False)
        ss_near_high_pct = 1.0
        ss_near_low_pct = 1.0
        if ss_enabled:
            ss_near_high_pct = st.slider(
                "Near High threshold %",
                0.1,
                5.0,
                1.0,
                0.1,
                key="ss_high_pct",
            )
            ss_near_low_pct = st.slider(
                "Near Low threshold %",
                0.1,
                5.0,
                1.0,
                0.1,
                key="ss_low_pct",
            )

        st.markdown("---")
        max_workers = st.slider("Parallel threads", 4, 32, 16, 4)

    # ── Build filter dict ───────────────────────────────────────────────────
    filters = {
        "close_near_high_enabled": f1_enabled,
        "close_near_high_pct": f1_pct,
        "trade_side": side_upper,
        "breakout_enabled": f2_enabled,
        "breakout_lookback": f2_lookback,
        "breakout_max_range_pct": f2_max_range,
        "breakout_buffer_pct": f2_buffer,
        "breakout_enforce_structure": f2_structure,
        "volume_spike_enabled": f3_enabled,
        "volume_lookback": f3_lookback,
        "consolidation_enabled": f4_enabled,
        "consol_lookback": f4_lookback,
        "consol_range_pct": f4_range,
        "rsi_enabled": f5_enabled,
        "rsi_length": f5_len,
        "rsi_threshold": f5_threshold,
        "mm_enabled": f6_enabled,
        "mm_stages": f6_stages,
    }

    # ── Run Scan ────────────────────────────────────────────────────────────
    if st.button("Run Scan", type="primary", use_container_width=True):
        if f6_enabled and side_upper == "SELL":
            st.warning("'MM Stocks' is a long-side quality setup filter. Switch Signal Side to Buy.")
            return
        if f6_enabled and not f6_stages:
            st.warning("Select at least one stage in MM Stocks filter.")
            return

        enabled = []
        if f1_enabled:
            enabled.append("Close near Low" if side_upper == "SELL" else "Close near High")
        if f2_enabled:
            enabled.append("Breakdown" if side_upper == "SELL" else "Breakout")
        if f3_enabled:
            enabled.append("Volume Spike")
        if f4_enabled:
            enabled.append("Prior Consolidation")
        if f5_enabled:
            enabled.append("RSI Overbought" if side_upper == "SELL" else "RSI Oversold")
        if f6_enabled:
            enabled.append("MM Stocks")
        if not enabled and not ss_enabled:
            st.warning("Enable at least one filter to scan.")
            return

        if enabled:
            st.info(f"Scanning **{len(NSE_SYMBOLS)}** stocks on **{timeframe}** timeframe with filters: {', '.join(enabled)}")

            progress = st.progress(0, text="Scanning...")
            results = []
            scanned = 0

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(scan_stock, sym, period, interval, filters): sym
                           for sym in NSE_SYMBOLS}
                for future in concurrent.futures.as_completed(futures):
                    scanned += 1
                    progress.progress(scanned / len(NSE_SYMBOLS),
                                      text=f"Scanned {scanned}/{len(NSE_SYMBOLS)}...")
                    res = future.result()
                    if res is not None:
                        results.append(res)

            progress.empty()

            if results:
                df_results = pd.DataFrame(results)
                if f6_enabled:
                    stage_order = {"Stage 1": 1, "Stage 2": 2, "Stage 3": 3, "Stage 4": 4}

                    def _mm_score_num(v):
                        s = str(v)
                        if "/" in s:
                            left = s.split("/", 1)[0].strip()
                            try:
                                return float(left)
                            except Exception:
                                return 0.0
                        try:
                            return float(s)
                        except Exception:
                            return 0.0

                    df_results["_stage_order"] = df_results.get("Stage", "").map(stage_order).fillna(99)
                    df_results["_mm_score_num"] = df_results.get("MM Score", "").map(_mm_score_num)
                    df_results["_stage_fit"] = pd.to_numeric(df_results.get("Stage Fit %", 0.0), errors="coerce").fillna(0.0)

                    df_results = df_results.sort_values(
                        ["_stage_order", "_stage_fit", "_mm_score_num", "Volume"],
                        ascending=[True, False, False, False],
                    ).reset_index(drop=True)
                    df_results["Stage Rank"] = (
                        df_results.groupby("Stage").cumcount() + 1
                    )
                    df_results = df_results.drop(columns=["_stage_order", "_mm_score_num", "_stage_fit"], errors="ignore")
                else:
                    # Sort by volume descending
                    df_results = df_results.sort_values("Volume", ascending=False).reset_index(drop=True)
                df_results.index += 1
                st.success(f"Found **{len(df_results)}** stocks matching all enabled filters.")
                st.dataframe(df_results, use_container_width=True, height=600)

                # Download button
                csv = df_results.to_csv(index=False)
                st.download_button("Download CSV", csv, "scan_results.csv", "text/csv")
            else:
                st.warning("No stocks matched all enabled filters. Try relaxing the parameters.")
        else:
            st.info("No technical filters enabled. Running Sector Summary only.")

        if ss_enabled:
            sec_progress = st.progress(0, text="Building sector summary...")
            proximity_rows: list[dict] = []
            scanned_sec = 0
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        scan_sector_proximity,
                        sym,
                        period,
                        interval,
                        ss_near_high_pct,
                        ss_near_low_pct,
                    ): sym
                    for sym in NSE_SYMBOLS
                }
                for future in concurrent.futures.as_completed(futures):
                    scanned_sec += 1
                    sec_progress.progress(
                        scanned_sec / len(NSE_SYMBOLS),
                        text=f"Sector snapshot {scanned_sec}/{len(NSE_SYMBOLS)}...",
                    )
                    rec = future.result()
                    if rec is not None:
                        proximity_rows.append(rec)
            sec_progress.empty()

            if not proximity_rows:
                st.warning("No sector summary data available.")
                st.session_state["sector_summary_payload"] = None
            else:
                summary_map: dict[str, dict] = {}
                for rec in proximity_rows:
                    sec = rec["Sector"]
                    bucket = summary_map.setdefault(
                        sec,
                        {
                            "Near High Count": 0,
                            "Near Low Count": 0,
                            "Near High Stocks": [],
                            "Near Low Stocks": [],
                        },
                    )
                    if rec["Near High"]:
                        bucket["Near High Count"] += 1
                        bucket["Near High Stocks"].append(
                            {
                                "Symbol": rec["Symbol"],
                                "Dist from High %": rec["Dist from High %"],
                            }
                        )
                    if rec["Near Low"]:
                        bucket["Near Low Count"] += 1
                        bucket["Near Low Stocks"].append(
                            {
                                "Symbol": rec["Symbol"],
                                "Dist from Low %": rec["Dist from Low %"],
                            }
                        )

                all_near_high_stocks = []
                all_near_low_stocks = []
                total_near_high = 0
                total_near_low = 0
                for sec_name, vals in summary_map.items():
                    total_near_high += int(vals["Near High Count"])
                    total_near_low += int(vals["Near Low Count"])
                    for item in vals["Near High Stocks"]:
                        all_near_high_stocks.append(
                            {
                                "Symbol": item["Symbol"],
                                "Sector": sec_name,
                                "Dist from High %": item["Dist from High %"],
                            }
                        )
                    for item in vals["Near Low Stocks"]:
                        all_near_low_stocks.append(
                            {
                                "Symbol": item["Symbol"],
                                "Sector": sec_name,
                                "Dist from Low %": item["Dist from Low %"],
                            }
                        )

                st.session_state["sector_summary_payload"] = {
                    "summary_map": summary_map,
                    "all_near_high_stocks": all_near_high_stocks,
                    "all_near_low_stocks": all_near_low_stocks,
                    "total_near_high": total_near_high,
                    "total_near_low": total_near_low,
                    "timeframe": timeframe,
                    "near_high_pct": ss_near_high_pct,
                    "near_low_pct": ss_near_low_pct,
                }

    if ss_enabled:
        payload = st.session_state.get("sector_summary_payload")
        if payload:
            summary_map = payload.get("summary_map", {})
            all_near_high_stocks = payload.get("all_near_high_stocks", [])
            all_near_low_stocks = payload.get("all_near_low_stocks", [])
            total_near_high = int(payload.get("total_near_high", 0))
            total_near_low = int(payload.get("total_near_low", 0))
            p_timeframe = payload.get("timeframe", timeframe)
            p_near_high = float(payload.get("near_high_pct", ss_near_high_pct))
            p_near_low = float(payload.get("near_low_pct", ss_near_low_pct))

            st.markdown("---")
            st.subheader("Sector Summary (Near High / Near Low)")
            st.caption(
                f"Based on selected timeframe: **{p_timeframe}**. "
                f"Near High <= {p_near_high:.1f}% | Near Low <= {p_near_low:.1f}%"
            )

            t1, t2, t3 = st.columns([3, 1.5, 1.5])
            t1.markdown("**ALL SECTORS (Total)**")
            if t2.button(str(total_near_high), key="near_high_all"):
                st.session_state["sector_pick"] = ("ALL", "HIGH")
            if t3.button(str(total_near_low), key="near_low_all"):
                st.session_state["sector_pick"] = ("ALL", "LOW")

            hdr = st.columns([3, 1.5, 1.5])
            hdr[0].markdown("**Sector**")
            hdr[1].markdown("**Near High Count**")
            hdr[2].markdown("**Near Low Count**")

            ordered = sorted(
                summary_map.items(),
                key=lambda x: (x[1]["Near High Count"] + x[1]["Near Low Count"], x[0]),
                reverse=True,
            )
            for sec, vals in ordered:
                c1, c2, c3 = st.columns([3, 1.5, 1.5])
                c1.write(sec)
                if c2.button(str(vals["Near High Count"]), key=f"near_high_{sec}"):
                    st.session_state["sector_pick"] = (sec, "HIGH")
                if c3.button(str(vals["Near Low Count"]), key=f"near_low_{sec}"):
                    st.session_state["sector_pick"] = (sec, "LOW")

            pick = st.session_state.get("sector_pick")
            if pick:
                sec, side = pick
                if sec == "ALL":
                    if side == "HIGH":
                        st.markdown(f"**All Sectors - Stocks Near High ({total_near_high})**")
                        detail = pd.DataFrame(all_near_high_stocks).sort_values(
                            "Dist from High %", ascending=True
                        )
                    else:
                        st.markdown(f"**All Sectors - Stocks Near Low ({total_near_low})**")
                        detail = pd.DataFrame(all_near_low_stocks).sort_values(
                            "Dist from Low %", ascending=True
                        )
                    if detail.empty:
                        st.info("No stocks for this selection.")
                    else:
                        detail.index += 1
                        st.dataframe(detail, use_container_width=True, height=320)
                else:
                    vals = summary_map.get(sec)
                    if vals is None:
                        st.info("Selected sector not available in current summary.")
                        vals = {"Near High Count": 0, "Near Low Count": 0, "Near High Stocks": [], "Near Low Stocks": []}
                    if side == "HIGH":
                        st.markdown(f"**{sec} - Stocks Near High ({vals['Near High Count']})**")
                        detail = pd.DataFrame(vals["Near High Stocks"]).sort_values(
                            "Dist from High %", ascending=True
                        )
                    else:
                        st.markdown(f"**{sec} - Stocks Near Low ({vals['Near Low Count']})**")
                        detail = pd.DataFrame(vals["Near Low Stocks"]).sort_values(
                            "Dist from Low %", ascending=True
                        )
                    if detail.empty:
                        st.info("No stocks for this selection.")
                    else:
                        detail.index += 1
                        st.dataframe(detail, use_container_width=True, height=320)
        else:
            st.info("Run Scan once to build Sector Summary.")


if __name__ == "__main__":
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        has_ctx = get_script_run_ctx() is not None
    except Exception:
        has_ctx = False

    if has_ctx:
        main()
    else:
        print("Launching with Streamlit...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", __file__], check=False)
