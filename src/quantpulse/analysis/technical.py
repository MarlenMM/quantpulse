"""Technical analysis: indicators, candlestick patterns, support/resistance,
relative strength, sector rotation, and anomaly detection (Section 7.1).

Geometric chart-pattern detection (head-and-shoulders, triangles, etc.) is a
separate, harder problem living in `analysis/patterns.py` (Phase 2's Opus half)
-- candlestick patterns here are just normalized output from a library with
60+ native detectors, no custom geometry.

Every function is pure (DataFrame/Series in, DataFrame/Series out) and has no
storage or network dependency, so the nightly pipeline that eventually calls
these can be tested independently of them.
"""

import numpy as np
import pandas as pd
import pandas_ta_classic as ta

_REQUIRED_OHLCV = ("open", "high", "low", "close", "volume")
# pandas-ta-classic's cdl_pattern crashes (AttributeError, not a clean skip)
# below this length rather than degrading gracefully -- a real case, not a
# hypothetical one: a freshly-listed or newly-added ticker legitimately has
# under 10 days of history when this first runs on it.
_MIN_ROWS_FOR_CANDLESTICK_PATTERNS = 10


def _require_columns(prices: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [c for c in columns if c not in prices.columns]
    if missing:
        raise ValueError(f"prices is missing required column(s): {missing}")


def compute_indicators(prices: pd.DataFrame) -> pd.DataFrame:
    """Trend/momentum/volatility/volume indicators (Section 7.1), appended as columns.

    `prices` must be indexed by date with lowercase open/high/low/close/volume
    columns (e.g. `price_history` rows set to a DatetimeIndex). SMA50/SMA200
    are included because the Market Regime Index (Section 5) needs "% of
    constituents above their 50/200-day moving average" as a breadth input.
    """
    _require_columns(prices, _REQUIRED_OHLCV)
    df = prices.copy()

    df["sma_20"] = ta.sma(df["close"], length=20)
    df["sma_50"] = ta.sma(df["close"], length=50)
    df["sma_200"] = ta.sma(df["close"], length=200)
    df["ema_12"] = ta.ema(df["close"], length=12)
    df["ema_26"] = ta.ema(df["close"], length=26)

    macd = ta.macd(df["close"])
    df["macd"] = macd["MACD_12_26_9"]
    df["macd_signal"] = macd["MACDs_12_26_9"]
    df["macd_hist"] = macd["MACDh_12_26_9"]

    adx = ta.adx(df["high"], df["low"], df["close"])
    df["adx_14"] = adx["ADX_14"]
    df["plus_di_14"] = adx["DMP_14"]
    df["minus_di_14"] = adx["DMN_14"]

    df["rsi_14"] = ta.rsi(df["close"])

    stoch = ta.stoch(df["high"], df["low"], df["close"])
    df["stoch_k"] = stoch["STOCHk_14_3_3"]
    df["stoch_d"] = stoch["STOCHd_14_3_3"]

    df["ao"] = ta.ao(df["high"], df["low"])

    bbands = ta.bbands(df["close"], length=20)
    df["bb_lower"] = bbands["BBL_20_2.0"]
    df["bb_mid"] = bbands["BBM_20_2.0"]
    df["bb_upper"] = bbands["BBU_20_2.0"]
    df["bb_bandwidth"] = bbands["BBB_20_2.0"]
    df["bb_percent"] = bbands["BBP_20_2.0"]

    df["atr_14"] = ta.atr(df["high"], df["low"], df["close"])
    df["obv"] = ta.obv(df["close"], df["volume"])
    df["vwap"] = ta.vwap(df["high"], df["low"], df["close"], df["volume"])
    df["cmf_20"] = ta.cmf(df["high"], df["low"], df["close"], df["volume"], length=20)

    return df


def detect_candlestick_patterns(prices: pd.DataFrame, symbol: str | None = None) -> pd.DataFrame:
    """Candlestick patterns (doji, hammer, engulfing, etc.), normalized to `pattern_signals`.

    pandas-ta-classic encodes each of its 60+ pattern columns as 0 (no
    signal) or a signed magnitude (bullish positive, bearish negative; most
    patterns use +/-100, a handful use other fixed magnitudes to encode a
    stronger or weaker version of the same shape). Confidence is that
    magnitude capped at 100 -- an honest, if coarse, per-pattern strength
    rather than a fabricated continuous score (Section 7.1's "confidence
    score... rather than a binary yes/no").
    """
    _require_columns(prices, ("open", "high", "low", "close"))
    columns = (["symbol"] if symbol else []) + ["date", "pattern_type", "direction", "confidence"]

    if len(prices) < _MIN_ROWS_FOR_CANDLESTICK_PATTERNS:
        return pd.DataFrame(columns=columns)

    raw = ta.cdl_pattern(prices["open"], prices["high"], prices["low"], prices["close"], name="all")

    stacked = raw.stack()
    stacked = stacked[stacked != 0]
    if stacked.empty:
        return pd.DataFrame(columns=columns)

    result = stacked.reset_index()
    result.columns = ["date", "pattern_type", "raw_value"]
    result["pattern_type"] = result["pattern_type"].str.removeprefix("CDL_").str.lower()
    result["direction"] = np.where(result["raw_value"] > 0, "bullish", "bearish")
    result["confidence"] = result["raw_value"].abs().clip(upper=100.0)
    result = result.drop(columns="raw_value")

    if symbol is not None:
        result.insert(0, "symbol", symbol)

    return result.sort_values("date").reset_index(drop=True)


def find_support_resistance_levels(
    prices: pd.DataFrame,
    order: int = 5,
    proximity_pct: float = 0.015,
    min_touches: int = 2,
) -> pd.DataFrame:
    """Price levels the market has repeatedly respected (Section 7.1).

    Local highs/lows are found with `scipy.signal.argrelextrema` (a peak
    every `order` bars on each side), then greedily merged into levels: any
    extremum within `proximity_pct` of a level's running mean joins it.
    Only levels touched at least `min_touches` times are returned -- a level
    tested once isn't support or resistance, it's a data point.
    """
    from scipy.signal import argrelextrema

    _require_columns(prices, ("high", "low"))
    high = prices["high"].to_numpy()
    low = prices["low"].to_numpy()

    peak_idx = argrelextrema(high, np.greater_equal, order=order)[0]
    trough_idx = argrelextrema(low, np.less_equal, order=order)[0]
    levels = np.sort(np.concatenate([high[peak_idx], low[trough_idx]]))

    if levels.size == 0:
        return pd.DataFrame(columns=["level", "touches"])

    clusters: list[list[float]] = [[float(levels[0])]]
    for lvl in levels[1:]:
        cluster_mean = float(np.mean(clusters[-1]))
        if cluster_mean != 0 and abs(lvl - cluster_mean) / cluster_mean <= proximity_pct:
            clusters[-1].append(float(lvl))
        else:
            clusters.append([float(lvl)])

    rows = [
        {"level": float(np.mean(c)), "touches": len(c)} for c in clusters if len(c) >= min_touches
    ]
    return (
        pd.DataFrame(rows, columns=["level", "touches"]).sort_values("level").reset_index(drop=True)
    )


def compute_relative_strength(symbol_close: pd.Series, benchmark_close: pd.Series) -> pd.Series:
    """Price-ratio "relative strength line" vs a benchmark, normalized to start at 100.

    Not the RSI oscillator -- a different classic concept with a confusingly
    similar name (Section 7.1). A rising line means the stock is
    outperforming the benchmark, independent of whether the stock itself is
    up or down.
    """
    aligned = pd.concat(
        [symbol_close.rename("symbol"), benchmark_close.rename("benchmark")], axis=1, join="inner"
    ).dropna()
    if aligned.empty:
        raise ValueError("symbol_close and benchmark_close have no overlapping dates")

    ratio = aligned["symbol"] / aligned["benchmark"]
    normalized = ratio / ratio.iloc[0] * 100
    normalized.name = "relative_strength"
    return normalized


def compute_sector_rotation(
    prices_by_symbol: dict[str, pd.Series],
    sector_by_symbol: dict[str, str],
    benchmark: pd.Series,
    lookback_days: int = 21,
) -> pd.DataFrame:
    """Which sectors have been gaining/losing relative strength recently (Section 7.1).

    Each sector's line is the equal-weighted average of its members'
    relative-strength lines; the output is each sector's % change in that
    line over `lookback_days` (~1 month), sorted so the top row is the
    sector money has been rotating into -- ready to render as a heatmap.
    """
    lines_by_sector: dict[str, list[pd.Series]] = {}
    for symbol, prices in prices_by_symbol.items():
        sector = sector_by_symbol.get(symbol)
        if sector is None:
            continue
        try:
            rs = compute_relative_strength(prices, benchmark)
        except ValueError:
            continue
        lines_by_sector.setdefault(sector, []).append(rs)

    rows = []
    for sector, lines in lines_by_sector.items():
        sector_line = pd.concat(lines, axis=1).mean(axis=1).dropna()
        if len(sector_line) < lookback_days + 1:
            continue
        change_pct = (sector_line.iloc[-1] / sector_line.iloc[-(lookback_days + 1)] - 1) * 100
        rows.append({"sector": sector, "relative_strength_change_pct": change_pct})

    return (
        pd.DataFrame(rows, columns=["sector", "relative_strength_change_pct"])
        .sort_values("relative_strength_change_pct", ascending=False)
        .reset_index(drop=True)
    )


def detect_anomalies(series: pd.Series, window: int = 20, z_threshold: float = 3.0) -> pd.DataFrame:
    """Flag statistically unusual values (volume spikes, outsized returns -- Section 7.1).

    The rolling baseline is computed from the `window` days *strictly
    before* each point (`shift(1)`), never including the point itself --
    otherwise a huge move would inflate its own baseline and could mask its
    own anomaly, and every score would be quietly look-ahead-biased.
    """
    rolling_mean = series.shift(1).rolling(window).mean()
    rolling_std = series.shift(1).rolling(window).std()
    z_score = (series - rolling_mean) / rolling_std

    return pd.DataFrame(
        {
            "value": series,
            "rolling_mean": rolling_mean,
            "rolling_std": rolling_std,
            "z_score": z_score,
            "is_anomaly": z_score.abs() >= z_threshold,
        }
    )
