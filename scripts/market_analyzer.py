from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Literal
from urllib.parse import quote_plus
from xml.etree import ElementTree as ET

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

FINANCE_ROOT = Path(os.environ.get("OPENCODE_FINANCE_ROOT", Path.home() / ".config" / "opencode" / "finance"))
KRONOS_ROOT = FINANCE_ROOT / "Kronos"
REPORTS_DIR = FINANCE_ROOT / "reports"
DATA_DIR = FINANCE_ROOT / "data"
CACHE_DIR = FINANCE_ROOT / "cache"

os.environ.setdefault("HF_HOME", str(CACHE_DIR / "hf"))
os.environ.setdefault("TORCH_HOME", str(CACHE_DIR / "torch"))

if str(KRONOS_ROOT) not in sys.path:
    sys.path.insert(0, str(KRONOS_ROOT))

import ccxt
import numpy as np
import pandas as pd
import requests
import yfinance as yf


Market = Literal["crypto", "bist"]


@dataclass
class AnalysisResult:
    market: Market
    symbol: str
    timeframe: str
    rows: int
    last_price: float
    score: int
    class_name: str
    bias: str
    setup: str
    evidence: list[str]
    trigger: str
    invalidation: str
    risk: str
    confidence: str
    kronos: dict | None
    plan: dict
    technical: dict
    fundamental: dict | None
    pre_news: dict
    news_impact: dict
    news_links: list[tuple[str, str]]


def fetch_crypto(symbol: str, timeframe: str, limit: int) -> tuple[pd.DataFrame, str]:
    errors: list[str] = []
    for exchange_id in ["binance", "okx", "bybit"]:
        try:
            exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True})
            rows = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df["amount"] = df["close"] * df["volume"]
            return df, exchange_id
        except Exception as exc:
            errors.append(f"{exchange_id}:{type(exc).__name__}")

    yahoo_symbol = symbol.split("/")[0].upper() + "-USD"
    interval = {"15m": "15m", "1h": "60m", "4h": "1h", "1d": "1d"}.get(timeframe, "60m")
    period = "60d" if interval != "1d" else "1y"
    raw = yf.download(yahoo_symbol, period=period, interval=interval, progress=False, auto_adjust=False)
    if raw.empty:
        raise RuntimeError("Crypto data failed: " + "; ".join(errors))
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0] for c in raw.columns]
    raw = raw.tail(limit).reset_index()
    time_col = "Datetime" if "Datetime" in raw.columns else "Date"
    df = raw.rename(columns={time_col: "timestamp", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
    df["amount"] = df["close"] * df["volume"].fillna(0)
    return df[["timestamp", "open", "high", "low", "close", "volume", "amount"]].dropna(), "yfinance"


def fetch_bist(symbol: str, period: str, interval: str) -> tuple[pd.DataFrame, str]:
    ticker = symbol.upper()
    if not ticker.endswith(".IS"):
        ticker += ".IS"
    raw = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
    if raw.empty:
        raise RuntimeError(f"BIST data not found for {ticker}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0] for c in raw.columns]
    raw = raw.reset_index()
    time_col = "Datetime" if "Datetime" in raw.columns else "Date"
    df = raw.rename(columns={time_col: "timestamp", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
    df["amount"] = df["close"] * df["volume"].fillna(0)
    return df[["timestamp", "open", "high", "low", "close", "volume", "amount"]].dropna(subset=["open", "high", "low", "close"]), "yfinance"


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["close"]
    high = out["high"]
    low = out["low"]
    volume = out["volume"].fillna(0)
    out["ret_pct"] = close.pct_change() * 100
    out["sma20"] = close.rolling(20).mean()
    out["sma50"] = close.rolling(50).mean()
    out["ema12"] = close.ewm(span=12, adjust=False).mean()
    out["ema26"] = close.ewm(span=26, adjust=False).mean()
    out["macd"] = out["ema12"] - out["ema26"]
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    out["rsi14"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    out["atr14"] = tr.rolling(14).mean()
    out["atr_pct"] = (out["atr14"] / close) * 100
    out["vol_z"] = (volume - volume.rolling(20).mean()) / volume.rolling(20).std().replace(0, np.nan)

    out["support20"] = low.rolling(20).min()
    out["resistance20"] = high.rolling(20).max()

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr_wilder = tr.ewm(alpha=1 / 14, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=out.index).ewm(alpha=1 / 14, adjust=False).mean() / atr_wilder.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=out.index).ewm(alpha=1 / 14, adjust=False).mean() / atr_wilder.replace(0, np.nan)
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    out["plus_di14"] = plus_di
    out["minus_di14"] = minus_di
    out["adx14"] = dx.ewm(alpha=1 / 14, adjust=False).mean()

    typical = (high + low + close) / 3
    vol_sum = volume.replace(0, np.nan).cumsum()
    out["vwap"] = (typical * volume).cumsum() / vol_sum
    out["rolling_vwap20"] = (typical * volume).rolling(20).sum() / volume.rolling(20).sum().replace(0, np.nan)

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    out["bb_mid20"] = bb_mid
    out["bb_upper20"] = bb_mid + 2 * bb_std
    out["bb_lower20"] = bb_mid - 2 * bb_std
    out["bb_width_pct"] = (out["bb_upper20"] - out["bb_lower20"]) / close * 100

    kc_mid = close.ewm(span=20, adjust=False).mean()
    out["kc_mid20"] = kc_mid
    out["kc_upper20"] = kc_mid + 1.5 * out["atr14"]
    out["kc_lower20"] = kc_mid - 1.5 * out["atr14"]
    out["squeeze_on"] = (out["bb_upper20"] < out["kc_upper20"]) & (out["bb_lower20"] > out["kc_lower20"])

    direction = np.sign(close.diff()).fillna(0)
    out["obv"] = (direction * volume).cumsum()
    money_flow_mult = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    out["accdist"] = (money_flow_mult.fillna(0) * volume).cumsum()

    out["pivot_p"] = (high.shift(1) + low.shift(1) + close.shift(1)) / 3
    out["pivot_r1"] = 2 * out["pivot_p"] - low.shift(1)
    out["pivot_s1"] = 2 * out["pivot_p"] - high.shift(1)
    out["pivot_r2"] = out["pivot_p"] + (high.shift(1) - low.shift(1))
    out["pivot_s2"] = out["pivot_p"] - (high.shift(1) - low.shift(1))

    left = 3
    swing_high_raw = high.eq(high.rolling(left * 2 + 1, center=True).max())
    swing_low_raw = low.eq(low.rolling(left * 2 + 1, center=True).min())
    confirmed_swing_high = swing_high_raw.shift(left, fill_value=False).astype(bool)
    confirmed_swing_low = swing_low_raw.shift(left, fill_value=False).astype(bool)
    out["swing_high"] = np.where(confirmed_swing_high, high.shift(left), np.nan)
    out["swing_low"] = np.where(confirmed_swing_low, low.shift(left), np.nan)
    out["last_swing_high"] = pd.Series(out["swing_high"], index=out.index).ffill()
    out["last_swing_low"] = pd.Series(out["swing_low"], index=out.index).ffill()

    if "timestamp" in out.columns:
        try:
            tmp = out[["timestamp", "high", "low", "close"]].copy()
            tmp["timestamp"] = pd.to_datetime(tmp["timestamp"])
            weekly = (
                tmp.set_index("timestamp")
                .resample("W-FRI")
                .agg({"high": "max", "low": "min", "close": "last"})
                .dropna()
            )
            weekly["weekly_pivot_p"] = (weekly["high"] + weekly["low"] + weekly["close"]) / 3
            weekly["weekly_pivot_r1"] = 2 * weekly["weekly_pivot_p"] - weekly["low"]
            weekly["weekly_pivot_s1"] = 2 * weekly["weekly_pivot_p"] - weekly["high"]
            weekly_pivots = weekly[["weekly_pivot_p", "weekly_pivot_r1", "weekly_pivot_s1"]].shift(1).reset_index()
            merged = pd.merge_asof(
                tmp[["timestamp"]].sort_values("timestamp"),
                weekly_pivots.sort_values("timestamp"),
                on="timestamp",
                direction="backward",
            )
            out[["weekly_pivot_p", "weekly_pivot_r1", "weekly_pivot_s1"]] = merged[
                ["weekly_pivot_p", "weekly_pivot_r1", "weekly_pivot_s1"]
            ].to_numpy()
        except Exception:
            out["weekly_pivot_p"] = np.nan
            out["weekly_pivot_r1"] = np.nan
            out["weekly_pivot_s1"] = np.nan
    return out


def classify(df: pd.DataFrame) -> tuple[int, str, str, str, list[str], str, str, str, str]:
    latest = df.dropna(subset=["close"]).iloc[-1]
    score = 50
    evidence: list[str] = []

    close = float(latest["close"])
    sma20 = latest.get("sma20")
    sma50 = latest.get("sma50")
    rsi = latest.get("rsi14")
    macd = latest.get("macd")
    macd_signal = latest.get("macd_signal")
    atr_pct = latest.get("atr_pct")
    vol_z = latest.get("vol_z")
    adx = latest.get("adx14")
    plus_di = latest.get("plus_di14")
    minus_di = latest.get("minus_di14")
    rvwap = latest.get("rolling_vwap20")
    squeeze_on = bool(latest.get("squeeze_on")) if pd.notna(latest.get("squeeze_on")) else False
    support = latest.get("support20")
    resistance = latest.get("resistance20")

    if pd.notna(sma20) and close > sma20:
        score += 10
        evidence.append("Fiyat SMA20 ustunde; kisa trend destekli.")
    else:
        score -= 8
        evidence.append("Fiyat SMA20 altinda veya kisa trend belirsiz.")

    if pd.notna(sma20) and pd.notna(sma50) and sma20 > sma50:
        score += 10
        evidence.append("SMA20, SMA50 uzerinde; orta trend pozitif.")
    elif pd.notna(sma20) and pd.notna(sma50):
        score -= 6
        evidence.append("SMA20, SMA50 altinda; orta trend zayif.")

    if pd.notna(macd) and pd.notna(macd_signal) and macd > macd_signal:
        score += 8
        evidence.append("MACD sinyal ustunde; momentum lehine.")
    elif pd.notna(macd) and pd.notna(macd_signal):
        score -= 5
        evidence.append("MACD sinyal altinda; momentum temkinli.")

    if pd.notna(rsi):
        if 45 <= rsi <= 68:
            score += 8
            evidence.append(f"RSI {rsi:.1f}; saglikli momentum bolgesi.")
        elif rsi > 75:
            score -= 8
            evidence.append(f"RSI {rsi:.1f}; asiri isinma riski.")
        elif rsi < 35:
            score -= 5
            evidence.append(f"RSI {rsi:.1f}; zayif/asiri satim bolgesi.")

    if pd.notna(vol_z) and vol_z > 1.0:
        score += 6
        evidence.append("Hacim son ortalamanin uzerinde; hareket teyidi gucleniyor.")
    elif pd.notna(vol_z) and vol_z < -0.8:
        score -= 4
        evidence.append("Hacim zayif; hareket teyidi eksik.")

    if pd.notna(atr_pct):
        if atr_pct > 10:
            score -= 8
            evidence.append(f"ATR% {atr_pct:.1f}; volatilite yuksek.")
        elif atr_pct < 5:
            score += 3
            evidence.append(f"ATR% {atr_pct:.1f}; oynaklik kontrol edilebilir.")

    if pd.notna(adx) and pd.notna(plus_di) and pd.notna(minus_di):
        if adx >= 22 and plus_di > minus_di:
            score += 7
            evidence.append(f"ADX {adx:.1f}; trend gucu alici lehine.")
        elif adx >= 22 and minus_di > plus_di:
            score -= 7
            evidence.append(f"ADX {adx:.1f}; trend gucu satici lehine.")
        elif adx < 16:
            score -= 3
            evidence.append(f"ADX {adx:.1f}; trend zayif, range riski var.")

    if pd.notna(rvwap):
        if close > rvwap:
            score += 4
            evidence.append("Fiyat 20 mum VWAP ustunde; ortalama maliyet lehine.")
        else:
            score -= 4
            evidence.append("Fiyat 20 mum VWAP altinda; long kovalamak riskli.")

    if squeeze_on:
        evidence.append("Bollinger/Keltner squeeze var; kirilim bekleyen sikisma.")

    score = int(max(0, min(100, score)))
    if score >= 75:
        class_name = "Guclu izle"
        bias = "bullish/pozitif"
        confidence = "medium-high"
    elif score >= 55:
        class_name = "Takip"
        bias = "neutral-pozitif" if score >= 63 else "neutral"
        confidence = "medium"
    elif score >= 35:
        class_name = "Riskli"
        bias = "neutral-negatif"
        confidence = "low-medium"
    else:
        class_name = "Uzak dur"
        bias = "bearish/negatif"
        confidence = "low"

    setup = "trend devam" if score >= 65 else "teyit bekleyen kurulum" if score >= 45 else "zayif/riskli yapi"
    trigger = f"20 mum direnci uzeri kapanis: {float(resistance):.4f}" if pd.notna(resistance) else "Hacimli kapanis teyidi beklenir."
    invalidation = f"20 mum destek altina sarkma: {float(support):.4f}" if pd.notna(support) else "Son dip altina sarkma."
    risk = "Haber/olay riski ve volatilite takip edilmeli."
    return score, class_name, bias, setup, evidence[:6], trigger, invalidation, risk, confidence


def fmt_price(value: float, market: Market) -> str:
    return f"{value:.2f}" if market == "bist" else f"{value:.4f}"


def _clean_float(value, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _level_map(levels: dict[str, float], market: Market) -> dict[str, str]:
    return {name: fmt_price(value, market) for name, value in levels.items() if value and not pd.isna(value)}


def _recent_swing_points(df: pd.DataFrame) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    highs = [(int(i), float(v)) for i, v in df["swing_high"].dropna().tail(6).items()] if "swing_high" in df else []
    lows = [(int(i), float(v)) for i, v in df["swing_low"].dropna().tail(6).items()] if "swing_low" in df else []
    return highs, lows


def build_technical_context(df: pd.DataFrame, market: Market) -> dict:
    latest = df.dropna(subset=["close"]).iloc[-1]
    close = _clean_float(latest.get("close"))
    atr = _clean_float(latest.get("atr14"), close * 0.02)
    highs, lows = _recent_swing_points(df)

    last_high = highs[-1][1] if highs else _clean_float(latest.get("resistance20"), close)
    last_low = lows[-1][1] if lows else _clean_float(latest.get("support20"), close)
    prev_high = highs[-2][1] if len(highs) >= 2 else np.nan
    prev_low = lows[-2][1] if len(lows) >= 2 else np.nan

    higher_high = pd.notna(prev_high) and last_high > float(prev_high)
    lower_high = pd.notna(prev_high) and last_high < float(prev_high)
    higher_low = pd.notna(prev_low) and last_low > float(prev_low)
    lower_low = pd.notna(prev_low) and last_low < float(prev_low)

    if higher_high and higher_low:
        structure_bias = "bullish"
        structure_note = "HH/HL yapisi; yukari trend yapisi korunuyor."
    elif lower_high and lower_low:
        structure_bias = "bearish"
        structure_note = "LH/LL yapisi; dusen trend yapisi baskin."
    elif higher_low and lower_high:
        structure_bias = "range"
        structure_note = "Sikisan yapi; karar kirilimla gelecek."
    else:
        structure_bias = "mixed"
        structure_note = "Yapi karisik; tek basina guvenli yon vermiyor."

    bos = "none"
    if close > last_high:
        bos = "bos_up"
        structure_note = "Son swing tepe kirildi; yukari BOS teyidi var."
    elif close < last_low:
        bos = "bos_down"
        structure_note = "Son swing dip kirildi; asagi BOS teyidi var."

    adx = _clean_float(latest.get("adx14"))
    plus_di = _clean_float(latest.get("plus_di14"))
    minus_di = _clean_float(latest.get("minus_di14"))
    if adx >= 22 and plus_di > minus_di:
        trend_label = "strong_up"
        trend_note = "Trend gucu yukari."
    elif adx >= 22 and minus_di > plus_di:
        trend_label = "strong_down"
        trend_note = "Trend gucu asagi."
    elif adx < 16:
        trend_label = "range"
        trend_note = "ADX dusuk; range/sahte kirilim riski."
    else:
        trend_label = "weak"
        trend_note = "Trend gucu orta/zayif; teyit beklenmeli."

    rolling_vwap = _clean_float(latest.get("rolling_vwap20"), np.nan)
    vwap = _clean_float(latest.get("vwap"), np.nan)
    vwap_position = "above" if pd.notna(rolling_vwap) and close > rolling_vwap else "below" if pd.notna(rolling_vwap) else "unknown"

    swing_low = last_low
    swing_high = last_high
    swing_direction = "up" if lows and highs and lows[-1][0] < highs[-1][0] else "down"
    swing_range = max(abs(swing_high - swing_low), atr, close * 0.005)
    if swing_direction == "up":
        fib_retracements = {
            "0.382": swing_high - 0.382 * swing_range,
            "0.500": swing_high - 0.500 * swing_range,
            "0.618": swing_high - 0.618 * swing_range,
        }
        fib_extensions = {
            "1.272": swing_high + 0.272 * swing_range,
            "1.618": swing_high + 0.618 * swing_range,
        }
    else:
        fib_retracements = {
            "0.382": swing_low + 0.382 * swing_range,
            "0.500": swing_low + 0.500 * swing_range,
            "0.618": swing_low + 0.618 * swing_range,
        }
        fib_extensions = {
            "1.272": swing_low - 0.272 * swing_range,
            "1.618": swing_low - 0.618 * swing_range,
        }

    gann_levels = {f"{i}/8": swing_low + (swing_range * i / 8) for i in range(1, 8)}
    if swing_direction == "down":
        gann_levels = {f"{i}/8": swing_high - (swing_range * i / 8) for i in range(1, 8)}

    pivot_levels = {
        "P": _clean_float(latest.get("pivot_p"), np.nan),
        "R1": _clean_float(latest.get("pivot_r1"), np.nan),
        "R2": _clean_float(latest.get("pivot_r2"), np.nan),
        "S1": _clean_float(latest.get("pivot_s1"), np.nan),
        "S2": _clean_float(latest.get("pivot_s2"), np.nan),
    }
    weekly_pivots = {
        "WP": _clean_float(latest.get("weekly_pivot_p"), np.nan),
        "WR1": _clean_float(latest.get("weekly_pivot_r1"), np.nan),
        "WS1": _clean_float(latest.get("weekly_pivot_s1"), np.nan),
    }

    squeeze_on = bool(latest.get("squeeze_on")) if pd.notna(latest.get("squeeze_on")) else False
    bb_width = _clean_float(latest.get("bb_width_pct"), np.nan)
    obv_slope = _clean_float(df["obv"].diff(5).iloc[-1]) if "obv" in df and len(df) > 6 else 0.0
    accdist_slope = _clean_float(df["accdist"].diff(5).iloc[-1]) if "accdist" in df and len(df) > 6 else 0.0

    notes = [structure_note, trend_note]
    if vwap_position == "above":
        notes.append("Fiyat 20 mum VWAP ustunde; geri cekilme/retest daha saglikli.")
    elif vwap_position == "below":
        notes.append("Fiyat 20 mum VWAP altinda; long kovalamak zayif.")
    if squeeze_on:
        notes.append("Squeeze aktif; yonlu kirilim gelmeden islem riski artar.")
    if obv_slope > 0 and close < _clean_float(latest.get("sma20"), close):
        notes.append("OBV son 5 mumda toparlaniyor; tepki ihtimali izlenir.")
    elif obv_slope < 0:
        notes.append("OBV son 5 mumda zayif; hacim destegi eksik.")

    return {
        "structure": {
            "bias": structure_bias,
            "bos": bos,
            "last_swing_high": last_high,
            "last_swing_low": last_low,
            "higher_high": higher_high,
            "higher_low": higher_low,
            "lower_high": lower_high,
            "lower_low": lower_low,
            "note": structure_note,
        },
        "trend_strength": {
            "label": trend_label,
            "adx14": adx,
            "plus_di14": plus_di,
            "minus_di14": minus_di,
            "note": trend_note,
        },
        "vwap": {
            "vwap": vwap,
            "rolling_vwap20": rolling_vwap,
            "position": vwap_position,
        },
        "fib": {
            "direction": swing_direction,
            "retracements": fib_retracements,
            "extensions": fib_extensions,
        },
        "gann": {"levels": gann_levels},
        "pivots": {"daily": pivot_levels, "weekly": weekly_pivots},
        "squeeze": {"on": squeeze_on, "bb_width_pct": bb_width},
        "volume_flow": {"obv_slope_5": obv_slope, "accdist_slope_5": accdist_slope},
        "notes": notes[:8],
    }


def build_pre_news_signal(df: pd.DataFrame, market: Market) -> dict:
    clean = df.dropna(subset=["open", "high", "low", "close"]).copy()
    if len(clean) < 30:
        return {
            "enabled": True,
            "level": "none",
            "direction": "neutral",
            "effect": "On fiyatlama icin veri kisa; sinyal uretilmedi.",
            "reasons": [],
            "metrics": {},
        }

    latest = clean.iloc[-1]
    prev = clean.iloc[-2]
    close = _clean_float(latest.get("close"))
    open_ = _clean_float(latest.get("open"), close)
    high = _clean_float(latest.get("high"), close)
    low = _clean_float(latest.get("low"), close)
    prev_close = _clean_float(prev.get("close"), close)
    atr = _clean_float(latest.get("atr14"), max(close * 0.02, 0.0001))
    atr = max(atr, abs(close) * 0.002, 0.0001)
    atr_pct = _clean_float(latest.get("atr_pct"), (atr / close * 100) if close else 0.0)
    vol_z = _clean_float(latest.get("vol_z"))
    ret1 = _clean_float(latest.get("ret_pct"))
    ret3 = ((close / _clean_float(clean["close"].iloc[-4], close)) - 1) * 100 if len(clean) >= 4 and close else 0.0
    ret5 = ((close / _clean_float(clean["close"].iloc[-6], close)) - 1) * 100 if len(clean) >= 6 and close else 0.0
    gap_pct = ((open_ / prev_close) - 1) * 100 if prev_close else 0.0
    day_range = max(high - low, abs(close) * 0.0005)
    range_atr = day_range / atr
    close_location = max(0.0, min(1.0, (close - low) / day_range))
    prev_resistance = _clean_float(clean["resistance20"].shift(1).iloc[-1], np.nan) if "resistance20" in clean else np.nan
    prev_support = _clean_float(clean["support20"].shift(1).iloc[-1], np.nan) if "support20" in clean else np.nan
    breakout = pd.notna(prev_resistance) and close > float(prev_resistance)
    breakdown = pd.notna(prev_support) and close < float(prev_support)
    recent = clean.tail(6)
    up_days = int((recent["close"].diff() > 0).sum())
    down_days = int((recent["close"].diff() < 0).sum())
    volume_20 = _clean_float(clean["volume"].tail(20).mean()) if "volume" in clean else 0.0
    obv_slope = _clean_float(clean["obv"].diff(5).iloc[-1]) if "obv" in clean and len(clean) > 6 else 0.0
    accdist_slope = _clean_float(clean["accdist"].diff(5).iloc[-1]) if "accdist" in clean and len(clean) > 6 else 0.0
    flow_scale = max(volume_20 * 5, 1.0)
    obv_norm = obv_slope / flow_scale
    accdist_norm = accdist_slope / flow_scale
    squeeze_recent = bool(clean["squeeze_on"].tail(10).any()) if "squeeze_on" in clean else False

    pos = 0
    neg = 0
    reasons: list[str] = []
    big_ret = max(1.2, atr_pct * 0.55)
    multi_ret = max(2.0, atr_pct * 0.90)

    if vol_z >= 1.8 and ret1 > 0:
        pos += 3
        reasons.append(f"Hacim anormal yuksek ve fiyat yukari kapatti: vol_z {vol_z:.2f}, gunluk {ret1:.2f}%.")
    elif vol_z >= 1.8 and ret1 < 0:
        neg += 3
        reasons.append(f"Hacim anormal yuksek ve fiyat asagi kapatti: vol_z {vol_z:.2f}, gunluk {ret1:.2f}%.")
    elif vol_z >= 1.2 and abs(ret1) >= big_ret:
        if ret1 > 0:
            pos += 2
        else:
            neg += 2
        reasons.append(f"Hacim ve fiyat ayni anda hizlandi: vol_z {vol_z:.2f}, gunluk {ret1:.2f}%.")

    if ret3 >= multi_ret or ret5 >= multi_ret * 1.15:
        pos += 2
        reasons.append(f"Son 3/5 mum pozitif ivme tasiyor: 3 mum {ret3:.2f}%, 5 mum {ret5:.2f}%.")
    elif ret3 <= -multi_ret or ret5 <= -multi_ret * 1.15:
        neg += 2
        reasons.append(f"Son 3/5 mum negatif ivme tasiyor: 3 mum {ret3:.2f}%, 5 mum {ret5:.2f}%.")

    if close_location >= 0.72 and range_atr >= 0.90 and ret1 > 0:
        pos += 2
        reasons.append(f"Mum gun araliginin ust bolgesinde kapandi; alici baskisi var ({close_location:.2f}).")
    elif close_location <= 0.28 and range_atr >= 0.90 and ret1 < 0:
        neg += 2
        reasons.append(f"Mum gun araliginin alt bolgesinde kapandi; satici baskisi var ({close_location:.2f}).")

    if breakout:
        pos += 3
        reasons.append("Fiyat onceki 20 mum direncinin ustune cikti; olasi haber/katalizor oncesi kirilim izlenir.")
        if vol_z >= 1.0:
            pos += 1
            reasons.append("Kirilim hacimle destekleniyor.")
    elif breakdown:
        neg += 3
        reasons.append("Fiyat onceki 20 mum desteginin altina indi; olasi risk oncesi satis baskisi izlenir.")
        if vol_z >= 1.0:
            neg += 1
            reasons.append("Destek kirilimi hacimle destekleniyor.")

    if gap_pct >= max(1.0, atr_pct * 0.40):
        pos += 2
        reasons.append(f"Yukari gap var: {gap_pct:.2f}%.")
    elif gap_pct <= -max(1.0, atr_pct * 0.40):
        neg += 2
        reasons.append(f"Asagi gap var: {gap_pct:.2f}%.")

    if obv_norm > 0.12 and accdist_norm > 0.04:
        pos += 2
        reasons.append("OBV ve Acc/Dist son 5 mumda para girisi tarafini destekliyor.")
    elif obv_norm < -0.12 and accdist_norm < -0.04:
        neg += 2
        reasons.append("OBV ve Acc/Dist son 5 mumda para cikisi tarafini destekliyor.")

    if up_days >= 4 and ret5 > 0:
        pos += 1
        reasons.append(f"Son 6 mumda {up_days} yukari kapanis var; sessiz toplama ihtimali izlenir.")
    elif down_days >= 4 and ret5 < 0:
        neg += 1
        reasons.append(f"Son 6 mumda {down_days} asagi kapanis var; sessiz dagitim ihtimali izlenir.")

    if squeeze_recent and (breakout or breakdown):
        if breakout:
            pos += 1
        else:
            neg += 1
        reasons.append("Yakinda sikisma vardi ve fiyat araliktan cikiyor; hareketin habere hassasiyeti artar.")

    if pos >= neg + 2:
        direction = "positive"
        dominant = pos
    elif neg >= pos + 2:
        direction = "negative"
        dominant = neg
    elif pos + neg >= 5:
        direction = "mixed"
        dominant = max(pos, neg)
    else:
        direction = "neutral"
        dominant = max(pos, neg)

    if direction == "neutral" or dominant < 3:
        level = "none"
    elif dominant >= 8:
        level = "high"
    elif dominant >= 5:
        level = "medium"
    else:
        level = "low"

    if level == "none":
        direction = "neutral"
        reasons = []
        effect = "Resmi haber/KAP oncesi belirgin on fiyatlama anomalisi yok."
    elif direction == "positive":
        effect = "Alici tarafi resmi haber gelmeden bir katalizoru fiyatliyor olabilir; bu iceriden bilgi degil, sadece fiyat/hacim anomalisi."
    elif direction == "negative":
        effect = "Satici tarafi resmi haber gelmeden bir risk fiyatliyor olabilir; yeni long icin haber/KAP ve teknik teyit beklenmeli."
    else:
        effect = "Tahtada anormal hareket var ama yon temiz degil; haber/KAP veya net teknik teyit gelmeden islem riski yuksek."

    return {
        "enabled": True,
        "level": level,
        "direction": direction,
        "score": {"positive": pos, "negative": neg, "net": pos - neg},
        "effect": effect,
        "reasons": reasons[:8],
        "metrics": {
            "vol_z": vol_z,
            "ret1_pct": ret1,
            "ret3_pct": ret3,
            "ret5_pct": ret5,
            "gap_pct": gap_pct,
            "range_atr": range_atr,
            "close_location": close_location,
            "obv_norm_5": obv_norm,
            "accdist_norm_5": accdist_norm,
            "breakout20": bool(breakout),
            "breakdown20": bool(breakdown),
        },
    }


def build_trade_plan(df: pd.DataFrame, market: Market, score: int, kronos_data: dict | None, technical_data: dict | None = None) -> dict:
    latest = df.dropna(subset=["close"]).iloc[-1]
    close = float(latest["close"])
    atr = float(latest["atr14"]) if pd.notna(latest.get("atr14")) and latest["atr14"] > 0 else close * 0.02
    support = float(latest["support20"]) if pd.notna(latest.get("support20")) else close - 2 * atr
    resistance = float(latest["resistance20"]) if pd.notna(latest.get("resistance20")) else close + 2 * atr
    has_kronos = bool(kronos_data and kronos_data.get("enabled", True) is not False)
    kronos_change = float(kronos_data["forecast_change_pct"]) if has_kronos else 0.0

    if score >= 60 and (not has_kronos or kronos_change >= -0.3):
        primary = "long"
        reason = "Teknik yapi long lehine."
        if has_kronos:
            reason += " Kronos ciddi sekilde ters dusmuyor."
    elif score <= 42 and (not has_kronos or kronos_change <= 0.3):
        primary = "short"
        reason = "Teknik yapi zayif."
        if has_kronos:
            reason += " Kronos yukari senaryoyu desteklemiyor."
    elif has_kronos and kronos_change > 0.8 and score >= 45:
        primary = "long"
        reason = "Kronos yukari sinyal veriyor, teknik yapi teyit bekliyor."
    elif has_kronos and kronos_change < -0.8 and score <= 58:
        primary = "short"
        reason = "Kronos asagi sinyal veriyor, teknik yapi temkinli."
    else:
        primary = "wait"
        reason = "Sinyaller karisik; net giris yerine teyit beklenmeli."

    long_zone_low = support + 0.15 * atr
    long_zone_high = support + 0.65 * atr
    if long_zone_high >= close:
        long_zone_low = max(support, close - 1.10 * atr)
        long_zone_high = max(long_zone_low, close - 0.45 * atr)

    long_breakout = resistance + 0.10 * atr
    long_invalid = support - 0.45 * atr
    long_target_base = max(close, long_breakout)
    long_targets = [
        max(resistance + 0.80 * atr, long_target_base + 0.80 * atr, close + 1.00 * atr),
        max(resistance + 1.60 * atr, long_target_base + 1.60 * atr, close + 1.80 * atr),
        max(resistance + 2.50 * atr, long_target_base + 2.50 * atr, close + 2.80 * atr),
    ]

    short_trigger = support - 0.15 * atr
    short_invalid = resistance + 0.45 * atr
    short_target_base = min(close, short_trigger)
    short_targets = [
        min(support - 0.80 * atr, short_target_base - 0.80 * atr, close - 1.00 * atr),
        min(support - 1.60 * atr, short_target_base - 1.60 * atr, close - 1.80 * atr),
        min(support - 2.50 * atr, short_target_base - 2.50 * atr, close - 2.80 * atr),
    ]

    long_pullback_rr = (long_targets[0] - long_zone_high) / _safe_risk(long_zone_high, long_invalid)
    long_breakout_rr = (long_targets[0] - long_breakout) / _safe_risk(long_breakout, long_invalid)
    short_rr = (short_trigger - short_targets[0]) / _safe_risk(short_trigger, short_invalid)
    rr_floor = 0.50

    if primary == "long" and max(long_pullback_rr, long_breakout_rr) < rr_floor:
        primary = "wait"
        reason += " Long R/R T1 yetersiz; kovalamak yerine daha iyi seviye beklenmeli."
    elif primary == "short" and short_rr < rr_floor:
        primary = "wait"
        reason += " Short R/R T1 yetersiz; destek kirilimi tek basina yeterli degil."

    if technical_data:
        structure_bias = technical_data.get("structure", {}).get("bias")
        trend_label = technical_data.get("trend_strength", {}).get("label")
        vwap_position = technical_data.get("vwap", {}).get("position")
        squeeze_on = bool(technical_data.get("squeeze", {}).get("on"))
        if primary == "long" and structure_bias == "bearish" and trend_label == "strong_down":
            primary = "wait"
            reason += " Market structure ve ADX satici lehine; long icin CHOCH/BOS teyidi beklenmeli."
        elif primary == "short" and structure_bias == "bullish" and trend_label == "strong_up":
            primary = "wait"
            reason += " Market structure ve ADX alici lehine; short icin yapinin bozulmasi beklenmeli."
        elif primary == "long" and vwap_position == "below" and trend_label != "strong_up":
            primary = "wait"
            reason += " Fiyat VWAP altinda; long kovalamak yerine retest/geri alim beklenmeli."
        if squeeze_on and primary in {"long", "short"}:
            reason += " Squeeze aktif; yonlu kirilim teyidi kritik."

    if primary == "long":
        action = "Duzeltme long veya direnc kirilimi bekle."
        wait_note = "Fiyat long bolgesinden uzaksa kovalamak yerine geri cekilme veya hacimli kirilim bekle."
    elif primary == "short":
        action = "Destek kirilimi sonrasi short senaryoyu izle."
        wait_note = "Destek kirilmadan short kovalamak riskli; tepki alimlari gelebilir."
    else:
        action = "Bekle."
        wait_note = "Long/short icin net teyit ve kabul edilebilir R/R yok; fiyat seviyelerden birine gelene kadar izleme modu."

    plan = {
        "current_price": close,
        "primary": primary,
        "reason": reason,
        "action": action,
        "long_pullback_zone": [long_zone_low, long_zone_high],
        "long_breakout_trigger": long_breakout,
        "long_invalidation": long_invalid,
        "long_targets": long_targets,
        "short_trigger": short_trigger,
        "short_invalidation": short_invalid,
        "short_targets": short_targets,
        "rr": {
            "long_pullback_t1": long_pullback_rr,
            "long_breakout_t1": long_breakout_rr,
            "short_t1": short_rr,
        },
        "wait_note": wait_note,
    }
    plan["strategy_switch"] = build_strategy_switch(plan)
    return plan


def _safe_risk(entry: float, stop: float) -> float:
    return max(abs(entry - stop), abs(entry) * 0.0005)


def _rr_label(value: float) -> str:
    if value >= 2:
        return "iyi"
    if value >= 1:
        return "idare eder"
    if value > 0:
        return "zayif"
    return "yok"


def build_strategy_switch(plan: dict) -> dict:
    primary = str(plan.get("primary", "wait"))
    rr = plan.get("rr", {})
    long_zone = plan.get("long_pullback_zone", [None, None])
    long_targets = plan.get("long_targets", [])
    short_targets = plan.get("short_targets", [])
    pullback_rr = float(rr.get("long_pullback_t1", 0.0) or 0.0)
    breakout_rr = float(rr.get("long_breakout_t1", 0.0) or 0.0)

    base = {
        "primary": primary,
        "mode": "conditional",
        "time_horizon": "maksimum 1 hafta",
        "rules": [],
        "summary": "",
    }

    if primary == "long":
        base["summary"] = "Ana strateji long, ama giris sadece seviye ve mum teyidiyle yapilir."
        base["rules"] = [
            {
                "step": 1,
                "name": "Baslangic stratejisi",
                "strategy": "pullback_long",
                "condition": "Fiyat long bolgesine iner ve tepki mumu/VWAP20 ustune donus/guculu yesil kapanis verir.",
                "levels": {"zone": long_zone},
                "action": "Geri cekilme long denenir.",
                "risk": f"R/R {pullback_rr:.2f} ({_rr_label(pullback_rr)}).",
            },
            {
                "step": 2,
                "name": "Diger stratejiye gec",
                "strategy": "breakout_long",
                "condition": "Fiyat geri cekilmeden long kirilim seviyesinin ustunde hacimli kapanis yapar.",
                "levels": {"trigger": plan.get("long_breakout_trigger")},
                "action": "Plan A bekleme biter, kirilim long stratejisine gecilir.",
                "risk": f"R/R {breakout_rr:.2f} ({_rr_label(breakout_rr)}).",
            },
            {
                "step": 3,
                "name": "Kar yonetimi",
                "strategy": "scale_out",
                "condition": "T1/T2/T3 hedeflerinden biri gelir.",
                "levels": {"targets": long_targets},
                "action": "Ilk hedefte risk azalt; 1 hafta icinde hedef/tepki yoksa fikri yeniden oku.",
                "risk": "Hedefe geldikten sonra kar geri verilmez.",
            },
            {
                "step": 4,
                "name": "Savunmaya don",
                "strategy": "defense_or_short_risk",
                "condition": "Long iptal seviyesi veya short tetik seviyesi altinda kapanis gelir.",
                "levels": {"long_invalidation": plan.get("long_invalidation"), "short_trigger": plan.get("short_trigger"), "short_targets": short_targets},
                "action": "Long fikri iptal; yeni alım yerine savunma/short-risk senaryosu izlenir.",
                "risk": "Bu kisimda inat yok.",
            },
        ]
    elif primary == "short":
        base["summary"] = "Ana strateji savunma/short-risk; long icin once yapi toparlanmali."
        base["rules"] = [
            {
                "step": 1,
                "name": "Baslangic stratejisi",
                "strategy": "short_risk",
                "condition": "Fiyat short tetik seviyesi altinda kapanis yapar.",
                "levels": {"trigger": plan.get("short_trigger"), "targets": short_targets},
                "action": "Destek kirilimi sonrasi short-risk senaryosu izlenir.",
                "risk": "Destek kirilmadan short kovalanmaz.",
            },
            {
                "step": 2,
                "name": "Long stratejiye gec",
                "strategy": "breakout_long",
                "condition": "Fiyat long kirilim seviyesinin ustunde hacimli kapanis yapar.",
                "levels": {"trigger": plan.get("long_breakout_trigger")},
                "action": "Short fikri zayiflar; kirilim long plani yeniden degerlendirilir.",
                "risk": f"Kirilim R/R {breakout_rr:.2f} ({_rr_label(breakout_rr)}).",
            },
            {
                "step": 3,
                "name": "Guvenli long sarti",
                "strategy": "pullback_long",
                "condition": "Fiyat long bolgesinde tepki verir ve VWAP20 ustune geri doner.",
                "levels": {"zone": long_zone},
                "action": "Dusus sonrasi toparlanma teyidi varsa long tekrar masaya gelir.",
                "risk": f"Duzeltme R/R {pullback_rr:.2f} ({_rr_label(pullback_rr)}).",
            },
            {
                "step": 4,
                "name": "Iptal",
                "strategy": "no_long",
                "condition": "Long iptal seviyesi altinda kapanis devam eder.",
                "levels": {"long_invalidation": plan.get("long_invalidation")},
                "action": "Long tarafi tamamen beklemeye alinir.",
                "risk": "Yapi toparlanmadan yeni alim yok.",
            },
        ]
    else:
        base["summary"] = "Bekle modu; sadece iki net kosuldan biri gelirse stratejiye gec."
        base["rules"] = [
            {
                "step": 1,
                "name": "Baslangic",
                "strategy": "wait",
                "condition": "Fiyat ne long bolgesinde tepki verdi ne de kirilim teyidi uretmis durumda.",
                "levels": {},
                "action": "Islem yok; fiyat kovalanmaz.",
                "risk": "Beklemek de pozisyondur.",
            },
            {
                "step": 2,
                "name": "Plan A'ya gec",
                "strategy": "pullback_long",
                "condition": "Fiyat long bolgesine iner ve tepki/VWAP20 geri alimi/guculu yesil kapanis verir.",
                "levels": {"zone": long_zone},
                "action": "Geri cekilme long stratejisi acilir.",
                "risk": f"R/R {pullback_rr:.2f} ({_rr_label(pullback_rr)}).",
            },
            {
                "step": 3,
                "name": "Plan B'ye gec",
                "strategy": "breakout_long",
                "condition": "Fiyat long kirilim seviyesinin ustunde hacimli kapanis yapar.",
                "levels": {"trigger": plan.get("long_breakout_trigger")},
                "action": "Kirilim long stratejisi izlenir.",
                "risk": f"R/R {breakout_rr:.2f} ({_rr_label(breakout_rr)}).",
            },
            {
                "step": 4,
                "name": "Savunmaya gec",
                "strategy": "defense_or_short_risk",
                "condition": "Fiyat short tetik seviyesi altinda kapanis yapar.",
                "levels": {"trigger": plan.get("short_trigger"), "targets": short_targets},
                "action": "Long fikri zayiflar; savunma/short-risk senaryosu izlenir.",
                "risk": "Bu bolgede long kovalanmaz.",
            },
        ]

    return base


KAP_BASE_URL = "https://www.kap.org.tr"
KAP_HEADERS = {
    "Origin": KAP_BASE_URL,
    "Referer": f"{KAP_BASE_URL}/tr/bildirim-sorgu",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "tr-TR,tr;q=0.9,en;q=0.8",
    "Content-Type": "application/json",
}
KAP_MEMBER_TYPES = ["IGS", "HT", "YK", "PYS", "BDK", "DCS", "DDK", "DK", "KVH"]

HIGH_IMPACT_KEYWORDS = {
    "positive": [
        "net kar",
        "net kâr",
        "kar payi",
        "kar payı",
        "temettu",
        "temettü",
        "geri alim",
        "geri alım",
        "pay alimi",
        "pay alımı",
        "yeni is",
        "yeni iş",
        "is iliskisi",
        "iş ilişkisi",
        "ihale kazan",
        "sozlesme",
        "sözleşme",
        "onay",
        "bedelsiz",
        "sermaye artirimi",
        "sermaye artırımı",
        "buyback",
        "partnership",
        "listing",
        "etf approval",
    ],
    "negative": [
        "net zarar",
        "zarar acikladi",
        "zarar açıkladı",
        "zarar beklentisi",
        "ceza",
        "dava",
        "sorusturma",
        "soruşturma",
        "spk",
        "tedbir",
        "iptal",
        "ret",
        "temerrut",
        "temerrüt",
        "konkordato",
        "iflas",
        "haciz",
        "borc yapilandirma",
        "borç yapılandırma",
        "borc odeme guclugu",
        "borç ödeme güçlüğü",
        "not indirimi",
        "derecelendirme notu",
        "delist",
        "delisting",
        "hack",
        "exploit",
        "unlock",
        "lawsuit",
        "sec",
        "regulation",
        "bankruptcy",
    ],
    "neutral_high": [
        "finansal rapor",
        "financial report",
        "faaliyet raporu",
        "operating review",
        "ozel durum",
        "özel durum",
        "bilanco",
        "bilanço",
        "sermaye",
        "birlesme",
        "birleşme",
        "bolunme",
        "bölünme",
        "yonetim kurulu",
        "yönetim kurulu",
        "pay satis",
        "pay satış",
        "tahvil ihraci",
        "tahvil ihracı",
        "borclanma araci",
        "borçlanma aracı",
        "kira sertifikasi",
        "kira sertifikası",
        "basel iii",
        "fiyatlanmasi",
        "fiyatlanması",
        "material event",
        "earnings",
    ],
}


def _plain_text(value: str) -> str:
    text = html.unescape(value or "")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _parse_any_datetime(value: str) -> datetime | None:
    if not value:
        return None
    for fmt in ("%d.%m.%Y %H:%M:%S", "%d.%m.%Y", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(value[:19], fmt)
        except ValueError:
            continue
    try:
        parsed = parsedate_to_datetime(value)
        return parsed.replace(tzinfo=None)
    except Exception:
        return None


def _extract_list(data) -> list[dict]:
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        for key in ("data", "list", "result", "resultList", "items"):
            value = data.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]
        return [data]
    return []


def _kap_get_json(url: str, timeout: int = 12) -> list[dict]:
    response = requests.get(url, headers=KAP_HEADERS, timeout=timeout)
    response.raise_for_status()
    return _extract_list(response.json())


def _kap_post_json(url: str, body: dict, timeout: int = 15) -> list[dict]:
    response = requests.post(url, headers=KAP_HEADERS, json=body, timeout=timeout)
    response.raise_for_status()
    return _extract_list(response.json())


def fetch_kap_company_map(refresh: bool = False) -> dict[str, dict]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / "kap_company_map.json"
    if not refresh and cache_path.exists():
        age_hours = (datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)).total_seconds() / 3600
        if age_hours < 24 * 7:
            try:
                return json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception:
                pass

    companies: dict[str, dict] = {}
    for member_type in KAP_MEMBER_TYPES:
        try:
            rows = _kap_get_json(f"{KAP_BASE_URL}/tr/api/company/items/{member_type}/A")
        except Exception:
            continue
        for row in rows:
            oid = str(row.get("memberOid") or row.get("mkkMemberOid") or row.get("kapMemberOid") or "")
            title = _plain_text(str(row.get("memberTitle") or row.get("kapMemberTitle") or ""))
            stock_codes = str(row.get("stockCodes") or row.get("stockCode") or "")
            if not oid or not stock_codes:
                continue
            for code in stock_codes.split(","):
                clean = code.strip().upper()
                if clean:
                    companies[clean] = {"oid": oid, "name": title, "ticker": clean}

    if companies:
        cache_path.write_text(json.dumps(companies, ensure_ascii=False, indent=2), encoding="utf-8")
    return companies


def fetch_kap_events(symbol: str, days: int = 14, limit: int = 12) -> tuple[list[dict], str]:
    base = symbol.upper().replace(".IS", "")
    try:
        companies = fetch_kap_company_map()
    except Exception as exc:
        return [], f"KAP sirket listesi alinamadi: {type(exc).__name__}"

    company = companies.get(base)
    if not company:
        return [], "KAP sirket eslesmesi bulunamadi."

    end = datetime.now()
    start = end - timedelta(days=days)
    body = {
        "fromDate": start.strftime("%Y-%m-%d"),
        "toDate": end.strftime("%Y-%m-%d"),
        "memberType": "",
        "mkkMemberOidList": [company["oid"]],
        "inactiveMkkMemberOidList": [],
        "disclosureClass": "",
        "subjectList": [],
        "isLate": "",
        "mainSector": "",
        "sector": "",
        "subSector": "",
        "marketOid": "",
        "index": "",
        "bdkReview": "",
        "bdkMemberOidList": [],
        "year": "",
        "term": "",
        "ruleType": "",
        "period": "",
        "fromSrc": False,
        "srcCategory": "",
        "disclosureIndexList": [],
    }

    try:
        rows = _kap_post_json(f"{KAP_BASE_URL}/tr/api/disclosure/members/byCriteria", body)
    except Exception as exc:
        return [], f"KAP bildirimleri alinamadi: {type(exc).__name__}"

    events: list[dict] = []
    for row in rows:
        index = row.get("disclosureIndex")
        subject = _plain_text(str(row.get("subject") or ""))
        summary = _plain_text(str(row.get("summary") or ""))
        title = subject if not summary else f"{subject}: {summary}"
        published = _parse_any_datetime(str(row.get("publishDate") or ""))
        events.append(
            {
                "source": "KAP",
                "published": published.isoformat(sep=" ") if published else str(row.get("publishDate") or ""),
                "title": title or "KAP bildirimi",
                "url": f"{KAP_BASE_URL}/tr/Bildirim/{index}" if index else f"{KAP_BASE_URL}/tr/ara/{base}",
                "category": str(row.get("disclosureType") or row.get("disclosureClass") or "KAP"),
                "raw_source": company.get("name", base),
            }
        )

    events.sort(key=lambda x: x.get("published", ""), reverse=True)
    return events[:limit], "ok"


def fetch_google_news_events(market: Market, symbol: str, company_name: str = "", days: int = 3, limit: int = 12) -> tuple[list[dict], str]:
    base = symbol.upper().replace(".IS", "")
    if market == "crypto":
        asset = base.split("/")[0]
        query = f"{asset} crypto OR coin OR market"
    else:
        name_part = f' OR "{company_name}"' if company_name else ""
        query = f'{base}{name_part} Borsa Istanbul OR KAP'

    url = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=tr&gl=TR&ceid=TR:tr"
    try:
        response = requests.get(url, headers={"User-Agent": KAP_HEADERS["User-Agent"]}, timeout=12)
        response.raise_for_status()
        root = ET.fromstring(response.content)
    except Exception as exc:
        return [], f"Google News alinamadi: {type(exc).__name__}"

    cutoff = datetime.now() - timedelta(days=days)
    events: list[dict] = []
    for item in root.findall(".//item"):
        title = _plain_text(item.findtext("title") or "")
        link = item.findtext("link") or ""
        pub_dt = _parse_any_datetime(item.findtext("pubDate") or "")
        source_node = item.find("source")
        source = _plain_text(source_node.text if source_node is not None and source_node.text else "Google News")
        if pub_dt and pub_dt < cutoff:
            continue
        events.append(
            {
                "source": source,
                "published": pub_dt.isoformat(sep=" ") if pub_dt else "",
                "title": title,
                "url": link,
                "category": "NEWS",
                "raw_source": source,
            }
        )
        if len(events) >= limit:
            break
    return events, "ok"


def classify_event_impact(event: dict) -> dict:
    text = f"{event.get('title', '')} {event.get('category', '')}".lower()
    pos = [kw for kw in HIGH_IMPACT_KEYWORDS["positive"] if kw in text]
    neg = [kw for kw in HIGH_IMPACT_KEYWORDS["negative"] if kw in text]
    neu = [kw for kw in HIGH_IMPACT_KEYWORDS["neutral_high"] if kw in text]

    if pos and neg:
        direction = "mixed"
    elif neg:
        direction = "negative"
    elif pos:
        direction = "positive"
    elif neu:
        direction = "watch"
    else:
        direction = "neutral"

    impact_points = len(pos) * 3 + len(neg) * 3 + len(neu) * 2
    if event.get("source") == "KAP" and (pos or neg or neu):
        impact_points += 2

    if direction in {"negative", "mixed"} and impact_points >= 3:
        level = "high" if impact_points >= 5 else "medium"
    elif direction in {"positive", "watch"} and impact_points >= 3:
        level = "high" if impact_points >= 5 else "medium"
    else:
        level = "low"

    return {
        **event,
        "impact_level": level,
        "direction": direction,
        "matched_keywords": sorted(set(pos + neg + neu)),
    }


def build_news_impact(market: Market, symbol: str, news_days: int, kap_days: int) -> dict:
    base = symbol.upper().replace(".IS", "")
    kap_events: list[dict] = []
    kap_status = "not_used"
    company_name = ""
    if market == "bist":
        kap_events, kap_status = fetch_kap_events(symbol, days=kap_days)
        if kap_events:
            company_name = str(kap_events[0].get("raw_source") or "")

    news_events, news_status = fetch_google_news_events(market, symbol, company_name=company_name, days=news_days)
    classified = [classify_event_impact(e) for e in kap_events + news_events]
    important = [e for e in classified if e["impact_level"] in {"high", "medium"}]

    def _event_sort_key(event: dict) -> tuple:
        published = _parse_any_datetime(str(event.get("published") or ""))
        ts = published.timestamp() if published else 0
        return (
            0 if event.get("source") == "KAP" else 1,
            0 if event.get("impact_level") == "high" else 1,
            -ts,
        )

    important.sort(key=_event_sort_key)
    important = important[:5]

    directions = {e["direction"] for e in important}
    if not important:
        level = "none"
        direction = "neutral"
        effect = "Kritik fiyat etkili haber bulunmadi. Onemsiz haberler rapora alinmadi."
    elif "mixed" in directions or ("positive" in directions and "negative" in directions):
        level = "high"
        direction = "mixed"
        effect = "Cakisan etkili haberler var; gun ici islemde teyitsiz giris riskli."
    elif "negative" in directions:
        level = "high" if any(e["impact_level"] == "high" and e["direction"] == "negative" for e in important) else "medium"
        direction = "negative"
        effect = "Negatif haber/KAP riski var; long plan zayiflar, teyitsiz islemden kacin."
    elif "positive" in directions:
        level = "high" if any(e["impact_level"] == "high" and e["direction"] == "positive" for e in important) else "medium"
        direction = "positive"
        effect = "Pozitif katalizor var; yine de fiyat kovalanmaz, seviye/teyit beklenir."
    else:
        level = "medium"
        direction = "watch"
        effect = "Fiyat etkisi dogurabilecek izleme haberi var; teknik teyitle birlikte okunmali."

    return {
        "symbol": base,
        "enabled": True,
        "level": level,
        "direction": direction,
        "effect": effect,
        "important_events": important,
        "raw_event_count": len(classified),
        "statuses": {"kap": kap_status, "google_news": news_status},
    }


def apply_news_to_plan(plan: dict, news_impact: dict) -> dict:
    adjusted = dict(plan)
    level = news_impact.get("level")
    direction = news_impact.get("direction")
    if level in {"high", "medium"} and direction in {"negative", "mixed"}:
        adjusted["primary"] = "wait"
        adjusted["action"] = "Bekle; haber riski sindirilmeden islem yok."
        adjusted["reason"] = f"{adjusted.get('reason', '')} Haber/KAP uyarisi plana fren koyuyor."
        adjusted["wait_note"] = "Fiyat etkili negatif/karisik haber var; gun ici kar hedefinde once riskin sindigini gormek gerekir."
        adjusted["news_adjusted"] = True
    elif level == "high" and direction == "positive":
        adjusted["reason"] = f"{adjusted.get('reason', '')} Pozitif haber/KAP katalizoru var; sadece teknik teyitle anlamli."
        adjusted["wait_note"] = f"{adjusted.get('wait_note', '')} Pozitif haber fiyatı gapli oynatabilir; kovalamadan retest/teyit bekle."
        adjusted["news_adjusted"] = True
    else:
        adjusted["news_adjusted"] = False
    adjusted["strategy_switch"] = build_strategy_switch(adjusted)
    return adjusted


def apply_pre_news_to_plan(plan: dict, pre_news: dict) -> dict:
    adjusted = dict(plan)
    level = pre_news.get("level")
    direction = pre_news.get("direction")
    if level in {"high", "medium"} and direction in {"negative", "mixed"}:
        adjusted["primary"] = "wait"
        adjusted["action"] = "Bekle; on fiyatlama riski netlesmeden islem yok."
        adjusted["reason"] = f"{adjusted.get('reason', '')} On fiyatlama uyarisi plana fren koyuyor."
        adjusted["wait_note"] = "Fiyat/hacim anomalisi risk tarafinda; haber/KAP veya net teknik teyit gelmeden yeni pozisyon acma."
        adjusted["pre_news_adjusted"] = True
    elif level == "high" and direction == "positive":
        adjusted["reason"] = f"{adjusted.get('reason', '')} Pozitif on fiyatlama izi var; fiyat kovalanmaz, retest/kirilim teyidi aranir."
        adjusted["wait_note"] = f"{adjusted.get('wait_note', '')} Pozitif on fiyatlama sert gap/ani geri alma yapabilir; giris seviyesi disiplinli olmali."
        adjusted["pre_news_adjusted"] = True
    elif level == "medium" and direction == "positive":
        adjusted["reason"] = f"{adjusted.get('reason', '')} Orta seviye pozitif on fiyatlama izi var; teyit kalitesi artmadan agresif davranma."
        adjusted["pre_news_adjusted"] = True
    else:
        adjusted["pre_news_adjusted"] = False
    adjusted["strategy_switch"] = build_strategy_switch(adjusted)
    return adjusted


def kronos_forecast(df: pd.DataFrame, timeframe: str, model_size: Literal["mini", "small"], lookback: int, pred_len: int) -> dict:
    import torch
    from model import Kronos, KronosPredictor, KronosTokenizer

    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    if model_size == "mini":
        tokenizer_id = "NeoQuasar/Kronos-Tokenizer-2k"
        model_id = "NeoQuasar/Kronos-mini"
        max_context = 2048
    else:
        tokenizer_id = "NeoQuasar/Kronos-Tokenizer-base"
        model_id = "NeoQuasar/Kronos-small"
        max_context = 512

    tokenizer = KronosTokenizer.from_pretrained(tokenizer_id)
    model = Kronos.from_pretrained(model_id)
    model.eval()
    predictor = KronosPredictor(model, tokenizer, max_context=max_context, device="cpu")

    use_len = min(lookback, max_context, len(df))
    x = df.tail(use_len).copy()
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        if col not in x:
            x[col] = 0.0

    freq = {"1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min", "1h": "1h", "4h": "4h", "1d": "1D", "1wk": "1W"}.get(timeframe, "1D")
    y_ts = pd.Series(pd.date_range(pd.to_datetime(x["timestamp"].iloc[-1]), periods=pred_len + 1, freq=freq)[1:])

    with torch.inference_mode():
        pred = predictor.predict(
            df=x[["open", "high", "low", "close", "volume", "amount"]],
            x_timestamp=pd.Series(pd.to_datetime(x["timestamp"])),
            y_timestamp=y_ts,
            pred_len=pred_len,
            T=0.9,
            top_p=0.9,
            sample_count=1,
            verbose=False,
        )

    last_close = float(df["close"].iloc[-1])
    final_close = float(pred["close"].iloc[-1])
    change_pct = (final_close / last_close - 1) * 100
    return {
        "model": f"Kronos-{model_size}",
        "lookback": use_len,
        "pred_len": pred_len,
        "last_close": last_close,
        "forecast_final_close": final_close,
        "forecast_change_pct": change_pct,
        "forecast_head": pred[["open", "high", "low", "close"]].head().round(6).to_dict(orient="records"),
    }


def kronos_auto_forecast(df: pd.DataFrame, timeframe: str, model_choice: str, lookback: int, pred_len: int) -> dict | None:
    if model_choice in {"mini", "small"}:
        return kronos_forecast(df, timeframe, model_choice, lookback, pred_len)

    forecasts: list[dict] = []
    errors: list[str] = []
    for model_size in ["mini", "small"]:
        try:
            forecasts.append(kronos_forecast(df, timeframe, model_size, lookback, pred_len))
        except Exception as exc:
            errors.append(f"{model_size}: {type(exc).__name__}: {exc}")

    if not forecasts:
        return {
            "enabled": False,
            "model": "Kronos-ensemble(mini+small)",
            "lookback": min(lookback, len(df)),
            "pred_len": pred_len,
            "last_close": float(df["close"].iloc[-1]),
            "forecast_final_close": float(df["close"].iloc[-1]),
            "forecast_change_pct": 0.0,
            "agreement": "failed",
            "confidence": "none",
            "members": [],
            "errors": errors,
        }

    if len(forecasts) == 1:
        only = dict(forecasts[0])
        only["model"] = f"{only.get('model', 'Kronos')} (tek model; digeri calismadi)"
        only["members"] = forecasts
        only["agreement"] = "single_model"
        only["confidence"] = "low-medium"
        only["errors"] = errors
        return only

    changes = [float(item["forecast_change_pct"]) for item in forecasts]
    avg_change = float(np.mean(changes))
    last_close = float(df["close"].iloc[-1])
    final_close = last_close * (1 + avg_change / 100)
    signs = [1 if x > 0 else -1 if x < 0 else 0 for x in changes]
    if all(x > 0 for x in signs):
        agreement = "bullish_agreement"
    elif all(x < 0 for x in signs):
        agreement = "bearish_agreement"
    elif all(x == 0 for x in signs):
        agreement = "flat"
    else:
        agreement = "split"

    spread = max(changes) - min(changes)
    if agreement in {"bullish_agreement", "bearish_agreement"} and spread <= 0.80 and abs(avg_change) >= 0.35:
        confidence = "medium-high"
    elif agreement in {"bullish_agreement", "bearish_agreement"}:
        confidence = "medium"
    elif agreement == "split":
        confidence = "low"
    else:
        confidence = "low-medium"

    return {
        "enabled": True,
        "model": "Kronos-ensemble(mini+small)",
        "lookback": min(int(item.get("lookback", lookback)) for item in forecasts),
        "pred_len": pred_len,
        "last_close": last_close,
        "forecast_final_close": final_close,
        "forecast_change_pct": avg_change,
        "agreement": agreement,
        "confidence": confidence,
        "members": forecasts,
        "errors": errors,
    }


def news_links(market: Market, symbol: str) -> list[tuple[str, str]]:
    if market == "bist":
        base = symbol.upper().replace(".IS", "")
        return [
            ("KAP ara", f"https://www.kap.org.tr/tr/ara/{base}"),
            ("Google News", f"https://news.google.com/search?q={base}%20Borsa%20Istanbul"),
        ]
    base = symbol.split("/")[0].upper()
    return [
        ("Google News", f"https://news.google.com/search?q={base}%20crypto"),
        ("CoinMarketCap", f"https://coinmarketcap.com/currencies/{base.lower()}/"),
    ]


def build_markdown(result: AnalysisResult) -> str:
    kronos_text = "Calistirilmadi."
    if result.kronos:
        if result.kronos.get("enabled") is False:
            kronos_text = "Kronos calistirilmak istendi ama model sonucu alinamadi: " + "; ".join(result.kronos.get("errors", []))
        elif result.kronos.get("members"):
            member_lines = []
            for member in result.kronos.get("members", []):
                member_direction = "pozitif" if member.get("forecast_change_pct", 0) > 0 else "negatif"
                member_lines.append(f"- {member.get('model')}: {float(member.get('forecast_change_pct', 0)):.2f}% {member_direction}")
            avg_direction = "pozitif" if result.kronos["forecast_change_pct"] > 0 else "negatif"
            kronos_text = (
                f"{result.kronos['model']} {result.kronos['pred_len']} mum sonunda ortalama "
                f"{result.kronos['forecast_change_pct']:.2f}% {avg_direction} sinyal uretti. "
                f"Model uyumu: {result.kronos.get('agreement', 'unknown')} / guven {result.kronos.get('confidence', 'unknown')}.\n"
                + "\n".join(member_lines)
                + "\nBu sinyal tek basina karar degildir."
            )
        else:
            direction = "pozitif" if result.kronos["forecast_change_pct"] > 0 else "negatif"
            kronos_text = (
                f"{result.kronos['model']} {result.kronos['pred_len']} mum sonunda "
                f"{result.kronos['forecast_change_pct']:.2f}% {direction} sinyal uretti. "
                "Bu sinyal tek basina karar degildir."
            )

    links = "\n".join([f"- [{name}]({url})" for name, url in result.news_links])
    evidence = "\n".join([f"- {x}" for x in result.evidence])
    pre_news = result.pre_news or {
        "level": "none",
        "direction": "neutral",
        "effect": "On fiyatlama sinyali uretilmedi.",
        "reasons": [],
        "metrics": {},
    }
    pre_reasons = "\n".join([f"- {x}" for x in pre_news.get("reasons", [])]) or "- Belirgin on fiyatlama nedeni yok."
    pre_metrics = pre_news.get("metrics", {})
    pre_metric_text = (
        f"vol_z {_clean_float(pre_metrics.get('vol_z')):.2f}, "
        f"1g {_clean_float(pre_metrics.get('ret1_pct')):.2f}%, "
        f"3g {_clean_float(pre_metrics.get('ret3_pct')):.2f}%, "
        f"5g {_clean_float(pre_metrics.get('ret5_pct')):.2f}%, "
        f"gap {_clean_float(pre_metrics.get('gap_pct')):.2f}%, "
        f"range/ATR {_clean_float(pre_metrics.get('range_atr')):.2f}, "
        f"kapanis yeri {_clean_float(pre_metrics.get('close_location')):.2f}"
    )
    news_impact = result.news_impact or {"level": "none", "effect": "Kritik fiyat etkili haber bulunmadi.", "important_events": []}
    news_lines = []
    for event in news_impact.get("important_events", []):
        keywords = ", ".join(event.get("matched_keywords", [])[:4])
        keyword_text = f" ({keywords})" if keywords else ""
        news_lines.append(
            f"- [{event.get('source', 'NEWS')}] {event.get('published', '')} - "
            f"{event.get('direction', 'neutral')}/{event.get('impact_level', 'low')}: "
            f"[{event.get('title', 'Haber')}]({event.get('url', '#')}){keyword_text}"
        )
    news_text = "\n".join(news_lines) if news_lines else "- Kritik fiyat etkili haber bulunmadi; onemsiz haberler rapora alinmadi."
    plan = result.plan
    technical = result.technical or {}
    fundamental = result.fundamental or {}
    structure = technical.get("structure", {})
    trend_strength = technical.get("trend_strength", {})
    vwap_data = technical.get("vwap", {})
    fib = technical.get("fib", {})
    pivots = technical.get("pivots", {})
    squeeze = technical.get("squeeze", {})
    gann = technical.get("gann", {})
    tech_notes = "\n".join([f"- {x}" for x in technical.get("notes", [])]) or "- Teknik okuma notu yok."
    fib_retracements = _level_map(fib.get("retracements", {}), result.market)
    fib_extensions = _level_map(fib.get("extensions", {}), result.market)
    gann_levels = _level_map(gann.get("levels", {}), result.market)
    daily_pivots = _level_map(pivots.get("daily", {}), result.market)
    weekly_pivots = _level_map(pivots.get("weekly", {}), result.market)
    fundamental_metrics = fundamental.get("metrics", {}) if fundamental else {}
    fundamental_notes = "\n".join([f"- {x}" for x in fundamental.get("notes", [])]) if fundamental else "- Temel analiz calistirilmadi."
    long_zone = plan["long_pullback_zone"]
    long_targets = plan["long_targets"]
    short_targets = plan["short_targets"]
    rr = plan.get("rr", {})
    strategy_switch = plan.get("strategy_switch") or build_strategy_switch(plan)
    switch_lines = []
    for rule in strategy_switch.get("rules", []):
        level_parts = []
        levels = rule.get("levels", {}) or {}
        if levels.get("zone"):
            zone = levels["zone"]
            if isinstance(zone, (list, tuple)) and len(zone) >= 2:
                level_parts.append(f"bolge {fmt_price(float(zone[0]), result.market)} - {fmt_price(float(zone[1]), result.market)}")
        if levels.get("trigger") is not None:
            level_parts.append(f"tetik {fmt_price(float(levels['trigger']), result.market)}")
        if levels.get("long_invalidation") is not None:
            level_parts.append(f"long iptal {fmt_price(float(levels['long_invalidation']), result.market)}")
        if levels.get("targets"):
            targets = levels["targets"]
            if isinstance(targets, (list, tuple)) and targets:
                level_parts.append("hedef " + " / ".join(fmt_price(float(x), result.market) for x in targets[:3]))
        level_text = f" Seviyeler: {', '.join(level_parts)}." if level_parts else ""
        switch_lines.append(
            f"{rule.get('step', '-')}. {rule.get('name', '-')}: {rule.get('strategy', '-')}\n"
            f"   - Kosul: {rule.get('condition', '-')}{level_text}\n"
            f"   - Yap: {rule.get('action', '-')}\n"
            f"   - Risk: {rule.get('risk', '-')}"
        )
    switch_text = "\n".join(switch_lines) if switch_lines else "Strateji gecis plani uretilmedi."
    return f"""# {result.symbol} Piyasa Analizi

Piyasa: {result.market}
Zaman dilimi: {result.timeframe}
Veri satiri: {result.rows}
Son fiyat: {result.last_price:.6f}

## Karar

Kalite: {result.class_name}
Bias: {result.bias}
Kurulum: {result.setup}
Guven: {result.confidence}

## Islem Plani

Ana senaryo: {plan['primary']}
Plan: {plan['action']}
Gerekce: {plan['reason']}

Mevcut fiyat: {fmt_price(plan['current_price'], result.market)}

Long duzeltme bolgesi: {fmt_price(long_zone[0], result.market)} - {fmt_price(long_zone[1], result.market)}
Long kirilim teyidi: {fmt_price(plan['long_breakout_trigger'], result.market)} ustu kapanis
Long iptal: {fmt_price(plan['long_invalidation'], result.market)} alti kapanis
Long hedefleri: {fmt_price(long_targets[0], result.market)} / {fmt_price(long_targets[1], result.market)} / {fmt_price(long_targets[2], result.market)}
Long R/R T1: duzeltme {rr.get('long_pullback_t1', 0):.2f}, kirilim {rr.get('long_breakout_t1', 0):.2f}

Short senaryo: {fmt_price(plan['short_trigger'], result.market)} alti kapanis
Short iptal: {fmt_price(plan['short_invalidation'], result.market)} ustu kapanis
Short hedefleri: {fmt_price(short_targets[0], result.market)} / {fmt_price(short_targets[1], result.market)} / {fmt_price(short_targets[2], result.market)}
Short R/R T1: {rr.get('short_t1', 0):.2f}

Not: {plan['wait_note']}

## Strateji Gecis Plani

Ozet: {strategy_switch.get('summary', '-')}
Zaman ufku: {strategy_switch.get('time_horizon', 'maksimum 1 hafta')}

{switch_text}

## On Fiyatlama Uyarisi

Seviye: {pre_news.get('level', 'none')}
Yon: {pre_news.get('direction', 'neutral')}
Skor: {pre_news.get('score', {})}
Etkisi: {pre_news.get('effect', 'On fiyatlama sinyali yok.')}
Metrikler: {pre_metric_text}

{pre_reasons}

## Haber/KAP Etki Uyarisi

Seviye: {news_impact.get('level', 'none')}
Yon: {news_impact.get('direction', 'neutral')}
Etkisi: {news_impact.get('effect', 'Kritik fiyat etkili haber bulunmadi.')}

{news_text}

## Teknik Okuma

Market structure: {structure.get('bias', 'unknown')} / {structure.get('bos', 'none')}
Son swing tepe/dip: {fmt_price(float(structure.get('last_swing_high', 0) or 0), result.market)} / {fmt_price(float(structure.get('last_swing_low', 0) or 0), result.market)}

ADX/DI: {trend_strength.get('label', 'unknown')} | ADX {float(trend_strength.get('adx14', 0) or 0):.1f} | +DI {float(trend_strength.get('plus_di14', 0) or 0):.1f} | -DI {float(trend_strength.get('minus_di14', 0) or 0):.1f}
VWAP: {vwap_data.get('position', 'unknown')} | VWAP20 {fmt_price(float(vwap_data.get('rolling_vwap20', 0) or 0), result.market)}
Squeeze: {'aktif' if squeeze.get('on') else 'yok'} | BB width {float(squeeze.get('bb_width_pct', 0) or 0):.2f}%

Fib retracement: {json.dumps(fib_retracements, ensure_ascii=False)}
Fib extension: {json.dumps(fib_extensions, ensure_ascii=False)}
Pivot gunluk: {json.dumps(daily_pivots, ensure_ascii=False)}
Pivot haftalik: {json.dumps(weekly_pivots, ensure_ascii=False)}
Gann 1/8-7/8: {json.dumps(gann_levels, ensure_ascii=False)}

Teknik notlar:
{tech_notes}

## Temel Analiz

Durum: {fundamental.get('label', 'not_run') if fundamental else 'not_run'}
Skor: {fundamental.get('score', '-') if fundamental else '-'}
Hasılat YoY: {fundamental_metrics.get('revenue_yoy_pct')}
Net kâr YoY: {fundamental_metrics.get('net_income_yoy_pct')}
Net kâr QoQ: {fundamental_metrics.get('net_income_qoq_pct')}
Borç/Özsermaye: {fundamental_metrics.get('debt_to_equity')}
Nakit/Borç: {fundamental_metrics.get('cash_to_debt')}

Temel notlar:
{fundamental_notes}

## Kanitlar

{evidence}

## Kronos

{kronos_text}

## Tetikleyici

{result.trigger}

## Gecersiz Kilan Seviye

{result.invalidation}

## Risk

{result.risk}

## Haber/KAP Linkleri

{links}

Not: Bu cikti yatirim tavsiyesi degildir. Al-sat emri vermez, sadece arastirma ve risk takibi icindir.
"""


def save_outputs(symbol: str, df: pd.DataFrame, markdown: str, result: AnalysisResult) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_symbol = symbol.replace("/", "_").replace(".", "_")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = DATA_DIR / f"{safe_symbol}_{result.timeframe}_{stamp}.csv"
    md_path = REPORTS_DIR / f"{safe_symbol}_{result.timeframe}_{stamp}.md"
    json_path = REPORTS_DIR / f"{safe_symbol}_{result.timeframe}_{stamp}.json"
    df.to_csv(csv_path, index=False)
    md_path.write_text(markdown, encoding="utf-8")
    json_payload = {
        "market": result.market,
        "symbol": result.symbol,
        "timeframe": result.timeframe,
        "score": result.score,
        "class_name": result.class_name,
        "bias": result.bias,
        "setup": result.setup,
        "trigger": result.trigger,
        "invalidation": result.invalidation,
        "risk": result.risk,
        "confidence": result.confidence,
        "kronos": result.kronos,
        "plan": result.plan,
        "technical": result.technical,
        "fundamental": result.fundamental,
        "pre_news": result.pre_news,
        "news_impact": result.news_impact,
    }
    json_path.write_text(json.dumps(json_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return md_path


def main() -> int:
    parser = argparse.ArgumentParser(description="OpenCode Kronos market analyst for crypto and BIST.")
    parser.add_argument("--market", choices=["crypto", "bist"], required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--limit", type=int, default=360)
    parser.add_argument("--period", default="1y")
    parser.add_argument("--kronos", dest="kronos", action="store_true", default=True)
    parser.add_argument("--no-kronos", dest="kronos", action="store_false")
    parser.add_argument("--model", choices=["auto", "both", "mini", "small"], default="auto")
    parser.add_argument("--lookback", type=int, default=128)
    parser.add_argument("--pred-len", type=int, default=12)
    parser.add_argument("--no-news", action="store_true")
    parser.add_argument("--news-days", type=int, default=3)
    parser.add_argument("--kap-days", type=int, default=14)
    parser.add_argument("--fundamental", action="store_true")
    args = parser.parse_args()

    if args.market == "crypto":
        df, source = fetch_crypto(args.symbol, args.timeframe, args.limit)
    else:
        df, source = fetch_bist(args.symbol, args.period, args.timeframe)

    if len(df) < 60:
        raise RuntimeError(f"Data too short: {len(df)} rows from {source}")

    df = add_indicators(df)
    technical_data = build_technical_context(df, args.market)
    score, class_name, bias, setup, evidence, trigger, invalidation, risk, confidence = classify(df)
    kronos_data = kronos_auto_forecast(df, args.timeframe, args.model, args.lookback, args.pred_len) if args.kronos else None
    plan = build_trade_plan(df, args.market, score, kronos_data, technical_data)
    pre_news_signal = build_pre_news_signal(df, args.market)
    plan = apply_pre_news_to_plan(plan, pre_news_signal)
    fundamental_data = None
    if args.fundamental and args.market == "bist":
        try:
            from fundamental_analyzer import fetch_fundamental_snapshot

            fundamental_data = fetch_fundamental_snapshot(args.symbol)
        except Exception as exc:
            fundamental_data = {
                "available": False,
                "score": 0,
                "label": "error",
                "notes": [f"Temel analiz alinamadi: {type(exc).__name__}"],
                "metrics": {},
            }
    news_impact = (
        {"enabled": False, "level": "none", "direction": "neutral", "effect": "Haber taramasi kapali.", "important_events": []}
        if args.no_news
        else build_news_impact(args.market, args.symbol, args.news_days, args.kap_days)
    )
    plan = apply_news_to_plan(plan, news_impact)

    result = AnalysisResult(
        market=args.market,
        symbol=args.symbol,
        timeframe=args.timeframe,
        rows=len(df),
        last_price=float(df["close"].iloc[-1]),
        score=score,
        class_name=class_name,
        bias=bias,
        setup=setup,
        evidence=evidence,
        trigger=trigger,
        invalidation=invalidation,
        risk=risk,
        confidence=confidence,
        kronos=kronos_data,
        plan=plan,
        technical=technical_data,
        fundamental=fundamental_data,
        pre_news=pre_news_signal,
        news_impact=news_impact,
        news_links=news_links(args.market, args.symbol),
    )
    markdown = build_markdown(result)
    report_path = save_outputs(args.symbol, df, markdown, result)
    print(markdown)
    print(f"\nREPORT_PATH={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
