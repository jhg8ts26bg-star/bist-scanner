from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from fundamental_analyzer import fetch_fundamental_snapshot
from export_top5_to_desktop_note import build_block, note_path_from_args, prepend_to_note
from performance_tracker import update_performance_memory
from market_analyzer import (
    REPORTS_DIR,
    add_indicators,
    apply_news_to_plan,
    apply_pre_news_to_plan,
    build_news_impact,
    build_pre_news_signal,
    build_technical_context,
    build_trade_plan,
    classify,
    fetch_bist,
)


DEFAULT_BIST100 = [
    "AEFES", "AGHOL", "AKBNK", "AKSA", "AKSEN", "ALARK", "ARCLK", "ASELS", "ASTOR", "BIMAS",
    "BRSAN", "CANTE", "CCOLA", "CIMSA", "DOAS", "DOHOL", "EKGYO", "ENJSA", "ENKAI", "EREGL",
    "FROTO", "GARAN", "GESAN", "GUBRF", "HALKB", "HEKTS", "ISCTR", "ISMEN", "KARSN", "KCAER",
    "KCHOL", "KONTR", "KOZAA", "KOZAL", "KRDMD", "MGROS", "MIATK", "ODAS", "OYAKC", "PETKM",
    "PGSUS", "QUAGR", "SAHOL", "SASA", "SISE", "SMRTG", "SOKM", "TAVHL", "TCELL", "THYAO",
    "TKFEN", "TOASO", "TSKB", "TTKOM", "TTRAK", "TUPRS", "ULKER", "VAKBN", "VESTL", "YKBNK",
    "ZOREN", "ALFAS", "ANSGR", "BAGFS", "BERA", "BIENY", "BOBET", "BRYAT", "BUCIM", "CWENE",
    "DOHOL", "ECILC", "EGEEN", "ENERY", "EUPWR", "GENIL", "GLYHO", "GWIND", "ISGYO", "KLSER",
    "KONYA", "MAVI", "MPARK", "OTKAR", "PENTA", "REEDR", "SELEC", "SKBNK", "TABGD", "TMSN",
    "TURSG", "YEOTK", "YYLGD", "OBAMS", "KTLEV", "AKFYE", "ANHYT", "CLEBI", "ECZYT", "ULUUN",
]

SECTOR_MAP = {
    "AEFES": "gida_perakende", "BIMAS": "gida_perakende", "CCOLA": "gida_perakende", "MGROS": "gida_perakende",
    "SOKM": "gida_perakende", "TABGD": "gida_perakende", "ULKER": "gida_perakende", "OBAMS": "gida_perakende",
    "AKBNK": "banka_finans", "ANHYT": "banka_finans", "ANSGR": "banka_finans", "GARAN": "banka_finans",
    "HALKB": "banka_finans", "ISCTR": "banka_finans", "ISMEN": "banka_finans", "SKBNK": "banka_finans",
    "TSKB": "banka_finans", "TURSG": "banka_finans", "VAKBN": "banka_finans", "YKBNK": "banka_finans",
    "KTLEV": "banka_finans",
    "AGHOL": "holding", "DOHOL": "holding", "GLYHO": "holding", "KCHOL": "holding", "SAHOL": "holding",
    "AKFYE": "enerji", "AKSEN": "enerji", "ALFAS": "enerji", "ASTOR": "enerji", "CWENE": "enerji",
    "ENJSA": "enerji", "ENKAI": "enerji", "ENERY": "enerji", "EUPWR": "enerji", "GWIND": "enerji",
    "ODAS": "enerji", "PETKM": "enerji", "TUPRS": "enerji", "YEOTK": "enerji", "ZOREN": "enerji",
    "BRSAN": "metal_maden", "EREGL": "metal_maden", "GUBRF": "metal_maden", "KCAER": "metal_maden",
    "KOZAA": "metal_maden", "KOZAL": "metal_maden", "KRDMD": "metal_maden",
    "BOBET": "insaat_cimento", "BUCIM": "insaat_cimento", "CIMSA": "insaat_cimento", "EKGYO": "insaat_cimento",
    "ISGYO": "insaat_cimento", "KONYA": "insaat_cimento", "OYAKC": "insaat_cimento",
    "ASELS": "teknoloji_savunma", "KONTR": "teknoloji_savunma", "MIATK": "teknoloji_savunma",
    "PENTA": "teknoloji_savunma", "REEDR": "teknoloji_savunma", "SMRTG": "teknoloji_savunma",
    "ARCLK": "sanayi_otomotiv", "DOAS": "sanayi_otomotiv", "FROTO": "sanayi_otomotiv", "KARSN": "sanayi_otomotiv",
    "OTKAR": "sanayi_otomotiv", "TOASO": "sanayi_otomotiv", "TTRAK": "sanayi_otomotiv", "VESTL": "sanayi_otomotiv",
    "CLEBI": "havacilik_turizm", "PGSUS": "havacilik_turizm", "TAVHL": "havacilik_turizm", "THYAO": "havacilik_turizm",
    "ECILC": "saglik", "ECZYT": "saglik", "GENIL": "saglik", "MPARK": "saglik", "SELEC": "saglik",
    "BAGFS": "kimya_cam", "HEKTS": "kimya_cam", "KLSER": "kimya_cam", "QUAGR": "kimya_cam", "SASA": "kimya_cam",
    "SISE": "kimya_cam",
    "MAVI": "tekstil", "TKFEN": "tarim_insaat", "ULUUN": "tarim_insaat", "YYLGD": "tarim_insaat",
}


@dataclass
class ScanRow:
    symbol: str
    price: float
    scanner_score: float
    technical_score: int
    class_name: str
    primary: str
    action: str
    long_rr: float
    short_rr: float
    ret5_pct: float
    ret20_pct: float
    structure: str
    bos: str
    adx: float
    trend_label: str
    vwap_position: str
    squeeze: bool
    pre_news_level: str
    pre_news_direction: str
    news_level: str
    news_direction: str
    news_quality_label: str
    news_quality_score: float
    news_gate: str
    fundamental_label: str
    fundamental_score: int
    entry_quality_score: float
    best_entry: str
    entry_window: str
    actionable_route: str
    trade_horizon: str
    holding_days: str
    nearest_entry_distance_pct: float
    daily_trigger_pct: float
    atr_pct: float
    chase_label: str
    chase_score: float
    chase_reason: str
    plan_a_distance_pct: float
    plan_b_distance_pct: float
    market_regime_label: str
    market_regime_bias: str
    market_regime_score: float
    sector: str
    sector_strength_label: str
    sector_strength_score: float
    prep_score: float
    prep_label: str
    prep_reason: str
    agent_consensus_score: float
    agent_verdict: str
    agent_votes: str
    agent_vetoes: str
    risk_manager_note: str
    alert_summary: str
    reason: str
    error: str = ""


def normalize_symbol(symbol: str) -> str:
    clean = symbol.strip().upper().replace(".IS", "")
    return clean


def parse_symbols(value: str) -> list[str]:
    if not value:
        return []
    return [normalize_symbol(x) for x in value.split(",") if x.strip()]


def scanner_rank(row: dict) -> float:
    plan = row["plan"]
    tech = row["technical"]
    news = row.get("news_impact") or {}
    pre_news = row.get("pre_news") or {}
    fundamental = row.get("fundamental") or {}
    rr = plan.get("rr", {})
    long_rr = float(rr.get("long_pullback_t1", 0.0))
    short_rr = float(rr.get("short_t1", 0.0))
    score = float(row["technical_score"])

    if plan.get("primary") == "long":
        score += min(long_rr, 3.0) * 4
    elif plan.get("primary") == "short":
        score += min(short_rr, 3.0) * 3
    else:
        score -= 4

    structure = tech.get("structure", {}).get("bias")
    trend_label = tech.get("trend_strength", {}).get("label")
    vwap_position = tech.get("vwap", {}).get("position")
    if structure == "bullish":
        score += 5
    elif structure == "bearish":
        score -= 5
    if trend_label == "strong_up":
        score += 4
    elif trend_label == "strong_down":
        score -= 4
    if vwap_position == "above":
        score += 3
    elif vwap_position == "below":
        score -= 3
    if tech.get("squeeze", {}).get("on"):
        score += 2

    entry_quality = row.get("entry_quality") or {}
    eq_score = float(entry_quality.get("score", 50.0) or 50.0)
    if eq_score >= 75:
        score += 8
    elif eq_score >= 60:
        score += 4
    elif eq_score < 30:
        score -= 14
    elif eq_score < 45:
        score -= 8

    market_regime = row.get("market_regime") or {}
    regime_bias = str(market_regime.get("bias", "neutral"))
    if regime_bias == "risk_on":
        if plan.get("primary") == "long":
            score += 8
        elif plan.get("primary") == "wait":
            score += 2
    elif regime_bias == "neutral_positive":
        if plan.get("primary") == "long":
            score += 4
    elif regime_bias == "cautious":
        if plan.get("primary") == "long":
            score -= 5
        if eq_score < 65:
            score -= 4
    elif regime_bias == "risk_off":
        if plan.get("primary") == "long":
            score -= 14
        elif plan.get("primary") == "short":
            score += 5
        else:
            score -= 3
        if eq_score < 75:
            score -= 8

    sector_strength = row.get("sector_strength") or {}
    sector_score = float(sector_strength.get("score", 50.0) or 50.0)
    if sector_score >= 72:
        if plan.get("primary") == "long":
            score += 7
        elif plan.get("primary") == "wait":
            score += 3
    elif sector_score >= 58:
        if plan.get("primary") == "long":
            score += 4
    elif sector_score < 32:
        if plan.get("primary") == "long":
            score -= 9
        elif plan.get("primary") == "wait":
            score -= 4
    elif sector_score < 44:
        if plan.get("primary") == "long":
            score -= 5

    news_quality = row.get("news_quality") or {}
    news_quality_score = float(news_quality.get("score", 50.0) or 50.0)
    news_gate = str(news_quality.get("gate", "temiz"))
    if news_gate == "risk_freni":
        score -= 8
    elif news_gate == "teyit_bekle" and eq_score < 70:
        score -= 4
    elif news_gate == "katalizor_var" and plan.get("primary") == "long":
        score += 4
    if news_gate != "not_checked":
        score += (news_quality_score - 50.0) * 0.08

    chase_state = row.get("chase_state") or {}
    chase_label = str(chase_state.get("label", "normal"))
    chase_score = float(chase_state.get("score", 0.0) or 0.0)
    if chase_label == "kovalanmaz":
        score -= 14
    elif chase_label == "geri_cekilme_bekle":
        score -= 7
    score -= min(chase_score, 70.0) * 0.04

    if pre_news.get("direction") in {"negative", "mixed"} and pre_news.get("level") in {"high", "medium"}:
        score -= 12
    elif pre_news.get("direction") == "positive" and pre_news.get("level") == "high":
        score += 6
    elif pre_news.get("direction") == "positive" and pre_news.get("level") == "medium":
        score += 3

    if fundamental:
        score += (float(fundamental.get("score", 50)) - 50) * 0.20
    if news.get("direction") in {"negative", "mixed"} and news.get("level") in {"high", "medium"}:
        score -= 15
    elif news.get("direction") == "positive" and news.get("level") == "high":
        score += 8
    agent_review = row.get("agent_review") or {}
    verdict = str(agent_review.get("risk_manager_verdict", "unknown"))
    consensus = float(agent_review.get("consensus_score", 50.0) or 50.0)
    if verdict == "allow_daily_candidate":
        score += 6
    elif verdict == "watch_only":
        score -= 5
    elif verdict == "block_trade":
        score -= 18
    if verdict != "unknown":
        score += max(-5.0, min(6.0, (consensus - 50.0) * 0.08))
    return round(score, 2)


def build_entry_quality(price: float, plan: dict, technical: dict, atr_pct: float = 0.0) -> dict:
    rr = plan.get("rr", {})
    long_rr = float(rr.get("long_pullback_t1", 0.0) or 0.0)
    breakout_rr = float(rr.get("long_breakout_t1", 0.0) or 0.0)
    zone = plan.get("long_pullback_zone") or []
    zone_low = float(zone[0]) if len(zone) >= 2 else price
    zone_high = float(zone[1]) if len(zone) >= 2 else price
    breakout = float(plan.get("long_breakout_trigger", price) or price)
    stop = float(plan.get("long_invalidation", price) or price)

    if zone_low <= price <= zone_high:
        plan_a_distance = 0.0
    elif price > zone_high:
        plan_a_distance = (price - zone_high) / price * 100
    else:
        plan_a_distance = (zone_low - price) / price * 100
    plan_b_distance = max((breakout - price) / price * 100, 0.0)
    stop_distance = abs(price - stop) / price * 100 if price else 0.0
    nearest_distance = min(plan_a_distance, plan_b_distance)
    atr_pct = float(atr_pct or 0.0)
    daily_trigger_pct = min(5.0, max(2.5, atr_pct * 1.25))
    near_watch_pct = min(8.0, max(daily_trigger_pct + 1.5, atr_pct * 1.8))

    plan_a_score = 50.0
    reasons: list[str] = []
    if plan_a_distance <= 3:
        plan_a_score += 22
        reasons.append("Plan A bolgesi fiyata yakin; isleme donus ihtimali iyi.")
    elif plan_a_distance <= 7:
        plan_a_score += 12
        reasons.append("Plan A bolgesi makul mesafede.")
    elif plan_a_distance <= 12:
        reasons.append("Plan A bolgesi biraz uzak; sabir gerekir.")
    else:
        plan_a_score -= 18
        reasons.append("Plan A bolgesi fiyattan uzak; bugun isleme donmeyebilir.")

    if long_rr >= 2:
        plan_a_score += 18
        reasons.append("Plan A R/R guclu.")
    elif long_rr >= 1:
        plan_a_score += 8
    elif long_rr < 0.5:
        plan_a_score -= 18
        reasons.append("Plan A R/R zayif.")

    plan_b_score = 45.0
    if plan_b_distance <= 3:
        plan_b_score += 15
        reasons.append("Plan B kirilimi yakin.")
    elif plan_b_distance <= 8:
        plan_b_score += 5
    else:
        plan_b_score -= 12
        reasons.append("Plan B kirilimi uzak.")

    if breakout_rr >= 1:
        plan_b_score += 15
    elif breakout_rr >= 0.5:
        plan_b_score += 5
    else:
        plan_b_score -= 20
        reasons.append("Plan B kirilim R/R zayif; kirilim kovalamaya uygun degil.")

    trend_label = technical.get("trend_strength", {}).get("label")
    vwap_position = technical.get("vwap", {}).get("position")
    structure = technical.get("structure", {}).get("bias")
    if trend_label == "strong_up":
        plan_a_score += 6
        plan_b_score += 6
    elif trend_label == "strong_down":
        plan_a_score -= 8
        plan_b_score -= 8
        reasons.append("Trend gucu satici lehine; kalite puani dustu.")
    if vwap_position == "above":
        plan_a_score += 5
        plan_b_score += 3
    elif vwap_position == "below":
        plan_a_score -= 6
        plan_b_score -= 4
        reasons.append("Fiyat VWAP altinda; long kalitesi dusuk.")
    if structure == "bearish":
        plan_a_score -= 6
        plan_b_score -= 6
    elif structure == "bullish":
        plan_a_score += 5
        plan_b_score += 5

    if plan_a_distance <= daily_trigger_pct and plan_a_score >= plan_b_score - 8:
        best_entry = "plan_a_pullback"
        entry_window = "bugun"
        actionable_route = "plan_a_today"
        reasons.append(f"Plan A gunluk makul mesafede: %{plan_a_distance:.1f} <= %{daily_trigger_pct:.1f}.")
    elif plan_b_distance <= daily_trigger_pct and breakout_rr >= 0.5:
        best_entry = "plan_b_breakout"
        entry_window = "bugun"
        actionable_route = "plan_b_today"
        reasons.append(f"Plan B kirilimi gunluk makul mesafede: %{plan_b_distance:.1f} <= %{daily_trigger_pct:.1f}.")
    elif nearest_distance <= near_watch_pct:
        best_entry = "plan_a_pullback" if plan_a_score >= plan_b_score else "plan_b_breakout"
        entry_window = "yakin"
        actionable_route = "radar_near"
        reasons.append(f"Giris yakinda ama bugun icin sinirda: en yakin %{nearest_distance:.1f}.")
    else:
        best_entry = "plan_a_pullback" if plan_a_score >= plan_b_score else "plan_b_breakout"
        entry_window = "uzak"
        actionable_route = "radar_far"
        reasons.append(f"Giris seviyesi uzak: en yakin %{nearest_distance:.1f}; bugun islem listesine alinmaz.")

    score = max(plan_a_score, plan_b_score)
    if entry_window == "yakin":
        score = min(score, 72)
    elif entry_window == "uzak":
        score = min(score, 55)
    if plan.get("primary") == "wait":
        score = min(score, 72)
        if best_entry == "plan_a_pullback":
            best_entry = "wait_for_plan_a"
        else:
            best_entry = "wait_for_plan_b"
    elif plan.get("primary") == "short":
        best_entry = "short_risk_or_wait"
        score = min(score, 45)

    score = max(0.0, min(100.0, score))
    return {
        "score": round(score, 1),
        "best_entry": best_entry,
        "entry_window": entry_window,
        "actionable_route": actionable_route,
        "nearest_entry_distance_pct": round(nearest_distance, 2),
        "daily_trigger_pct": round(daily_trigger_pct, 2),
        "atr_pct": round(atr_pct, 2),
        "plan_a_score": round(max(0.0, min(100.0, plan_a_score)), 1),
        "plan_b_score": round(max(0.0, min(100.0, plan_b_score)), 1),
        "plan_a_distance_pct": round(plan_a_distance, 2),
        "plan_b_distance_pct": round(plan_b_distance, 2),
        "stop_distance_pct": round(stop_distance, 2),
        "reasons": reasons[:6],
    }


def build_trade_horizon(entry_quality: dict, chase_state: dict, primary: str) -> dict:
    entry_window = str(entry_quality.get("entry_window", "unknown"))
    actionable_route = str(entry_quality.get("actionable_route", "-"))
    chase_label = str(chase_state.get("label", "normal"))
    nearest = float(entry_quality.get("nearest_entry_distance_pct", 999.0) or 999.0)
    daily_trigger = float(entry_quality.get("daily_trigger_pct", 3.0) or 3.0)

    if primary == "short":
        return {
            "trade_horizon": "bekle",
            "holding_days": "islem yok",
            "horizon_note": "Short/long tarafi net degil; yeni tarama beklenir.",
        }
    if chase_label == "kovalanmaz":
        return {
            "trade_horizon": "haftalik_radar",
            "holding_days": "max 1 hafta izleme",
            "horizon_note": "Hisse uzamis; ayni gun kovalanmaz, geri cekilme veya yeni setup beklenir.",
        }
    if primary != "long":
        if entry_window == "bugun":
            return {
                "trade_horizon": "gunluk_teyit",
                "holding_days": "1 gun teyit",
                "horizon_note": "Seviye bugune yakin ama ana karar bekle; teyit gelmeden islem degil.",
            }
        if entry_window == "yakin":
            return {
                "trade_horizon": "bir_kac_gun_teyit",
                "holding_days": "2-5 gun teyit",
                "horizon_note": "Seviye yakinda ama ana karar bekle; teyit ve yeni tarama gerekir.",
            }
        return {
            "trade_horizon": "haftalik_radar",
            "holding_days": "max 1 hafta izleme",
            "horizon_note": "Ana karar bekle; bu dogrudan islem degil, radar takibi.",
        }
    if entry_window == "bugun":
        return {
            "trade_horizon": "gunluk",
            "holding_days": "1 gun",
            "horizon_note": f"{actionable_route} bugun icin makul mesafede; teyit gelirse gunluk plan.",
        }
    if entry_window == "yakin":
        return {
            "trade_horizon": "bir_kac_gunluk",
            "holding_days": "2-5 gun",
            "horizon_note": f"Giris yakin ama tam tetikte degil; {nearest:.1f}% mesafe, sinir {daily_trigger:.1f}%.",
        }
    return {
        "trade_horizon": "haftalik_radar",
        "holding_days": "max 1 hafta izleme",
        "horizon_note": "Giris seviyesi uzak; bu dogrudan islem degil, radar/alarm takibi.",
    }


def _pct_change(series: pd.Series, periods: int) -> float:
    try:
        if len(series) <= periods:
            return 0.0
        current = float(series.iloc[-1])
        previous = float(series.iloc[-1 - periods])
        if not previous:
            return 0.0
        return (current / previous - 1) * 100
    except Exception:
        return 0.0


def build_market_regime(args) -> dict:
    symbol = str(getattr(args, "regime_symbol", "XU100.IS") or "XU100.IS").strip().upper()
    if not symbol.endswith(".IS"):
        symbol = f"{symbol}.IS"
    if getattr(args, "no_market_regime", False):
        return {
            "symbol": symbol,
            "label": "disabled",
            "bias": "neutral",
            "score": 50.0,
            "effect": "Piyasa rejimi filtresi kapali.",
            "reasons": [],
        }
    try:
        df, source = fetch_bist(symbol, str(getattr(args, "period", "1y") or "1y"), "1d")
        if len(df) < 60:
            raise RuntimeError(f"rejim verisi kisa: {len(df)}")
        df = add_indicators(df)
        latest = df.iloc[-1]
        close = float(latest["close"])
        sma20 = float(latest["sma20"]) if pd.notna(latest.get("sma20")) else close
        sma50 = float(latest["sma50"]) if pd.notna(latest.get("sma50")) else close
        rvwap = float(latest["rolling_vwap20"]) if pd.notna(latest.get("rolling_vwap20")) else close
        adx = float(latest["adx14"]) if pd.notna(latest.get("adx14")) else 0.0
        atr_pct = float(latest["atr_pct"]) if pd.notna(latest.get("atr_pct")) else 0.0
        ret1 = _pct_change(df["close"], 1)
        ret5 = _pct_change(df["close"], 5)
        ret20 = _pct_change(df["close"], 20)

        score = 50.0
        reasons: list[str] = []
        if close > sma20:
            score += 10
            reasons.append("BIST100 SMA20 ustunde.")
        else:
            score -= 10
            reasons.append("BIST100 SMA20 altinda.")
        if sma20 > sma50:
            score += 10
            reasons.append("Kisa ortalama uzun ortalamanin ustunde.")
        else:
            score -= 10
            reasons.append("Kisa ortalama uzun ortalamanin altinda.")
        if close > rvwap:
            score += 8
            reasons.append("Endeks VWAP20 ustunde.")
        else:
            score -= 8
            reasons.append("Endeks VWAP20 altinda.")
        if ret5 > 1.0:
            score += 8
            reasons.append(f"5 gunluk momentum pozitif: %{ret5:.1f}.")
        elif ret5 < -1.0:
            score -= 8
            reasons.append(f"5 gunluk momentum negatif: %{ret5:.1f}.")
        if ret20 > 2.0:
            score += 7
        elif ret20 < -2.0:
            score -= 7
        if ret1 < -2.5:
            score -= 12
            reasons.append(f"Gunluk sert satis var: %{ret1:.1f}.")
        elif ret1 > 2.0:
            score += 6
            reasons.append(f"Gunluk guclu tepki var: %{ret1:.1f}.")
        if atr_pct > 5.5:
            score -= 5
            reasons.append(f"Endeks oynakligi yuksek: ATR% {atr_pct:.1f}.")
        if adx >= 22 and close > sma20 and sma20 > sma50:
            score += 4
        elif adx >= 22 and close < sma20 and sma20 < sma50:
            score -= 4

        score = max(0.0, min(100.0, score))
        if score >= 72:
            bias = "risk_on"
            label = "guclu_piyasa"
            effect = "Long adaylar icin zemin destekleyici; kaliteli girisler yukari tasinir."
        elif score >= 55:
            bias = "neutral_positive"
            label = "pozitif_notr"
            effect = "Piyasa long icin fena degil; yine de giris seviyesi beklenir."
        elif score >= 40:
            bias = "cautious"
            label = "temkinli"
            effect = "Piyasa karisik; zayif girisler asagi suzulur."
        else:
            bias = "risk_off"
            label = "riskli_piyasa"
            effect = "Piyasa zemini zayif; long adaylar daha sert cezalandirilir."

        return {
            "symbol": symbol,
            "source": source,
            "label": label,
            "bias": bias,
            "score": round(score, 1),
            "close": round(close, 2),
            "ret1_pct": round(ret1, 2),
            "ret5_pct": round(ret5, 2),
            "ret20_pct": round(ret20, 2),
            "adx14": round(adx, 2),
            "atr_pct": round(atr_pct, 2),
            "effect": effect,
            "reasons": reasons[:7],
        }
    except Exception as exc:
        return {
            "symbol": symbol,
            "label": "unavailable",
            "bias": "neutral",
            "score": 50.0,
            "effect": f"Piyasa rejimi okunamadi; scanner neutral modda devam etti: {type(exc).__name__}: {exc}",
            "reasons": [],
        }


def apply_market_regime(result: dict, market_regime: dict) -> dict:
    result["market_regime"] = market_regime
    plan = result.get("plan", {})
    if plan is not None:
        plan["market_regime_note"] = market_regime.get("effect", "")
    return result


def sector_for_symbol(symbol: str) -> str:
    return SECTOR_MAP.get(normalize_symbol(symbol), "diger")


def build_sector_strength(results: list[dict]) -> dict:
    groups: dict[str, list[dict]] = {}
    for result in results:
        sector = sector_for_symbol(result.get("symbol", ""))
        groups.setdefault(sector, []).append(result)

    out: dict[str, dict] = {}
    for sector, items in groups.items():
        if not items:
            continue
        ret5_values = [float((x.get("momentum") or {}).get("ret5_pct", 0.0) or 0.0) for x in items]
        ret20_values = [float((x.get("momentum") or {}).get("ret20_pct", 0.0) or 0.0) for x in items]
        above_sma = [bool((x.get("momentum") or {}).get("above_sma20", False)) for x in items]
        primary_long = [str((x.get("plan") or {}).get("primary", "wait")) == "long" for x in items]
        primary_short = [str((x.get("plan") or {}).get("primary", "wait")) == "short" for x in items]
        avg_ret5 = sum(ret5_values) / len(ret5_values)
        avg_ret20 = sum(ret20_values) / len(ret20_values)
        above_ratio = sum(1 for x in above_sma if x) / len(above_sma)
        long_ratio = sum(1 for x in primary_long if x) / len(primary_long)
        short_ratio = sum(1 for x in primary_short if x) / len(primary_short)

        score = 50.0
        score += max(-18.0, min(18.0, avg_ret5 * 2.0))
        score += max(-14.0, min(14.0, avg_ret20 * 0.75))
        score += (above_ratio - 0.5) * 18.0
        score += long_ratio * 8.0
        score -= short_ratio * 10.0
        if len(items) == 1:
            score = 50.0 + (score - 50.0) * 0.65
        score = max(0.0, min(100.0, score))

        if score >= 72:
            label = "guclu_sektor"
            effect = "Sektor momentumu guclu; kaliteli long adaylar desteklenir."
        elif score >= 58:
            label = "toparlanan_sektor"
            effect = "Sektor ortalama ustu; giris kalitesi iyi olanlar izlenir."
        elif score >= 44:
            label = "notr_sektor"
            effect = "Sektor karisik; hisse bazli teyit onemli."
        elif score >= 32:
            label = "zayif_sektor"
            effect = "Sektor zayif; long adaylar secici suzulur."
        else:
            label = "cok_zayif_sektor"
            effect = "Sektor baski altinda; long kovalamak riskli."

        out[sector] = {
            "sector": sector,
            "label": label,
            "score": round(score, 1),
            "count": len(items),
            "avg_ret5_pct": round(avg_ret5, 2),
            "avg_ret20_pct": round(avg_ret20, 2),
            "above_sma20_ratio": round(above_ratio, 2),
            "effect": effect,
        }
    return out


def apply_sector_strength(result: dict, sector_strength: dict) -> dict:
    sector = sector_for_symbol(result.get("symbol", ""))
    data = sector_strength.get(sector) or {
        "sector": sector,
        "label": "unknown",
        "score": 50.0,
        "count": 0,
        "effect": "Sektor gucu hesaplanamadi.",
    }
    result["sector"] = sector
    result["sector_strength"] = data
    plan = result.get("plan", {})
    if plan is not None:
        plan["sector_strength_note"] = data.get("effect", "")
    return result


def build_news_quality(news_impact: dict, pre_news: dict) -> dict:
    news_impact = news_impact or {}
    pre_news = pre_news or {}
    level = str(news_impact.get("level", "none"))
    direction = str(news_impact.get("direction", "neutral"))
    events = list(news_impact.get("important_events", []) or [])
    raw_count = int(news_impact.get("raw_event_count", 0) or 0)
    kap_count = sum(1 for event in events if event.get("source") == "KAP")
    high_count = sum(1 for event in events if event.get("impact_level") == "high")
    enabled = bool(news_impact.get("enabled", False))
    effect_text = str(news_impact.get("effect", ""))

    if not enabled and "atlandi" in effect_text.lower():
        return {
            "label": "haber_taranmadi",
            "score": 50.0,
            "gate": "not_checked",
            "direction": "neutral",
            "summary": "Haber taramasi bu geciste calismadi; teknik ve on fiyatlama ile devam edildi.",
            "reasons": [],
            "important_count": 0,
            "kap_count": 0,
            "raw_event_count": 0,
        }

    score = 56.0
    label = "temiz"
    gate = "temiz"
    reasons: list[str] = []

    if level in {"high", "medium"} and direction in {"negative", "mixed"}:
        score = 18.0 if level == "high" else 28.0
        label = "haber_riski"
        gate = "risk_freni"
        reasons.append("Negatif/karisik etkili haber var; long plani teyitsiz acilmaz.")
    elif level == "high" and direction == "positive":
        score = 72.0
        label = "pozitif_katalizor"
        gate = "katalizor_var"
        reasons.append("Pozitif katalizor var; fiyat kovalamadan retest/kirilim teyidi beklenir.")
    elif level == "medium" and direction == "positive":
        score = 64.0
        label = "pozitif_izleme"
        gate = "teyit_bekle"
        reasons.append("Orta seviye pozitif haber var; teknik teyit kalitesi onemli.")
    elif direction == "watch":
        score = 46.0
        label = "izleme_haberi"
        gate = "teyit_bekle"
        reasons.append("Fiyat etkisi dogurabilecek izleme haberi var.")
    elif not events:
        label = "kritik_haber_yok"
        gate = "temiz"
        score = 58.0 if raw_count else 55.0
        reasons.append("Raporlanacak kadar fiyat etkili haber/KAP bulunmadi.")

    if kap_count:
        reasons.append(f"KAP kaynakli etkili olay sayisi: {kap_count}.")
        if gate == "katalizor_var":
            score += 5
        elif gate == "risk_freni":
            score -= 5
    if high_count >= 2 and gate == "risk_freni":
        score -= 5
        reasons.append("Birden fazla yuksek etkili olay var; risk artiyor.")
    if raw_count >= 8 and not events:
        score -= 3
        reasons.append("Haber kalabaligi var ama fiyat etkili olay suzulmedi.")

    pre_level = str(pre_news.get("level", "none"))
    pre_direction = str(pre_news.get("direction", "neutral"))
    if pre_level in {"high", "medium"} and pre_direction in {"negative", "mixed"}:
        score -= 10
        if gate == "temiz":
            label = "on_fiyat_riski"
            gate = "teyit_bekle"
        reasons.append("Haber oncesi negatif/karisik fiyatlama izi var.")
    elif pre_level == "high" and pre_direction == "positive":
        score += 4
        if gate == "temiz":
            label = "pozitif_on_fiyat"
            gate = "teyit_bekle"
        reasons.append("Pozitif on fiyatlama izi var; haber/kirilim teyidi beklenir.")

    score = max(0.0, min(100.0, score))
    return {
        "label": label,
        "score": round(score, 1),
        "gate": gate,
        "direction": direction,
        "summary": news_impact.get("effect", "Kritik fiyat etkili haber bulunmadi."),
        "reasons": reasons[:7],
        "important_count": len(events),
        "kap_count": kap_count,
        "raw_event_count": raw_count,
    }


def build_trade_alerts(result: dict) -> list[dict]:
    symbol = str(result.get("symbol", "-"))
    price = float(result.get("price", 0.0) or 0.0)
    plan = result.get("plan") or {}
    entry_quality = result.get("entry_quality") or {}
    news_quality = result.get("news_quality") or {}
    market_regime = result.get("market_regime") or {}
    sector_strength = result.get("sector_strength") or {}
    alerts: list[dict] = []

    def add_alert(kind: str, urgency: str, short: str, detail: str) -> None:
        alerts.append({"symbol": symbol, "kind": kind, "urgency": urgency, "short": short, "detail": detail})

    plan_a_distance = float(entry_quality.get("plan_a_distance_pct", 999.0) or 999.0)
    plan_b_distance = float(entry_quality.get("plan_b_distance_pct", 999.0) or 999.0)
    zone = plan.get("long_pullback_zone") or []
    breakout = plan.get("long_breakout_trigger")
    stop = plan.get("long_invalidation")
    short_trigger = plan.get("short_trigger")
    best_entry = str(entry_quality.get("best_entry", "-"))

    if plan_a_distance <= 1.5 and zone:
        add_alert(
            "plan_a_near",
            "high",
            "Plan A bolgesinde",
            f"{symbol} fiyat Plan A geri cekilme bolgesine cok yakin; tepki mumu/VWAP teyidi bekle.",
        )
    elif plan_a_distance <= 4.0 and zone and best_entry in {"plan_a_pullback", "wait_for_plan_a"}:
        add_alert(
            "plan_a_watch",
            "medium",
            "Plan A yaklasiyor",
            f"{symbol} Plan A bolgesine yaklasiyor; acele etmeden seviye ve mum teyidi izle.",
        )

    if plan_b_distance <= 1.5 and breakout:
        add_alert(
            "plan_b_near",
            "high",
            "Plan B kirilim yakin",
            f"{symbol} kirilim tetigine yakin; hacimli kapanis yoksa kovalamak yok.",
        )
    elif plan_b_distance <= 4.0 and breakout and best_entry in {"plan_b_breakout", "wait_for_plan_b"}:
        add_alert(
            "plan_b_watch",
            "medium",
            "Plan B izlenir",
            f"{symbol} kirilim seviyesine yaklasiyor; hacim ve kapanis teyidi ara.",
        )

    try:
        stop_distance = abs(price - float(stop)) / price * 100 if price and stop else 999.0
    except Exception:
        stop_distance = 999.0
    try:
        short_distance = abs(price - float(short_trigger)) / price * 100 if price and short_trigger else 999.0
    except Exception:
        short_distance = 999.0
    if stop_distance <= 2.0:
        add_alert("stop_near", "high", "Stop/iptal yakin", f"{symbol} long iptal seviyesine cok yakin; yeni long icin savunma modu.")
    if short_distance <= 2.0:
        add_alert("short_risk_near", "medium", "Short-risk tetigi yakin", f"{symbol} short-risk tetigine yakin; destek kirilirsa long zayiflar.")

    gate = str(news_quality.get("gate", "temiz"))
    if gate == "risk_freni":
        add_alert("news_risk", "high", "Haber freni", f"{symbol} haber/KAP riski nedeniyle teyitsiz islem riskli.")
    elif gate == "teyit_bekle":
        add_alert("news_watch", "medium", "Haber teyidi bekle", f"{symbol} haber/on fiyatlama izleme modunda; seviye teyidi olmadan girme.")

    if market_regime.get("bias") == "risk_off":
        add_alert("market_risk", "medium", "Piyasa riskli", "BIST100 zemini risk_off; long adaylarda daha secici ol.")
    if float(sector_strength.get("score", 50.0) or 50.0) < 35:
        add_alert("sector_weak", "medium", "Sektor zayif", f"{symbol} sektoru zayif; pozisyon boyu ve teyit kalitesi onemli.")

    priority = {"high": 0, "medium": 1, "low": 2}
    alerts.sort(key=lambda item: priority.get(item.get("urgency", "low"), 2))
    return alerts[:5]


def build_agent_review(result: dict) -> dict:
    plan = result.get("plan") or {}
    technical = result.get("technical") or {}
    news_quality = result.get("news_quality") or {}
    fundamental = result.get("fundamental") or {}
    market_regime = result.get("market_regime") or {}
    sector_strength = result.get("sector_strength") or {}
    entry_quality = result.get("entry_quality") or {}
    chase_state = result.get("chase_state") or {}
    horizon = result.get("trade_horizon") or {}

    def number(value: object, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def vote(agent: str, decision: str, score: float, reason: str, veto: bool = False) -> dict:
        return {
            "agent": agent,
            "decision": decision,
            "score": round(max(0.0, min(100.0, score)), 1),
            "reason": reason,
            "veto": bool(veto),
        }

    symbol = str(result.get("symbol", "-"))
    primary = str(plan.get("primary", "wait"))
    class_name = str(result.get("class_name", ""))
    technical_score = number(result.get("technical_score"), 0.0)
    structure = str(technical.get("structure", {}).get("bias", "unknown"))
    trend = str(technical.get("trend_strength", {}).get("label", "unknown"))
    vwap = str(technical.get("vwap", {}).get("position", "unknown"))
    eq_score = number(entry_quality.get("score"), 0.0)
    entry_window = str(entry_quality.get("entry_window", "unknown"))
    trade_horizon = str(horizon.get("trade_horizon", "unknown"))
    news_gate = str(news_quality.get("gate", "temiz"))
    news_score = number(news_quality.get("score"), 50.0)
    fund_score = number(fundamental.get("score"), 50.0) if fundamental else 50.0
    fund_label = str(fundamental.get("label", "not_run")) if fundamental else "not_run"
    market_bias = str(market_regime.get("bias", "neutral"))
    sector_score = number(sector_strength.get("score"), 50.0)
    chase_label = str(chase_state.get("label", "normal"))

    votes: list[dict] = []
    if class_name in {"Riskli", "Uzak dur"} or primary == "short" or (structure == "bearish" and trend == "strong_down"):
        votes.append(vote("technical", "veto", 24.0, f"{class_name}/{primary}/{structure}/{trend}", True))
    elif primary == "long" and technical_score >= 60 and structure != "bearish":
        boost = 7.0 if vwap == "above" else 0.0
        votes.append(vote("technical", "approve", 70.0 + boost, f"{structure}, {trend}, VWAP {vwap}"))
    else:
        votes.append(vote("technical", "wait", 48.0, f"{structure}, {trend}, ana senaryo {primary}"))

    if primary == "long" and entry_window == "bugun" and trade_horizon == "gunluk" and eq_score >= 65:
        votes.append(vote("entry", "approve", min(88.0, eq_score + 8.0), "bugun ve 1 haftadan kisa giris penceresi var"))
    elif primary == "long" and entry_window == "bugun" and trade_horizon == "gunluk" and eq_score >= 55:
        votes.append(vote("entry", "wait", eq_score, "bugun yakin ama teyit kalitesi sinirda"))
    else:
        votes.append(vote("entry", "veto", max(15.0, min(eq_score, 45.0)), f"bugun islem degil: {entry_window}/{trade_horizon}", True))

    if news_gate == "risk_freni":
        votes.append(vote("news_kap", "veto", news_score, "haber/KAP riski long planini kilitledi", True))
    elif news_gate == "not_checked":
        votes.append(vote("news_kap", "wait", 50.0, "haber/KAP bu geciste taranmadi"))
    elif news_gate == "teyit_bekle":
        votes.append(vote("news_kap", "wait", news_score, "haber/on fiyatlama teyidi bekleniyor"))
    else:
        votes.append(vote("news_kap", "approve", max(58.0, news_score), f"haber kapisi {news_gate}"))

    if fundamental:
        if fund_score < 30:
            votes.append(vote("fundamental", "veto", fund_score, f"temel zayif: {fund_label}", True))
        elif fund_score >= 65:
            votes.append(vote("fundamental", "approve", fund_score, f"temel destek: {fund_label}"))
        else:
            votes.append(vote("fundamental", "wait", fund_score, f"temel not: {fund_label}"))
    else:
        votes.append(vote("fundamental", "wait", 50.0, "temel veri bu geciste yok"))

    if market_bias == "risk_off" and sector_score < 45:
        votes.append(vote("market_sector", "veto", 28.0, f"piyasa {market_bias}, sektor {sector_score:.1f}", True))
    elif sector_score < 35:
        votes.append(vote("market_sector", "veto", sector_score, f"sektor cok zayif: {sector_score:.1f}", True))
    elif market_bias in {"risk_on", "neutral_positive"} and sector_score >= 55:
        votes.append(vote("market_sector", "approve", min(82.0, 50.0 + sector_score * 0.45), f"piyasa {market_bias}, sektor {sector_score:.1f}"))
    else:
        votes.append(vote("market_sector", "wait", max(35.0, min(65.0, sector_score)), f"piyasa {market_bias}, sektor {sector_score:.1f}"))

    if chase_label == "kovalanmaz":
        votes.append(vote("chase", "veto", 20.0, "fiyat kacmis; kovalamak yasak", True))
    elif chase_label == "geri_cekilme_bekle":
        votes.append(vote("chase", "wait", 45.0, "geri cekilme bekle"))
    else:
        votes.append(vote("chase", "approve", 68.0, "kovalama riski normal"))

    scores = [number(item.get("score"), 0.0) for item in votes]
    consensus = sum(scores) / len(scores) if scores else 0.0
    vetoes = [item for item in votes if item.get("veto")]
    approvals = sum(1 for item in votes if item.get("decision") == "approve")
    daily_requirements = (
        primary == "long"
        and entry_window == "bugun"
        and trade_horizon == "gunluk"
        and eq_score >= 55
        and chase_label == "normal"
        and class_name not in {"Riskli", "Uzak dur"}
        and news_gate not in {"risk_freni", "not_checked"}
    )

    if vetoes:
        verdict = "block_trade"
        note = "Veto var; islem onerisi degil. " + "; ".join(
            f"{item['agent']}: {item['reason']}" for item in vetoes[:3]
        )
    elif daily_requirements and consensus >= 60.0 and approvals >= 3:
        verdict = "allow_daily_candidate"
        note = f"{symbol} gunluk aday olabilir; yine de seviye/mum/hacim teyidi sart."
    else:
        verdict = "watch_only"
        note = "Izleme/radar; bugun islem onerisi gibi sunma."

    vote_summary = "; ".join(f"{item['agent']}={item['decision']}({item['score']:.1f})" for item in votes)
    veto_summary = "; ".join(f"{item['agent']}: {item['reason']}" for item in vetoes) or "veto yok"
    return {
        "risk_manager_verdict": verdict,
        "consensus_score": round(consensus, 1),
        "risk_manager_note": note[:260],
        "vote_summary": vote_summary[:300],
        "veto_summary": veto_summary[:260],
        "agents": votes,
    }


def build_chase_state(entry_quality: dict, momentum: dict, plan: dict, technical: dict) -> dict:
    ret5 = float(momentum.get("ret5_pct", 0.0) or 0.0)
    ret20 = float(momentum.get("ret20_pct", 0.0) or 0.0)
    plan_a_distance = float(entry_quality.get("plan_a_distance_pct", 999.0) or 999.0)
    nearest = float(entry_quality.get("nearest_entry_distance_pct", 999.0) or 999.0)
    daily_trigger = float(entry_quality.get("daily_trigger_pct", 3.0) or 3.0)
    entry_window = str(entry_quality.get("entry_window", "unknown"))
    trend = str(technical.get("trend_strength", {}).get("label", "unknown"))
    vwap = str(technical.get("vwap", {}).get("position", "unknown"))
    primary = str(plan.get("primary", "wait"))

    risk = 0.0
    reasons: list[str] = []
    if ret5 > 8:
        risk += 28
        reasons.append(f"5 gunluk hareket yuksek: %{ret5:.1f}")
    elif ret5 > 5:
        risk += 14
        reasons.append(f"5 gunluk hareket hizli: %{ret5:.1f}")
    if ret20 > 22:
        risk += 25
        reasons.append(f"20 gunluk hareket sismis: %{ret20:.1f}")
    elif ret20 > 14:
        risk += 12
    if plan_a_distance > daily_trigger * 2:
        risk += 22
        reasons.append(f"Plan A gunluk sinirin cok disinda: %{plan_a_distance:.1f}")
    elif plan_a_distance > daily_trigger:
        risk += 12
        reasons.append(f"Plan A bugun icin uzak: %{plan_a_distance:.1f}")
    if nearest > daily_trigger * 1.8:
        risk += 12
    if trend == "strong_up" and vwap == "above" and entry_window == "uzak":
        risk += 10
        reasons.append("guc var ama giris kacmis")
    if primary == "wait":
        risk += 4

    risk = max(0.0, min(100.0, risk))
    if risk >= 58:
        label = "kovalanmaz"
    elif risk >= 35:
        label = "geri_cekilme_bekle"
    else:
        label = "normal"

    return {
        "label": label,
        "score": round(risk, 1),
        "reason": ", ".join(reasons[:5]) or "kovalama riski dusuk",
    }


def scan_symbol(symbol: str, args, include_fundamental: bool = False, include_news: bool = False) -> dict:
    ticker = normalize_symbol(symbol)
    df, source = fetch_bist(f"{ticker}.IS", args.period, args.timeframe)
    if len(df) < args.min_rows:
        raise RuntimeError(f"veri kisa: {len(df)}")
    df = add_indicators(df)
    technical = build_technical_context(df, "bist")
    technical_score, class_name, bias, setup, evidence, trigger, invalidation, risk, confidence = classify(df)
    plan = build_trade_plan(df, "bist", technical_score, None, technical)
    pre_news = build_pre_news_signal(df, "bist")
    plan = apply_pre_news_to_plan(plan, pre_news)

    news_impact = {
        "enabled": False,
        "level": "none",
        "direction": "neutral",
        "effect": "Haber taramasi scanner icin atlandi.",
        "important_events": [],
    }
    if include_news:
        news_impact = build_news_impact("bist", f"{ticker}.IS", args.news_days, args.kap_days)
        plan = apply_news_to_plan(plan, news_impact)
    news_quality = build_news_quality(news_impact, pre_news)
    atr_pct = float(df["atr_pct"].iloc[-1]) if pd.notna(df["atr_pct"].iloc[-1]) else 0.0
    momentum = {
        "ret5_pct": round(_pct_change(df["close"], 5), 2),
        "ret20_pct": round(_pct_change(df["close"], 20), 2),
        "above_sma20": bool(df["close"].iloc[-1] > df["sma20"].iloc[-1]) if pd.notna(df["sma20"].iloc[-1]) else False,
    }
    entry_quality = build_entry_quality(float(df["close"].iloc[-1]), plan, technical, atr_pct=atr_pct)
    chase_state = build_chase_state(entry_quality, momentum, plan, technical)
    horizon = build_trade_horizon(entry_quality, chase_state, str(plan.get("primary", "wait")))

    fundamental = None
    if include_fundamental:
        fundamental = fetch_fundamental_snapshot(f"{ticker}.IS")

    return {
        "symbol": ticker,
        "source": source,
        "price": float(df["close"].iloc[-1]),
        "atr_pct": round(atr_pct, 2),
        "momentum": momentum,
        "technical_score": int(technical_score),
        "class_name": class_name,
        "bias": bias,
        "setup": setup,
        "confidence": confidence,
        "evidence": evidence,
        "trigger": trigger,
        "invalidation": invalidation,
        "risk": risk,
        "plan": plan,
        "technical": technical,
        "pre_news": pre_news,
        "news_impact": news_impact,
        "news_quality": news_quality,
        "entry_quality": entry_quality,
        "chase_state": chase_state,
        "trade_horizon": horizon,
        "fundamental": fundamental,
    }


def row_from_result(result: dict) -> ScanRow:
    plan = result["plan"]
    tech = result["technical"]
    news = result.get("news_impact") or {}
    pre_news = result.get("pre_news") or {}
    news_quality = result.get("news_quality") or {}
    fundamental = result.get("fundamental") or {}
    entry_quality = result.get("entry_quality") or {}
    chase_state = result.get("chase_state") or {}
    horizon = result.get("trade_horizon") or {}
    market_regime = result.get("market_regime") or {}
    sector_strength = result.get("sector_strength") or {}
    agent_review = result.get("agent_review") or {}
    alerts = result.get("alerts") or []
    alert_summary = "; ".join(str(a.get("short", "")) for a in alerts[:3] if a.get("short")) or "alarm yok"
    momentum = result.get("momentum") or {}
    rr = plan.get("rr", {})
    row_data = ScanRow(
        symbol=result["symbol"],
        price=float(result["price"]),
        scanner_score=float(result["scanner_score"]),
        technical_score=int(result["technical_score"]),
        class_name=result["class_name"],
        primary=str(plan.get("primary", "wait")),
        action=str(plan.get("action", "")),
        long_rr=float(rr.get("long_pullback_t1", 0.0)),
        short_rr=float(rr.get("short_t1", 0.0)),
        ret5_pct=float(momentum.get("ret5_pct", 0.0) or 0.0),
        ret20_pct=float(momentum.get("ret20_pct", 0.0) or 0.0),
        structure=str(tech.get("structure", {}).get("bias", "unknown")),
        bos=str(tech.get("structure", {}).get("bos", "none")),
        adx=float(tech.get("trend_strength", {}).get("adx14", 0.0) or 0.0),
        trend_label=str(tech.get("trend_strength", {}).get("label", "unknown")),
        vwap_position=str(tech.get("vwap", {}).get("position", "unknown")),
        squeeze=bool(tech.get("squeeze", {}).get("on")),
        pre_news_level=str(pre_news.get("level", "none")),
        pre_news_direction=str(pre_news.get("direction", "neutral")),
        news_level=str(news.get("level", "none")),
        news_direction=str(news.get("direction", "neutral")),
        news_quality_label=str(news_quality.get("label", "unknown")),
        news_quality_score=float(news_quality.get("score", 50.0) or 50.0),
        news_gate=str(news_quality.get("gate", "temiz")),
        fundamental_label=str(fundamental.get("label", "not_run")) if fundamental else "not_run",
        fundamental_score=int(fundamental.get("score", 0)) if fundamental else 0,
        entry_quality_score=float(entry_quality.get("score", 0.0) or 0.0),
        best_entry=str(entry_quality.get("best_entry", "-")),
        entry_window=str(entry_quality.get("entry_window", "unknown")),
        actionable_route=str(entry_quality.get("actionable_route", "-")),
        trade_horizon=str(horizon.get("trade_horizon", "unknown")),
        holding_days=str(horizon.get("holding_days", "-")),
        nearest_entry_distance_pct=float(entry_quality.get("nearest_entry_distance_pct", 0.0) or 0.0),
        daily_trigger_pct=float(entry_quality.get("daily_trigger_pct", 0.0) or 0.0),
        atr_pct=float(entry_quality.get("atr_pct", result.get("atr_pct", 0.0)) or 0.0),
        chase_label=str(chase_state.get("label", "normal")),
        chase_score=float(chase_state.get("score", 0.0) or 0.0),
        chase_reason=str(chase_state.get("reason", "")),
        plan_a_distance_pct=float(entry_quality.get("plan_a_distance_pct", 0.0) or 0.0),
        plan_b_distance_pct=float(entry_quality.get("plan_b_distance_pct", 0.0) or 0.0),
        market_regime_label=str(market_regime.get("label", "unknown")),
        market_regime_bias=str(market_regime.get("bias", "neutral")),
        market_regime_score=float(market_regime.get("score", 50.0) or 50.0),
        sector=str(result.get("sector") or sector_strength.get("sector") or sector_for_symbol(result["symbol"])),
        sector_strength_label=str(sector_strength.get("label", "unknown")),
        sector_strength_score=float(sector_strength.get("score", 50.0) or 50.0),
        prep_score=0.0,
        prep_label="unknown",
        prep_reason="",
        agent_consensus_score=float(agent_review.get("consensus_score", 0.0) or 0.0),
        agent_verdict=str(agent_review.get("risk_manager_verdict", "unknown")),
        agent_votes=str(agent_review.get("vote_summary", ""))[:240],
        agent_vetoes=str(agent_review.get("veto_summary", ""))[:240],
        risk_manager_note=str(agent_review.get("risk_manager_note", ""))[:240],
        alert_summary=alert_summary[:180],
        reason=str(plan.get("reason", ""))[:240],
    )
    prep_score, prep_label, prep_reason = setup_prep_score(row_data)
    row_data.prep_score = prep_score
    row_data.prep_label = prep_label
    row_data.prep_reason = prep_reason
    return row_data


def entry_ready_score(row: ScanRow | dict) -> float:
    if isinstance(row, dict):
        get = row.get
    else:
        data = asdict(row)
        get = data.get

    entry_quality = float(get("entry_quality_score", 0.0) or 0.0)
    scanner_score = float(get("scanner_score", 0.0) or 0.0)
    plan_a_distance = float(get("plan_a_distance_pct", 999.0) or 999.0)
    plan_b_distance = float(get("plan_b_distance_pct", 999.0) or 999.0)
    long_rr = float(get("long_rr", 0.0) or 0.0)
    sector_score = float(get("sector_strength_score", 50.0) or 50.0)
    market_bias = str(get("market_regime_bias", "neutral"))
    news_gate = str(get("news_gate", "temiz"))
    primary = str(get("primary", "wait"))
    alert = str(get("alert_summary", ""))
    best_entry = str(get("best_entry", ""))
    entry_window = str(get("entry_window", "unknown"))
    chase_label = str(get("chase_label", "normal"))
    agent_verdict = str(get("agent_verdict", "unknown"))
    agent_consensus = float(get("agent_consensus_score", 50.0) or 50.0)

    score = entry_quality * 0.50
    score += min(max(scanner_score, 0.0), 120.0) / 120.0 * 20.0
    score += min(max(long_rr, 0.0), 4.0) / 4.0 * 10.0

    if plan_a_distance <= 1.5:
        score += 12
    elif plan_a_distance <= 4:
        score += 9
    elif plan_a_distance <= 8:
        score += 6
    elif plan_a_distance <= 12:
        score += 2
    elif plan_a_distance > 15:
        score -= 8

    if plan_b_distance <= 1.5:
        score += 8
    elif plan_b_distance <= 4:
        score += 5
    elif plan_b_distance <= 8:
        score += 2
    elif plan_b_distance > 12 and best_entry in {"plan_b_breakout", "wait_for_plan_b"}:
        score -= 8

    if primary == "long":
        score += 5
    elif primary == "wait":
        score -= 2
    elif primary == "short":
        score -= 18

    if news_gate == "risk_freni":
        score -= 30
    elif news_gate == "teyit_bekle":
        score -= 8
    elif news_gate == "katalizor_var":
        score += 4
    elif news_gate == "temiz":
        score += 3

    if market_bias == "risk_off":
        score -= 10
    elif market_bias == "cautious":
        score -= 3
    elif market_bias == "risk_on":
        score += 4

    score += max(-6.0, min(6.0, (sector_score - 50.0) * 0.12))

    if "Haber freni" in alert:
        score -= 15
    if "Plan A bolgesinde" in alert or "Plan A yaklasiyor" in alert:
        score += 5
    if "Plan B kirilim yakin" in alert:
        score += 4
    if entry_window != "bugun":
        score -= 35
    if chase_label == "kovalanmaz":
        score -= 35
    elif chase_label == "geri_cekilme_bekle":
        score -= 10

    if agent_verdict == "allow_daily_candidate":
        score += 8
    elif agent_verdict == "watch_only":
        score -= 15
    elif agent_verdict == "block_trade":
        score -= 50
    if agent_verdict != "unknown":
        score += max(-5.0, min(8.0, (agent_consensus - 50.0) * 0.12))

    return round(max(0.0, min(100.0, score)), 1)


def entry_ready_rows(rows: list[ScanRow], top: int = 5) -> list[ScanRow]:
    candidates = [
        row
        for row in rows
        if row.primary == "long"
        and row.news_gate != "risk_freni"
        and row.entry_quality_score >= 55
        and row.entry_window == "bugun"
        and row.trade_horizon == "gunluk"
        and row.chase_label != "kovalanmaz"
        and row.class_name not in {"Riskli", "Uzak dur"}
        and row.scanner_score >= 55
        and row.news_gate != "not_checked"
        and row.agent_verdict == "allow_daily_candidate"
    ]
    return sorted(candidates, key=entry_ready_score, reverse=True)[:top]


def setup_prep_score(row: ScanRow | dict) -> tuple[float, str, str]:
    if isinstance(row, dict):
        get = row.get
    else:
        data = asdict(row)
        get = data.get

    ret5 = float(get("ret5_pct", 0.0) or 0.0)
    ret20 = float(get("ret20_pct", 0.0) or 0.0)
    scanner_score = float(get("scanner_score", 0.0) or 0.0)
    entry_quality = float(get("entry_quality_score", 0.0) or 0.0)
    nearest = float(get("nearest_entry_distance_pct", 999.0) or 999.0)
    plan_b = float(get("plan_b_distance_pct", 999.0) or 999.0)
    sector_score = float(get("sector_strength_score", 50.0) or 50.0)
    atr_pct = float(get("atr_pct", 0.0) or 0.0)
    structure = str(get("structure", "unknown"))
    trend = str(get("trend_label", "unknown"))
    vwap = str(get("vwap_position", "unknown"))
    primary = str(get("primary", "wait"))
    news_gate = str(get("news_gate", "temiz"))
    market_bias = str(get("market_regime_bias", "neutral"))
    chase_label = str(get("chase_label", "normal"))
    squeeze = bool(get("squeeze", False))
    class_name = str(get("class_name", ""))

    score = 40.0
    reasons: list[str] = []

    if -4.0 <= ret5 <= 4.0:
        score += 16
        reasons.append("son 5 gun sakin")
    elif 4.0 < ret5 <= 8.0:
        score += 6
        reasons.append("son 5 gun guclu ama henuz asiri degil")
    elif ret5 > 8.0:
        score -= 22
        reasons.append("son 5 gun fazla kosmus")
    elif ret5 < -8.0:
        score -= 12
        reasons.append("son 5 gun sert zayif")

    if -8.0 <= ret20 <= 14.0:
        score += 10
    elif ret20 > 22.0:
        score -= 15
        reasons.append("20 gunluk hareket sismis")
    elif ret20 < -15.0:
        score -= 8
        reasons.append("20 gunluk zemin zayif")

    if squeeze:
        score += 14
        reasons.append("sikisma var")
    if structure in {"bullish", "mixed", "range"}:
        score += 8
    elif structure == "bearish":
        score -= 8
    if trend in {"range", "weak"}:
        score += 8
        reasons.append("patlamadan onceki zayif/range trend")
    elif trend == "strong_up" and ret5 <= 6:
        score += 4
    elif trend == "strong_up" and ret5 > 8:
        score -= 8
    elif trend == "strong_down":
        score -= 12

    if vwap == "above":
        score += 5
    elif vwap == "below" and structure == "bullish":
        score += 2
    elif vwap == "below":
        score -= 5

    if 2.0 <= nearest <= 8.0:
        score += 10
        reasons.append("giris alanina makul radar mesafesi")
    elif nearest < 2.0 and primary == "wait":
        score += 5
    elif nearest > 12.0:
        score -= 10
        reasons.append("kurulum seviyesi uzak")

    if 3.0 <= plan_b <= 10.0:
        score += 7
        reasons.append("kirilim tetigi izlenebilir mesafede")
    elif plan_b > 16.0:
        score -= 6

    score += min(max(scanner_score, 0.0), 100.0) * 0.08
    score += min(max(entry_quality, 0.0), 80.0) * 0.08
    score += max(-6.0, min(7.0, (sector_score - 50.0) * 0.14))

    if 2.0 <= atr_pct <= 6.5:
        score += 5
    elif atr_pct > 9.0:
        score -= 8
        reasons.append("oynaklik fazla")

    if primary == "short":
        score -= 25
    if class_name == "Uzak dur":
        score -= 18
    elif class_name == "Riskli":
        score -= 8
    if news_gate == "risk_freni":
        score -= 25
        reasons.append("haber freni var")
    elif news_gate == "teyit_bekle":
        score -= 5
    if market_bias == "risk_off":
        score -= 8
        reasons.append("piyasa risk_off")
    elif market_bias == "risk_on":
        score += 4
    if chase_label == "kovalanmaz":
        score -= 30
        reasons.append("kovalanmaz etiketi var")
    elif chase_label == "geri_cekilme_bekle":
        score -= 8

    score = round(max(0.0, min(100.0, score)), 1)
    if score >= 72:
        label = "hazirlik_guclu"
    elif score >= 60:
        label = "hazirlik_izle"
    elif score >= 48:
        label = "radar_zayif"
    else:
        label = "hazirlik_yok"
    reason = ", ".join(reasons[:5]) or "normal teknik radar"
    return score, label, reason


def setup_prep_rows(rows: list[ScanRow], top: int = 5) -> list[ScanRow]:
    candidates = []
    for row in rows:
        score, label, _ = setup_prep_score(row)
        if (
            score >= 60
            and label != "hazirlik_yok"
            and row.news_gate != "risk_freni"
            and row.primary != "short"
            and row.trade_horizon != "gunluk"
            and row.chase_label != "kovalanmaz"
            and row.ret5_pct <= 8.0
            and row.ret20_pct <= 22.0
            and row.class_name not in {"Riskli", "Uzak dur"}
        ):
            candidates.append(row)
    return sorted(candidates, key=lambda item: setup_prep_score(item)[0], reverse=True)[:top]


def build_trade_gate(ready_rows_: list[ScanRow], prep_rows_: list[ScanRow], general_rows_: list[ScanRow]) -> dict:
    if ready_rows_:
        return {
            "status": "ACTIONABLE",
            "actionable_count": len(ready_rows_),
            "message": "Bugun girise yakin aday var. Sadece Bugun Isleme En Yakin tablosu islem adayi sayilir.",
            "rule": "Hazirlik/Radar ve Genel Guc tablolari islem onerisi degildir.",
            "allowed_table": "entry_ready_rows",
            "forbidden_tables": ["setup_prep_rows", "rows"],
            "violation_if": "entry_ready_rows disindan sembol islem onerisi gibi sunulursa kural ihlalidir.",
            "radar_count": len(prep_rows_),
            "general_count": len(general_rows_),
        }
    return {
        "status": "ISLEM_YOK",
        "actionable_count": 0,
        "message": "Bugun temiz giris yok. Islem onerisi uretme; sadece radar/alarm listesi ver.",
        "rule": "Hazirlik/Radar ve Genel Guc tablolari aday degil, izleme listesidir.",
        "allowed_table": "none",
        "forbidden_tables": ["entry_ready_rows", "setup_prep_rows", "rows"],
        "violation_if": "Herhangi bir sembol bugun islem onerisi gibi sunulursa kural ihlalidir.",
        "radar_count": len(prep_rows_),
        "general_count": len(general_rows_),
    }


def build_markdown(args, rows: list[ScanRow], errors: list[dict]) -> str:
    top = rows[: min(args.top, len(rows))]
    ready = entry_ready_rows(rows, top=5)
    prep = setup_prep_rows(rows, top=5)
    trade_gate = build_trade_gate(ready, prep, top)
    market_text = "unknown"
    if rows:
        market_text = f"{rows[0].market_regime_label}/{rows[0].market_regime_bias} ({rows[0].market_regime_score:.1f})"

    def _table(source_rows: list[ScanRow], include_ready_score: bool = False) -> str:
        table_rows = []
        for i, r in enumerate(source_rows):
            ready_cell = f" | {entry_ready_score(r):.1f}" if include_ready_score else ""
            table_rows.append(
                f"| {i+1} | {r.symbol} | {r.trade_horizon}/{r.holding_days} | "
                f"{r.scanner_score:.1f}{ready_cell} | {r.primary} | {r.class_name} | "
                f"{r.entry_quality_score:.1f} | {r.entry_window}/{r.actionable_route} | "
                f"{r.nearest_entry_distance_pct:.1f} | {r.daily_trigger_pct:.1f} | "
                f"{r.chase_label}/{r.chase_score:.1f} | {r.best_entry} | {r.plan_a_distance_pct:.1f} | "
                f"{r.plan_b_distance_pct:.1f} | {r.sector} | {r.sector_strength_score:.1f} | "
                f"{r.price:.2f} | {r.long_rr:.2f} | {r.structure} | {r.trend_label} | "
                f"{r.vwap_position} | {r.pre_news_level}/{r.pre_news_direction} | "
                f"{r.news_level}/{r.news_direction} | {r.news_quality_label}/{r.news_gate}/{r.news_quality_score:.1f} | "
                f"{r.alert_summary} | {r.fundamental_label}/{r.fundamental_score} |"
            )
        return "\n".join(table_rows)

    def _prep_table(source_rows: list[ScanRow]) -> str:
        table_rows = []
        for i, r in enumerate(source_rows):
            table_rows.append(
                f"| {i+1} | {r.symbol} | {r.trade_horizon}/{r.holding_days} | "
                f"{r.prep_score:.1f} | {r.prep_label} | "
                f"{r.ret5_pct:.1f} | {r.ret20_pct:.1f} | {r.squeeze} | {r.structure} | {r.trend_label} | "
                f"{r.vwap_position} | {r.nearest_entry_distance_pct:.1f} | {r.plan_b_distance_pct:.1f} | "
                f"{r.sector} | {r.sector_strength_score:.1f} | {r.news_quality_label}/{r.news_gate} | "
                f"{r.chase_label}/{r.chase_score:.1f} | {r.prep_reason} | {r.alert_summary} |"
            )
        return "\n".join(table_rows)

    def _safe_cell(value: object, limit: int = 120) -> str:
        return str(value or "-").replace("|", "/").replace("\n", " ")[:limit]

    def _agent_table(source_rows: list[ScanRow]) -> str:
        table_rows = []
        for i, r in enumerate(source_rows[:10]):
            table_rows.append(
                f"| {i+1} | {r.symbol} | {r.agent_verdict} | {r.agent_consensus_score:.1f} | "
                f"{_safe_cell(r.agent_vetoes, 120)} | {_safe_cell(r.risk_manager_note, 160)} |"
            )
        return "\n".join(table_rows) or "| - | - | - | - | - | - |"

    table = _table(top)
    ready_table = _table(ready, include_ready_score=True)
    prep_table = _prep_table(prep)
    agent_table = _agent_table(top)
    if not ready_table:
        ready_table = "| - | Bugun temiz giris yok | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |"
    if not prep_table:
        prep_table = "| - | Hazirlik adayi yok | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |"
    if not table:
        table = "| - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |"
    error_text = "\n".join([f"- {x['symbol']}: {x['error']}" for x in errors[:20]]) or "- Yok."
    return f"""# BIST Scanner

Zaman: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Timeframe: {args.timeframe}
Period: {args.period}
Taranan: {len(rows)}
Hata: {len(errors)}
Fundamental top: {args.fundamental_top}
News top: {args.news_top}
Piyasa rejimi: {market_text}

## Gunluk Islem Kilidi

Durum: {trade_gate["status"]}
Mesaj: {trade_gate["message"]}
Kural: {trade_gate["rule"]}

## Ajan Kurulu / Risk Manager

| # | Sembol | Karar | Konsensus | Veto | Not |
|---:|---|---|---:|---|---|
{agent_table}

## En Iyi Adaylar - Genel Guc (ISLEM DEGIL / Sadece Radar)

| # | Sembol | Vade | Skor | Senaryo | Kalite | Giris kalite | Giris durumu | En yakin % | Gunluk sinir % | Kovalama | En iyi giris | Plan A uzak % | Plan B uzak % | Sektor | Sektor gucu | Fiyat | Long R/R | Structure | Trend | VWAP | On fiyat | Haber | Haber kalite | Alarm | Temel |
|---:|---|---|---:|---|---|---:|---|---:|---:|---|---|---:|---:|---|---:|---:|---:|---|---|---|---|---|---|---|---|
{table}

## Bugun Isleme En Yakin Adaylar - Giris Kalitesi

| # | Sembol | Vade | Genel skor | Girise yakin skor | Senaryo | Kalite | Giris kalite | Giris durumu | En yakin % | Gunluk sinir % | Kovalama | En iyi giris | Plan A uzak % | Plan B uzak % | Sektor | Sektor gucu | Fiyat | Long R/R | Structure | Trend | VWAP | On fiyat | Haber | Haber kalite | Alarm | Temel |
|---:|---|---|---:|---:|---|---|---:|---|---:|---:|---|---|---:|---:|---|---:|---:|---:|---|---|---|---|---|---|---|---|
{ready_table}

## Hazirlik / Patlamamis Radar (ISLEM DEGIL / Alarm Listesi)

| # | Sembol | Vade | Hazirlik skor | Etiket | 5g % | 20g % | Squeeze | Structure | Trend | VWAP | En yakin % | Kirilim uzak % | Sektor | Sektor gucu | Haber kalite | Kovalama | Neden | Alarm |
|---:|---|---|---:|---|---:|---:|---|---|---|---|---:|---:|---|---:|---|---|---|---|
{prep_table}

## Hatalar

{error_text}

Not: Scanner emir acmaz. Gunluk islem adayi sadece `Bugun Isleme En Yakin Adaylar` tablosundan gelir. Bu tablo bos ise bugun islem onerisi yoktur; Hazirlik/Radar ve Genel Guc sadece izleme/alarm listesidir.
"""


def save_outputs(args, rows: list[ScanRow], raw_results: list[dict], errors: list[dict], markdown: str) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = REPORTS_DIR / f"BIST_SCANNER_{args.timeframe}_{stamp}"
    md_path = base.with_suffix(".md")
    json_path = base.with_suffix(".json")
    csv_path = base.with_suffix(".csv")
    market_regime = next((x.get("market_regime") for x in raw_results if x.get("market_regime")), {})
    sector_strength = {}
    alerts = []
    for item in raw_results:
        data = item.get("sector_strength")
        if data and data.get("sector"):
            sector_strength[str(data["sector"])] = data
        alerts.extend(item.get("alerts") or [])
    row_dicts = []
    for row in rows:
        data = asdict(row)
        data["entry_ready_score"] = entry_ready_score(row)
        row_dicts.append(data)
    ready_scan_rows = entry_ready_rows(rows, top=5)
    prep_scan_rows = setup_prep_rows(rows, top=5)
    trade_gate = build_trade_gate(ready_scan_rows, prep_scan_rows, rows[: min(getattr(args, "top", 15), len(rows))])
    ready_dicts = []
    for row in ready_scan_rows:
        data = asdict(row)
        data["entry_ready_score"] = entry_ready_score(row)
        ready_dicts.append(data)
    prep_dicts = []
    for row in prep_scan_rows:
        data = asdict(row)
        data["entry_ready_score"] = entry_ready_score(row)
        prep_dicts.append(data)
    payload = {
        "trade_gate": trade_gate,
        "rows": row_dicts,
        "entry_ready_rows": ready_dicts,
        "setup_prep_rows": prep_dicts,
        "raw_results": raw_results,
        "errors": errors,
        "market_regime": market_regime or {},
        "sector_strength": sector_strength,
        "alerts": alerts,
    }
    md_path.write_text(markdown, encoding="utf-8")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(row_dicts).to_csv(csv_path, index=False)
    if not getattr(args, "no_desktop_note", False):
        try:
            note_path = note_path_from_args(getattr(args, "note_path", ""))
            block = build_block(payload, json_path, 5)
            prepend_to_note(note_path, block, json_path.name, allow_duplicate=False)
        except Exception as exc:
            print(f"DESKTOP_NOTE_ERROR={type(exc).__name__}: {exc}")
    if not getattr(args, "no_performance_memory", False):
        try:
            perf = update_performance_memory(
                payload,
                json_path,
                top=int(getattr(args, "performance_top", 5) or 5),
                update_days=int(getattr(args, "performance_update_days", 45) or 45),
            )
            print(
                "PERFORMANCE_MEMORY="
                f"added:{perf['added']} updated:{perf['updated']} path:{perf['memory_path']}"
            )
        except Exception as exc:
            print(f"PERFORMANCE_MEMORY_ERROR={type(exc).__name__}: {exc}")
    return md_path


def run_scan(args) -> tuple[list[ScanRow], list[dict], list[dict], str]:
    symbols = parse_symbols(args.symbols) if args.symbols else DEFAULT_BIST100.copy()
    if args.max_symbols:
        symbols = symbols[: args.max_symbols]

    market_regime = build_market_regime(args)
    first_pass: list[dict] = []
    errors: list[dict] = []
    for symbol in symbols:
        try:
            result = scan_symbol(symbol, args, include_fundamental=False, include_news=False)
            result = apply_market_regime(result, market_regime)
            result["scanner_score"] = scanner_rank(result)
            first_pass.append(result)
        except Exception as exc:
            errors.append({"symbol": symbol, "error": f"{type(exc).__name__}: {exc}"})

    sector_strength = build_sector_strength(first_pass)
    for result in first_pass:
        apply_sector_strength(result, sector_strength)
        result["alerts"] = build_trade_alerts(result)
        result["agent_review"] = build_agent_review(result)
        result["scanner_score"] = scanner_rank(result)

    first_pass.sort(key=lambda x: x["scanner_score"], reverse=True)
    enrich_symbols = {x["symbol"] for x in first_pass[: max(args.fundamental_top, args.news_top)]}

    enriched: list[dict] = []
    for result in first_pass:
        symbol = result["symbol"]
        if symbol in enrich_symbols:
            try:
                result = scan_symbol(
                    symbol,
                    args,
                    include_fundamental=symbol in {x["symbol"] for x in first_pass[: args.fundamental_top]},
                    include_news=symbol in {x["symbol"] for x in first_pass[: args.news_top]},
                )
                result = apply_market_regime(result, market_regime)
                result = apply_sector_strength(result, sector_strength)
                result["alerts"] = build_trade_alerts(result)
                result["agent_review"] = build_agent_review(result)
                result["scanner_score"] = scanner_rank(result)
            except Exception as exc:
                errors.append({"symbol": symbol, "error": f"enrich {type(exc).__name__}: {exc}"})
                continue
        enriched.append(result)

    enriched.sort(key=lambda x: x["scanner_score"], reverse=True)
    rows = [row_from_result(x) for x in enriched]
    markdown = build_markdown(args, rows, errors)
    return rows, enriched, errors, markdown


def main() -> int:
    parser = argparse.ArgumentParser(description="BIST technical + fundamental + news scanner.")
    parser.add_argument("--symbols", default="")
    parser.add_argument("--timeframe", default="1d")
    parser.add_argument("--period", default="1y")
    parser.add_argument("--min-rows", type=int, default=120)
    parser.add_argument("--max-symbols", type=int, default=30)
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--fundamental-top", type=int, default=8)
    parser.add_argument("--news-top", type=int, default=8)
    parser.add_argument("--news-days", type=int, default=3)
    parser.add_argument("--kap-days", type=int, default=14)
    parser.add_argument("--note-path", default="")
    parser.add_argument("--no-desktop-note", action="store_true")
    parser.add_argument("--performance-top", type=int, default=5)
    parser.add_argument("--performance-update-days", type=int, default=45)
    parser.add_argument("--no-performance-memory", action="store_true")
    parser.add_argument("--regime-symbol", default="XU100.IS")
    parser.add_argument("--no-market-regime", action="store_true")
    args = parser.parse_args()

    rows, raw_results, errors, markdown = run_scan(args)
    path = save_outputs(args, rows, raw_results, errors, markdown)
    print(markdown)
    print(f"\nBIST_SCANNER_PATH={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
