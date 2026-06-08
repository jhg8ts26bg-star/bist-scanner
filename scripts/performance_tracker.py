from __future__ import annotations

import json
import re
import shutil
from datetime import date, datetime
from pathlib import Path
from statistics import mean

import pandas as pd


FINANCE_ROOT = Path.home() / ".config" / "opencode" / "finance"
DATA_DIR = FINANCE_ROOT / "data"
REPORTS_DIR = FINANCE_ROOT / "reports"
MEMORY_PATH = DATA_DIR / "performance_memory.json"
SUMMARY_PATH = REPORTS_DIR / "PERFORMANCE_MEMORY.md"


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _pct(now: float, base: float) -> float:
    return ((now / base) - 1) * 100 if base else 0.0


def _parse_report_time(path: Path) -> datetime:
    match = re.search(r"(\d{8})_(\d{6})", path.name)
    if match:
        return datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S")
    return datetime.fromtimestamp(path.stat().st_mtime)


def _load_memory() -> dict:
    if not MEMORY_PATH.exists():
        return {"version": 1, "updated_at": None, "records": [], "summary": {}}
    memory = json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
    for record in memory.get("records", []):
        if not record.get("record_type"):
            record["record_type"] = "legacy_general"
        record.setdefault("trade_horizon", "unknown")
        record.setdefault("holding_days", "")
        record.setdefault("entry_window", "unknown")
        record.setdefault("actionable_route", "")
    return memory


def _save_memory(memory: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if MEMORY_PATH.exists():
        backup = MEMORY_PATH.with_suffix(".json." + datetime.now().strftime("%Y%m%d_%H%M%S") + ".bak")
        shutil.copy2(MEMORY_PATH, backup)
    tmp = MEMORY_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(memory, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(MEMORY_PATH)


def _raw_map(payload: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for item in payload.get("raw_results", []):
        symbol = str(item.get("symbol", "")).upper()
        if symbol:
            out[symbol] = item
    return out


def _make_record(row: dict, raw: dict, scanner_path: Path, rank: int, report_time: datetime, record_type: str) -> dict:
    symbol = str(row.get("symbol", "")).upper()
    plan = raw.get("plan", {}) if raw else {}
    switch = plan.get("strategy_switch", {}) if plan else {}
    return {
        "id": f"{scanner_path.name}:{record_type}:{symbol}",
        "source_report": scanner_path.name,
        "record_type": record_type,
        "rank": rank,
        "symbol": symbol,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "report_time": report_time.isoformat(timespec="seconds"),
        "report_date": report_time.date().isoformat(),
        "entry_price": _safe_float(row.get("price")),
        "primary": str(row.get("primary", "wait")),
        "trade_horizon": str(row.get("trade_horizon", "unknown")),
        "holding_days": str(row.get("holding_days", "")),
        "entry_window": str(row.get("entry_window", "unknown")),
        "actionable_route": str(row.get("actionable_route", "")),
        "scanner_score": _safe_float(row.get("scanner_score")),
        "class_name": str(row.get("class_name", "")),
        "long_rr": _safe_float(row.get("long_rr")),
        "plan": {
            "long_pullback_zone": plan.get("long_pullback_zone"),
            "long_breakout_trigger": plan.get("long_breakout_trigger"),
            "long_invalidation": plan.get("long_invalidation"),
            "long_targets": plan.get("long_targets"),
            "short_trigger": plan.get("short_trigger"),
            "short_targets": plan.get("short_targets"),
            "strategy_summary": switch.get("summary", ""),
        },
        "signals": {
            "structure": str(row.get("structure", "")),
            "trend_label": str(row.get("trend_label", "")),
            "vwap_position": str(row.get("vwap_position", "")),
            "pre_news": f"{row.get('pre_news_level', 'none')}/{row.get('pre_news_direction', 'neutral')}",
            "news": f"{row.get('news_level', 'none')}/{row.get('news_direction', 'neutral')}",
            "fundamental": f"{row.get('fundamental_label', 'not_run')}/{row.get('fundamental_score', 0)}",
        },
        "outcomes": {
            "status": "pending",
            "latest_price": None,
            "latest_pct": None,
            "day_1_pct": None,
            "day_3_pct": None,
            "day_5_pct": None,
            "plan_a_touched": False,
            "plan_b_triggered": False,
            "target_1_hit": False,
            "target_2_hit": False,
            "target_3_hit": False,
            "stop_hit": False,
            "first_target_date": None,
            "first_stop_date": None,
            "last_checked": None,
        },
        "manual": {"my_trade": "", "pnl": "", "note": ""},
    }


def add_new_records(memory: dict, payload: dict, scanner_path: Path, top: int) -> int:
    existing_ids = {item.get("id") for item in memory.get("records", [])}
    raw = _raw_map(payload)
    report_time = _parse_report_time(scanner_path)
    added = 0
    actionable_rows = payload.get("entry_ready_rows", [])[:top]
    for rank, row in enumerate(actionable_rows, start=1):
        symbol = str(row.get("symbol", "")).upper()
        if not symbol:
            continue
        record = _make_record(row, raw.get(symbol, {}), scanner_path, rank, report_time, "actionable_trade")
        if record["id"] in existing_ids:
            continue
        memory.setdefault("records", []).append(record)
        existing_ids.add(record["id"])
        added += 1
    return added


def _first_date(masked: pd.DataFrame) -> str | None:
    if masked.empty:
        return None
    return pd.to_datetime(masked["timestamp"].iloc[0]).date().isoformat()


def _update_record(record: dict, df: pd.DataFrame) -> None:
    if df.empty:
        return
    entry = _safe_float(record.get("entry_price"))
    if not entry:
        return
    rec_date = date.fromisoformat(record["report_date"])
    data = df.copy()
    data["date"] = pd.to_datetime(data["timestamp"]).dt.date
    data = data.sort_values("timestamp")
    after = data[data["date"] > rec_date].copy()
    latest = data.iloc[-1]
    outcomes = record.setdefault("outcomes", {})
    outcomes["latest_price"] = _safe_float(latest.get("close"))
    outcomes["latest_pct"] = round(_pct(outcomes["latest_price"], entry), 2)
    outcomes["last_checked"] = datetime.now().isoformat(timespec="seconds")

    for horizon in [1, 3, 5]:
        key = f"day_{horizon}_pct"
        if len(after) >= horizon:
            close = _safe_float(after.iloc[horizon - 1].get("close"))
            outcomes[key] = round(_pct(close, entry), 2)

    future = after
    plan = record.get("plan", {})
    zone = plan.get("long_pullback_zone") or []
    if len(zone) >= 2 and not future.empty:
        z_low = min(_safe_float(zone[0]), _safe_float(zone[1]))
        z_high = max(_safe_float(zone[0]), _safe_float(zone[1]))
        outcomes["plan_a_touched"] = bool(((future["low"] <= z_high) & (future["high"] >= z_low)).any())

    breakout = plan.get("long_breakout_trigger")
    if breakout is not None and not future.empty:
        trigger = _safe_float(breakout)
        outcomes["plan_b_triggered"] = bool(((future["high"] >= trigger) | (future["close"] >= trigger)).any())

    stop = plan.get("long_invalidation")
    if stop is not None and not future.empty:
        stop_value = _safe_float(stop)
        stop_rows = future[(future["low"] <= stop_value) | (future["close"] <= stop_value)]
        outcomes["stop_hit"] = bool(not stop_rows.empty)
        outcomes["first_stop_date"] = _first_date(stop_rows)

    targets = plan.get("long_targets") or []
    target_dates = []
    for idx, target in enumerate(targets[:3], start=1):
        target_value = _safe_float(target)
        hit_rows = future[future["high"] >= target_value] if not future.empty else future
        outcomes[f"target_{idx}_hit"] = bool(not hit_rows.empty)
        if idx == 1:
            outcomes["first_target_date"] = _first_date(hit_rows)
        if not hit_rows.empty:
            target_dates.append(pd.to_datetime(hit_rows["timestamp"].iloc[0]).date())

    stop_date = date.fromisoformat(outcomes["first_stop_date"]) if outcomes.get("first_stop_date") else None
    target_date = date.fromisoformat(outcomes["first_target_date"]) if outcomes.get("first_target_date") else None
    if stop_date and (not target_date or stop_date <= target_date):
        outcomes["status"] = "stop_hit"
    elif target_date:
        outcomes["status"] = "target_1_hit"
    elif outcomes.get("latest_pct") is None:
        outcomes["status"] = "pending"
    elif outcomes["latest_pct"] > 0:
        outcomes["status"] = "active_positive"
    elif outcomes["latest_pct"] < 0:
        outcomes["status"] = "active_negative"
    else:
        outcomes["status"] = "flat"


def update_outcomes(memory: dict, update_days: int) -> int:
    from market_analyzer import fetch_bist

    records = memory.get("records", [])
    cutoff = datetime.now().date().toordinal() - update_days
    update_records = [
        r for r in records
        if date.fromisoformat(r["report_date"]).toordinal() >= cutoff or r.get("outcomes", {}).get("status") in {"pending", "active_positive", "active_negative", "flat"}
    ]
    symbols = sorted({r["symbol"] for r in update_records if r.get("symbol")})
    fetched: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        try:
            fetched[symbol] = fetch_bist(f"{symbol}.IS", "1y", "1d")[0]
        except Exception:
            continue
    updated = 0
    for record in update_records:
        df = fetched.get(record.get("symbol"))
        if df is None:
            continue
        _update_record(record, df)
        updated += 1
    return updated


def build_summary(memory: dict) -> dict:
    all_records = memory.get("records", [])
    records = [r for r in all_records if r.get("record_type") == "actionable_trade"]
    outcomes = [r.get("outcomes", {}) for r in records]
    day_stats = {}
    for key in ["day_1_pct", "day_3_pct", "day_5_pct", "latest_pct"]:
        vals = [_safe_float(o.get(key), None) for o in outcomes if o.get(key) is not None]
        day_stats[key] = round(mean(vals), 2) if vals else None
    summary = {
        "total_records": len(all_records),
        "actionable_records": len(records),
        "excluded_legacy_or_radar_records": len(all_records) - len(records),
        "target_1_hits": sum(1 for o in outcomes if o.get("target_1_hit")),
        "stop_hits": sum(1 for o in outcomes if o.get("stop_hit")),
        "active_or_pending": sum(1 for o in outcomes if o.get("status") in {"pending", "active_positive", "active_negative", "flat"}),
        "averages": day_stats,
    }
    memory["summary"] = summary
    return summary


def write_markdown_summary(memory: dict) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = memory.get("summary", {})
    records = sorted(
        [r for r in memory.get("records", []) if r.get("record_type") == "actionable_trade"],
        key=lambda r: r.get("report_time", ""),
        reverse=True,
    )
    rows = []
    for record in records[:25]:
        out = record.get("outcomes", {})
        rows.append(
            f"| {record.get('report_date')} | {record.get('rank')} | {record.get('symbol')} | "
            f"{record.get('trade_horizon', '-')} | {record.get('primary')} | {record.get('entry_price'):.2f} | {out.get('latest_pct')} | "
            f"{out.get('day_1_pct')} | {out.get('day_3_pct')} | {out.get('day_5_pct')} | "
            f"{out.get('status')} | {record.get('source_report')} |"
        )
    table = "\n".join(rows) or "| - | - | - | - | - | - | - | - | - | - | - | - |"
    text = f"""# BIST Performans Hafizasi

Guncelleme: {memory.get('updated_at')}

Toplam kayit: {summary.get('total_records', 0)}
Gercek islem adayi kaydi: {summary.get('actionable_records', 0)}
Basari hesabindan cikarilan legacy/radar kaydi: {summary.get('excluded_legacy_or_radar_records', 0)}
T1 hit: {summary.get('target_1_hits', 0)}
Stop hit: {summary.get('stop_hits', 0)}
Aktif/bekleyen: {summary.get('active_or_pending', 0)}

Ortalama son durum: {summary.get('averages', {}).get('latest_pct')}
Ortalama 1 gun: {summary.get('averages', {}).get('day_1_pct')}
Ortalama 3 gun: {summary.get('averages', {}).get('day_3_pct')}
Ortalama 5 gun: {summary.get('averages', {}).get('day_5_pct')}

Not: Bu tablo sadece `Bugun Isleme En Yakin Adaylar` icinden gelen gercek islem adaylarini sayar. Radar/genel guc kayitlari basari oranina dahil edilmez.

| Tarih | Sira | Hisse | Vade | Karar | Giris | Son % | 1g % | 3g % | 5g % | Durum | Kaynak |
|---|---:|---|---|---|---:|---:|---:|---:|---:|---|---|
{table}
"""
    SUMMARY_PATH.write_text(text, encoding="utf-8")


def update_performance_memory(payload: dict, scanner_path: Path, top: int = 5, update_days: int = 45) -> dict:
    memory = _load_memory()
    added = add_new_records(memory, payload, scanner_path, top)
    updated = update_outcomes(memory, update_days=update_days)
    memory["updated_at"] = datetime.now().isoformat(timespec="seconds")
    summary = build_summary(memory)
    _save_memory(memory)
    write_markdown_summary(memory)
    return {"added": added, "updated": updated, "summary": summary, "memory_path": str(MEMORY_PATH), "summary_path": str(SUMMARY_PATH)}
