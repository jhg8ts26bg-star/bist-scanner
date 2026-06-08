from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path


FINANCE_ROOT = Path.home() / ".config" / "opencode" / "finance"
REPORTS_DIR = FINANCE_ROOT / "reports"
DESKTOP = Path.home() / "Desktop"
DEFAULT_NOTE = DESKTOP / "OpenCode BIST Takip Notlari.txt"
LEGACY_NOTE = DESKTOP / "BIST Analiz 4 Haziran 2026.txt"


def latest_scanner_json() -> Path:
    files = sorted(REPORTS_DIR.glob("BIST_SCANNER_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError("BIST scanner JSON raporu bulunamadi.")
    return files[0]


def note_path_from_args(value: str) -> Path:
    if value:
        return Path(value).expanduser()
    return DEFAULT_NOTE if DEFAULT_NOTE.exists() or not LEGACY_NOTE.exists() else LEGACY_NOTE


def fmt_price(value) -> str:
    try:
        return f"{float(value):.2f}"
    except Exception:
        return "-"


def fmt_zone(values) -> str:
    if isinstance(values, list) and len(values) >= 2:
        return f"{fmt_price(values[0])}-{fmt_price(values[1])}"
    return "-"


def fmt_targets(values) -> str:
    if isinstance(values, list) and values:
        return "/".join(fmt_price(x) for x in values[:3])
    return "-"


def short_text(value: str, limit: int = 72) -> str:
    clean = " ".join(str(value or "").replace("|", "/").split())
    return clean if len(clean) <= limit else clean[: limit - 3].rstrip() + "..."


def raw_by_symbol(payload: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for item in payload.get("raw_results", []):
        symbol = str(item.get("symbol", "")).upper()
        if symbol:
            out[symbol] = item
    return out


def row_line(index: int, row: dict, raw: dict | None) -> str:
    symbol = str(row.get("symbol", "-")).upper()
    horizon = short_text(f"{row.get('trade_horizon', '-')}/{row.get('holding_days', '-')}", 28)
    plan = (raw or {}).get("plan", {})
    switch = plan.get("strategy_switch", {})
    plan_a = fmt_zone(plan.get("long_pullback_zone"))
    plan_b = f"{fmt_price(plan.get('long_breakout_trigger'))} ustu"
    stop = f"{fmt_price(plan.get('long_invalidation'))} alti"
    targets = fmt_targets(plan.get("long_targets"))
    summary = switch.get("summary") or row.get("reason", "")
    primary = str(row.get("primary", "-")).upper()
    score = fmt_price(row.get("scanner_score"))
    ready_score = fmt_price(row.get("entry_ready_score"))
    entry_quality = fmt_price(row.get("entry_quality_score"))
    entry_window = short_text(f"{row.get('entry_window', '-')}/{row.get('actionable_route', '-')}", 28)
    nearest_distance = fmt_price(row.get("nearest_entry_distance_pct"))
    daily_trigger = fmt_price(row.get("daily_trigger_pct"))
    chase = short_text(f"{row.get('chase_label', 'normal')}/{fmt_price(row.get('chase_score'))}", 28)
    agent = short_text(f"{row.get('agent_verdict', 'unknown')}/{fmt_price(row.get('agent_consensus_score'))}", 38)
    best_entry = short_text(row.get("best_entry", "-"), 28)
    sector = short_text(row.get("sector", "-"), 22)
    sector_strength = fmt_price(row.get("sector_strength_score"))
    plan_a_distance = fmt_price(row.get("plan_a_distance_pct"))
    plan_b_distance = fmt_price(row.get("plan_b_distance_pct"))
    price = fmt_price(row.get("price"))
    rr = fmt_price(row.get("long_rr"))
    news = f"{row.get('news_level', 'none')}/{row.get('news_direction', 'neutral')}"
    news_quality = short_text(
        f"{row.get('news_quality_label', 'unknown')}/{row.get('news_gate', '-')}/{fmt_price(row.get('news_quality_score'))}",
        42,
    )
    alarm = short_text(row.get("alert_summary", "alarm yok"), 54)
    pre = f"{row.get('pre_news_level', 'none')}/{row.get('pre_news_direction', 'neutral')}"
    return (
        f"| {index} | {symbol} | {horizon} | {primary} | {score} | {ready_score} | {entry_quality} | "
        f"{entry_window} | {nearest_distance} | {daily_trigger} | {chase} | {best_entry} | "
        f"{agent} | {sector} | {sector_strength} | {price} | {rr} | {plan_a_distance} | {plan_b_distance} | {plan_a} | {plan_b} | "
        f"{stop} | {targets} | {pre} | {news} | {news_quality} | {alarm} | "
        f"{short_text(summary)} |  |  |  |"
    )


def prep_row_line(index: int, row: dict) -> str:
    symbol = str(row.get("symbol", "-")).upper()
    horizon = short_text(f"{row.get('trade_horizon', '-')}/{row.get('holding_days', '-')}", 28)
    agent = short_text(f"{row.get('agent_verdict', 'unknown')}/{fmt_price(row.get('agent_consensus_score'))}", 38)
    return (
        f"| {index} | {symbol} | {horizon} | {fmt_price(row.get('prep_score'))} | {row.get('prep_label', '-')} | "
        f"{fmt_price(row.get('ret5_pct'))} | {fmt_price(row.get('ret20_pct'))} | {row.get('squeeze', False)} | "
        f"{row.get('structure', '-')} | {row.get('trend_label', '-')} | {row.get('vwap_position', '-')} | "
        f"{fmt_price(row.get('nearest_entry_distance_pct'))} | {fmt_price(row.get('plan_b_distance_pct'))} | "
        f"{short_text(row.get('sector', '-'), 22)} | {fmt_price(row.get('sector_strength_score'))} | "
        f"{short_text(row.get('news_quality_label', '-'))}/{row.get('news_gate', '-')} | "
        f"{short_text(row.get('chase_label', 'normal'))}/{fmt_price(row.get('chase_score'))} | {agent} | "
        f"{short_text(row.get('prep_reason', '-'), 80)} | {short_text(row.get('alert_summary', 'alarm yok'), 54)} |"
    )


def table_header(title: str) -> list[str]:
    return [
        "-" * 120,
        title,
        "| # | Hisse | Vade | Karar | Skor | Girise yakin skor | Giris kalite | Giris durumu | En yakin % | Gunluk sinir % | Kovalama | En iyi giris | Ajan | Sektor | Sektor gucu | Fiyat | Long R/R | Plan A uzak % | Plan B uzak % | Plan A bolge | Plan B kirilim | Stop | Hedefler | On fiyat | Haber | Haber kalite | Alarm | Strateji gecisi | Islemim | K/Z | Notum |",
        "|---:|---|---|---|---:|---:|---:|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]


def prep_table_header(title: str) -> list[str]:
    return [
        "-" * 120,
        title,
        "| # | Hisse | Vade | Hazirlik skor | Etiket | 5g % | 20g % | Squeeze | Structure | Trend | VWAP | En yakin % | Kirilim uzak % | Sektor | Sektor gucu | Haber kalite | Kovalama | Ajan | Neden | Alarm |",
        "|---:|---|---|---:|---|---:|---:|---|---|---|---|---:|---:|---|---:|---|---|---|---|---|",
    ]


def empty_row(message: str) -> str:
    return (
        f"| - | {short_text(message, 80)} | - | - | - | - | - | - | - | - | - | - | - | - | - | "
        "- | - | - | - | - | - | - | - | - | - | - | - |  |  |  |"
    )


def fallback_note_path(note_path: Path) -> Path:
    return note_path.with_name(f"{note_path.stem} - bekleyen kayitlar{note_path.suffix}")


def build_block(payload: dict, scanner_path: Path, top: int) -> str:
    rows = payload.get("rows", [])[:top]
    ready_rows = payload.get("entry_ready_rows", [])[:top]
    prep_rows = payload.get("setup_prep_rows", [])[:top]
    trade_gate = payload.get("trade_gate") or {}
    gate_status = trade_gate.get("status") or ("ACTIONABLE" if ready_rows else "ISLEM_YOK")
    gate_message = trade_gate.get("message") or (
        "Bugun girise yakin aday var." if ready_rows else "Bugun temiz giris yok. Radar listeleri islem onerisi degildir."
    )
    gate_rule = trade_gate.get("rule") or "Sadece BUGUN ISLEME EN YAKIN tablosu islem adayi sayilir."
    raw_map = raw_by_symbol(payload)
    regime = payload.get("market_regime") or {}
    regime_text = (
        f"{regime.get('label', 'unknown')}/{regime.get('bias', 'neutral')} "
        f"({fmt_price(regime.get('score', 50))})"
    )
    now = datetime.now()
    header = [
        "",
        "=" * 120,
        f"OPENCODE BIST TOP {top} TAKIP TABLOSU - {now.strftime('%Y-%m-%d %H:%M')}",
        f"Kaynak rapor: {scanner_path.name}",
        f"Piyasa rejimi: {regime_text}",
        f"GUNLUK ISLEM KILIDI: {gate_status} - {gate_message}",
        f"Kural: {gate_rule}",
        "Not: Islemim / K-Z / Notum sutunlari bos birakildi; sen dolduracaksin.",
    ]
    lines = header[:]
    lines.extend(table_header(f"BUGUN ISLEME EN YAKIN TOP {min(top, len(ready_rows)) or top}"))
    for idx, row in enumerate(ready_rows, start=1):
        raw = raw_map.get(str(row.get("symbol", "")).upper())
        lines.append(row_line(idx, row, raw))
    if not ready_rows:
        lines.append(empty_row("Bugun temiz giris yok; uzak seviyeler radar listesinde kalir."))

    lines.extend(prep_table_header(f"HAZIRLIK / PATLAMAMIS RADAR TOP {min(top, len(prep_rows)) or top} - ISLEM DEGIL / ALARM"))
    for idx, row in enumerate(prep_rows, start=1):
        lines.append(prep_row_line(idx, row))
    if not prep_rows:
        lines.append("| - | Hazirlik adayi yok | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |")

    lines.extend(table_header(f"GENEL GUC TOP {min(top, len(rows)) or top} - ISLEM DEGIL / SADECE RADAR"))
    for idx, row in enumerate(rows, start=1):
        raw = raw_map.get(str(row.get("symbol", "")).upper())
        lines.append(row_line(idx, row, raw))
    if not rows:
        lines.append(empty_row("Rapor bos geldi."))
    lines.extend(["=" * 120, ""])
    return "\n".join(lines)


def prepend_to_note(note_path: Path, block: str, source_name: str, allow_duplicate: bool) -> None:
    note_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        old_text = note_path.read_text(encoding="utf-8", errors="replace") if note_path.exists() else ""
        if not allow_duplicate and f"Kaynak rapor: {source_name}" in old_text:
            print(f"Zaten eklenmis: {note_path}")
            return
        if note_path.exists():
            backup = note_path.with_name(note_path.name + "." + datetime.now().strftime("%Y%m%d_%H%M%S") + ".bak")
            shutil.copy2(note_path, backup)
        note_path.write_text(block + old_text.lstrip("\ufeff\r\n"), encoding="utf-8")
        print(f"NOTE_PATH={note_path}")
    except PermissionError:
        fallback = fallback_note_path(note_path)
        old_text = fallback.read_text(encoding="utf-8", errors="replace") if fallback.exists() else ""
        if not allow_duplicate and f"Kaynak rapor: {source_name}" in old_text:
            print(f"Zaten bekleyen kayitlara eklenmis: {fallback}")
            return
        fallback.write_text(block + old_text.lstrip("\ufeff\r\n"), encoding="utf-8")
        print(f"NOTE_PATH_LOCKED={note_path}")
        print(f"NOTE_FALLBACK_PATH={fallback}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Son BIST scanner ilk 5 sonucunu masaustu not dosyasinin en ustune ekler.")
    parser.add_argument("--scanner-json", default="")
    parser.add_argument("--note-path", default="")
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--allow-duplicate", action="store_true")
    args = parser.parse_args()

    scanner_path = Path(args.scanner_json).expanduser() if args.scanner_json else latest_scanner_json()
    payload = json.loads(scanner_path.read_text(encoding="utf-8"))
    note_path = note_path_from_args(args.note_path)
    block = build_block(payload, scanner_path, max(1, args.top))
    prepend_to_note(note_path, block, scanner_path.name, args.allow_duplicate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
