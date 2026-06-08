from __future__ import annotations

import argparse
import json
from pathlib import Path


FINANCE_ROOT = Path.home() / ".config" / "opencode" / "finance"
REPORTS_DIR = FINANCE_ROOT / "reports"


def latest_scanner_json() -> Path:
    files = sorted(REPORTS_DIR.glob("BIST_SCANNER_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError("BIST scanner JSON raporu bulunamadi.")
    return files[0]


def fmt(value, digits: int = 2) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "-"


def row_line(row: dict, index: int) -> str:
    symbol = str(row.get("symbol", "-")).upper()
    horizon = f"{row.get('trade_horizon', '-')}/{row.get('holding_days', '-')}"
    window = f"{row.get('entry_window', '-')}/{row.get('actionable_route', '-')}"
    chase = f"{row.get('chase_label', 'normal')}/{fmt(row.get('chase_score'), 1)}"
    agent = f"{row.get('agent_verdict', 'unknown')}/{fmt(row.get('agent_consensus_score'), 1)}"
    return (
        f"{index}. {symbol} | Vade: {horizon} | Durum: {window} | "
        f"En yakin: %{fmt(row.get('nearest_entry_distance_pct'), 1)} | "
        f"Kovalama: {chase} | Ajan: {agent} | Not: ISLEM DEGIL, alarm/radar"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="BIST scanner raporunu guvenli yorum kilidiyle ozetler.")
    parser.add_argument("--scanner-json", default="")
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()

    path = Path(args.scanner_json).expanduser() if args.scanner_json else latest_scanner_json()
    payload = json.loads(path.read_text(encoding="utf-8"))
    ready = payload.get("entry_ready_rows", [])[: args.top]
    ready_symbols = {str(row.get("symbol", "")).upper() for row in ready}
    prep = payload.get("setup_prep_rows", [])[: args.top]
    rows = payload.get("rows", [])[: args.top]
    gate = payload.get("trade_gate") or {}
    status = gate.get("status") or ("ACTIONABLE" if ready else "ISLEM_YOK")

    print(f"SAFE_BIST_SUMMARY={path}")
    print(f"GUNLUK_ISLEM_KILIDI={status}")
    print(gate.get("message") or ("Bugun girise yakin aday var." if ready else "Bugun temiz giris yok."))
    print(gate.get("rule") or "Sadece Bugun Isleme En Yakin tablosu islem adayi sayilir.")
    print()

    if status != "ACTIONABLE" or not ready:
        print("BUGUN_ISLEM_ONERISI: YOK")
        print("Kesin yorum: Bugun temiz giris yoksa sembol onerme. Asagidakiler sadece alarm/radar listesidir.")
    else:
        print("BUGUN_ISLEM_ADAYLARI:")
        for i, row in enumerate(ready, start=1):
            symbol = str(row.get("symbol", "-")).upper()
            print(
                f"{i}. {symbol} | Vade: {row.get('trade_horizon', '-')}/{row.get('holding_days', '-')} | "
                f"Giris: {row.get('entry_window', '-')}/{row.get('actionable_route', '-')} | "
                f"En yakin: %{fmt(row.get('nearest_entry_distance_pct'), 1)} | "
                f"Ajan: {row.get('agent_verdict', 'unknown')}/{fmt(row.get('agent_consensus_score'), 1)}"
            )

    print()
    print("RADAR_ALARM_LISTESI_ISLEM_DEGIL:")
    prep_only = [row for row in prep if str(row.get("symbol", "")).upper() not in ready_symbols]
    if not prep_only:
        print("- Radar adayi yok.")
    for i, row in enumerate(prep_only, start=1):
        print(row_line(row, i))

    print()
    print("GENEL_GUC_ISLEM_DEGIL:")
    general_only = [row for row in rows if str(row.get("symbol", "")).upper() not in ready_symbols]
    if not general_only:
        print("- Genel liste bos.")
    for i, row in enumerate(general_only, start=1):
        print(row_line(row, i))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
