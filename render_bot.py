import os
import sys
import json
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

from bist_scanner import run_scan, save_outputs

TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
if not TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN env var gerekli")

FINANCE_ROOT = Path("/tmp/opencode_finance")
REPORTS_DIR = FINANCE_ROOT / "reports"
os.environ["OPENCODE_FINANCE_ROOT"] = str(FINANCE_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

application = Application.builder().token(TOKEN).build()


class BotArgs:
    symbols = ""
    timeframe = "1d"
    period = "1y"
    min_rows = 120
    max_symbols = 20
    top = 15
    fundamental_top = 8
    news_top = 8
    news_days = 3
    kap_days = 14
    note_path = ""
    no_desktop_note = True
    performance_top = 5
    performance_update_days = 45
    no_performance_memory = True
    regime_symbol = "XU100.IS"
    no_market_regime = False


def format_summary(ready_list: list, gate_status: str, max_items: int = 5) -> str:
    lines = ["BIST Tarama Sonucu", f"Kilit: {gate_status}", ""]
    if gate_status == "ACTIONABLE" and ready_list:
        lines.append("BUGUN ISLEME EN YAKIN ADAYLAR:")
        for i, r in enumerate(ready_list[:max_items], 1):
            s = r.get("symbol", "?")
            score = r.get("entry_ready_score", 0)
            horizon = r.get("trade_horizon", "-")
            window = r.get("entry_window", "-")
            price = r.get("price", "-")
            dist = r.get("nearest_entry_distance_pct", 0)
            lines.append(f"{i}. {s} | Skor: {score:.1f}")
            lines.append(f"   Vade: {horizon} | Giris: {window}")
            lines.append(f"   Fiyat: {price} | Mesafe: %{dist:.1f}")
            lines.append("")
    else:
        lines.append("Bugun temiz giris yok. /radar ile liste.")
    return "\n".join(lines)


def format_radar(payload: dict, max_items: int = 5) -> str:
    prep = payload.get("setup_prep_rows", [])
    if not prep:
        return "Radar adayi yok."
    lines = ["RADAR / HAZIRLIK (islem degil):"]
    for i, r in enumerate(prep[:max_items], 1):
        s = r.get("symbol", "?")
        score = r.get("prep_score", 0)
        reason = r.get("prep_reason", "-")
        lines.append(f"{i}. {s} | Skor: {score:.1f}")
        lines.append(f"   {str(reason)[:80]}")
        lines.append("")
    return "\n".join(lines)


def run_scanner():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (FINANCE_ROOT / "cache").mkdir(parents=True, exist_ok=True)
    (FINANCE_ROOT / "data").mkdir(parents=True, exist_ok=True)

    args = BotArgs()
    rows, raw_results, errors, markdown = run_scan(args)
    md_path = save_outputs(args, rows, raw_results, errors, markdown)

    json_path = md_path.with_suffix(".json")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    ready = payload.get("entry_ready_rows", [])
    gate = payload.get("trade_gate", {})
    status = gate.get("status") or ("ACTIONABLE" if ready else "ISLEM_YOK")
    return {"summary": format_summary(ready, status), "radar": format_radar(payload), "payload": payload}


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "BIST Scanner Bot\n\n"
        "/scan - BIST taramasi yap\n"
        "/radar - Radar listesini goster\n"
        "/yardim - Bu mesaj"
    )


async def cmd_scan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("BIST taranıyor... biraz bekle (30-60sn)...")
    try:
        result = run_scanner()
        await msg.edit_text(result["summary"])
    except Exception as e:
        logging.exception("Scan failed")
        await msg.edit_text(f"Hata: {e}")


async def cmd_radar(update: Update, context: ContextTypes.DEFAULT_TYPE):
    files = sorted(REPORTS_DIR.glob("BIST_SCANNER_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        await update.message.reply_text("Henuz tarama yapilmamis. Once /scan dene.")
        return
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    msg = format_radar(payload)
    await update.message.reply_text(msg)


async def error(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logging.error(f"Update {update} caused error {context.error}")


application.add_handler(CommandHandler("start", cmd_start))
application.add_handler(CommandHandler("yardim", cmd_start))
application.add_handler(CommandHandler("scan", cmd_scan))
application.add_handler(CommandHandler("tarama", cmd_scan))
application.add_handler(CommandHandler("radar", cmd_radar))
application.add_error_handler(error)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    webhook_url = os.environ.get("WEBHOOK_URL", "")
    application.run_webhook(
        listen="0.0.0.0",
        port=port,
        url_path=TOKEN,
        webhook_url=f"{webhook_url}/{TOKEN}" if webhook_url else None,
    )
