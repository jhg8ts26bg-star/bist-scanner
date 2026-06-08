from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

FINANCE_ROOT = Path.home() / ".config" / "opencode" / "finance"
REPORTS_DIR = FINANCE_ROOT / "reports"


INCOME_ROWS = {
    "revenue": ["Total Revenue", "Operating Revenue"],
    "net_income": [
        "Net Income",
        "Net Income Common Stockholders",
        "Net Income From Continuing Operation Net Minority Interest",
        "Normalized Income",
    ],
    "operating_income": ["Operating Income", "Operating Income or Loss"],
    "ebitda": ["EBITDA", "Normalized EBITDA"],
    "net_interest_income": ["Net Interest Income"],
}

BALANCE_ROWS = {
    "total_debt": ["Total Debt", "Net Debt"],
    "cash": ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"],
    "equity": ["Common Stock Equity", "Stockholders Equity", "Total Equity Gross Minority Interest"],
    "assets": ["Total Assets"],
}

CASHFLOW_ROWS = {
    "operating_cash_flow": ["Operating Cash Flow", "Total Cash From Operating Activities"],
    "free_cash_flow": ["Free Cash Flow"],
    "capex": ["Capital Expenditure", "Capital Expenditures"],
}


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        if math.isinf(float(value)):
            return default
        return float(value)
    except Exception:
        return default


def _safe_pct(new: float, old: float) -> float | None:
    if old is None or abs(old) < 1e-9:
        return None
    return (new / old - 1.0) * 100.0


def _find_row(df: pd.DataFrame, names: list[str]) -> pd.Series | None:
    if df is None or df.empty:
        return None
    index_map = {str(idx).lower(): idx for idx in df.index}
    for name in names:
        idx = index_map.get(name.lower())
        if idx is not None:
            return df.loc[idx]
    return None


def _series_values(df: pd.DataFrame, names: list[str], max_items: int = 6) -> list[float]:
    row = _find_row(df, names)
    if row is None:
        return []
    values = []
    for value in list(row.values)[:max_items]:
        values.append(_safe_float(value))
    return values


def _latest(values: list[float]) -> float | None:
    return values[0] if values else None


def _prev(values: list[float], offset: int) -> float | None:
    return values[offset] if len(values) > offset else None


def _trend_label(values: list[float]) -> str:
    cleaned = [x for x in values[:4] if x is not None]
    if len(cleaned) < 3:
        return "unknown"
    positives = sum(1 for x in cleaned if x > 0)
    if cleaned[0] > cleaned[-1] and positives >= 3:
        return "improving"
    if cleaned[0] < cleaned[-1] or positives <= 1:
        return "weakening"
    return "mixed"


def _score_snapshot(metrics: dict) -> tuple[int, str, list[str]]:
    score = 50
    notes: list[str] = []

    rev_yoy = metrics.get("revenue_yoy_pct")
    ni_yoy = metrics.get("net_income_yoy_pct")
    ni_qoq = metrics.get("net_income_qoq_pct")
    ocf = metrics.get("operating_cash_flow")
    fcf = metrics.get("free_cash_flow")
    debt_to_equity = metrics.get("debt_to_equity")
    cash_to_debt = metrics.get("cash_to_debt")
    profit_trend = metrics.get("net_income_trend")

    if rev_yoy is not None:
        if rev_yoy > 15:
            score += 10
            notes.append(f"Hasılat yıllık {rev_yoy:.1f}% artıyor.")
        elif rev_yoy < -10:
            score -= 10
            notes.append(f"Hasılat yıllık {rev_yoy:.1f}% düşüyor.")

    if ni_yoy is not None:
        if ni_yoy > 20:
            score += 12
            notes.append(f"Net kâr yıllık {ni_yoy:.1f}% artıyor.")
        elif ni_yoy < -20:
            score -= 12
            notes.append(f"Net kâr yıllık {ni_yoy:.1f}% zayıflıyor.")

    if ni_qoq is not None:
        if ni_qoq > 10:
            score += 5
            notes.append(f"Net kâr çeyreklik {ni_qoq:.1f}% toparlıyor.")
        elif ni_qoq < -15:
            score -= 6
            notes.append(f"Net kâr çeyreklik {ni_qoq:.1f}% geriliyor.")

    if profit_trend == "improving":
        score += 8
        notes.append("Son çeyreklerde kâr trendi toparlanıyor.")
    elif profit_trend == "weakening":
        score -= 8
        notes.append("Son çeyreklerde kâr trendi zayıflıyor.")

    if ocf is not None:
        if ocf > 0:
            score += 6
            notes.append("Operasyonel nakit akışı pozitif.")
        elif ocf < 0:
            score -= 8
            notes.append("Operasyonel nakit akışı negatif.")

    if fcf is not None:
        if fcf > 0:
            score += 4
            notes.append("Serbest nakit akışı pozitif.")
        elif fcf < 0:
            score -= 5
            notes.append("Serbest nakit akışı negatif.")

    if debt_to_equity is not None:
        if debt_to_equity > 2.5:
            score -= 8
            notes.append(f"Borç/özsermaye yüksek: {debt_to_equity:.2f}.")
        elif debt_to_equity < 0.8:
            score += 5
            notes.append(f"Borç/özsermaye kontrollü: {debt_to_equity:.2f}.")

    if cash_to_debt is not None:
        if cash_to_debt > 0.7:
            score += 4
            notes.append(f"Nakit/borç tamponu iyi: {cash_to_debt:.2f}.")
        elif cash_to_debt < 0.15:
            score -= 4
            notes.append(f"Nakit/borç tamponu zayıf: {cash_to_debt:.2f}.")

    score = int(max(0, min(100, score)))
    if score >= 70:
        label = "strong"
    elif score >= 55:
        label = "watch"
    elif score >= 40:
        label = "mixed"
    else:
        label = "weak"
    return score, label, notes[:8]


def fetch_fundamental_snapshot(symbol: str, max_quarters: int = 6) -> dict:
    ticker = symbol.upper()
    if not ticker.endswith(".IS") and "/" not in ticker:
        ticker += ".IS"

    yf_ticker = yf.Ticker(ticker)
    income = yf_ticker.quarterly_income_stmt
    balance = yf_ticker.quarterly_balance_sheet
    cashflow = yf_ticker.quarterly_cashflow

    if income is None or income.empty:
        return {
            "symbol": ticker,
            "available": False,
            "score": 0,
            "label": "missing",
            "notes": ["Temel veri bulunamadi."],
            "metrics": {},
            "source": "yfinance",
        }

    series = {name: _series_values(income, rows, max_quarters) for name, rows in INCOME_ROWS.items()}
    balance_series = {name: _series_values(balance, rows, max_quarters) for name, rows in BALANCE_ROWS.items()}
    cash_series = {name: _series_values(cashflow, rows, max_quarters) for name, rows in CASHFLOW_ROWS.items()}

    revenue = _latest(series.get("revenue", []))
    revenue_prev = _prev(series.get("revenue", []), 1)
    revenue_yoy_base = _prev(series.get("revenue", []), 4)
    net_income = _latest(series.get("net_income", []))
    net_income_prev = _prev(series.get("net_income", []), 1)
    net_income_yoy_base = _prev(series.get("net_income", []), 4)
    total_debt = _latest(balance_series.get("total_debt", []))
    cash = _latest(balance_series.get("cash", []))
    equity = _latest(balance_series.get("equity", []))
    operating_cash_flow = _latest(cash_series.get("operating_cash_flow", []))
    free_cash_flow = _latest(cash_series.get("free_cash_flow", []))

    metrics = {
        "revenue": revenue,
        "revenue_qoq_pct": _safe_pct(revenue, revenue_prev) if revenue is not None and revenue_prev is not None else None,
        "revenue_yoy_pct": _safe_pct(revenue, revenue_yoy_base) if revenue is not None and revenue_yoy_base is not None else None,
        "net_income": net_income,
        "net_income_qoq_pct": _safe_pct(net_income, net_income_prev) if net_income is not None and net_income_prev is not None else None,
        "net_income_yoy_pct": _safe_pct(net_income, net_income_yoy_base) if net_income is not None and net_income_yoy_base is not None else None,
        "net_income_trend": _trend_label(series.get("net_income", [])),
        "operating_income": _latest(series.get("operating_income", [])),
        "ebitda": _latest(series.get("ebitda", [])),
        "net_interest_income": _latest(series.get("net_interest_income", [])),
        "total_debt": total_debt,
        "cash": cash,
        "equity": equity,
        "debt_to_equity": total_debt / equity if total_debt is not None and equity and equity > 0 else None,
        "cash_to_debt": cash / total_debt if cash is not None and total_debt and total_debt > 0 else None,
        "operating_cash_flow": operating_cash_flow,
        "free_cash_flow": free_cash_flow,
    }
    score, label, notes = _score_snapshot(metrics)

    return {
        "symbol": ticker,
        "available": True,
        "score": score,
        "label": label,
        "notes": notes,
        "metrics": metrics,
        "quarters": [str(x.date()) if hasattr(x, "date") else str(x) for x in list(income.columns)[:max_quarters]],
        "source": "yfinance",
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }


def build_markdown(snapshot: dict) -> str:
    metrics = snapshot.get("metrics", {})
    notes = "\n".join([f"- {x}" for x in snapshot.get("notes", [])]) or "- Not yok."
    return f"""# Temel Analiz - {snapshot.get('symbol')}

Veri var mı: {snapshot.get('available')}
Kaynak: {snapshot.get('source')}
Skor: {snapshot.get('score')}/100
Etiket: {snapshot.get('label')}

## Metrikler

Hasılat YoY: {metrics.get('revenue_yoy_pct')}
Net kâr YoY: {metrics.get('net_income_yoy_pct')}
Net kâr QoQ: {metrics.get('net_income_qoq_pct')}
Kâr trendi: {metrics.get('net_income_trend')}
Borç/Özsermaye: {metrics.get('debt_to_equity')}
Nakit/Borç: {metrics.get('cash_to_debt')}
Operasyonel nakit akışı: {metrics.get('operating_cash_flow')}
Serbest nakit akışı: {metrics.get('free_cash_flow')}

## Notlar

{notes}

Not: Bu temel analiz modülü hızlı risk filtresidir; tek başına yatırım kararı değildir.
"""


def save_snapshot(snapshot: dict, markdown: str) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_symbol = snapshot.get("symbol", "SYMBOL").replace(".", "_").replace("/", "_")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = REPORTS_DIR / f"FUNDAMENTAL_{safe_symbol}_{stamp}"
    md_path = base.with_suffix(".md")
    json_path = base.with_suffix(".json")
    md_path.write_text(markdown, encoding="utf-8")
    json_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    return md_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Fast BIST fundamental analysis snapshot.")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--max-quarters", type=int, default=6)
    args = parser.parse_args()
    snapshot = fetch_fundamental_snapshot(args.symbol, args.max_quarters)
    markdown = build_markdown(snapshot)
    path = save_snapshot(snapshot, markdown)
    print(markdown)
    print(f"\nFUNDAMENTAL_PATH={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
