from __future__ import annotations

import html
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import pandas_market_calendars as mcal
import requests
import yfinance as yf


ETFS = [
    "IGV", "SOXX", "SMH", "CIBR", "IBB", "XBI", "IHI", "IHF",
    "KRE", "KBE", "KIE", "ITA", "XAR", "IYT", "JETS", "XOP",
    "OIH", "XME", "GDX", "COPX", "URA", "XRT", "ITB", "XHB",
    "PBJ", "TAN",
]

NY_TZ = ZoneInfo("America/New_York")


@dataclass
class GapResult:
    symbol: str
    prev_close: float
    today_open: float
    latest_price: float | None
    gap_pct: float
    first_bar_time: str


def env_float(name: str, default: float) -> float:
    value = os.getenv(name, "").strip()
    return float(value) if value else default


def env_int(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    return int(value) if value else default


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name, "").strip().lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "y", "on"}


GAP_THRESHOLD_PCT = env_float("GAP_THRESHOLD_PCT", 1.0)
ALERT_WINDOW_MINUTES = env_int("ALERT_WINDOW_MINUTES", 90)
NOTIFY_WHEN_NONE = env_bool("NOTIFY_WHEN_NONE", False)
FORCE_RUN = env_bool("FORCE_RUN", False)


def get_today_nyse_session(now_ny: datetime):
    """Return today's NYSE open/close in New York time, or None if market closed."""
    nyse = mcal.get_calendar("NYSE")
    schedule = nyse.schedule(
        start_date=now_ny.date().isoformat(),
        end_date=now_ny.date().isoformat(),
    )

    if schedule.empty:
        return None

    market_open = schedule.iloc[0]["market_open"].to_pydatetime().astimezone(NY_TZ)
    market_close = schedule.iloc[0]["market_close"].to_pydatetime().astimezone(NY_TZ)
    return market_open, market_close


def normalize_index_to_ny(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    if not isinstance(df.index, pd.DatetimeIndex):
        return df

    if df.index.tz is None:
        df = df.tz_localize(NY_TZ)
    else:
        df = df.tz_convert(NY_TZ)

    return df


def get_previous_close(symbol: str, today_date) -> float | None:
    """Previous regular-session close before today."""
    df = yf.Ticker(symbol).history(
        period="10d",
        interval="1d",
        auto_adjust=False,
        prepost=False,
    )

    if df.empty or "Close" not in df.columns:
        return None

    df = df.dropna(subset=["Close"])

    if isinstance(df.index, pd.DatetimeIndex):
        df = df[[idx.date() < today_date for idx in df.index]]

    if df.empty:
        return None

    return float(df["Close"].iloc[-1])


def get_today_open_and_latest(symbol: str, today_date, market_open_ny: datetime):
    """
    Uses today's first 1-minute regular-session bar as the opening price.
    yfinance may be delayed; running at 09:45 ET is usually safer than 09:31.
    """
    df = yf.Ticker(symbol).history(
        period="1d",
        interval="1m",
        auto_adjust=False,
        prepost=False,
    )

    if df.empty or "Open" not in df.columns:
        return None, None, None

    df = normalize_index_to_ny(df)
    df = df.dropna(subset=["Open"])

    if df.empty:
        return None, None, None

    rows = df[
        (df.index.date == today_date)
        & (df.index >= market_open_ny)
    ]

    if rows.empty:
        return None, None, None

    first_row = rows.iloc[0]
    first_bar_time = rows.index[0].strftime("%H:%M")

    today_open = float(first_row["Open"])

    latest_price = None
    if "Close" in rows.columns:
        close_series = rows["Close"].dropna()
        if not close_series.empty:
            latest_price = float(close_series.iloc[-1])

    return today_open, latest_price, first_bar_time


def scan_symbol(symbol: str, today_date, market_open_ny: datetime) -> GapResult | None:
    prev_close = get_previous_close(symbol, today_date)
    if prev_close is None or prev_close <= 0:
        return None

    today_open, latest_price, first_bar_time = get_today_open_and_latest(
        symbol,
        today_date,
        market_open_ny,
    )

    if today_open is None or today_open <= 0:
        return None

    gap_pct = (today_open / prev_close - 1.0) * 100.0

    if gap_pct >= GAP_THRESHOLD_PCT:
        return GapResult(
            symbol=symbol,
            prev_close=prev_close,
            today_open=today_open,
            latest_price=latest_price,
            gap_pct=gap_pct,
            first_bar_time=first_bar_time or "N/A",
        )

    return None


def build_alert_message(results: list[GapResult], now_ny: datetime) -> str:
    lines = [
        "🚀 <b>ETF Gap-Up Alert</b>",
        f"日期: {html.escape(now_ny.strftime('%Y-%m-%d %H:%M %Z'))}",
        f"条件: 今日第一根常规交易 K 线 Open ≥ 前收 + {GAP_THRESHOLD_PCT:.2f}%",
        "",
    ]

    for r in sorted(results, key=lambda x: x.gap_pct, reverse=True):
        latest = f"{r.latest_price:.2f}" if r.latest_price is not None else "N/A"
        lines.append(
            f"• <b>{html.escape(r.symbol)}</b>: "
            f"+{r.gap_pct:.2f}% | "
            f"Prev Close {r.prev_close:.2f} → Open {r.today_open:.2f} "
            f"({html.escape(r.first_bar_time)} ET) | Latest {latest}"
        )

    return "\n".join(lines)


def build_no_signal_message(now_ny: datetime) -> str:
    return (
        "📭 <b>ETF Gap-Up Scan</b>\n"
        f"日期: {html.escape(now_ny.strftime('%Y-%m-%d %H:%M %Z'))}\n"
        f"结果: 没有 ETF 满足跳空高开 ≥ {GAP_THRESHOLD_PCT:.2f}%"
    )


def chunk_text(text: str, max_len: int = 3800) -> list[str]:
    if len(text) <= max_len:
        return [text]

    chunks = []
    current = []
    current_len = 0

    for line in text.splitlines():
        line_len = len(line) + 1
        if current and current_len + line_len > max_len:
            chunks.append("\n".join(current))
            current = []
            current_len = 0

        current.append(line)
        current_len += line_len

    if current:
        chunks.append("\n".join(current))

    return chunks


def send_telegram(text: str) -> None:
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    chat_id = os.environ["TELEGRAM_CHAT_ID"]

    url = f"https://api.telegram.org/bot{token}/sendMessage"

    for chunk in chunk_text(text):
        resp = requests.post(
            url,
            json={
                "chat_id": chat_id,
                "text": chunk,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data.get("ok"):
            raise RuntimeError(f"Telegram API error: {data}")


def main() -> int:
    now_ny = datetime.now(NY_TZ)
    today_date = now_ny.date()

    session = get_today_nyse_session(now_ny)
    if session is None:
        print(f"{today_date} is not an NYSE trading day. Skip.")
        return 0

    market_open_ny, market_close_ny = session
    latest_allowed_time = min(
        market_open_ny + timedelta(minutes=ALERT_WINDOW_MINUTES),
        market_close_ny,
    )

    if not FORCE_RUN and not (market_open_ny <= now_ny <= latest_allowed_time):
        print(
            "Outside alert window. "
            f"Now={now_ny}, allowed={market_open_ny} to {latest_allowed_time}. Skip."
        )
        return 0

    alerts: list[GapResult] = []
    errors: list[str] = []

    for symbol in ETFS:
        try:
            result = scan_symbol(symbol, today_date, market_open_ny)
            if result:
                alerts.append(result)
            time.sleep(0.2)  # be gentle with the free data source
        except Exception as exc:
            errors.append(f"{symbol}: {type(exc).__name__}: {exc}")

    print(f"Scanned {len(ETFS)} ETFs. Alerts={len(alerts)} Errors={len(errors)}")

    if errors:
        print("Errors:")
        for e in errors:
            print(f"  - {e}")

    if alerts:
        send_telegram(build_alert_message(alerts, now_ny))
    elif NOTIFY_WHEN_NONE:
        send_telegram(build_no_signal_message(now_ny))

    if len(errors) == len(ETFS):
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
