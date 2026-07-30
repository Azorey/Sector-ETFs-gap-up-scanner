from __future__ import annotations

import html
import json
import math
import os
import sys
import time
import traceback
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
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
PRICE_COLUMNS = {
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
    "Dividends",
    "Stock Splits",
    "Capital Gains",
}


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
PREMARKET_THRESHOLD_PCT = env_float(
    "PREMARKET_THRESHOLD_PCT",
    GAP_THRESHOLD_PCT,
)
PREMARKET_LOOKBACK_MINUTES = env_int("PREMARKET_LOOKBACK_MINUTES", 15)
PREMARKET_MAX_STALENESS_MINUTES = env_int(
    "PREMARKET_MAX_STALENESS_MINUTES",
    15,
)
PREMARKET_MIN_VALID_BARS = env_int("PREMARKET_MIN_VALID_BARS", 3)
PREMARKET_MIN_RECENT_VOLUME = env_int("PREMARKET_MIN_RECENT_VOLUME", 100)
PREMARKET_MAX_LEAD_MINUTES = env_int("PREMARKET_MAX_LEAD_MINUTES", 120)
PREMARKET_MIN_COVERAGE_PCT = env_float(
    "PREMARKET_MIN_COVERAGE_PCT",
    80.0,
)
MIN_REFERENCE_COVERAGE_PCT = env_float(
    "MIN_REFERENCE_COVERAGE_PCT",
    80.0,
)
OPEN_MIN_COVERAGE_PCT = env_float("OPEN_MIN_COVERAGE_PCT", 80.0)
OPEN_PROVISIONAL_AFTER_MINUTES = env_int(
    "OPEN_PROVISIONAL_AFTER_MINUTES",
    15,
)
OPEN_HARD_DEADLINE_MINUTES = env_int("OPEN_HARD_DEADLINE_MINUTES", 35)
OPEN_DATA_RETRY_COUNT = env_int("OPEN_DATA_RETRY_COUNT", 6)
OPEN_DATA_RETRY_SECONDS = env_int("OPEN_DATA_RETRY_SECONDS", 30)
YAHOO_MAX_THREADS = env_int("YAHOO_MAX_THREADS", 8)
NOTIFY_WHEN_NONE = env_bool("NOTIFY_WHEN_NONE", False)
PREMARKET_NOTIFY_WHEN_NONE = env_bool("PREMARKET_NOTIFY_WHEN_NONE", False)
FORCE_RUN = env_bool("FORCE_RUN", False)
RESEND = env_bool("RESEND", False)
STATE_FILE = Path(os.getenv("STATE_FILE", ".state/etf-gap-up.json"))


def validate_config() -> None:
    finite_nonnegative = {
        "GAP_THRESHOLD_PCT": GAP_THRESHOLD_PCT,
        "PREMARKET_THRESHOLD_PCT": PREMARKET_THRESHOLD_PCT,
    }
    for name, value in finite_nonnegative.items():
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a finite number >= 0; got {value!r}")

    positive_integers = {
        "PREMARKET_LOOKBACK_MINUTES": PREMARKET_LOOKBACK_MINUTES,
        "PREMARKET_MAX_STALENESS_MINUTES": PREMARKET_MAX_STALENESS_MINUTES,
        "PREMARKET_MIN_VALID_BARS": PREMARKET_MIN_VALID_BARS,
        "PREMARKET_MAX_LEAD_MINUTES": PREMARKET_MAX_LEAD_MINUTES,
        "OPEN_PROVISIONAL_AFTER_MINUTES": OPEN_PROVISIONAL_AFTER_MINUTES,
        "OPEN_HARD_DEADLINE_MINUTES": OPEN_HARD_DEADLINE_MINUTES,
        "OPEN_DATA_RETRY_COUNT": OPEN_DATA_RETRY_COUNT,
        "OPEN_DATA_RETRY_SECONDS": OPEN_DATA_RETRY_SECONDS,
        "YAHOO_MAX_THREADS": YAHOO_MAX_THREADS,
    }
    for name, value in positive_integers.items():
        if value < 1:
            raise ValueError(f"{name} must be >= 1; got {value!r}")

    if PREMARKET_MIN_RECENT_VOLUME < 0:
        raise ValueError(
            "PREMARKET_MIN_RECENT_VOLUME must be >= 0; "
            f"got {PREMARKET_MIN_RECENT_VOLUME!r}"
        )

    percentages = {
        "PREMARKET_MIN_COVERAGE_PCT": PREMARKET_MIN_COVERAGE_PCT,
        "MIN_REFERENCE_COVERAGE_PCT": MIN_REFERENCE_COVERAGE_PCT,
        "OPEN_MIN_COVERAGE_PCT": OPEN_MIN_COVERAGE_PCT,
    }
    for name, value in percentages.items():
        if not math.isfinite(value) or not 0 < value <= 100:
            raise ValueError(f"{name} must be in (0, 100]; got {value!r}")

    if OPEN_HARD_DEADLINE_MINUTES <= OPEN_PROVISIONAL_AFTER_MINUTES:
        raise ValueError(
            "OPEN_HARD_DEADLINE_MINUTES must be greater than "
            "OPEN_PROVISIONAL_AFTER_MINUTES."
        )


@dataclass(frozen=True)
class PremarketResult:
    symbol: str
    prev_close: float
    robust_price: float
    latest_price: float
    robust_gap_pct: float
    latest_gap_pct: float
    latest_time: datetime
    valid_bar_count: int
    recent_volume: int


@dataclass(frozen=True)
class OpenResult:
    symbol: str
    prev_close: float
    today_open: float
    latest_price: float | None
    gap_pct: float
    first_bar_time: datetime


def get_today_nyse_session(now_ny: datetime) -> tuple[datetime, datetime] | None:
    """Return today's NYSE open and close in New York time."""
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
    if df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return df

    normalized = df.copy()
    if normalized.index.tz is None:
        normalized.index = normalized.index.tz_localize(NY_TZ)
    else:
        normalized.index = normalized.index.tz_convert(NY_TZ)
    return normalized.sort_index()


def download_batch(
    period: str,
    interval: str,
    prepost: bool,
    tickers: list[str] | None = None,
    actions: bool = False,
) -> pd.DataFrame:
    """Download a ticker group while preserving symbol-labelled columns."""
    requested_tickers = list(ETFS if tickers is None else tickers)
    if not requested_tickers:
        return pd.DataFrame()

    last_error: Exception | None = None
    for attempt in range(1, 3):
        try:
            data = yf.download(
                tickers=requested_tickers,
                period=period,
                interval=interval,
                group_by="ticker",
                auto_adjust=False,
                prepost=prepost,
                actions=actions,
                threads=min(YAHOO_MAX_THREADS, len(requested_tickers)),
                progress=False,
                timeout=20,
                multi_level_index=True,
            )
            if isinstance(data, pd.DataFrame) and not data.empty:
                data.attrs["_requested_tickers"] = requested_tickers
                return data
            print(
                f"Yahoo returned an empty {interval} batch "
                f"(attempt {attempt}/2)."
            )
        except Exception as exc:
            last_error = exc
            print(
                f"Yahoo {interval} batch failed "
                f"(attempt {attempt}/2): {type(exc).__name__}: {exc}"
            )
        if attempt < 2:
            time.sleep(2)

    if last_error is not None:
        print(f"Last Yahoo error: {type(last_error).__name__}: {last_error}")
    return pd.DataFrame()


def normalized_column_name(column: Any) -> str:
    if isinstance(column, tuple):
        for item in column:
            if str(item) in PRICE_COLUMNS:
                return str(item)
        return str(column[-1])
    return str(column)


def extract_symbol_frame(batch: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if batch.empty:
        return pd.DataFrame()

    frame: pd.DataFrame
    if isinstance(batch.columns, pd.MultiIndex):
        level_zero = {str(value) for value in batch.columns.get_level_values(0)}
        level_one = {str(value) for value in batch.columns.get_level_values(1)}
        if symbol in level_zero:
            frame = batch.xs(symbol, axis=1, level=0, drop_level=True)
        elif symbol in level_one:
            frame = batch.xs(symbol, axis=1, level=1, drop_level=True)
        else:
            return pd.DataFrame()
    else:
        requested_tickers = batch.attrs.get("_requested_tickers", [])
        if requested_tickers != [symbol]:
            # A flat result from a multi-ticker request has no safe ownership
            # information. Returning it for every symbol would create false alerts.
            return pd.DataFrame()
        frame = batch.copy()

    if isinstance(frame, pd.Series):
        frame = frame.to_frame()
    frame = frame.copy()
    frame.columns = [normalized_column_name(column) for column in frame.columns]
    frame = frame.loc[:, ~frame.columns.duplicated()]
    return frame.dropna(how="all")


def get_previous_closes(
    daily_batch: pd.DataFrame,
    today_date: date,
    symbols: list[str] | None = None,
) -> tuple[dict[str, float], dict[str, str]]:
    closes: dict[str, float] = {}
    missing: dict[str, str] = {}

    for symbol in ETFS if symbols is None else symbols:
        frame = extract_symbol_frame(daily_batch, symbol)
        if frame.empty or "Close" not in frame.columns:
            missing[symbol] = "missing_daily_close"
            continue

        if (
            "Stock Splits" in frame.columns
            and isinstance(frame.index, pd.DatetimeIndex)
        ):
            split_series = pd.to_numeric(
                frame["Stock Splits"],
                errors="coerce",
            ).fillna(0)
            todays_splits = split_series[
                [timestamp.date() == today_date for timestamp in split_series.index]
            ]
            nonzero_splits = todays_splits[todays_splits != 0]
            if not nonzero_splits.empty:
                ratio = float(nonzero_splits.iloc[-1])
                missing[symbol] = f"split_detected_ratio_{ratio:g}"
                continue

        close_series = pd.to_numeric(frame["Close"], errors="coerce").dropna()
        if isinstance(close_series.index, pd.DatetimeIndex):
            before_today = [
                timestamp.date() < today_date for timestamp in close_series.index
            ]
            close_series = close_series[before_today]

        if close_series.empty:
            missing[symbol] = "missing_previous_session"
            continue

        previous_close = float(close_series.iloc[-1])
        if not math.isfinite(previous_close) or previous_close <= 0:
            missing[symbol] = "invalid_previous_close"
            continue
        closes[symbol] = previous_close

    return closes, missing


def get_premarket_snapshot(
    symbol: str,
    frame: pd.DataFrame,
    previous_close: float,
    now_ny: datetime,
    market_open_ny: datetime,
) -> tuple[PremarketResult | None, str | None]:
    if frame.empty or "Close" not in frame.columns or "Volume" not in frame.columns:
        return None, "missing_intraday_columns"
    if not isinstance(frame.index, pd.DatetimeIndex):
        return None, "invalid_intraday_index"
    if frame.index.tz is None:
        return None, "missing_intraday_timezone"

    frame = normalize_index_to_ny(frame)

    work = pd.DataFrame(index=frame.index)
    work["Close"] = pd.to_numeric(frame["Close"], errors="coerce")
    work["Volume"] = pd.to_numeric(frame["Volume"], errors="coerce").fillna(0)

    early_session_start = market_open_ny.replace(
        hour=4,
        minute=0,
        second=0,
        microsecond=0,
    )
    traded = work[
        (work.index >= early_session_start)
        & (work.index < market_open_ny)
        & (work.index <= now_ny)
        & (work["Close"] > 0)
        & (work["Volume"] > 0)
        & work["Close"].map(math.isfinite)
        & work["Volume"].map(math.isfinite)
    ]
    if traded.empty:
        return None, "no_premarket_trade"

    latest_time = traded.index[-1].to_pydatetime()
    age_minutes = (now_ny - latest_time).total_seconds() / 60.0
    if age_minutes > PREMARKET_MAX_STALENESS_MINUTES:
        return None, "stale_premarket_trade"

    recent_start = max(
        now_ny - timedelta(minutes=PREMARKET_LOOKBACK_MINUTES),
        early_session_start,
    )
    recent = traded[traded.index >= recent_start]
    if len(recent) < PREMARKET_MIN_VALID_BARS:
        return None, "too_few_recent_bars"
    pricing_sample = recent.tail(PREMARKET_MIN_VALID_BARS)
    pricing_volume = int(pricing_sample["Volume"].sum())
    if pricing_volume < PREMARKET_MIN_RECENT_VOLUME:
        return None, "too_little_recent_volume"

    price_ordered = pricing_sample.sort_values("Close")
    cumulative_volume = price_ordered["Volume"].cumsum()
    weighted_median_rows = price_ordered[
        cumulative_volume >= pricing_volume / 2.0
    ]
    robust_price = float(weighted_median_rows["Close"].iloc[0])
    latest_price = float(recent["Close"].iloc[-1])
    robust_gap_pct = (robust_price / previous_close - 1.0) * 100.0
    latest_gap_pct = (latest_price / previous_close - 1.0) * 100.0

    return (
        PremarketResult(
            symbol=symbol,
            prev_close=previous_close,
            robust_price=robust_price,
            latest_price=latest_price,
            robust_gap_pct=robust_gap_pct,
            latest_gap_pct=latest_gap_pct,
            latest_time=latest_time,
            valid_bar_count=len(pricing_sample),
            recent_volume=pricing_volume,
        ),
        None,
    )


def get_open_snapshot(
    symbol: str,
    frame: pd.DataFrame,
    previous_close: float,
    now_ny: datetime,
    market_open_ny: datetime,
    market_close_ny: datetime,
) -> tuple[OpenResult | None, str | None]:
    if frame.empty or "Open" not in frame.columns or "Volume" not in frame.columns:
        return None, "missing_intraday_columns"
    if not isinstance(frame.index, pd.DatetimeIndex):
        return None, "invalid_intraday_index"
    if frame.index.tz is None:
        return None, "missing_intraday_timezone"

    frame = normalize_index_to_ny(frame)

    work = frame.copy()
    work["Open"] = pd.to_numeric(work["Open"], errors="coerce")
    work["Volume"] = pd.to_numeric(work["Volume"], errors="coerce").fillna(0)
    rows = work[
        (work.index >= market_open_ny)
        & (work.index <= min(now_ny, market_close_ny))
        & (work["Open"] > 0)
        & (work["Volume"] > 0)
        & work["Open"].map(math.isfinite)
        & work["Volume"].map(math.isfinite)
    ]
    if rows.empty:
        return None, "no_regular_session_trade"

    first_timestamp = rows.index[0]
    latest_price: float | None = None
    if "Close" in rows.columns:
        close_series = pd.to_numeric(rows["Close"], errors="coerce").dropna()
        close_series = close_series[close_series > 0]
        if not close_series.empty:
            latest_price = float(close_series.iloc[-1])

    today_open = float(rows.iloc[0]["Open"])
    return (
        OpenResult(
            symbol=symbol,
            prev_close=previous_close,
            today_open=today_open,
            latest_price=latest_price,
            gap_pct=(today_open / previous_close - 1.0) * 100.0,
            first_bar_time=first_timestamp.to_pydatetime(),
        ),
        None,
    )


def default_state(trading_date: date) -> dict[str, Any]:
    return {
        "trading_date": trading_date.isoformat(),
        "premarket_alerts": {},
        "premarket_no_signal_sent": False,
        "open_confirmation_sent": False,
        "open_confirmed_symbols": [],
        "open_symbol_notifications": {},
        "data_failure_alerts": [],
        "previous_closes": {},
    }


def load_state(path: Path, trading_date: date) -> dict[str, Any]:
    if not path.exists():
        return default_state(trading_date)

    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"State file is unreadable; starting fresh: {type(exc).__name__}: {exc}")
        return default_state(trading_date)

    if (
        not isinstance(state, dict)
        or state.get("trading_date") != trading_date.isoformat()
    ):
        return default_state(trading_date)

    state.setdefault("premarket_alerts", {})
    state.setdefault("premarket_no_signal_sent", False)
    state.setdefault("open_confirmation_sent", False)
    state.setdefault("open_confirmed_symbols", [])
    state.setdefault("open_symbol_notifications", {})
    state.setdefault("data_failure_alerts", [])
    state.setdefault("previous_closes", {})
    if not isinstance(state["premarket_alerts"], dict):
        state["premarket_alerts"] = {}
    if not isinstance(state["open_confirmed_symbols"], list):
        state["open_confirmed_symbols"] = []
    if not isinstance(state["open_symbol_notifications"], dict):
        state["open_symbol_notifications"] = {}
    if not isinstance(state["data_failure_alerts"], list):
        state["data_failure_alerts"] = []
    if not isinstance(state["previous_closes"], dict):
        state["previous_closes"] = {}
    return state


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary_path, path)


def chunk_text(text: str, max_len: int = 3800) -> list[str]:
    if len(text) <= max_len:
        return [text]

    chunks: list[str] = []
    current: list[str] = []
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
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must both be configured."
        )

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    for chunk_number, chunk in enumerate(chunk_text(text), start=1):
        for attempt in range(1, 4):
            try:
                response = requests.post(
                    url,
                    json={
                        "chat_id": chat_id,
                        "text": chunk,
                        "parse_mode": "HTML",
                        "disable_web_page_preview": True,
                    },
                    timeout=20,
                )
                try:
                    payload = response.json()
                except ValueError:
                    payload = {}

                if response.ok and payload.get("ok"):
                    break

                description = str(payload.get("description", "unknown error"))
                error_code = payload.get("error_code", response.status_code)
                retry_after = (
                    payload.get("parameters", {}).get("retry_after", 0)
                    if isinstance(payload.get("parameters"), dict)
                    else 0
                )
                retryable = response.status_code == 429 or response.status_code >= 500
                if not retryable or attempt == 3:
                    raise RuntimeError(
                        "Telegram rejected message "
                        f"chunk {chunk_number}: code={error_code}, "
                        f"description={description}"
                    )
                delay_seconds = max(int(retry_after or 0), 2**attempt)
                print(
                    f"Telegram temporary error {error_code}; "
                    f"retrying in {delay_seconds}s."
                )
                time.sleep(delay_seconds)
            except requests.RequestException as exc:
                if attempt == 3:
                    raise RuntimeError(
                        "Telegram network request failed after 3 attempts: "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc
                delay_seconds = 2**attempt
                print(
                    f"Telegram network error; retrying in {delay_seconds}s: "
                    f"{type(exc).__name__}"
                )
                time.sleep(delay_seconds)


def summarize_missing(missing: dict[str, str]) -> str:
    if not missing:
        return "none"
    counts = Counter(missing.values())
    return ", ".join(
        f"{reason}={count}" for reason, count in sorted(counts.items())
    )


def build_premarket_message(
    results: list[PremarketResult],
    now_ny: datetime,
    coverage_count: int,
    missing: dict[str, str],
) -> str:
    lines = [
        "🟡 <b>ETF 盘前 Gap 预警（尚未确认）</b>",
        f"时间: {html.escape(now_ny.strftime('%Y-%m-%d %H:%M %Z'))}",
        (
            "条件: 最近有效盘前成交量加权中位价与最新成交价均相对前收 "
            f"≥ {PREMARKET_THRESHOLD_PCT:.2f}%；最近 "
            f"{PREMARKET_LOOKBACK_MINUTES} 分钟至少 "
            f"{PREMARKET_MIN_VALID_BARS} 根且累计成交量 "
            f"≥ {PREMARKET_MIN_RECENT_VOLUME:,}"
        ),
        "",
    ]
    for result in sorted(
        results,
        key=lambda item: item.robust_gap_pct,
        reverse=True,
    ):
        lines.append(
            f"• <b>{html.escape(result.symbol)}</b>: "
            f"+{result.robust_gap_pct:.2f}% | "
            f"{result.prev_close:.2f} → {result.robust_price:.2f} | "
            f"最新 {result.latest_price:.2f} "
            f"({result.latest_time.strftime('%H:%M')} ET, "
            f"{result.latest_gap_pct:+.2f}%) | "
            f"{result.valid_bar_count} bars / "
            f"vol {result.recent_volume:,}"
        )

    lines.extend(
        [
            "",
            (
                "⚠️ 这是盘前成交数据的预估，不是 09:30 正式开盘价；"
                "开盘后会发送确认或撤销结果。"
            ),
            (
                f"有效盘前样本: {coverage_count}/{len(ETFS)}；"
                f"不可用: {len(missing)}"
            ),
        ]
    )
    return "\n".join(lines)


def build_premarket_no_signal_message(
    now_ny: datetime,
    coverage_count: int,
    missing: dict[str, str],
) -> str:
    return "\n".join(
        [
            "📭 <b>ETF 盘前 Gap 扫描</b>",
            f"时间: {html.escape(now_ny.strftime('%Y-%m-%d %H:%M %Z'))}",
            (
                "结果: 没有有效盘前样本同时满足 "
                f"≥ {PREMARKET_THRESHOLD_PCT:.2f}%"
            ),
            (
                f"有效盘前样本: {coverage_count}/{len(ETFS)}；"
                f"不可用: {len(missing)}"
            ),
            "提示: 无成交或过期数据不会被当作“没有 gap”。",
        ]
    )


def state_gap_pct(snapshot: Any) -> float:
    try:
        return float(snapshot.get("gap_pct", 0.0))
    except (AttributeError, TypeError, ValueError):
        return 0.0


def meets_threshold(value: float, threshold: float) -> bool:
    return value + 1e-9 >= threshold


def build_open_incremental_message(
    results: list[OpenResult],
    now_ny: datetime,
    premarket_alerts: dict[str, Any],
    coverage_count: int,
    missing: dict[str, str],
) -> str:
    lines = [
        "⚡ <b>ETF Gap 开盘即时更新</b>",
        f"时间: {html.escape(now_ny.strftime('%Y-%m-%d %H:%M %Z'))}",
        "口径: 当日首根有成交常规盘 1 分钟 K 线 Open（Yahoo 代理值）",
        "",
    ]
    for result in sorted(results, key=lambda item: item.gap_pct, reverse=True):
        if meets_threshold(result.gap_pct, GAP_THRESHOLD_PCT):
            category = (
                "盘前预警已确认"
                if result.symbol in premarket_alerts
                else "开盘新增"
            )
        else:
            category = (
                "盘前信号已消失"
                if result.symbol in premarket_alerts
                else "此前开盘信号已更正"
            )
        lines.append(
            f"• <b>{html.escape(result.symbol)}</b> [{category}]: "
            f"{result.gap_pct:+.2f}% | "
            f"{result.prev_close:.2f} → {result.today_open:.2f} "
            f"({result.first_bar_time.strftime('%H:%M')} ET)"
        )

    lines.extend(
        [
            "",
            (
                f"当前覆盖 {coverage_count}/{len(ETFS)}；"
                f"仍缺 {len(missing)}。系统会继续补齐并发送最终汇总。"
            ),
        ]
    )
    return "\n".join(lines)


def build_open_message(
    observations: list[OpenResult],
    now_ny: datetime,
    market_open_ny: datetime,
    premarket_alerts: dict[str, Any],
    missing: dict[str, str],
    corrected_events: list[OpenResult] | None = None,
) -> str:
    by_symbol = {result.symbol: result for result in observations}
    official = {
        result.symbol: result
        for result in observations
        if meets_threshold(result.gap_pct, GAP_THRESHOLD_PCT)
    }
    premarket_symbols = set(premarket_alerts)
    official_symbols = set(official)
    observed_symbols = set(by_symbol)
    resolved_premarket = premarket_symbols & observed_symbols
    unresolved_premarket = sorted(premarket_symbols - observed_symbols)

    confirmed = sorted(
        resolved_premarket & official_symbols,
        key=lambda symbol: official[symbol].gap_pct,
        reverse=True,
    )
    faded = sorted(resolved_premarket - official_symbols)
    new_at_open = sorted(
        official_symbols - premarket_symbols,
        key=lambda symbol: official[symbol].gap_pct,
        reverse=True,
    )

    delay_minutes = max(
        0,
        int((now_ny - market_open_ny).total_seconds() // 60),
    )
    lines = [
        "✅ <b>ETF Gap 开盘最终汇总</b>",
        f"时间: {html.escape(now_ny.strftime('%Y-%m-%d %H:%M %Z'))}",
        (
            "确认口径: 当日首根有成交的常规盘 1 分钟 K 线 Open "
            "（Yahoo 代理值）"
            f"≥ 前收 + {GAP_THRESHOLD_PCT:.2f}%"
        ),
        f"任务执行于开盘后 {delay_minutes} 分钟；数据覆盖 {len(observations)}/{len(ETFS)}",
        "",
    ]

    if premarket_symbols:
        lines.append("<b>盘前预警且开盘确认</b>")
        if confirmed:
            for symbol in confirmed:
                result = official[symbol]
                lines.append(
                    f"• <b>{html.escape(symbol)}</b>: "
                    f"盘前 +{state_gap_pct(premarket_alerts[symbol]):.2f}% "
                    f"→ 开盘 +{result.gap_pct:.2f}% "
                    f"({result.prev_close:.2f} → {result.today_open:.2f}, "
                    f"{result.first_bar_time.strftime('%H:%M')} ET)"
                )
        else:
            lines.append("• 无")

        lines.extend(["", "<b>盘前有、开盘消失</b>"])
        if faded:
            for symbol in faded:
                premarket_gap = state_gap_pct(premarket_alerts[symbol])
                result = by_symbol[symbol]
                lines.append(
                    f"• <b>{html.escape(symbol)}</b>: "
                    f"盘前 +{premarket_gap:.2f}% "
                    f"→ 开盘 {result.gap_pct:+.2f}% "
                    f"({result.first_bar_time.strftime('%H:%M')} ET)"
                )
        else:
            lines.append("• 无")

        if unresolved_premarket:
            lines.extend(["", "<b>仍待确认（不会标记完成）</b>"])
            for symbol in unresolved_premarket:
                lines.append(f"• <b>{html.escape(symbol)}</b>: 开盘数据缺失")

        lines.extend(["", "<b>开盘新增</b>"])
        if new_at_open:
            for symbol in new_at_open:
                result = official[symbol]
                lines.append(
                    f"• <b>{html.escape(symbol)}</b>: "
                    f"+{result.gap_pct:.2f}% "
                    f"({result.prev_close:.2f} → {result.today_open:.2f}, "
                    f"{result.first_bar_time.strftime('%H:%M')} ET)"
                )
        else:
            lines.append("• 无")
    elif official:
        lines.append("<b>正式跳空</b>")
        for result in sorted(
            official.values(),
            key=lambda item: item.gap_pct,
            reverse=True,
        ):
            latest = (
                f"{result.latest_price:.2f}"
                if result.latest_price is not None
                else "N/A"
            )
            lines.append(
                f"• <b>{html.escape(result.symbol)}</b>: "
                f"+{result.gap_pct:.2f}% | "
                f"{result.prev_close:.2f} → {result.today_open:.2f} | "
                f"{result.first_bar_time.strftime('%H:%M')} ET | Latest {latest}"
            )
    else:
        if missing:
            lines.append(
                f"结果: 在已取得开盘数据的 {len(observations)} 个 ETF 中，"
                f"没有发现跳空 ≥ {GAP_THRESHOLD_PCT:.2f}%；扫描并非全覆盖。"
            )
        else:
            lines.append(
                f"结果: 没有 ETF 开盘跳空高于 {GAP_THRESHOLD_PCT:.2f}%"
            )

    if corrected_events:
        lines.extend(["", "<b>对先前即时通知的更正</b>"])
        for result in sorted(
            corrected_events,
            key=lambda item: item.symbol,
        ):
            if meets_threshold(result.gap_pct, GAP_THRESHOLD_PCT):
                outcome = f"更新为 {result.gap_pct:+.2f}%"
            else:
                outcome = f"已撤销，最终为 {result.gap_pct:+.2f}%"
            lines.append(
                f"• <b>{html.escape(result.symbol)}</b>: {outcome} "
                f"({result.first_bar_time.strftime('%H:%M')} ET)"
            )

    if missing:
        lines.extend(
            [
                "",
                (
                    f"数据缺失 {len(missing)} 个: "
                    f"{html.escape(summarize_missing(missing))}"
                ),
            ]
        )
    lines.extend(["", "提示: 若曾收到“开盘即时更新”，以本最终汇总为准。"])
    return "\n".join(lines)


def send_data_issue_once(
    key: str,
    stage: str,
    details: str,
    now_ny: datetime,
    state: dict[str, Any],
    will_retry: bool = True,
) -> None:
    sent_keys = state.setdefault("data_failure_alerts", [])
    if key in sent_keys and not RESEND:
        print(f"Data issue '{key}' was already reported today.")
        return

    message = "\n".join(
        [
            "⚠️ <b>ETF Scanner 数据异常</b>",
            f"时间: {html.escape(now_ny.strftime('%Y-%m-%d %H:%M %Z'))}",
            f"阶段: {html.escape(stage)}",
            f"详情: {html.escape(details)}",
            (
                "本次不会把数据缺失误判为“没有 gap”。"
                + (
                    "后续定时任务会继续重试。"
                    if will_retry
                    else "已到最终截止时间，请查看日志或手动重跑。"
                )
            ),
        ]
    )
    send_telegram(message)
    if key not in sent_keys:
        sent_keys.append(key)
    save_state(STATE_FILE, state)


def run_premarket_scan(
    now_ny: datetime,
    market_open_ny: datetime,
    market_close_ny: datetime,
    previous_closes: dict[str, float],
    state: dict[str, Any],
    allow_no_signal: bool = True,
) -> int:
    batch = download_batch(period="2d", interval="1m", prepost=True)
    if batch.empty:
        send_data_issue_once(
            key=f"premarket_batch:{now_ny.date().isoformat()}",
            stage="盘前扫描",
            details="Yahoo 返回的 1 分钟盘前批量数据为空。",
            now_ny=now_ny,
            state=state,
        )
        return 1

    observations: list[PremarketResult] = []
    missing: dict[str, str] = {}
    for symbol in ETFS:
        previous_close = previous_closes.get(symbol)
        if previous_close is None:
            missing[symbol] = "missing_previous_close"
            continue
        result, reason = get_premarket_snapshot(
            symbol=symbol,
            frame=extract_symbol_frame(batch, symbol),
            previous_close=previous_close,
            now_ny=now_ny,
            market_open_ny=market_open_ny,
        )
        if result is None:
            missing[symbol] = reason or "unknown"
        else:
            observations.append(result)

    phase_check_ny = datetime.now(NY_TZ)
    if phase_check_ny >= market_open_ny:
        print(
            "The market opened while premarket data was being processed; "
            "switching to open confirmation before sending."
        )
        return run_open_scan(
            now_ny=phase_check_ny,
            market_open_ny=market_open_ny,
            market_close_ny=market_close_ny,
            previous_closes=previous_closes,
            state=state,
            reference_coverage_ok=(
                len(previous_closes) / len(ETFS) * 100.0
                >= MIN_REFERENCE_COVERAGE_PCT
            ),
        )

    candidates = [
        result
        for result in observations
        if (
            meets_threshold(
                result.robust_gap_pct,
                PREMARKET_THRESHOLD_PCT,
            )
            and meets_threshold(
                result.latest_gap_pct,
                PREMARKET_THRESHOLD_PCT,
            )
        )
    ]
    alerted = state.setdefault("premarket_alerts", {})
    if not isinstance(alerted, dict):
        alerted = {}
        state["premarket_alerts"] = alerted
    new_candidates = (
        candidates
        if RESEND
        else [result for result in candidates if result.symbol not in alerted]
    )
    coverage_pct = len(observations) / len(ETFS) * 100.0

    print(
        f"Premarket scan: valid={len(observations)}/{len(ETFS)}, "
        f"candidates={len(candidates)}, new={len(new_candidates)}, "
        f"missing={summarize_missing(missing)}"
    )

    if new_candidates:
        send_telegram(
            build_premarket_message(
                new_candidates,
                now_ny,
                len(observations),
                missing,
            )
        )
        for result in new_candidates:
            alerted[result.symbol] = {
                "gap_pct": result.robust_gap_pct,
                "latest_gap_pct": result.latest_gap_pct,
                "price": result.robust_price,
                "latest_price": result.latest_price,
                "time": result.latest_time.isoformat(),
            }
        save_state(STATE_FILE, state)
    elif (
        not candidates
        and PREMARKET_NOTIFY_WHEN_NONE
        and allow_no_signal
        and coverage_pct >= PREMARKET_MIN_COVERAGE_PCT
        and (RESEND or not state.get("premarket_no_signal_sent"))
    ):
        send_telegram(
            build_premarket_no_signal_message(
                now_ny,
                len(observations),
                missing,
            )
        )
        state["premarket_no_signal_sent"] = True
        save_state(STATE_FILE, state)
    else:
        if (
            not candidates
            and PREMARKET_NOTIFY_WHEN_NONE
            and (
                not allow_no_signal
                or coverage_pct < PREMARKET_MIN_COVERAGE_PCT
            )
        ):
            print(
                "Premarket no-signal message suppressed because reference "
                "or premarket coverage is below its configured threshold "
                f"(premarket={coverage_pct:.1f}%)."
            )
        print("No new premarket Telegram message is needed.")

    return 0


def run_open_scan(
    now_ny: datetime,
    market_open_ny: datetime,
    market_close_ny: datetime,
    previous_closes: dict[str, float],
    state: dict[str, Any],
    reference_coverage_ok: bool = True,
) -> int:
    resolved: dict[str, OpenResult] = {}
    latest_missing: dict[str, str] = {}
    premarket_alerts = state.get("premarket_alerts", {})
    if not isinstance(premarket_alerts, dict):
        premarket_alerts = {}
    notifications = state.setdefault("open_symbol_notifications", {})
    if not isinstance(notifications, dict):
        notifications = {}
        state["open_symbol_notifications"] = notifications
    previously_notified_gaps = {
        symbol
        for symbol, snapshot in notifications.items()
        if isinstance(snapshot, dict) and snapshot.get("status") == "gap"
    }
    must_resolve_symbols = set(premarket_alerts) | previously_notified_gaps

    required_count = max(
        1,
        int(len(ETFS) * OPEN_MIN_COVERAGE_PCT / 100.0 + 0.9999),
    )
    attempt_now_ny = now_ny

    for attempt in range(1, OPEN_DATA_RETRY_COUNT + 1):
        attempt_now_ny = datetime.now(NY_TZ)
        provisional_before_attempt = {
            symbol
            for symbol, result in resolved.items()
            if result.first_bar_time.replace(second=0, microsecond=0)
            > market_open_ny.replace(second=0, microsecond=0)
        }
        unresolved_symbols = [
            symbol
            for symbol in ETFS
            if (
                symbol in previous_closes
                and (
                    symbol not in resolved
                    or symbol in provisional_before_attempt
                )
            )
        ]
        batch = download_batch(
            period="2d",
            interval="1m",
            prepost=False,
            tickers=unresolved_symbols,
        )
        for symbol in unresolved_symbols:
            previous_close = previous_closes.get(symbol)
            if previous_close is None:
                continue
            result, reason = get_open_snapshot(
                symbol=symbol,
                frame=extract_symbol_frame(batch, symbol),
                previous_close=previous_close,
                now_ny=attempt_now_ny,
                market_open_ny=market_open_ny,
                market_close_ny=market_close_ny,
            )
            if result is None:
                if symbol not in resolved:
                    latest_missing[symbol] = reason or "unknown"
            else:
                previous_result = resolved.get(symbol)
                if (
                    previous_result is None
                    or result.first_bar_time <= previous_result.first_bar_time
                ):
                    resolved[symbol] = result
                latest_missing.pop(symbol, None)

        delay_minutes = max(
            0,
            int((attempt_now_ny - market_open_ny).total_seconds() // 60),
        )
        provisional_symbols = {
            symbol
            for symbol, result in resolved.items()
            if result.first_bar_time.replace(second=0, microsecond=0)
            > market_open_ny.replace(second=0, microsecond=0)
        }
        provisional_deadline_reached = (
            delay_minutes >= OPEN_PROVISIONAL_AFTER_MINUTES
        )
        hard_deadline_reached = delay_minutes >= OPEN_HARD_DEADLINE_MINUTES
        accepted_symbols = (
            set(resolved)
            if provisional_deadline_reached
            else set(resolved) - provisional_symbols
        )
        missing = {
            symbol: (
                "missing_previous_close"
                if symbol not in previous_closes
                else "provisional_late_first_trade"
                if symbol in provisional_symbols
                and not provisional_deadline_reached
                else latest_missing.get(symbol, "missing_open_data")
            )
            for symbol in ETFS
            if symbol not in accepted_symbols
        }
        unresolved_required = sorted(must_resolve_symbols - accepted_symbols)
        full_coverage = len(accepted_symbols) == len(ETFS)
        minimum_coverage = (
            len(accepted_symbols) >= required_count
            and reference_coverage_ok
        )
        safe_to_finalize_partial = (
            hard_deadline_reached
            and minimum_coverage
            and not unresolved_required
        )

        print(
            f"Open data attempt {attempt}/{OPEN_DATA_RETRY_COUNT}: "
            f"best={len(resolved)}/{len(ETFS)}, "
            f"accepted={len(accepted_symbols)}, "
            f"provisional={sorted(provisional_symbols) or 'none'}, "
            f"required={required_count}, "
            f"unresolved_required={unresolved_required or 'none'}, "
            f"missing={summarize_missing(missing)}"
        )
        if full_coverage or safe_to_finalize_partial:
            break
        if attempt < OPEN_DATA_RETRY_COUNT:
            time.sleep(OPEN_DATA_RETRY_SECONDS)

    delay_minutes = max(
        0,
        int((attempt_now_ny - market_open_ny).total_seconds() // 60),
    )
    provisional_deadline_reached = (
        delay_minutes >= OPEN_PROVISIONAL_AFTER_MINUTES
    )
    hard_deadline_reached = delay_minutes >= OPEN_HARD_DEADLINE_MINUTES
    provisional_symbols = {
        symbol
        for symbol, result in resolved.items()
        if result.first_bar_time.replace(second=0, microsecond=0)
        > market_open_ny.replace(second=0, microsecond=0)
    }
    accepted_symbols = (
        set(resolved)
        if provisional_deadline_reached
        else set(resolved) - provisional_symbols
    )
    observations = [
        resolved[symbol] for symbol in ETFS if symbol in accepted_symbols
    ]
    missing = {
        symbol: (
            "missing_previous_close"
            if symbol not in previous_closes
            else "provisional_late_first_trade"
            if symbol in provisional_symbols
            and not provisional_deadline_reached
            else latest_missing.get(symbol, "missing_open_data")
        )
        for symbol in ETFS
        if symbol not in accepted_symbols
    }
    unresolved_required = sorted(must_resolve_symbols - accepted_symbols)
    coverage_is_acceptable = (
        len(observations) >= required_count
        and reference_coverage_ok
    )
    official = [
        result
        for result in observations
        if meets_threshold(result.gap_pct, GAP_THRESHOLD_PCT)
    ]

    event_results: list[OpenResult] = []
    event_snapshots: dict[str, dict[str, Any]] = {}
    for result in observations:
        is_gap = meets_threshold(result.gap_pct, GAP_THRESHOLD_PCT)
        is_premarket_resolution = result.symbol in premarket_alerts
        was_previously_notified = result.symbol in notifications
        if not (is_gap or is_premarket_resolution or was_previously_notified):
            continue
        status = (
            "gap"
            if is_gap
            else "premarket_faded"
            if is_premarket_resolution
            else "corrected_below_threshold"
        )
        snapshot = {
            "status": status,
            "gap_pct": round(result.gap_pct, 6),
            "today_open": round(result.today_open, 6),
            "first_bar_time": result.first_bar_time.isoformat(),
        }
        if RESEND or notifications.get(result.symbol) != snapshot:
            event_results.append(result)
            event_snapshots[result.symbol] = snapshot
    changed_prior_results = [
        result for result in event_results if result.symbol in notifications
    ]

    ready_to_finalize = not missing or (
        hard_deadline_reached
        and coverage_is_acceptable
        and not unresolved_required
    )
    if not ready_to_finalize:
        if event_results:
            send_telegram(
                build_open_incremental_message(
                    results=event_results,
                    now_ny=attempt_now_ny,
                    premarket_alerts=premarket_alerts,
                    coverage_count=len(observations),
                    missing=missing,
                )
            )
            notifications.update(event_snapshots)
            save_state(STATE_FILE, state)

        issue_key = (
            f"open_coverage_final:{now_ny.date().isoformat()}"
            if hard_deadline_reached
            else f"open_coverage:{now_ny.date().isoformat()}"
        )
        send_data_issue_once(
            key=issue_key,
            stage=(
                "开盘确认最终失败"
                if hard_deadline_reached
                else "开盘确认仍在补数据"
            ),
            details=(
                f"累计取得 {len(observations)}/{len(ETFS)} 个首笔常规盘开盘价，"
                f"最低要求 {required_count}；"
                f"前收覆盖门槛={'满足' if reference_coverage_ok else '未满足'}；"
                f"待确认的已通知标的={unresolved_required or '无'}；"
                f"{summarize_missing(missing)}"
            ),
            now_ny=attempt_now_ny,
            state=state,
            will_retry=not hard_deadline_reached,
        )
        return 1

    should_send = (
        bool(official)
        or bool(premarket_alerts)
        or bool(event_results)
        or NOTIFY_WHEN_NONE
        or RESEND
    )
    print(
        f"Open scan complete: official_alerts={len(official)}, "
        f"premarket_alerts={len(premarket_alerts)}, send={should_send}"
    )
    if should_send:
        send_telegram(
            build_open_message(
                observations=observations,
                now_ny=attempt_now_ny,
                market_open_ny=market_open_ny,
                premarket_alerts=premarket_alerts,
                missing=missing,
                corrected_events=changed_prior_results,
            )
        )
        notifications.update(event_snapshots)

    state["open_confirmation_sent"] = True
    state["open_confirmed_symbols"] = sorted(
        result.symbol for result in official
    )
    save_state(STATE_FILE, state)
    return 0


def main() -> int:
    validate_config()
    now_ny = datetime.now(NY_TZ)
    today_date = now_ny.date()
    session = get_today_nyse_session(now_ny)
    if session is None:
        print(f"{today_date} is not an NYSE trading day. Skip.")
        return 0

    market_open_ny, market_close_ny = session
    state = load_state(STATE_FILE, today_date)

    requested_phase = os.getenv("SCAN_PHASE", "auto").strip().lower() or "auto"
    if requested_phase not in {"auto", "premarket", "open"}:
        raise ValueError(
            "SCAN_PHASE must be one of: auto, premarket, open; "
            f"got {requested_phase!r}"
        )

    phase = (
        "premarket"
        if requested_phase == "auto" and now_ny < market_open_ny
        else "open"
        if requested_phase == "auto"
        else requested_phase
    )

    if phase == "premarket" and now_ny >= market_open_ny:
        print(
            "The requested premarket job started after 09:30 ET; "
            "automatically switching to formal open confirmation."
        )
        phase = "open"

    if phase == "open" and now_ny < market_open_ny:
        print(
            "Formal opening data does not exist yet. "
            f"Now={now_ny.isoformat()}, open={market_open_ny.isoformat()}. Skip."
        )
        return 0

    earliest_premarket_scan = market_open_ny - timedelta(
        minutes=PREMARKET_MAX_LEAD_MINUTES
    )
    if (
        phase == "premarket"
        and now_ny < earliest_premarket_scan
        and not FORCE_RUN
    ):
        print(
            "Too early for the configured premarket window. "
            f"Now={now_ny.isoformat()}, earliest={earliest_premarket_scan.isoformat()}."
        )
        return 0

    if phase == "open" and state.get("open_confirmation_sent") and not RESEND:
        print("Today's formal open confirmation is already complete. Skip.")
        return 0

    print(
        f"Starting phase={phase}, requested={requested_phase}, "
        f"now={now_ny.isoformat()}, open={market_open_ny.isoformat()}, "
        f"force={FORCE_RUN}, resend={RESEND}."
    )
    print(
        f"Thresholds: premarket={PREMARKET_THRESHOLD_PCT:.2f}%, "
        f"open={GAP_THRESHOLD_PCT:.2f}%; "
        f"premarket bars>={PREMARKET_MIN_VALID_BARS}, "
        f"volume>={PREMARKET_MIN_RECENT_VOLUME}, "
        f"staleness<={PREMARKET_MAX_STALENESS_MINUTES}m."
    )

    cached_closes_raw = state.get("previous_closes", {})
    previous_closes: dict[str, float] = {}
    if isinstance(cached_closes_raw, dict):
        for symbol in ETFS:
            try:
                value = float(cached_closes_raw[symbol])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(value) and value > 0:
                previous_closes[symbol] = value

    missing_reference_symbols = [
        symbol for symbol in ETFS if symbol not in previous_closes
    ]
    if not missing_reference_symbols:
        reference_missing: dict[str, str] = {}
        print("Using today's cached previous closes.")
    else:
        print(
            "Refreshing missing previous closes: "
            f"{', '.join(missing_reference_symbols)}"
        )
        daily_batch = download_batch(
            period="10d",
            interval="1d",
            prepost=False,
            tickers=missing_reference_symbols,
            actions=True,
        )
        refreshed_closes, refresh_missing = get_previous_closes(
            daily_batch,
            today_date,
            symbols=missing_reference_symbols,
        )
        previous_closes.update(refreshed_closes)
        reference_missing = {
            symbol: refresh_missing.get(symbol, "missing_previous_close")
            for symbol in ETFS
            if symbol not in previous_closes
        }
        state["previous_closes"] = previous_closes
        save_state(STATE_FILE, state)

    reference_coverage_pct = len(previous_closes) / len(ETFS) * 100.0
    print(
        f"Previous-close coverage: {len(previous_closes)}/{len(ETFS)} "
        f"({reference_coverage_pct:.1f}%). "
        f"Missing={summarize_missing(reference_missing)}"
    )
    reference_coverage_ok = (
        reference_coverage_pct >= MIN_REFERENCE_COVERAGE_PCT
    )
    if not reference_coverage_ok:
        print(
            "Previous-close coverage is below the completion threshold; "
            "available symbols will still be scanned for positive signals."
        )
        if phase == "premarket":
            send_data_issue_once(
                key=f"reference_coverage:{today_date.isoformat()}",
                stage="前收基准",
                details=(
                    f"前收覆盖率只有 {reference_coverage_pct:.1f}% "
                    f"({len(previous_closes)}/{len(ETFS)})，低于 "
                    f"{MIN_REFERENCE_COVERAGE_PCT:.1f}%；"
                    f"{summarize_missing(reference_missing)}"
                ),
                now_ny=now_ny,
                state=state,
            )

    if phase == "premarket":
        status = run_premarket_scan(
            now_ny=now_ny,
            market_open_ny=market_open_ny,
            market_close_ny=market_close_ny,
            previous_closes=previous_closes,
            state=state,
            allow_no_signal=reference_coverage_ok,
        )
        return 1 if status == 0 and not reference_coverage_ok else status

    return run_open_scan(
        now_ny=now_ny,
        market_open_ny=market_open_ny,
        market_close_ny=market_close_ny,
        previous_closes=previous_closes,
        state=state,
        reference_coverage_ok=reference_coverage_ok,
    )


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"Fatal error: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
