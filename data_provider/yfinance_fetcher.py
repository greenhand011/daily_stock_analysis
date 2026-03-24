# -*- coding: utf-8 -*-
"""Yahoo Finance fallback fetcher."""

import logging
import re

import pandas as pd
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .base import BaseFetcher, DataFetchError, STANDARD_COLUMNS

logger = logging.getLogger(__name__)


class YfinanceFetcher(BaseFetcher):
    name = "YfinanceFetcher"
    priority = 4

    def __init__(self):
        pass

    def _convert_stock_code(self, stock_code: str) -> str:
        code = stock_code.strip()
        upper_code = code.upper()

        if upper_code.endswith((".SS", ".SZ", ".HK")):
            return upper_code

        if re.fullmatch(r"HK\d{5}", upper_code):
            return f"{upper_code[2:]}.HK"

        if re.fullmatch(r"\d{5}", code):
            return f"{code}.HK"

        if re.fullmatch(r"[A-Z][A-Z0-9.-]{0,14}", upper_code):
            return upper_code

        cleaned_code = code.replace(".SH", "").replace(".sh", "")
        if cleaned_code.startswith(("600", "601", "603", "688")):
            return f"{cleaned_code}.SS"
        if cleaned_code.startswith(("000", "002", "300")):
            return f"{cleaned_code}.SZ"

        logger.warning("Unable to infer market for %s, passing through as-is", code)
        return upper_code

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _fetch_raw_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        import yfinance as yf

        yf_code = self._convert_stock_code(stock_code)
        logger.debug("Calling yfinance.download(%s, %s, %s)", yf_code, start_date, end_date)

        try:
            df = yf.download(
                tickers=yf_code,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=True,
            )
            if df.empty:
                raise DataFetchError(f"Yahoo Finance returned no data for {stock_code}")
            return df
        except Exception as exc:
            if isinstance(exc, DataFetchError):
                raise
            raise DataFetchError(f"Yahoo Finance fetch failed: {exc}") from exc

    def _normalize_data(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        df = df.copy()
        yf_code = self._convert_stock_code(stock_code)

        # yfinance may return a MultiIndex even for one ticker. Flatten it to a
        # single OHLCV table before any numeric operations.
        if isinstance(df.columns, pd.MultiIndex):
            if yf_code in df.columns.get_level_values(-1):
                df = df.xs(yf_code, axis=1, level=-1, drop_level=True)
            elif yf_code in df.columns.get_level_values(0):
                df = df.xs(yf_code, axis=1, level=0, drop_level=True)
            else:
                expected = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
                best_level = 0
                best_score = -1
                for level in range(df.columns.nlevels):
                    values = {str(v) for v in df.columns.get_level_values(level)}
                    score = len(values & expected)
                    if score > best_score:
                        best_level = level
                        best_score = score
                df.columns = [str(col) for col in df.columns.get_level_values(best_level)]

        df = df.reset_index()

        normalized_columns = []
        for idx, col in enumerate(df.columns):
            column_name = str(col).strip()
            if idx == 0 and column_name.lower() in {"date", "datetime", "index"}:
                normalized_columns.append("date")
            else:
                normalized_columns.append(column_name.lower())
        df.columns = normalized_columns

        if "adj close" in df.columns and "close" not in df.columns:
            df = df.rename(columns={"adj close": "close"})

        if "close" in df.columns:
            close_series = pd.to_numeric(df["close"], errors="coerce")
            df["close"] = close_series
            df["pct_chg"] = close_series.pct_change() * 100
            df["pct_chg"] = df["pct_chg"].fillna(0).round(2)

        if "volume" in df.columns and "close" in df.columns:
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
            df["amount"] = df["volume"] * df["close"]
        else:
            df["amount"] = 0

        df["code"] = stock_code

        keep_cols = ["code"] + STANDARD_COLUMNS
        existing_cols = [col for col in keep_cols if col in df.columns]
        return df[existing_cols]


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    fetcher = YfinanceFetcher()
    try:
        frame = fetcher.get_daily_data("600519")
        print(f"Fetched {len(frame)} rows")
        print(frame.tail())
    except Exception as exc:
        print(f"Fetch failed: {exc}")
