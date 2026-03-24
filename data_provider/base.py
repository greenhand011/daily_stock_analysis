# -*- coding: utf-8 -*-
"""
===================================
数据源基类与管理器
===================================
"""

import logging
import random
import re
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, List, Tuple

import pandas as pd
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

logger = logging.getLogger(__name__)

STANDARD_COLUMNS = ['date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'pct_chg']


class DataFetchError(Exception):
    pass


class RateLimitError(DataFetchError):
    pass


class DataSourceUnavailableError(DataFetchError):
    pass


class BaseFetcher(ABC):
    name: str = "BaseFetcher"
    priority: int = 99

    @abstractmethod
    def _fetch_raw_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        pass

    @abstractmethod
    def _normalize_data(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        pass

    def get_daily_data(
        self,
        stock_code: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        days: int = 30
    ) -> pd.DataFrame:
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if start_date is None:
            from datetime import timedelta
            start_dt = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=days * 2)
            start_date = start_dt.strftime('%Y-%m-%d')

        logger.info(f"[{self.name}] 获取 {stock_code} 数据: {start_date} ~ {end_date}")

        try:
            raw_df = self._fetch_raw_data(stock_code, start_date, end_date)
            if raw_df is None or raw_df.empty:
                raise DataFetchError(f"[{self.name}] 未获取到 {stock_code} 的数据")
            df = self._normalize_data(raw_df, stock_code)
            df = self._clean_data(df)
            df = self._calculate_indicators(df)
            logger.info(f"[{self.name}] {stock_code} 获取成功，共 {len(df)} 条数据")
            return df
        except Exception as e:
            logger.error(f"[{self.name}] 获取 {stock_code} 失败: {str(e)}")
            raise DataFetchError(f"[{self.name}] {stock_code}: {str(e)}") from e

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount', 'pct_chg']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=['close', 'volume'])
        df = df.sort_values('date', ascending=True).reset_index(drop=True)
        return df

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['ma5'] = df['close'].rolling(window=5, min_periods=1).mean()
        df['ma10'] = df['close'].rolling(window=10, min_periods=1).mean()
        df['ma20'] = df['close'].rolling(window=20, min_periods=1).mean()
        avg_volume_5 = df['volume'].rolling(window=5, min_periods=1).mean()
        df['volume_ratio'] = df['volume'] / avg_volume_5.shift(1)
        df['volume_ratio'] = df['volume_ratio'].fillna(1.0)
        for col in ['ma5', 'ma10', 'ma20', 'volume_ratio']:
            if col in df.columns:
                df[col] = df[col].round(2)
        return df

    @staticmethod
    def random_sleep(min_seconds: float = 1.0, max_seconds: float = 3.0) -> None:
        sleep_time = random.uniform(min_seconds, max_seconds)
        logger.debug(f"随机休眠 {sleep_time:.2f} 秒...")
        time.sleep(sleep_time)


class DataFetcherManager:
    def __init__(self, fetchers: Optional[List[BaseFetcher]] = None):
        self._fetchers: List[BaseFetcher] = []
        if fetchers:
            self._fetchers = sorted(fetchers, key=lambda f: f.priority)
        else:
            self._init_default_fetchers()

    def _init_default_fetchers(self) -> None:
        """初始化默认数据源，单个导入失败不影响整体"""
        fetchers = []

        # 1. EfinanceFetcher
        try:
            from .efinance_fetcher import EfinanceFetcher
            fetchers.append(EfinanceFetcher())
            logger.debug("已加载 EfinanceFetcher")
        except ImportError as e:
            logger.warning(f"EfinanceFetcher 导入失败: {e}")

        # 2. AkshareFetcher
        try:
            from .akshare_fetcher import AkshareFetcher
            fetchers.append(AkshareFetcher())
            logger.debug("已加载 AkshareFetcher")
        except ImportError as e:
            logger.warning(f"AkshareFetcher 导入失败: {e}")

        # 3. TushareFetcher
        try:
            from .tushare_fetcher import TushareFetcher
            fetchers.append(TushareFetcher())
            logger.debug("已加载 TushareFetcher")
        except ImportError as e:
            logger.warning(f"TushareFetcher 导入失败: {e}")

        # 4. BaostockFetcher
        try:
            from .baostock_fetcher import BaostockFetcher
            fetchers.append(BaostockFetcher())
            logger.debug("已加载 BaostockFetcher")
        except ImportError as e:
            logger.warning(f"BaostockFetcher 导入失败: {e}")

        # 5. YfinanceFetcher
        try:
            from .yfinance_fetcher import YfinanceFetcher
            fetchers.append(YfinanceFetcher())
            logger.debug("已加载 YfinanceFetcher")
        except ImportError as e:
            logger.warning(f"YfinanceFetcher 导入失败: {e}")

        # 6. TencentFetcher（可选，即使不存在也不影响）
        try:
            from .tencent_fetcher import TencentFetcher
            fetchers.append(TencentFetcher())
            logger.debug("已加载 TencentFetcher")
        except ImportError:
            logger.debug("TencentFetcher 未安装或不存在，跳过")
        except Exception as e:
            logger.warning(f"TencentFetcher 初始化失败: {e}，跳过")

        # 按优先级排序
        self._fetchers = sorted(fetchers, key=lambda f: f.priority)
        logger.info(f"已初始化 {len(self._fetchers)} 个数据源: " +
                    ", ".join([f.name for f in self._fetchers]))

    def add_fetcher(self, fetcher: BaseFetcher) -> None:
        self._fetchers.append(fetcher)
        self._fetchers.sort(key=lambda f: f.priority)

    def get_daily_data(
        self,
        stock_code: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        days: int = 30
    ) -> Tuple[pd.DataFrame, str]:
        errors = []
        fetchers = self._fetchers

        # US tickers should go to Yahoo Finance first. The mainland data
        # sources add long timeouts and noisy errors for symbols like MSFT/IVV.
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9.-]{0,14}", stock_code):
            preferred = [f for f in self._fetchers if f.name == "YfinanceFetcher"]
            if preferred:
                fetchers = preferred

        for fetcher in fetchers:
            try:
                logger.info(f"尝试使用 [{fetcher.name}] 获取 {stock_code}...")
                df = fetcher.get_daily_data(
                    stock_code=stock_code,
                    start_date=start_date,
                    end_date=end_date,
                    days=days
                )
                if df is not None and not df.empty:
                    logger.info(f"[{fetcher.name}] 成功获取 {stock_code}")
                    return df, fetcher.name
            except Exception as e:
                error_msg = f"[{fetcher.name}] 失败: {str(e)}"
                logger.warning(error_msg)
                errors.append(error_msg)
                continue

        error_summary = f"所有数据源获取 {stock_code} 失败:\n" + "\n".join(errors)
        logger.error(error_summary)
        raise DataFetchError(error_summary)

    @property
    def available_fetchers(self) -> List[str]:
        return [f.name for f in self._fetchers]
