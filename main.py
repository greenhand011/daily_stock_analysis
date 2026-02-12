#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CIO 多资产智能分析系统
支持：
- 腾讯实时行情
- 历史K线
- Gemini AI 分析
- Telegram 推送

增强健壮性：单个数据源/组件失败不影响整体运行
"""

import os
import sys
import argparse
import logging
import time
import json
import re
import pandas as pd

from datetime import datetime, date
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import List, Dict, Optional

# ================= 编码修复 =================
try:
    import codecs
    if sys.stdout.encoding != "UTF-8" and hasattr(sys.stdout, "buffer"):
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer)
    if sys.stderr.encoding != "UTF-8" and hasattr(sys.stderr, "buffer"):
        sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer)
except:
    pass

os.environ["PYTHONIOENCODING"] = "utf-8"

# ================= 核心模块导入（带降级处理）=================

from config import get_config, Config
from storage import get_db

# ---------- 数据源管理器：尝试初始化，失败则降级 ----------
try:
    from data_provider import DataFetcherManager
    _fetcher_manager = DataFetcherManager()
    logger = logging.getLogger(__name__)
    logger.info("✅ DataFetcherManager 初始化成功")
except Exception as e:
    logger = logging.getLogger(__name__)
    logger.error(f"❌ DataFetcherManager 初始化失败: {e}，将使用降级数据源")
    # 降级方案：创建一个仅包含基本 fetcher 的管理器
    try:
        from data_provider.efinance_fetcher import EfinanceFetcher
        from data_provider.akshare_fetcher import AkshareFetcher
        from data_provider.base import DataFetcherManager
        _fetcher_manager = DataFetcherManager(fetchers=[
            EfinanceFetcher(),
            AkshareFetcher(),
        ])
    except Exception as e2:
        logger.error(f"❌ 降级数据源也失败: {e2}，将无法获取历史数据")
        _fetcher_manager = None

# ---------- AI 分析器：尝试初始化，失败则标记不可用 ----------
try:
    from analyzer import GeminiAnalyzer, AnalysisResult
    _analyzer = GeminiAnalyzer()
except Exception as e:
    logger.error(f"❌ GeminiAnalyzer 初始化失败: {e}，AI分析功能不可用")
    GeminiAnalyzer = None
    AnalysisResult = None
    _analyzer = None

# ---------- 通知服务：尝试初始化，失败则标记不可用 ----------
try:
    from notification import NotificationService
    _notifier = NotificationService()
except Exception as e:
    logger.error(f"❌ NotificationService 初始化失败: {e}，推送功能不可用")
    NotificationService = None
    _notifier = None

# ================= 日志配置 =================

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_SECTOR = "Macro"

logger = logging.getLogger(__name__)


def setup_logging(debug: bool = False, log_dir: str = "./logs") -> None:
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    today_str = datetime.now().strftime("%Y%m%d")
    log_file = log_path / f"stock_analysis_{today_str}.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if debug else logging.INFO)

    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
    root_logger.addHandler(console_handler)

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
    root_logger.addHandler(file_handler)


# ================= Pipeline =================

class StockAnalysisPipeline:

    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()

        self.portfolio = self._load_portfolio_config()
        self.db = get_db()
        self.fetcher_manager = _fetcher_manager      # 使用全局降级后的管理器
        self.analyzer = _analyzer                    # 可能为 None
        self.notifier = _notifier                   # 可能为 None

        if self.analyzer is None:
            logger.warning("⚠️ AI分析器未初始化，将跳过AI分析")
        if self.notifier is None:
            logger.warning("⚠️ 通知服务未初始化，将跳过推送")

        logger.info("✅ AI-CIO 初始化完成（降级模式可能已启用）")

    # ---------- 加载持仓 ----------

    def _load_portfolio_config(self) -> dict:
        env_stock_list = os.getenv("STOCK_LIST", "")

        if env_stock_list:
            stock_codes = [
                code.strip() for code in env_stock_list.split(",") if code.strip()
            ]
            return {
                str(code): {
                    "code": str(code),
                    "cost": 0,
                    "shares": 0,
                    "name": f"资产{code}",
                    "sector": DEFAULT_SECTOR,
                }
                for code in stock_codes
            }

        path = "portfolio.json"
        if not os.path.exists(path):
            logger.warning("未找到 portfolio.json，将使用空持仓")
            return {}

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            final_data = {}
            if isinstance(data, dict):
                for code, info in data.items():
                    info = info if isinstance(info, dict) else {}
                    info["code"] = str(code)
                    info.setdefault("cost", 0)
                    info.setdefault("shares", 0)
                    info.setdefault("name", f"资产{code}")
                    info.setdefault("sector", DEFAULT_SECTOR)
                    final_data[str(code)] = info

            return final_data

        except Exception as e:
            logger.error(f"加载 portfolio.json 失败: {e}")
            return {}

    # ---------- 技术指标 ----------

    def _calculate_technical_indicators(self, df: pd.DataFrame) -> dict:
        df = df.sort_values("date")
        close = df["close"]

        return {
            "price": close.iloc[-1],
            "ma5": close.rolling(5).mean().iloc[-1],
            "ma20": close.rolling(20).mean().iloc[-1],
            "ma60": close.rolling(60).mean().iloc[-1],
            "rsi": None,
            "macd": None,
            "support": df["low"].tail(20).min(),
            "resistance": df["high"].tail(20).max(),
        }

    # ---------- 单资产处理 ----------

    def process_single_stock(self, code: str) -> Optional[object]:
        """
        处理单个资产分析流程
        :param code: 股票代码（可带前缀，如 'sz000858'；也可纯数字）
        """
        raw_code = str(code).strip()
        match = re.search(r"\d{6}", raw_code)
        if not match:
            logger.warning(f"无效代码: {raw_code}")
            return None

        asset_code = match.group(0)
        logger.info(f"========== 处理资产: {asset_code} (原始: {raw_code}) ==========")

        # ---------- 1. 腾讯实时行情（独立导入，异常不影响主流程）----------
        realtime_data = {}
        try:
            # 延迟导入，避免初始化阶段强制依赖
            from data_provider.tencent_fetcher import TencentFetcher
            tencent = TencentFetcher()
            realtime_data = tencent.get_realtime_quote(raw_code) or {}
            if realtime_data.get("name"):
                logger.info(f"腾讯行情获取成功: {realtime_data['name']}")
        except ImportError:
            logger.debug("TencentFetcher 未安装或不存在，跳过腾讯实时行情")
        except Exception as e:
            logger.warning(f"腾讯行情获取失败: {e}")

        # ---------- 2. 历史K线数据 ----------
        df = None
        source = None
        if self.fetcher_manager is not None:
            try:
                df, source = self.fetcher_manager.get_daily_data(asset_code, days=120)
                logger.info(f"历史数据源: {source}")
            except Exception as e:
                logger.warning(f"历史数据获取失败: {e}")
        else:
            logger.warning("数据源管理器不可用，无法获取历史数据")

        # ---------- 3. 资产基础信息 ----------
        stock_info = self.portfolio.get(
            asset_code,
            {
                "name": f"资产{asset_code}",
                "sector": DEFAULT_SECTOR,
                "cost": 0,
                "shares": 0,
            }
        )
        stock_info["code"] = asset_code

        if realtime_data.get("name"):
            stock_info["name"] = realtime_data["name"]

        # ---------- 4. 技术指标 ----------
        if df is not None and not df.empty:
            tech_data = self._calculate_technical_indicators(df)
        else:
            logger.warning("无历史K线，使用最小技术结构")
            tech_data = {
                "price": realtime_data.get("price", 0),
                "ma5": None,
                "ma20": None,
                "ma60": None,
                "rsi": None,
                "macd": None,
                "support": None,
                "resistance": None,
            }

        if realtime_data.get("price") is not None:
            tech_data["price"] = realtime_data["price"]

        # ---------- 5. AI 分析（如果分析器可用）----------
        if self.analyzer is None:
            logger.warning("AI分析器不可用，跳过分析")
            return None

        try:
            base_prompt = self.analyzer.generate_cio_prompt(
                stock_info,
                tech_data,
                {"macro": "", "sector": "", "target_sector": "Macro"},
            )

            context = {
                "code": asset_code,
                "stock_name": stock_info["name"],
                "date": date.today().strftime("%Y-%m-%d"),
            }

            result = self.analyzer.analyze(context, custom_prompt=base_prompt)
            return result
        except Exception as e:
            logger.error(f"AI分析失败: {e}", exc_info=True)
            return None

    # ---------- 主执行 ----------

    def run(self, stock_codes: Optional[List[str]] = None):
        if not stock_codes:
            stock_codes = list(self.portfolio.keys())

        results = []

        for code in stock_codes:
            res = self.process_single_stock(code)
            if res:
                results.append(res)
            time.sleep(5)

        if results and self.notifier is not None:
            try:
                report = self.notifier.generate_dashboard_report(results)
                if report and self.notifier.is_available():
                    self.notifier.send_to_telegram(report)
            except Exception as e:
                logger.error(f"推送报告失败: {e}")

        return results


# ================= CLI =================

def parse_args():
    parser = argparse.ArgumentParser(description="AI-CIO 多资产分析系统")
    parser.add_argument("--stocks", type=str, help="资产代码，用逗号分隔")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(args.debug)

    logger.info("🚀 启动 AI-CIO 多资产分析系统")

    try:
        pipeline = StockAnalysisPipeline()

        stock_codes = []
        if args.stocks:
            stock_codes = [
                code.strip() for code in args.stocks.split(",") if code.strip()
            ]

        results = pipeline.run(stock_codes)
        logger.info(f"🎉 分析完成，共 {len(results)} 个结果")
        return 0

    except Exception as e:
        logger.error(f"程序异常: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
