# -*- coding: utf-8 -*-
import os
import argparse
import logging
import sys
import time
import json
import re
import pandas as pd
from datetime import datetime, date
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import List, Dict, Any, Optional

from config import get_config, Config
from storage import get_db
from data_provider import DataFetcherManager
from analyzer import GeminiAnalyzer, AnalysisResult
from notification import NotificationService

# ================= 日志配置 =================

LOG_FORMAT = '%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
DEFAULT_SECTOR = "Macro"

def setup_logging(debug: bool = False, log_dir: str = "./logs") -> None:
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    today_str = datetime.now().strftime('%Y%m%d')
    log_file = log_path / f"stock_analysis_{today_str}.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if debug else logging.INFO)
    root_logger.handlers = []

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
    root_logger.addHandler(console_handler)

    file_handler = RotatingFileHandler(
        log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8'
    )
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
    root_logger.addHandler(file_handler)

logger = logging.getLogger(__name__)

# ================= Pipeline =================

class StockAnalysisPipeline:
    def __init__(self, config: Optional[Config] = None, max_workers: Optional[int] = None):
        self.config = config or get_config()
        self.max_workers = 1

        self.portfolio = self._load_portfolio_config()
        self.db = get_db()
        self.fetcher_manager = DataFetcherManager()
        self.analyzer = GeminiAnalyzer()
        self.notifier = NotificationService()

        logger.info("AI-CIO 初始化完成")

    # ---------- 配置加载 ----------

    def _load_portfolio_config(self) -> dict:
        path = "portfolio.json"
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                return {
                    str(code): {
                        "code": str(code),
                        "cost": 0,
                        "shares": 0,
                        "name": f"股票{code}",
                        "sector": "Unknown"
                    }
                    for code in data
                }

            if isinstance(data, dict):
                final_data = {}
                for code, info in data.items():
                    info = info if isinstance(info, dict) else {}
                    info["code"] = str(code)
                    info.setdefault("cost", 0)
                    info.setdefault("shares", 0)
                    info.setdefault("name", f"股票{code}")
                    info.setdefault("sector", "Unknown")
                    final_data[str(code)] = info
                return final_data

        except Exception as e:
            logger.error(f"加载 portfolio.json 失败: {e}")

        return {}

    # ---------- 原有新闻上下文 ----------

    def _get_trend_radar_context(self, code: str, json_path: str = "news_summary.json") -> dict:
        context = {"macro": "", "sector": "", "target_sector": DEFAULT_SECTOR}
        stock_info = self.portfolio.get(code, {})
        target_sector = stock_info.get("sector", DEFAULT_SECTOR)
        context["target_sector"] = target_sector

        if not os.path.exists(json_path):
            return context

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                news_items = json.load(f)

            macro_news, sector_news = [], []
            for item in news_items:
                cat = item.get("category", "Macro")
                line = f"- {item.get('title', '')}: {item.get('summary', '')}"
                if cat in ["Macro", "Finance"]:
                    macro_news.append(line)
                if cat == target_sector:
                    sector_news.append(line)

            context["macro"] = "\n".join(macro_news) if macro_news else "当前宏观面平静。"
            context["sector"] = (
                "\n".join(sector_news) if sector_news else f"当前{target_sector}板块无重大消息。"
            )
            return context
        except Exception as e:
            logger.warning(f"读取新闻上下文失败: {e}")
            return context

    # ---------- 🟢 新增：读取 TrendRadar AI 结论 ----------

    def _load_trendradar_ai_summary(self) -> Optional[str]:
        today = date.today().strftime("%Y-%m-%d")
        path = Path("trendradar_ai") / f"{today}.json"

        if not path.exists():
            logger.info("未发现 TrendRadar AI 结论文件，跳过增强")
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, dict):
                return data.get("summary") or json.dumps(data, ensure_ascii=False)
            if isinstance(data, str):
                return data

        except Exception as e:
            logger.warning(f"读取 TrendRadar AI 失败: {e}")

        return None

    # ---------- 🟢 新增：合并 AI 研究观点 ----------

    def _merge_trendradar_into_context(self, context: dict, ai_text: Optional[str]) -> dict:
    if not ai_text:
        return context

    context["trendradar_ai"] = ai_text
    return context

    # ---------- 技术指标 ----------

    def _calculate_technical_indicators(self, df: pd.DataFrame) -> dict:
        if df is None or df.empty:
            return {}

        df = df.sort_values("date")
        close = df["close"]

        ma5 = close.rolling(5).mean().iloc[-1]
        ma20 = close.rolling(20).mean().iloc[-1]
        ma60 = close.rolling(60).mean().iloc[-1]

        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rsi = 100 if loss.iloc[-1] == 0 else 100 - (100 / (1 + gain / loss)).iloc[-1]

        exp12 = close.ewm(span=12, adjust=False).mean()
        exp26 = close.ewm(span=26, adjust=False).mean()
        macd = (exp12 - exp26).iloc[-1]
        signal = (exp12 - exp26).ewm(span=9, adjust=False).mean().iloc[-1]

        return {
            "price": close.iloc[-1],
            "ma5": ma5,
            "ma20": ma20,
            "ma60": ma60,
            "rsi": rsi,
            "macd": macd,
            "macd_signal": signal,
            "support": df["low"].tail(20).min(),
            "resistance": df["high"].tail(20).max(),
        }

    # ---------- 单股处理 ----------

    def process_single_stock(self, code: str) -> Optional[AnalysisResult]:
        match = re.search(r"\d{6}", code)
        if not match:
            return None

        stock_code = match.group(0)
        logger.info(f"========== 处理 A 股: {stock_code} ==========")

        df = self.fetcher_manager.get_daily_data(stock_code, days=100)[0]
        if df is None or df.empty:
            return None

        stock_info = self.portfolio.get(
            stock_code,
            {"name": f"股票{stock_code}", "sector": DEFAULT_SECTOR, "cost": 0, "shares": 0}
        )
        stock_info["code"] = stock_code

        tech_data = self._calculate_technical_indicators(df)

        trend_context = self._get_trend_radar_context(stock_code)
        trendradar_ai = self._load_trendradar_ai_summary()
        trend_context = self._merge_trendradar_into_context(trend_context, trendradar_ai)

        # --- 🟢 TrendRadar AI 作为独立研究结论注入 Prompt ---
        extra_research = ""
        if trend_context.get("trendradar_ai"):
            extra_research = (
                "\n\n=== TrendRadar · AI 研究结论（高权重） ===\n"
                + trend_context["trendradar_ai"]
                + "\n【说明】以上结论为跨市场、多源信息综合判断，应优先纳入交易决策。"
            )

base_prompt = self.analyzer.generate_cio_prompt(
    stock_info, tech_data, trend_context
)

prompt = base_prompt + extra_research


        context = {
            "code": stock_code,
            "stock_name": stock_info["name"],
            "date": date.today().strftime("%Y-%m-%d"),
        }

        return self.analyzer.analyze(context, custom_prompt=prompt)

    # ---------- 主执行 ----------

    def run(self, stock_codes: List[str]):
        results = []
        for i, code in enumerate(stock_codes):
            res = self.process_single_stock(code)
            if res:
                results.append(res)
            if i < len(stock_codes) - 1:
                time.sleep(15)

        if results:
            report = self.notifier.generate_dashboard_report(results)
            if self.notifier.is_available():
                self.notifier.send_to_telegram(report)

        return results

# ================= CLI =================

def main():
    config = get_config()
    setup_logging(False, config.log_dir)

    pipeline = StockAnalysisPipeline(config)
    pipeline.run(list(pipeline.portfolio.keys()))

if __name__ == "__main__":
    sys.exit(main())
