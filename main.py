#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股自选股智能分析系统 - 主程序
支持 DeepSeek、Gemini 等多种 AI 分析引擎
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
from typing import List, Dict, Any, Optional

# ================= 编码修复（必须放在最前面） =================
# 修复标准输出编码，解决中文显示问题
try:
    # 方法1：使用codecs修复
    import codecs
    if sys.stdout.encoding != 'UTF-8' and hasattr(sys.stdout, 'buffer'):
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
    if sys.stderr.encoding != 'UTF-8' and hasattr(sys.stderr, 'buffer'):
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)
except:
    pass

# 方法2：设置环境变量（备用）
os.environ['PYTHONIOENCODING'] = 'utf-8'

# 方法3：尝试设置locale
try:
    import locale
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
except:
    try:
        locale.setlocale(locale.LC_ALL, 'C.UTF-8')
    except:
        pass

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
    """配置日志系统，确保使用UTF-8编码"""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    today_str = datetime.now().strftime('%Y%m%d')
    log_file = log_path / f"stock_analysis_{today_str}.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if debug else logging.INFO)
    
    # 清除现有的处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
    root_logger.addHandler(console_handler)

    # 文件处理器（确保使用UTF-8编码）
    try:
        file_handler = RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8'
        )
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
        root_logger.addHandler(file_handler)
    except Exception as e:
        logger.warning(f"创建日志文件失败: {e}")

logger = logging.getLogger(__name__)

# ================= Pipeline =================

class StockAnalysisPipeline:
    def __init__(self, config: Optional[Config] = None, max_workers: Optional[int] = None):
        self.config = config or get_config()
        self.max_workers = max_workers or 1

        # 打印AI引擎信息（调试用）
        self._print_ai_engine_info()
        
        self.portfolio = self._load_portfolio_config()
        self.db = get_db()
        self.fetcher_manager = DataFetcherManager()
        self.analyzer = GeminiAnalyzer()
        self.notifier = NotificationService()

        logger.info("AI-CIO 初始化完成")

    def _print_ai_engine_info(self):
        """打印当前使用的AI引擎信息"""
        import os
        
        # 检查配置了哪些AI引擎
        deepseek_key = os.getenv('DEEPSEEK_API_KEY')
        openai_key = os.getenv('OPENAI_API_KEY')
        gemini_key = os.getenv('GEMINI_API_KEY')
        
        if deepseek_key:
            logger.info(f"✅ 检测到 DeepSeek API 密钥 (长度: {len(deepseek_key)})")
        if openai_key:
            logger.info(f"✅ 检测到 OpenAI 兼容 API 密钥 (长度: {len(openai_key)})")
        if gemini_key:
            logger.info(f"✅ 检测到 Gemini API 密钥 (长度: {len(gemini_key)})")
            
        # 判断主要使用哪个引擎
        if deepseek_key or (openai_key and 'deepseek.com' in os.getenv('OPENAI_BASE_URL', '')):
            logger.info("✅ 使用 DeepSeek 作为 AI 分析引擎")
        elif gemini_key:
            logger.info("✅ 使用 Gemini 作为 AI 分析引擎")
        elif openai_key:
            logger.info("✅ 使用 OpenAI 兼容 API 作为 AI 分析引擎")
        else:
            logger.warning("⚠️  未检测到任何 AI API 密钥，AI 分析功能将不可用")

    # ---------- 配置加载 ----------

    def _load_portfolio_config(self) -> dict:
        """加载投资组合配置，支持多种格式"""
        # 优先使用环境变量中的股票列表
        env_stock_list = os.getenv('STOCK_LIST', '')
        if env_stock_list:
            stock_codes = [code.strip() for code in env_stock_list.split(',') if code.strip()]
            if stock_codes:
                return {
                    str(code): {
                        "code": str(code),
                        "cost": 0,
                        "shares": 0,
                        "name": f"股票{code}",
                        "sector": "Unknown"
                    }
                    for code in stock_codes
                }
        
        # 其次使用portfolio.json文件
        path = "portfolio.json"
        if not os.path.exists(path):
            logger.warning("未找到 portfolio.json 文件，使用默认股票列表")
            # 默认股票列表
            default_stocks = ['600519', '000858', '002415']
            return {
                str(code): {
                    "code": str(code),
                    "cost": 0,
                    "shares": 0,
                    "name": f"股票{code}",
                    "sector": "Unknown"
                }
                for code in default_stocks
            }
            
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
        """获取新闻数据上下文"""
        context = {"macro": "", "sector": "", "target_sector": DEFAULT_SECTOR}
        stock_info = self.portfolio.get(code, {})
        target_sector = stock_info.get("sector", DEFAULT_SECTOR)
        context["target_sector"] = target_sector

        if not os.path.exists(json_path):
            logger.debug(f"未找到新闻文件: {json_path}")
            return context

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                news_items = json.load(f)

            macro_news, sector_news = [], []
            for item in news_items:
                cat = item.get("category", "Macro")
                title = item.get('title', '')
                summary = item.get('summary', '')
                
                # 确保中文字符正确处理
                if isinstance(title, str):
                    title = title.encode('utf-8', errors='ignore').decode('utf-8')
                if isinstance(summary, str):
                    summary = summary.encode('utf-8', errors='ignore').decode('utf-8')
                    
                line = f"- {title}: {summary}"
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
        """加载 TrendRadar AI 分析结论"""
        today = date.today().strftime("%Y-%m-%d")
        path = Path("trendradar_ai") / f"{today}.json"

        if not path.exists():
            logger.info("未发现 TrendRadar AI 结论文件，跳过增强")
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, dict):
                result = data.get("summary") or json.dumps(data, ensure_ascii=False)
                # 确保返回UTF-8编码的字符串
                if isinstance(result, str):
                    return result.encode('utf-8', errors='ignore').decode('utf-8')
            if isinstance(data, str):
                return data.encode('utf-8', errors='ignore').decode('utf-8')

        except Exception as e:
            logger.warning(f"读取 TrendRadar AI 失败: {e}")

        return None

    # ---------- 🟢 新增：合并 AI 研究观点 ----------

    def _merge_trendradar_into_context(self, context: dict, ai_text: Optional[str]) -> dict:
        """将 TrendRadar AI 结论合并到上下文中"""
        if not ai_text:
            return context
    
        context["trendradar_ai"] = ai_text
        return context

    # ---------- 技术指标 ----------

    def _calculate_technical_indicators(self, df: pd.DataFrame) -> dict:
        """计算技术指标"""
        if df is None or df.empty:
            return {}

        try:
            df = df.sort_values("date")
            close = df["close"]

            ma5 = close.rolling(5).mean().iloc[-1]
            ma20 = close.rolling(20).mean().iloc[-1]
            ma60 = close.rolling(60).mean().iloc[-1]

            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = -delta.where(delta < 0, 0).rolling(14).mean()
            if loss.iloc[-1] == 0:
                rsi = 100
            else:
                rsi = 100 - (100 / (1 + gain.iloc[-1] / loss.iloc[-1]))

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
        except Exception as e:
            logger.error(f"计算技术指标失败: {e}")
            return {}

    # ---------- 单股处理 ----------

    def process_single_stock(self, code: str) -> Optional[AnalysisResult]:
        """处理单只股票分析"""
        match = re.search(r"\d{6}", str(code))
        if not match:
            logger.warning(f"无效的股票代码格式: {code}")
            return None

        stock_code = match.group(0)
        logger.info(f"========== 处理 A 股: {stock_code} ==========")

        try:
            # 获取股票数据
            df = self.fetcher_manager.get_daily_data(stock_code, days=100)[0]
            if df is None or df.empty:
                logger.warning(f"无法获取股票 {stock_code} 的数据")
                return None

            # 获取股票信息
            stock_info = self.portfolio.get(
                stock_code,
                {"name": f"股票{stock_code}", "sector": DEFAULT_SECTOR, "cost": 0, "shares": 0}
            )
            stock_info["code"] = stock_code

            # 计算技术指标
            tech_data = self._calculate_technical_indicators(df)

            # 获取新闻上下文
            trend_context = self._get_trend_radar_context(stock_code)
            
            # 加载并合并 TrendRadar AI 结论
            trendradar_ai = self._load_trendradar_ai_summary()
            trend_context = self._merge_trendradar_into_context(trend_context, trendradar_ai)

            # 构建提示词
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

            # 构建上下文
            context = {
                "code": stock_code,
                "stock_name": stock_info["name"],
                "date": date.today().strftime("%Y-%m-%d"),
            }

            # 调用 AI 分析
            logger.info(f"开始 AI 分析股票: {stock_code}")
            result = self.analyzer.analyze(context, custom_prompt=prompt)
            
            if result:
                logger.info(f"股票 {stock_code} 分析完成")
            else:
                logger.warning(f"股票 {stock_code} 分析失败")
                
            return result
            
        except Exception as e:
            logger.error(f"处理股票 {stock_code} 时发生错误: {e}", exc_info=True)
            return None

    # ---------- 主执行 ----------

    def run(self, stock_codes: Optional[List[str]] = None):
        """运行股票分析流水线"""
        if not stock_codes:
            stock_codes = list(self.portfolio.keys())
            
        if not stock_codes:
            logger.error("没有可分析的股票代码")
            return []

        logger.info(f"开始分析 {len(stock_codes)} 只股票: {', '.join(stock_codes)}")
        
        results = []
        for i, code in enumerate(stock_codes):
            logger.info(f"进度: {i+1}/{len(stock_codes)}")
            
            res = self.process_single_stock(code)
            if res:
                results.append(res)
                
            # 在股票之间添加延迟，避免请求过于频繁
            if i < len(stock_codes) - 1:
                delay_seconds = 15
                logger.debug(f"等待 {delay_seconds} 秒后处理下一只股票...")
                time.sleep(delay_seconds)

        # 生成并发送报告
        if results:
            logger.info(f"分析完成，共生成 {len(results)} 个结果")
            try:
                report = self.notifier.generate_dashboard_report(results)
                if report and self.notifier.is_available():
                    self.notifier.send_to_telegram(report)
                    logger.info("报告已发送到 Telegram")
                else:
                    logger.warning("报告生成失败或通知服务不可用")
            except Exception as e:
                logger.error(f"生成或发送报告时出错: {e}")
        else:
            logger.warning("没有生成任何分析结果")

        return results

# ================= CLI =================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='A股自选股智能分析系统')
    parser.add_argument('--stocks', type=str, help='要分析的股票代码，用逗号分隔')
    parser.add_argument('--market-review', action='store_true', help='运行大盘复盘分析')
    parser.add_argument('--workers', type=int, default=1, help='并发工作线程数')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    parser.add_argument('--log-dir', type=str, default='./logs', help='日志目录')
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    # 设置日志
    setup_logging(args.debug, args.log_dir)
    
    # 记录启动信息
    logger.info("=" * 50)
    logger.info("🚀 启动 AI-CIO 分析系统")
    logger.info(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Python 版本: {sys.version}")
    logger.info(f"工作目录: {os.getcwd()}")
    logger.info("=" * 50)
    
    try:
        # 创建分析流水线
        pipeline = StockAnalysisPipeline(max_workers=args.workers)
        
        # 确定要分析的股票
        stock_codes = []
        if args.stocks:
            stock_codes = [code.strip() for code in args.stocks.split(',') if code.strip()]
            logger.info(f"使用命令行指定的股票: {stock_codes}")
        
        # 运行分析
        results = pipeline.run(stock_codes)
        
        logger.info(f"🎉 分析完成，共处理 {len(results)} 只股票")
        return 0
        
    except KeyboardInterrupt:
        logger.info("用户中断程序执行")
        return 130
    except Exception as e:
        logger.error(f"程序执行失败: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)