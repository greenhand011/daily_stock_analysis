# -*- coding: utf-8 -*-
import logging
import json
import re
from dataclasses import dataclass
from typing import Optional, Dict, Any

from config import get_config

logger = logging.getLogger(__name__)

# ================= 数据结构 =================

@dataclass
class AnalysisResult:
    code: str
    name: str
    date: str
    sentiment_score: int
    operation_advice: str
    risk_alert: str
    trend_prediction: str
    analysis_summary: str
    buy_reason: str = ""
    sell_reason: str = ""

    def get_emoji(self):
        if self.sentiment_score >= 80:
            return "🔴"
        if self.sentiment_score <= 40:
            return "🟢"
        return "🟡"


# ================= DeepSeek Analyzer =================

class DeepSeekAnalyzer:
    """
    单一 DeepSeek Analyzer（无 fallback，CI 稳定）
    """

    def __init__(self):
        self.config = get_config()
        self.llm = None

        if not self.config.deepseek_api_key:
            logger.error("❌ 未配置 DEEPSEEK_API_KEY")
            return

        try:
            from openai import OpenAI

            self.llm = OpenAI(
                api_key=self.config.deepseek_api_key,
                base_url="https://api.deepseek.com/v1",
            )
            logger.info("✅ 使用 DeepSeek 作为唯一 AI 分析引擎")

        except Exception as e:
            logger.error(f"❌ DeepSeek 初始化失败: {e}")
            self.llm = None

    # ================= Prompt =================

    def generate_cio_prompt(
        self,
        stock_info: Dict[str, Any],
        tech_data: Dict[str, Any],
        trend_context: Dict[str, Any],
    ) -> str:

        stock_name = stock_info.get("name", "未知股票")
        stock_code = stock_info.get("code", "Unknown")

        cost = float(stock_info.get("cost", 0))
        shares = int(stock_info.get("shares", 0))
        current_price = float(tech_data.get("price", 0))

        if shares > 0 and cost > 0 and current_price > 0:
            profit_pct = (current_price - cost) / cost * 100
            position_context = (
                f"用户持仓 {shares} 股，成本 {cost} 元，"
                f"当前收益 {profit_pct:.2f}%。"
            )
        else:
            position_context = "用户当前为空仓，请评估安全边际与建仓方式。"

        macro_news = trend_context.get("macro", "当前宏观面平静")
        sector_news = trend_context.get("sector", "板块暂无重大消息")
        target_sector = trend_context.get("target_sector", "通用")

        return f"""
你是专业的 A 股首席投资官（CIO）。

请基于【消息面 + 技术面 + 用户真实持仓】，对 {stock_name}（{stock_code}）给出交易决策。

=== 市场环境（TrendRadar）===
宏观：{macro_news}
行业（{target_sector}）：{sector_news}

=== 技术面（日线）===
现价：{tech_data.get("price")}
MA5 / MA20 / MA60：{tech_data.get("ma5")} / {tech_data.get("ma20")} / {tech_data.get("ma60")}
RSI：{tech_data.get("rsi")}
MACD：{tech_data.get("macd")}
支撑 / 压力：{tech_data.get("support")} / {tech_data.get("resistance")}

=== 用户持仓 ===
{position_context}

=== 输出要求 ===
仅返回 JSON，不要 Markdown：
{{
  "stock_name": "{stock_name}",
  "sentiment_score": 0-100,
  "operation_advice": "操作建议",
  "core_view": "一句话核心逻辑",
  "analysis_summary": "详细分析（结合持仓）",
  "risk_alert": "主要风险",
  "trend_prediction": "未来1周走势"
}}
"""

    # ================= 核心分析 =================

    def analyze(
        self,
        context: Dict[str, Any],
        custom_prompt: str,
    ) -> Optional[AnalysisResult]:

        if not self.llm:
            return None

        try:
            resp = self.llm.chat.completions.create(
                model=self.config.deepseek_model,
                messages=[{"role": "user", "content": custom_prompt}],
                temperature=0.2,
            )

            content = resp.choices[0].message.content
            content = str(content).strip()
            content = content.replace("```json", "").replace("```", "")

            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                content = match.group(0)

            data = json.loads(content)

            score = int(data.get("sentiment_score", 50))
            score = max(0, min(100, score))
            core_view = data.get("core_view", "")

            return AnalysisResult(
                code=context.get("code", ""),
                name=data.get("stock_name", context.get("stock_name", "")),
                date=context.get("date", ""),
                sentiment_score=score,
                operation_advice=data.get("operation_advice", "观望"),
                risk_alert=data.get("risk_alert", ""),
                trend_prediction=data.get("trend_prediction", ""),
                analysis_summary=data.get("analysis_summary", ""),
                buy_reason=core_view,
                sell_reason=core_view,
            )

        except Exception as e:
            logger.error(f"AI 分析失败: {e}")
            return AnalysisResult(
                code=context.get("code", ""),
                name=context.get("stock_name", ""),
                date=context.get("date", ""),
                sentiment_score=50,
                operation_advice="人工复核",
                risk_alert=str(e),
                trend_prediction="不确定",
                analysis_summary="AI 返回异常",
            )


# === 向后兼容（旧代码仍然 import GeminiAnalyzer）===
GeminiAnalyzer = DeepSeekAnalyzer