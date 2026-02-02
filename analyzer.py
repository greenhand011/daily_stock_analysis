# -*- coding: utf-8 -*-
import logging
import json
import re
import requests
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


# ================= Analyzer =================

class DeepSeekAnalyzer:
    def __init__(self):
        self.config = get_config()
        self.api_key = self.config.deepseek_api_key
        self.model = self.config.deepseek_model

        if not self.api_key:
            logger.warning("DeepSeek API Key 未配置，AI 分析将被跳过")

    # ---------- Prompt ----------

    def generate_cio_prompt(
        self,
        stock_info: Dict[str, Any],
        tech_data: Dict[str, Any],
        trend_context: Dict[str, Any]
    ) -> str:

        stock_name = stock_info.get("name", "未知股票")
        stock_code = stock_info.get("code", "Unknown")

        cost = float(stock_info.get("cost", 0))
        shares = int(stock_info.get("shares", 0))
        current_price = float(tech_data.get("price", 0))

        if shares > 0 and cost > 0 and current_price > 0:
            profit_pct = (current_price - cost) / cost * 100
            position_context = (
                f"用户持有 {shares} 股，成本 {cost} 元，"
                f"当前浮动盈亏 {profit_pct:.2f}%。"
            )
        else:
            position_context = "用户当前为空仓，请评估建仓安全性。"

        macro_news = trend_context.get("macro", "宏观面平稳")
        sector_news = trend_context.get("sector", "板块暂无重大消息")

        return f"""
你是专业的 A 股投资 CIO。

股票：{stock_name}（{stock_code}）

【宏观】
{macro_news}

【行业】
{sector_news}

【技术面】
现价：{tech_data.get("price")}
MA5 / MA20 / MA60：
{tech_data.get("ma5")} / {tech_data.get("ma20")} / {tech_data.get("ma60")}
RSI：{tech_data.get("rsi")}
MACD：{tech_data.get("macd")}

【用户状态】
{position_context}

请严格返回 JSON，不要 Markdown，不要多余文字：

{{
  "stock_name": "{stock_name}",
  "sentiment_score": 0-100,
  "operation_advice": "",
  "core_view": "",
  "analysis_summary": "",
  "risk_alert": "",
  "trend_prediction": "看涨/看跌/震荡"
}}
"""

    # ---------- 核心分析 ----------

    def analyze(
        self,
        context: Dict[str, Any],
        custom_prompt: str
    ) -> Optional[AnalysisResult]:

        if not self.api_key:
            return None

        try:
            response = requests.post(
                "https://api.deepseek.com/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "user", "content": custom_prompt}
                    ],
                    "temperature": 0.2,
                },
                timeout=120,
            )

            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]

            content = content.replace("```json", "").replace("```", "").strip()
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if not match:
                raise ValueError("未找到 JSON")

            data = json.loads(match.group(0))

            score = int(data.get("sentiment_score", 50))
            score = max(0, min(100, score))

            core_view = data.get("core_view", "")

            return AnalysisResult(
                code=context.get("code", ""),
                name=data.get("stock_name", context.get("stock_name", "")),
                date=context.get("date", ""),
                sentiment_score=score,
                operation_advice=data.get("operation_advice", ""),
                risk_alert=data.get("risk_alert", ""),
                trend_prediction=data.get("trend_prediction", ""),
                analysis_summary=data.get("analysis_summary", ""),
                buy_reason=core_view,
                sell_reason=core_view,
            )

        except Exception as e:
            logger.error(f"DeepSeek 分析失败: {e}")

            return AnalysisResult(
                code=context.get("code", ""),
                name=context.get("stock_name", ""),
                date=context.get("date", ""),
                sentiment_score=50,
                operation_advice="人工复核",
                risk_alert=str(e),
                trend_prediction="不确定",
                analysis_summary="AI 分析失败",
            )