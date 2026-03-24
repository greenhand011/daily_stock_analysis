#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LLM-based stock analyzer using direct HTTP requests."""

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from config import get_config

logger = logging.getLogger(__name__)


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

    def get_emoji(self) -> str:
        if self.sentiment_score >= 80:
            return "🟢"
        if self.sentiment_score <= 40:
            return "🔴"
        return "🟡"


class DeepSeekAnalyzer:
    """Historical compatibility name for the project's main analyzer."""

    def __init__(self):
        self.config = get_config()
        self.session = requests.Session()
        self.model_name = self.config.openai_model or "gpt-4o-mini"
        self.api_key = None
        self.base_url = None

        if self.config.openai_api_key:
            self.api_key = self.config.openai_api_key
            self.base_url = (self.config.openai_base_url or "https://api.openai.com/v1").rstrip("/")
            self.model_name = self.config.openai_model or "gpt-4o-mini"
            logger.info("Using OpenAI-compatible endpoint: %s", self.base_url)
        elif self.config.deepseek_api_key:
            self.api_key = self.config.deepseek_api_key
            self.base_url = (self.config.openai_base_url or "https://api.deepseek.com/v1").rstrip("/")
            self.model_name = self.config.deepseek_model or "deepseek-chat"
            logger.info("Using DeepSeek-compatible endpoint: %s", self.base_url)
        else:
            logger.error("No AI API key configured")

    def _is_ready(self) -> bool:
        return bool(self.api_key and self.base_url)

    def generate_cio_prompt(
        self,
        stock_info: Dict[str, Any],
        tech_data: Dict[str, Any],
        trend_context: Dict[str, Any],
    ) -> str:
        stock_name = stock_info.get("name", "Unknown Asset")
        stock_code = stock_info.get("code", "UNKNOWN")

        cost = float(stock_info.get("cost", 0) or 0)
        shares = int(stock_info.get("shares", 0) or 0)
        current_price = float(tech_data.get("price", 0) or 0)

        if shares > 0 and cost > 0 and current_price > 0:
            profit_pct = (current_price - cost) / cost * 100
            status = "盈利" if profit_pct >= 0 else "亏损"
            position_context = (
                f"用户当前持有 {shares} 股，成本价 {cost:.2f}，"
                f"现价 {current_price:.2f}，当前{status} {abs(profit_pct):.2f}%。"
            )
        else:
            position_context = "用户当前没有持仓，请重点评估是否值得建仓，以及风险收益比。"

        macro_news = trend_context.get("macro", "暂无宏观新闻摘要")
        sector_news = trend_context.get("sector", "暂无行业新闻摘要")
        target_sector = trend_context.get("target_sector", "综合")

        def fmt(value: Any) -> str:
            if value is None:
                return "N/A"
            if isinstance(value, (int, float)):
                return f"{value:.2f}"
            return str(value)

        prompt = f"""
你是一位专业投资分析师，请基于市场环境、技术面和持仓信息，为 {stock_name} ({stock_code}) 输出一份简洁、明确、可执行的投资判断。

市场环境：
- 宏观：{macro_news}
- 行业（{target_sector}）：{sector_news}

技术面（日线）：
- 当前价格：{current_price:.2f}
- MA5：{fmt(tech_data.get("ma5"))}
- MA20：{fmt(tech_data.get("ma20"))}
- MA60：{fmt(tech_data.get("ma60"))}
- RSI：{fmt(tech_data.get("rsi"))}
- MACD：{fmt(tech_data.get("macd"))}
- 支撑位：{fmt(tech_data.get("support"))}
- 阻力位：{fmt(tech_data.get("resistance"))}

持仓情况：
{position_context}

请严格返回 JSON，不要输出额外解释。字段如下：
{{
  "stock_name": "{stock_name}",
  "sentiment_score": 0-100 的整数,
  "operation_advice": "明确操作建议",
  "core_view": "一句话核心观点",
  "analysis_summary": "分析摘要",
  "risk_alert": "主要风险",
  "trend_prediction": "未来 1 周判断"
}}
"""
        return prompt.strip()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _call_ai_api(self, prompt: str) -> str:
        if not self._is_ready():
            raise ValueError("AI client is not initialized")

        print("=== CALLING OPENAI ===")
        print("API KEY EXISTS:", bool(os.getenv("OPENAI_API_KEY")))
        print("BASE URL:", os.getenv("OPENAI_BASE_URL"))

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": prompt},
            ],
        }

        try:
            response = self.session.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=60,
            )
            print("=== OPENAI RESPONSE STATUS ===", response.status_code)
            print("=== OPENAI RESPONSE TEXT ===", response.text[:500])
            response.raise_for_status()
            data = response.json()
            result = data["choices"][0]["message"]["content"]
            if not result or len(result.strip()) < 10:
                raise Exception("Empty AI response")
            return result
        except requests.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else "unknown"
            body = exc.response.text[:500] if exc.response is not None else ""
            logger.warning("AI API HTTP error %s: %s", status_code, body)
            raise
        except requests.RequestException as exc:
            logger.warning("AI API request failed and will be retried: %s", exc)
            raise

    def analyze(self, context: Dict[str, Any], custom_prompt: Optional[str] = None) -> AnalysisResult:
        print("=== ANALYZER START ===")
        if not self._is_ready():
            logger.error("AI client is not initialized")
            raise Exception("AI client is not initialized")

        if not custom_prompt:
            raise Exception("Missing analysis prompt")

        try:
            logger.info("Starting AI analysis for %s", context.get("code", "UNKNOWN"))
            start_time = time.time()
            ai_response = self._call_ai_api(custom_prompt)
            elapsed_time = time.time() - start_time
            logger.debug("AI analysis finished in %.2fs", elapsed_time)

            if not ai_response:
                return self._create_error_result(context, "Empty AI response")

            result = self._parse_ai_response(ai_response, context)
            if result:
                logger.info(
                    "Analysis finished for %s with sentiment %s",
                    result.code,
                    result.sentiment_score,
                )
                return result

            raise Exception("Failed to parse AI response")
        except Exception as e:
            print("=== ANALYZER ERROR ===", str(e))
            raise e

    def _parse_ai_response(self, ai_response: str, context: Dict[str, Any]) -> Optional[AnalysisResult]:
        try:
            json_match = re.search(r"\{.*\}", ai_response, re.DOTALL)
            if not json_match:
                logger.error("No JSON object found in AI response: %s", ai_response[:200])
                raise Exception("No JSON object found in AI response")

            data = json.loads(json_match.group(0))
            required_fields = [
                "sentiment_score",
                "operation_advice",
                "core_view",
                "analysis_summary",
                "risk_alert",
                "trend_prediction",
            ]
            for field in required_fields:
                data.setdefault(field, "未知")

            try:
                score = int(data["sentiment_score"])
                score = max(0, min(100, score))
            except Exception:
                score = 50

            stock_name = data.get("stock_name") or context.get(
                "stock_name",
                f"资产{context.get('code', '')}",
            )

            return AnalysisResult(
                code=context.get("code", ""),
                name=stock_name,
                date=context.get("date", ""),
                sentiment_score=score,
                operation_advice=data.get("operation_advice", "观望"),
                risk_alert=data.get("risk_alert", ""),
                trend_prediction=data.get("trend_prediction", ""),
                analysis_summary=data.get("analysis_summary", ""),
                buy_reason=data.get("core_view", ""),
                sell_reason=data.get("core_view", ""),
            )
        except json.JSONDecodeError as e:
            print("=== ANALYZER ERROR ===", str(e))
            raise e
        except Exception as e:
            print("=== ANALYZER ERROR ===", str(e))
            raise e

    def _create_error_result(self, context: Dict[str, Any], error_msg: str) -> AnalysisResult:
        summary = "AI 分析失败，需要人工检查。"
        if "Connection error" in error_msg or "APIConnectionError" in error_msg:
            summary = "OpenAI 连接失败，请检查 GitHub Actions 到 OpenAI 官方接口的网络连通性。"
        if "Read timed out" in error_msg or "ConnectTimeout" in error_msg or "timed out" in error_msg:
            summary = "OpenAI 请求超时，建议减少股票数量，或改用更稳定的 OPENAI_BASE_URL。"

        return AnalysisResult(
            code=context.get("code", ""),
            name=context.get("stock_name", ""),
            date=context.get("date", ""),
            sentiment_score=50,
            operation_advice="人工复核",
            risk_alert=error_msg,
            trend_prediction="不确定",
            analysis_summary=summary,
        )


GeminiAnalyzer = DeepSeekAnalyzer
