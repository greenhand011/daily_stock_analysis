#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LLM-based stock analyzer."""

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

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
        self.llm = None
        self.model_name = self.config.openai_model or "gpt-4o-mini"

        try:
            from openai import OpenAI

            api_key = None
            base_url = None

            # Prefer explicit OpenAI configuration. This prevents OpenAI keys
            # from being sent to DeepSeek when users only want official OpenAI.
            if self.config.openai_api_key:
                api_key = self.config.openai_api_key
                base_url = self.config.openai_base_url or None
                self.model_name = self.config.openai_model or "gpt-4o-mini"
                if base_url:
                    logger.info("Using OpenAI-compatible endpoint: %s", base_url)
                else:
                    logger.info("Using official OpenAI endpoint")
            elif self.config.deepseek_api_key:
                api_key = self.config.deepseek_api_key
                base_url = self.config.openai_base_url or "https://api.deepseek.com/v1"
                self.model_name = self.config.deepseek_model or "deepseek-chat"
                logger.info("Using DeepSeek-compatible endpoint: %s", base_url)

            if not api_key:
                logger.error("No AI API key configured")
                return

            client_kwargs = {
                "api_key": api_key,
                "timeout": 30,
            }
            if base_url:
                client_kwargs["base_url"] = base_url

            self.llm = OpenAI(**client_kwargs)
            logger.info("Analyzer initialized with model: %s", self.model_name)

        except ImportError:
            logger.error("openai package is not installed")
        except Exception as exc:
            logger.error("Analyzer initialization failed: %s", exc)

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
                f"现价 {current_price:.2f}，当前{status} {abs(profit_pct):.2f}% 。"
            )
        else:
            position_context = "用户当前没有持仓，请重点评估是否值得建仓以及风险收益比。"

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
你是一位专业投资分析师，请基于市场环境、技术面和持仓信息，对 {stock_name} ({stock_code}) 输出一份简洁、明确、可执行的投资判断。

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
        if not self.llm:
            raise ValueError("AI client is not initialized")

        try:
            response = self.llm.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "你是一位专业的股票分析师。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=1000,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            logger.warning("AI API call failed and will be retried: %s", exc)
            raise

    def analyze(self, context: Dict[str, Any], custom_prompt: Optional[str] = None) -> AnalysisResult:
        if not self.llm:
            logger.error("AI client is not initialized")
            return self._create_error_result(context, "AI client is not initialized")

        if not custom_prompt:
            return self._create_error_result(context, "Missing analysis prompt")

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

            return self._create_error_result(context, "Failed to parse AI response")
        except Exception as exc:
            logger.error("AI analysis failed: %s", exc)
            return self._create_error_result(context, str(exc))

    def _parse_ai_response(self, ai_response: str, context: Dict[str, Any]) -> Optional[AnalysisResult]:
        try:
            json_match = re.search(r"\{.*\}", ai_response, re.DOTALL)
            if not json_match:
                logger.error("No JSON object found in AI response: %s", ai_response[:200])
                return None

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
                f"股票{context.get('code', '')}",
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
        except json.JSONDecodeError as exc:
            logger.error("Failed to decode AI JSON: %s", exc)
            return None
        except Exception as exc:
            logger.error("Failed to parse AI response: %s", exc)
            return None

    def _create_error_result(self, context: Dict[str, Any], error_msg: str) -> AnalysisResult:
        return AnalysisResult(
            code=context.get("code", ""),
            name=context.get("stock_name", ""),
            date=context.get("date", ""),
            sentiment_score=50,
            operation_advice="人工复核",
            risk_alert=error_msg,
            trend_prediction="不确定",
            analysis_summary="AI 分析失败，需要人工检查。",
        )


GeminiAnalyzer = DeepSeekAnalyzer
