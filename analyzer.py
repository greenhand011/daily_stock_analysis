# -*- coding: utf-8 -*-
import json
import re
import logging
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

class GeminiAnalyzer:
    """
    多模型 Analyzer
    优先级：DeepSeek → Gemini
    """

    def __init__(self):
        self.config = get_config()
        self.llm = None
        self.backend = None

        # ---------- 1️⃣ DeepSeek（首选） ----------
        if self.config.deepseek_api_key:
            try:
                from openai import OpenAI

                self.llm = OpenAI(
                    api_key=self.config.deepseek_api_key,
                    base_url="https://api.deepseek.com/v1",
                )
                self.backend = "deepseek"
                logger.info("✅ 使用 DeepSeek 作为 AI 分析引擎")
                return
            except Exception as e:
                logger.warning(f"DeepSeek 初始化失败，回退 Gemini: {e}")

        # ---------- 2️⃣ Gemini（备用） ----------
        if self.config.gemini_api_key:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI

                self.llm = ChatGoogleGenerativeAI(
                    model=self.config.gemini_model,
                    google_api_key=self.config.gemini_api_key,
                    temperature=0.2,
                    timeout=120,
                )
                self.backend = "gemini"
                logger.info("⚠️ 回退使用 Gemini 作为 AI 分析引擎")
                return
            except Exception as e:
                logger.error(f"Gemini 初始化失败: {e}")

        logger.error("❌ 未能初始化任何 AI 模型")

    # ================= JSON 解析兜底 =================

    @staticmethod
    def _safe_json_parse(text: str) -> Optional[Dict[str, Any]]:
        """
        尝试从模型输出中提取 JSON
        """
        try:
            text = text.replace("```json", "").replace("```", "").strip()
            match = re.search(r"\{.*\}", text, re.S)
            if not match:
                return None
            return json.loads(match.group(0))
        except Exception:
            return None

    # ================= 核心分析 =================

    def analyze(
        self,
        context: Dict[str, Any],
        custom_prompt: str,
    ) -> AnalysisResult:

        if not self.llm or not self.backend:
            return AnalysisResult(
                code=context.get("code", ""),
                name=context.get("stock_name", ""),
                date=context.get("date", ""),
                sentiment_score=50,
                operation_advice="跳过",
                risk_alert="AI 未初始化",
                trend_prediction="未知",
                analysis_summary="AI 引擎不可用",
            )

        try:
            # ===== DeepSeek：强制 JSON =====
            if self.backend == "deepseek":
                resp = self.llm.chat.completions.create(
                    model=self.config.deepseek_model,
                    messages=[{"role": "user", "content": custom_prompt}],
                    temperature=0.2,
                    response_format={"type": "json_object"},
                )
                content = resp.choices[0].message.content

            # ===== Gemini：弱约束 =====
            else:
                result = self.llm.invoke(custom_prompt)
                content = result.content

            data = self._safe_json_parse(str(content))

            # ---------- JSON 成功 ----------
            if data:
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

            # ---------- JSON 失败 → 文本兜底 ----------
            logger.warning("AI 返回非 JSON，使用文本兜底")

            return AnalysisResult(
                code=context.get("code", ""),
                name=context.get("stock_name", ""),
                date=context.get("date", ""),
                sentiment_score=50,
                operation_advice="人工判断",
                risk_alert="模型输出非结构化",
                trend_prediction="不确定",
                analysis_summary=str(content)[:1200],
            )

        except Exception as e:
            logger.error(f"AI 分析异常: {e}")

            return AnalysisResult(
                code=context.get("code", ""),
                name=context.get("stock_name", ""),
                date=context.get("date", ""),
                sentiment_score=50,
                operation_advice="人工复核",
                risk_alert=str(e),
                trend_prediction="不确定",
                analysis_summary="AI 调用异常",
            )