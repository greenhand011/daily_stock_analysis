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


# ================= Analyzer =================

class GeminiAnalyzer:
    """
    多模型自动切换 Analyzer
    优先级：
    DeepSeek → Gemini → OpenAI-compatible
    """

    def __init__(self):
        self.config = get_config()
        self.llm = None
        self.model_name = None

        self._init_llm()

    # ---------- 初始化模型 ----------

    def _init_llm(self):
        """
        按优先级尝试初始化可用模型
        """
        # 1️⃣ DeepSeek（OpenAI-compatible）
        if self.config.deepseek_api_key:
            try:
                from langchain_openai import ChatOpenAI

                self.llm = ChatOpenAI(
                    api_key=self.config.deepseek_api_key,
                    base_url="https://api.deepseek.com",
                    model=self.config.deepseek_model,
                    temperature=0.2,
                    timeout=120,
                )
                self.model_name = f"DeepSeek({self.config.deepseek_model})"
                logger.info(f"✅ 使用 {self.model_name}")
                return
            except Exception as e:
                logger.warning(f"DeepSeek 初始化失败，降级处理: {e}")

        # 2️⃣ Gemini
        if self.config.gemini_api_key:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI

                self.llm = ChatGoogleGenerativeAI(
                    model=self.config.gemini_model,
                    google_api_key=self.config.gemini_api_key,
                    temperature=0.2,
                    timeout=120,
                )
                self.model_name = f"Gemini({self.config.gemini_model})"
                logger.info(f"✅ 使用 {self.model_name}")
                return
            except Exception as e:
                logger.warning(f"Gemini 初始化失败，降级处理: {e}")

        # 3️⃣ OpenAI-compatible（兜底）
        if self.config.openai_api_key and self.config.openai_base_url:
            try:
                from langchain_openai import ChatOpenAI

                self.llm = ChatOpenAI(
                    api_key=self.config.openai_api_key,
                    base_url=self.config.openai_base_url,
                    model=self.config.openai_model,
                    temperature=0.2,
                    timeout=120,
                )
                self.model_name = f"OpenAI-compatible({self.config.openai_model})"
                logger.info(f"✅ 使用 {self.model_name}")
                return
            except Exception as e:
                logger.error(f"OpenAI-compatible 初始化失败: {e}")

        logger.error("❌ 无可用 AI 模型，分析功能将被跳过")
        self.llm = None

    # ---------- Prompt 生成（保持你原逻辑） ----------

    def generate_cio_prompt(
        self,
        stock_info: Dict[str, Any],
        tech_data: Dict[str, Any],
        trend_context: Dict[str, Any]
    ) -> str:
        # ⚠️ 这一段请你 **原封不动**
        # 直接复制你现在 analyzer.py 里的 generate_cio_prompt 实现
        ...
    
    # ---------- 核心分析 ----------

    def analyze(
        self,
        context: Dict[str, Any],
        custom_prompt: Optional[str] = None
    ) -> Optional[AnalysisResult]:

        if not self.llm:
            return self._fallback_result(context, "无可用 AI 模型")

        try:
            result = self.llm.invoke(custom_prompt or "请分析股票")
            content = result.content

            if isinstance(content, list):
                content = "\n".join(
                    str(x.get("text", x)) if isinstance(x, dict) else str(x)
                    for x in content
                )
            else:
                content = str(content)

            content = content.replace("```json", "").replace("```", "").strip()
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if not match:
                raise ValueError("未找到 JSON 结构")

            data = json.loads(match.group(0))

            score = int(data.get("sentiment_score", 50))
            score = max(0, min(100, score))

            core_view = data.get("core_view", "见详细分析")

            return AnalysisResult(
                code=context.get("code", "Unknown"),
                name=data.get("stock_name", context.get("stock_name", "Unknown")),
                date=context.get("date", ""),
                sentiment_score=score,
                operation_advice=data.get("operation_advice", "观望"),
                risk_alert=data.get("risk_alert", "暂无"),
                trend_prediction=data.get("trend_prediction", "震荡"),
                analysis_summary=data.get("analysis_summary", ""),
                buy_reason=core_view,
                sell_reason=core_view,
            )

        except Exception as e:
            logger.error(f"AI 分析失败 ({self.model_name}): {e}")
            return self._fallback_result(context, str(e))

    # ---------- 保底 ----------

    def _fallback_result(self, context: Dict[str, Any], reason: str) -> AnalysisResult:
        return AnalysisResult(
            code=context.get("code", "Unknown"),
            name=context.get("stock_name", "Unknown"),
            date=context.get("date", ""),
            sentiment_score=50,
            operation_advice="人工复核",
            risk_alert=f"AI 不可用: {reason}",
            trend_prediction="不确定",
            analysis_summary="AI 服务异常，建议人工查看。",
            buy_reason="N/A",
            sell_reason="N/A",
        )