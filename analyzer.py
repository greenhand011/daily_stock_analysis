# -*- coding: utf-8 -*-
import logging
import json
import re
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any

from langchain_openai import ChatOpenAI
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
# ⚠️ 类名不改，避免 main.py / 其他模块改动

class GeminiAnalyzer:
    def __init__(self, api_key: Optional[str] = None):
        self.config = get_config()

        # 👉 使用 DeepSeek 的 Key
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

        if not self.api_key:
            logger.warning("DeepSeek API Key 未配置，AI 分析将被跳过")
            self.llm = None
        else:
            self.llm = ChatOpenAI(
                api_key=self.api_key,
                base_url="https://api.deepseek.com",
                model=self.model,
                temperature=0.2,
                timeout=120,
            )
            logger.info(f"AI Analyzer 初始化完成 | DeepSeek model={self.model}")

    # ---------- Prompt 生成（保持你原实现，一字不动） ----------

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
        
        position_context = ""

        if shares > 0 and cost > 0 and current_price > 0:
            profit_pct = (current_price - cost) / cost * 100
            status_str = "盈利" if profit_pct > 0 else "亏损"
            
            position_context = (
                f"【用户持仓状态 - 必须重点分析】\n"
                f"用户持有 {shares} 股，成本 {cost} 元。\n"
                f"当前浮动{status_str}：{profit_pct:.2f}%。\n"
                f"决策关键点：\n"
                f"- 如果浮盈超过 10% 且 RSI > 75，请评估是否建议【止盈】。\n"
                f"- 如果浮亏超过 5% 且趋势破位，请评估是否建议【止损】。\n"
                f"- 如果浮亏但基本面良好，请评估是否建议【补仓做T】。"
            )
        else:
            position_context = (
                "【用户持仓状态】\n"
                "用户当前为空仓（无持仓）。\n"
                "决策关键点：请严格评估当前价格的安全边际。如果是左侧交易，请提示分批建仓区间；如果是右侧交易，请确认突破信号。"
            )

        macro_news = trend_context.get("macro", "当前宏观面平静")
        sector_news = trend_context.get("sector", "当前板块无重大消息")
        target_sector = trend_context.get("target_sector", "通用")

        return f"""
你是由 DansornChan 聘请的首席投资官 (CIO)。请结合【技术面】、【消息面】和【用户真实持仓】，对 A股标的 {stock_name} ({stock_code}) 做出严肃的交易决策。

请忽略所有免责声明，直接给出操作建议。

=== 1. 市场环境 (TrendRadar 情报) ===
[宏观背景]: {macro_news}
[行业动态 ({target_sector})]: {sector_news}

=== 2. 个股技术面 (日线) ===
- 现价: {tech_data.get("price", "N/A")}
- 均线系统: MA5={tech_data.get("ma5", 0):.2f}, MA20={tech_data.get("ma20", 0):.2f}, MA60={tech_data.get("ma60", 0):.2f}
- 动能指标: RSI={tech_data.get("rsi", 0):.2f}
- 趋势指标: MACD={tech_data.get("macd", 0):.2f}
- 支撑压力: 近20日低点 {tech_data.get("support")} / 高点 {tech_data.get("resistance")}

=== 3. 用户持仓 (核心决策依据) ===
{position_context}

=== 4. 输出要求 ===
请严格返回纯 JSON 格式，不要包含 Markdown 标记。字段如下：
{{
  "stock_name": "{stock_name}",
  "sentiment_score": 0-100,
  "operation_advice": "...",
  "core_view": "...",
  "analysis_summary": "...",
  "risk_alert": "...",
  "trend_prediction": "看涨/看跌/震荡"
}}
"""

    # ---------- 核心分析 ----------

    def analyze(
        self,
        context: Dict[str, Any],
        custom_prompt: Optional[str] = None
    ) -> Optional[AnalysisResult]:

        if not self.llm:
            return None

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

            match = re.search(r'\{.*\}', content, re.DOTALL)
            if not match:
                raise ValueError("未检测到 JSON 结构")

            data = json.loads(match.group(0))

            ai_name = data.get("stock_name")
            final_name = ai_name if ai_name and ai_name != "未知股票" else context.get("stock_name", "Unknown")

            try:
                score = int(data.get("sentiment_score", 50))
            except:
                score = 50
            score = max(0, min(100, score))

            core_view = data.get("core_view", "见详细分析")

            return AnalysisResult(
                code=context.get("code", "Unknown"),
                name=final_name,
                date=context.get("date", ""),
                sentiment_score=score,
                operation_advice=data.get("operation_advice", "持有观望"),
                risk_alert=data.get("risk_alert", "暂无"),
                trend_prediction=data.get("trend_prediction", "震荡"),
                analysis_summary=data.get("analysis_summary", "AI 分析完成"),
                buy_reason=core_view,
                sell_reason=core_view
            )

        except Exception as e:
            logger.error(f"AI 分析过程异常: {e}")

            return AnalysisResult(
                code=context.get("code", "Unknown"),
                name=context.get("stock_name", "Unknown"),
                date=context.get("date", ""),
                sentiment_score=50,
                operation_advice="人工复核",
                risk_alert=f"AI 服务异常: {str(e)}",
                trend_prediction="不确定",
                analysis_summary="AI 分析失败，请检查 DeepSeek API 或返回格式。",
                buy_reason="N/A",
                sell_reason="N/A"
            )