#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI 分析引擎 - 支持 DeepSeek 和多种模型
"""

import logging
import json
import re
import time
import os
import sys
from dataclasses import dataclass
from typing import Optional, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config import get_config

logger = logging.getLogger(__name__)

# ================= 编码修复辅助函数 =================

def safe_encode(text: Any, encoding: str = 'utf-8') -> str:
    """安全地编码文本，确保返回UTF-8字符串"""
    if text is None:
        return ""
    
    if isinstance(text, bytes):
        try:
            return text.decode(encoding, errors='ignore')
        except:
            return str(text)
    
    if not isinstance(text, str):
        text = str(text)
    
    try:
        # 确保文本是UTF-8编码
        encoded = text.encode(encoding, errors='ignore')
        return encoded.decode(encoding, errors='ignore')
    except:
        return text

def extract_json_from_text(text: str) -> Optional[str]:
    """从文本中提取JSON内容"""
    if not text:
        return None
    
    # 尝试找到JSON对象
    text = safe_encode(text)
    
    # 方法1：查找 {...} 模式
    json_pattern = r'\{[^{}]*\}'
    matches = re.findall(json_pattern, text, re.DOTALL)
    
    # 方法2：查找 ```json ... ``` 模式
    if not matches:
        json_block_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        matches = re.findall(json_block_pattern, text, re.DOTALL)
    
    # 方法3：直接尝试解析整个文本
    if matches:
        for match in matches:
            try:
                # 验证是否为有效JSON
                json.loads(match)
                return match
            except:
                continue
    
    # 如果没找到，返回原始文本
    return text

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
    DeepSeek 分析引擎，支持重试和错误处理
    """

    def __init__(self):
        self.config = get_config()
        self.llm = None
        self.model_name = "deepseek-chat"
        self.max_retries = 3
        self.timeout = 30  # 秒
        
        # 检查配置
        self._check_config()
        
        try:
            from openai import OpenAI
            
            # 支持多种配置方式
            api_key = None
            base_url = "https://api.deepseek.com/v1"
            
            # 方式1：使用 DEEPSEEK_API_KEY
            if self.config.deepseek_api_key:
                api_key = self.config.deepseek_api_key
                logger.info("✅ 使用 DEEPSEEK_API_KEY 配置")
            
            # 方式2：使用 OPENAI_API_KEY + OPENAI_BASE_URL
            elif self.config.openai_api_key:
                api_key = self.config.openai_api_key
                if self.config.openai_base_url:
                    base_url = self.config.openai_base_url
                logger.info(f"✅ 使用 OpenAI 兼容模式，Base URL: {base_url}")
            
            if not api_key:
                logger.error("❌ 未配置 AI API 密钥（需要 DEEPSEEK_API_KEY 或 OPENAI_API_KEY）")
                return
            
            # 设置模型名称
            if self.config.deepseek_model:
                self.model_name = self.config.deepseek_model
            elif self.config.openai_model:
                self.model_name = self.config.openai_model
            
            # 创建客户端
            self.llm = OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=self.timeout,
            )
            
            # 测试连接
            self._test_connection()
            
            logger.info(f"✅ DeepSeek 分析引擎初始化完成，使用模型: {self.model_name}")
            
        except ImportError as e:
            logger.error(f"❌ 未安装 openai 包: {e}")
            logger.info("请运行: pip install openai>=1.0.0")
            self.llm = None
        except Exception as e:
            logger.error(f"❌ DeepSeek 初始化失败: {e}")
            self.llm = None

    def _check_config(self):
        """检查配置并记录信息"""
        config_vars = [
            ("DEEPSEEK_API_KEY", self.config.deepseek_api_key),
            ("OPENAI_API_KEY", self.config.openai_api_key),
            ("OPENAI_BASE_URL", self.config.openai_base_url),
            ("DEEPSEEK_MODEL", self.config.deepseek_model),
            ("OPENAI_MODEL", self.config.openai_model),
        ]
        
        for var_name, var_value in config_vars:
            if var_value:
                masked_value = var_value[:8] + "..." + var_value[-4:] if len(var_value) > 12 else "***"
                logger.debug(f"   {var_name}: {masked_value}")
            else:
                logger.debug(f"   {var_name}: 未设置")

    def _test_connection(self):
        """测试API连接"""
        try:
            # 简单的模型列表请求测试
            models = self.llm.models.list()
            available_models = [model.id for model in models.data]
            logger.debug(f"可用模型: {available_models}")
            
            # 检查目标模型是否可用
            if self.model_name not in available_models:
                logger.warning(f"模型 {self.model_name} 不在可用列表中，将尝试使用")
        except Exception as e:
            logger.warning(f"API连接测试失败（可能不影响正常使用）: {e}")

    # ================= Prompt 生成 =================

    def generate_cio_prompt(
    self,
    stock_info: Dict[str, Any],
    tech_data: Dict[str, Any],
    trend_context: Dict[str, Any],
) -> str:
    """生成首席投资官提示词"""

        stock_name = safe_encode(stock_info.get("name", "未知股票"))
        stock_code = stock_info.get("code", "Unknown")
    
        cost = float(stock_info.get("cost", 0))
        shares = int(stock_info.get("shares", 0))
        current_price = float(tech_data.get("price", 0))

    # 持仓上下文
    if shares > 0 and cost > 0 and current_price > 0:
        profit_pct = (current_price - cost) / cost * 100
        profit_status = "盈利" if profit_pct > 0 else "亏损"
        position_context = (
            f"用户持仓 {shares} 股，成本 {cost:.2f} 元，当前价格 {current_price:.2f} 元，"
            f"当前{profit_status} {abs(profit_pct):.2f}%。"
        )
    else:
        position_context = "用户当前为空仓，请评估安全边际与建仓方式。"

    # 新闻上下文
    macro_news = safe_encode(trend_context.get("macro", "当前宏观面平静"))
    sector_news = safe_encode(trend_context.get("sector", "板块暂无重大消息"))
    target_sector = safe_encode(trend_context.get("target_sector", "通用"))

    # 简化格式化：直接转换为字符串，不进行复杂的条件格式化
    def simple_format(value):
        if value is None:
            return "N/A"
        if isinstance(value, (int, float)):
            # 简单的格式化，避免复杂逻辑
            return f"{value:.2f}"
        return str(value)

    # 格式化技术指标
    ma5_str = simple_format(tech_data.get("ma5"))
    ma20_str = simple_format(tech_data.get("ma20"))
    ma60_str = simple_format(tech_data.get("ma60"))
    rsi_str = simple_format(tech_data.get("rsi"))
    macd_str = simple_format(tech_data.get("macd"))
    support_str = simple_format(tech_data.get("support"))
    resistance_str = simple_format(tech_data.get("resistance"))

    prompt = f"""
你是一位专业的 A 股首席投资官（CIO），拥有 20 年投资经验。

请基于以下三方面信息，对 {stock_name}（{stock_code}）给出明确的交易决策：
1. 市场环境与新闻面
2. 技术分析指标
3. 用户实际持仓情况

=== 市场环境 ===
宏观面：{macro_news}
行业面（{target_sector}）：{sector_news}

=== 技术面分析（日线）===
当前价格：{current_price:.2f}
移动平均线：
  - MA5（短期）：{ma5_str}
  - MA20（中期）：{ma20_str}
  - MA60（长期）：{ma60_str}
相对强弱指数（RSI）：{rsi_str}
MACD：{macd_str}
关键技术位：
  - 支撑位：{support_str}
  - 阻力位：{resistance_str}

=== 用户持仓情况 ===
{position_context}

=== 输出要求 ===
请以JSON格式返回分析结果，包含以下字段：
{{
  "stock_name": "{stock_name}",
  "sentiment_score": 0-100的整数（80+积极，40-谨慎，中间中性）,
  "operation_advice": "具体的操作建议（如：加仓、减仓、持有、观望等）",
  "core_view": "一句话核心逻辑",
  "analysis_summary": "详细分析（结合持仓和当前市场环境）",
  "risk_alert": "需要关注的主要风险",
  "trend_prediction": "未来1周走势预测"
}}

请确保只返回有效的JSON格式，不要包含其他解释性文字。
"""

    return prompt

    # ================= 核心分析（带重试） =================

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((Exception,)),
        reraise=True
    )
    def _call_ai_api(self, prompt: str) -> Optional[str]:
        """调用AI API（带重试机制）"""
        if not self.llm:
            raise ValueError("AI客户端未初始化")
        
        try:
            logger.debug(f"调用AI API，prompt长度: {len(prompt)}")
            
            # 确保prompt是UTF-8编码
            prompt = safe_encode(prompt)
            
            response = self.llm.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "你是一位专业的股票分析师。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # 较低的temperature以获得更稳定的输出
                max_tokens=1000,
            )
            
            content = response.choices[0].message.content
            return safe_encode(content)
            
        except Exception as e:
            logger.warning(f"AI API调用失败，将重试: {e}")
            raise

    def analyze(
        self,
        context: Dict[str, Any],
        custom_prompt: str = None,
    ) -> Optional[AnalysisResult]:
        """分析股票数据"""
        
        if not self.llm:
            logger.error("AI客户端未初始化，无法进行分析")
            return self._create_error_result(context, "AI客户端未初始化")

        try:
            # 使用自定义提示词或生成默认提示词
            if custom_prompt:
                prompt = safe_encode(custom_prompt)
            else:
                # 如果没有提供custom_prompt，应该生成一个
                # 这里简单返回错误结果
                return self._create_error_result(context, "未提供分析提示词")
            
            # 调用AI API
            logger.info(f"开始AI分析，股票: {context.get('code', '未知')}")
            start_time = time.time()
            
            ai_response = self._call_ai_api(prompt)
            
            elapsed_time = time.time() - start_time
            logger.debug(f"AI分析完成，耗时: {elapsed_time:.2f}秒")
            
            if not ai_response:
                logger.error("AI返回空响应")
                return self._create_error_result(context, "AI返回空响应")
            
            # 解析AI响应
            result = self._parse_ai_response(ai_response, context)
            
            if result:
                logger.info(f"股票 {result.code} 分析完成，情绪分数: {result.sentiment_score}")
            else:
                logger.warning(f"股票 {context.get('code')} 分析结果解析失败")
                
            return result
            
        except Exception as e:
            logger.error(f"AI分析过程失败: {e}", exc_info=True)
            return self._create_error_result(context, str(e))

    def _parse_ai_response(self, ai_response: str, context: Dict[str, Any]) -> Optional[AnalysisResult]:
        """解析AI返回的响应"""
        try:
            # 从响应中提取JSON
            json_str = extract_json_from_text(ai_response)
            
            if not json_str:
                logger.error(f"无法从AI响应中提取JSON: {ai_response[:200]}...")
                return None
            
            # 解析JSON
            data = json.loads(json_str)
            
            # 验证必要字段
            required_fields = ["sentiment_score", "operation_advice", "core_view", 
                             "analysis_summary", "risk_alert", "trend_prediction"]
            
            for field in required_fields:
                if field not in data:
                    logger.warning(f"AI响应缺少字段: {field}")
                    data[field] = "未知"
            
            # 处理情绪分数
            try:
                score = int(data["sentiment_score"])
                score = max(0, min(100, score))
            except:
                score = 50
                logger.warning(f"情绪分数解析失败，使用默认值50: {data.get('sentiment_score')}")
            
            # 获取股票名称
            stock_name = data.get("stock_name", "")
            if not stock_name:
                stock_name = context.get("stock_name", f"股票{context.get('code', '')}")
            
            # 创建分析结果
            return AnalysisResult(
                code=context.get("code", ""),
                name=safe_encode(stock_name),
                date=context.get("date", ""),
                sentiment_score=score,
                operation_advice=safe_encode(data.get("operation_advice", "观望")),
                risk_alert=safe_encode(data.get("risk_alert", "")),
                trend_prediction=safe_encode(data.get("trend_prediction", "")),
                analysis_summary=safe_encode(data.get("analysis_summary", "")),
                buy_reason=safe_encode(data.get("core_view", "")),
                sell_reason=safe_encode(data.get("core_view", "")),
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {e}")
            logger.debug(f"原始响应: {ai_response[:500]}")
            return None
        except Exception as e:
            logger.error(f"解析AI响应时出错: {e}")
            return None

    def _create_error_result(self, context: Dict[str, Any], error_msg: str) -> AnalysisResult:
        """创建错误结果"""
        return AnalysisResult(
            code=context.get("code", ""),
            name=context.get("stock_name", ""),
            date=context.get("date", ""),
            sentiment_score=50,
            operation_advice="人工复核",
            risk_alert=safe_encode(error_msg),
            trend_prediction="不确定",
            analysis_summary="AI分析失败，需要人工检查",
        )

# ================= 向后兼容 =================

# 保持GeminiAnalyzer名称向后兼容
GeminiAnalyzer = DeepSeekAnalyzer

# ================= 工厂函数 =================

def create_analyzer(analyzer_type: str = "deepseek") -> DeepSeekAnalyzer:
    """创建分析器工厂函数"""
    if analyzer_type.lower() in ["deepseek", "openai", "gemini"]:
        return DeepSeekAnalyzer()
    else:
        logger.warning(f"不支持的分析器类型: {analyzer_type}，使用默认DeepSeek分析器")
        return DeepSeekAnalyzer()
