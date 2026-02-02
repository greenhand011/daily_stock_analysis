# -*- coding: utf-8 -*-
"""
===================================
A股自选股智能分析系统 - 配置管理模块
===================================

职责：
1. 使用单例模式管理全局配置
2. 从 .env / GitHub Secrets 加载配置
3. 支持多 AI 模型（DeepSeek / Gemini / OpenAI-compatible）
"""

import os
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv
from dataclasses import dataclass, field


# ================= 配置数据结构 =================

@dataclass
class Config:
    """
    系统配置类（单例）
    """

    # === 股票配置 ===
    stock_list: List[str] = field(default_factory=list)

    # === 数据源 ===
    tushare_token: Optional[str] = None

    # ================= AI 配置 =================

    # --- DeepSeek（最高优先级，OpenAI-compatible） ---
    deepseek_api_key: Optional[str] = None
    deepseek_model: str = "deepseek-chat"

    # --- Gemini ---
    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemini-3-flash-preview"
    gemini_model_fallback: str = "gemini-2.5-flash"

    gemini_request_delay: float = 2.0
    gemini_max_retries: int = 5
    gemini_retry_delay: float = 5.0

    # --- OpenAI-compatible（兜底） ---
    openai_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None
    openai_model: str = "gpt-4o-mini"

    # ================= 搜索引擎 =================
    bocha_api_keys: List[str] = field(default_factory=list)
    tavily_api_keys: List[str] = field(default_factory=list)
    serpapi_keys: List[str] = field(default_factory=list)

    # ================= 通知配置 =================
    wechat_webhook_url: Optional[str] = None
    feishu_webhook_url: Optional[str] = None

    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None

    email_sender: Optional[str] = None
    email_password: Optional[str] = None
    email_receivers: List[str] = field(default_factory=list)

    pushover_user_key: Optional[str] = None
    pushover_api_token: Optional[str] = None

    custom_webhook_urls: List[str] = field(default_factory=list)
    custom_webhook_bearer_token: Optional[str] = None

    # ================= 系统配置 =================
    single_stock_notify: bool = False
    database_path: str = "./data/stock_analysis.db"

    log_dir: str = "./logs"
    log_level: str = "INFO"

    max_workers: int = 3
    debug: bool = False

    schedule_enabled: bool = False
    schedule_time: str = "18:00"
    market_review_enabled: bool = True

    # WebUI
    webui_enabled: bool = False
    webui_host: str = "127.0.0.1"
    webui_port: int = 8000

    # 单例实例
    _instance: Optional["Config"] = None

    # ================= 单例入口 =================

    @classmethod
    def get_instance(cls) -> "Config":
        if cls._instance is None:
            cls._instance = cls._load_from_env()
        return cls._instance

    # ================= 加载逻辑 =================

    @classmethod
    def _load_from_env(cls) -> "Config":
        env_path = Path(__file__).parent / ".env"
        load_dotenv(dotenv_path=env_path)

        stock_list_str = os.getenv("STOCK_LIST", "")
        stock_list = [s.strip() for s in stock_list_str.split(",") if s.strip()]
        if not stock_list:
            stock_list = ["600519", "000001", "300750"]

        def split_keys(name: str) -> List[str]:
            return [k.strip() for k in os.getenv(name, "").split(",") if k.strip()]

        return cls(
            stock_list=stock_list,
            tushare_token=os.getenv("TUSHARE_TOKEN"),

            deepseek_api_key=os.getenv("DEEPSEEK_API_KEY"),
            deepseek_model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),

            gemini_api_key=os.getenv("GEMINI_API_KEY"),
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-3-flash-preview"),
            gemini_model_fallback=os.getenv("GEMINI_MODEL_FALLBACK", "gemini-2.5-flash"),
            gemini_request_delay=float(os.getenv("GEMINI_REQUEST_DELAY", "2.0")),
            gemini_max_retries=int(os.getenv("GEMINI_MAX_RETRIES", "5")),
            gemini_retry_delay=float(os.getenv("GEMINI_RETRY_DELAY", "5.0")),

            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_base_url=os.getenv("OPENAI_BASE_URL"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),

            bocha_api_keys=split_keys("BOCHA_API_KEYS"),
            tavily_api_keys=split_keys("TAVILY_API_KEYS"),
            serpapi_keys=split_keys("SERPAPI_API_KEYS"),

            wechat_webhook_url=os.getenv("WECHAT_WEBHOOK_URL"),
            feishu_webhook_url=os.getenv("FEISHU_WEBHOOK_URL"),

            telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN"),
            telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID"),

            email_sender=os.getenv("EMAIL_SENDER"),
            email_password=os.getenv("EMAIL_PASSWORD"),
            email_receivers=split_keys("EMAIL_RECEIVERS"),

            pushover_user_key=os.getenv("PUSHOVER_USER_KEY"),
            pushover_api_token=os.getenv("PUSHOVER_API_TOKEN"),

            custom_webhook_urls=split_keys("CUSTOM_WEBHOOK_URLS"),
            custom_webhook_bearer_token=os.getenv("CUSTOM_WEBHOOK_BEARER_TOKEN"),

            single_stock_notify=os.getenv("SINGLE_STOCK_NOTIFY", "false").lower() == "true",
            database_path=os.getenv("DATABASE_PATH", "./data/stock_analysis.db"),

            log_dir=os.getenv("LOG_DIR", "./logs"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),

            max_workers=int(os.getenv("MAX_WORKERS", "3")),
            debug=os.getenv("DEBUG", "false").lower() == "true",

            schedule_enabled=os.getenv("SCHEDULE_ENABLED", "false").lower() == "true",
            schedule_time=os.getenv("SCHEDULE_TIME", "18:00"),
            market_review_enabled=os.getenv("MARKET_REVIEW_ENABLED", "true").lower() == "true",

            webui_enabled=os.getenv("WEBUI_ENABLED", "false").lower() == "true",
            webui_host=os.getenv("WEBUI_HOST", "127.0.0.1"),
            webui_port=int(os.getenv("WEBUI_PORT", "8000")),
        )

    # ================= ★ 新增：数据库接口 =================

    def get_db_url(self) -> str:
        """
        提供给 storage.py 使用的数据库连接 URL
        默认 SQLite
        """
        db_path = Path(self.database_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{db_path.resolve()}"

    # ================= 校验 =================

    def validate(self) -> List[str]:
        warnings = []

        if not self.stock_list:
            warnings.append("未配置 STOCK_LIST")

        if not (self.deepseek_api_key or self.gemini_api_key or self.openai_api_key):
            warnings.append("未配置任何 AI API Key")

        if not (
            self.wechat_webhook_url
            or self.feishu_webhook_url
            or (self.telegram_bot_token and self.telegram_chat_id)
        ):
            warnings.append("未配置任何通知渠道")

        return warnings


# === 快捷访问 ===
def get_config() -> Config:
    return Config.get_instance()


if __name__ == "__main__":
    cfg = get_config()
    print("配置加载成功")
    print("股票:", cfg.stock_list)
    print("AI 优先级: DeepSeek → Gemini → OpenAI")