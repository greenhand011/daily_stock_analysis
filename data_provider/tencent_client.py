# coding=utf-8
"""
统一市场数据客户端
支持：
- A股
- ETF
- 指数
- 场外基金
"""

from tencent_stock import TencentStockAPI
from typing import Dict, Any


class MarketDataClient:
    def __init__(self):
        self.client = TencentStockAPI()

    def get_quote(self, code: str) -> Dict[str, Any]:
        """
        自动识别并获取行情
        """

        raw = self.client.get_stock_info(code)

        if not raw:
            raise ValueError(f"No data for {code}")

        return self._normalize(raw, code)

    def _normalize(self, raw: Dict[str, Any], code: str) -> Dict[str, Any]:
        """
        统一数据结构
        """

        return {
            "code": code,
            "name": raw.get("name"),
            "type": self._detect_type(code),

            "price": raw.get("price"),
            "change": raw.get("change"),
            "change_pct": raw.get("change_percent"),

            "volume": raw.get("volume"),
            "turnover": raw.get("turnover"),

            "pe": raw.get("pe"),
            "pb": raw.get("pb"),
            "market_cap": raw.get("market_cap"),

            "nav": raw.get("nav"),  # 场外基金净值
        }

    def _detect_type(self, code: str) -> str:
        """
        简单类型识别规则
        """

        if code.startswith(("5", "15", "16", "51", "56")):
            return "ETF/LOF"

        if code.startswith(("0", "1")) and len(code) == 6:
            return "Fund"

        if code.startswith(("000", "399")):
            return "Index"

        return "Stock"
