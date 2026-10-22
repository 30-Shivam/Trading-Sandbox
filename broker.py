"""Thin Alpaca PAPER-trading REST client (plain `requests`, no extra SDK).

Hard-wired to the paper endpoint: the constructor refuses any other host, so
a mis-set ALPACA_BASE_URL can never route orders to a live account. Credentials
come from ALPACA_PAPER_KEY / ALPACA_PAPER_SECRET (never from code or git).

Used only by paper_trade.py. Every method raises BrokerError with the HTTP
status and Alpaca's message on failure instead of returning partial data.
"""

import os
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv

load_dotenv()

PAPER_HOST = "paper-api.alpaca.markets"
PAPER_BASE_URL = f"https://{PAPER_HOST}"
REQUEST_TIMEOUT_SEC = 20


class BrokerError(RuntimeError):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class AlpacaPaperBroker:
    def __init__(self, key: str | None = None, secret: str | None = None,
                 base_url: str | None = None, session: requests.Session | None = None):
        key = key or os.getenv("ALPACA_PAPER_KEY")
        secret = secret or os.getenv("ALPACA_PAPER_SECRET")
        if not key or not secret:
            raise BrokerError("ALPACA_PAPER_KEY / ALPACA_PAPER_SECRET are not set.")
        base_url = (base_url or os.getenv("ALPACA_BASE_URL") or PAPER_BASE_URL).rstrip("/")
        if urlparse(base_url).hostname != PAPER_HOST:
            raise BrokerError(f"Refusing non-paper endpoint {base_url!r}; only {PAPER_HOST} is allowed.")
        self.base_url = base_url
        self._session = session or requests.Session()
        self._session.headers.update({"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret})

    def _request(self, method: str, path: str, **kwargs):
        try:
            resp = self._session.request(method, f"{self.base_url}{path}", timeout=REQUEST_TIMEOUT_SEC, **kwargs)
        except requests.RequestException as exc:
            raise BrokerError(f"{method} {path} failed: {exc}") from exc
        if resp.status_code >= 400:
            try:
                message = resp.json().get("message", resp.text)
            except ValueError:
                message = resp.text
            raise BrokerError(f"{method} {path} -> HTTP {resp.status_code}: {message}", status=resp.status_code)
        return resp.json() if resp.content else None

    def account(self) -> dict:
        return self._request("GET", "/v2/account")

    def clock(self) -> dict:
        return self._request("GET", "/v2/clock")

    def positions(self) -> list[dict]:
        return self._request("GET", "/v2/positions")

    def orders(self, status: str = "all", after: str | None = None, limit: int = 500) -> list[dict]:
        params = {"status": status, "limit": limit, "nested": "true", "direction": "desc"}
        if after:
            params["after"] = after
        return self._request("GET", "/v2/orders", params=params)

    def submit_bracket_limit(self, symbol: str, qty: int, limit_price: float, stop_price: float,
                             take_profit_price: float, client_order_id: str) -> dict:
        """Buy-limit entry with attached stop-loss and take-profit (OCO) legs.
        GTC so the entry persists until filled; paper_trade.py cancels stale ones."""
        body = {
            "symbol": symbol, "qty": str(int(qty)), "side": "buy", "type": "limit",
            "time_in_force": "gtc", "limit_price": f"{limit_price:.2f}", "order_class": "bracket",
            "take_profit": {"limit_price": f"{take_profit_price:.2f}"},
            "stop_loss": {"stop_price": f"{stop_price:.2f}"},
            "client_order_id": client_order_id,
        }
        return self._request("POST", "/v2/orders", json=body)

    def submit_market_notional(self, symbol: str, notional: float, side: str, client_order_id: str) -> dict:
        """Day market order sized in dollars (fractional shares allowed for market orders)."""
        body = {"symbol": symbol, "notional": f"{notional:.2f}", "side": side, "type": "market",
                "time_in_force": "day", "client_order_id": client_order_id}
        return self._request("POST", "/v2/orders", json=body)

    def portfolio_history(self, period: str = "3M", timeframe: str = "1D") -> dict:
        return self._request("GET", "/v2/account/portfolio/history", params={"period": period, "timeframe": timeframe})

    def cancel_order(self, order_id: str) -> None:
        self._request("DELETE", f"/v2/orders/{order_id}")

    def close_position(self, symbol: str) -> dict:
        return self._request("DELETE", f"/v2/positions/{symbol}")
