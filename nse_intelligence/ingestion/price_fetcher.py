"""
ingestion/price_fetcher.py
Two-tier price strategy:
  1. NSE internal JSON API (requires session cookie)
  2. yfinance fallback (15-min delayed, always available)
"""

import requests
import yfinance as yf
from loguru import logger
from datetime import datetime


NSE_BASE = "https://www.nseindia.com"
NSE_QUOTE_URL = "https://www.nseindia.com/api/quote-equity?symbol={symbol}"

_session = None


def _get_nse_session() -> requests.Session:
    """Create a requests session with NSE cookies by hitting the home page."""
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": NSE_BASE,
        })
        try:
            _session.get(NSE_BASE, timeout=10)  # seeds cookies
            logger.info("NSE session initialized.")
        except Exception as e:
            logger.warning(f"Could not initialize NSE session: {e}")
    return _session


def get_nse_quote(symbol: str) -> dict | None:
    """Try NSE internal API. Returns dict on success, None on failure."""
    session = _get_nse_session()
    url = NSE_QUOTE_URL.format(symbol=symbol.upper())
    try:
        resp = session.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            pd = data.get("priceInfo", {})
            return {
                "symbol":     symbol.upper(),
                "ltp":        pd.get("lastPrice"),
                "change":     pd.get("change"),
                "change_pct": pd.get("pChange"),
                "open":       pd.get("open"),
                "high":       pd.get("intraDayHighLow", {}).get("max"),
                "low":        pd.get("intraDayHighLow", {}).get("min"),
                "prev_close": pd.get("previousClose"),
                "52w_high":   pd.get("weekHighLow", {}).get("max"),
                "52w_low":    pd.get("weekHighLow", {}).get("min"),
                "source":     "NSE API (real-time)",
                "fetched_at": datetime.now().isoformat(),
            }
    except Exception as e:
        logger.warning(f"NSE API failed for {symbol}: {e}")
    return None


def get_yfinance_quote(symbol: str) -> dict:
    """yfinance fallback — always works, ~15 min delayed."""
    yf_symbol = f"{symbol.upper()}.NS"
    try:
        ticker = yf.Ticker(yf_symbol)
        info = ticker.fast_info
        prev = getattr(info, "previous_close", 0) or 0
        ltp  = getattr(info, "last_price", 0) or 0
        chg  = round(ltp - prev, 2) if prev else 0
        pchg = round((chg / prev) * 100, 2) if prev else 0

        return {
            "symbol":     symbol.upper(),
            "ltp":        round(ltp, 2),
            "change":     chg,
            "change_pct": pchg,
            "open":       round(getattr(info, "open", 0), 2),
            "high":       round(getattr(info, "high", 0), 2),
            "low":        round(getattr(info, "low", 0), 2),
            "prev_close": round(prev, 2),
            "52w_high":   round(getattr(info, "year_high", 0), 2),
            "52w_low":    round(getattr(info, "year_low", 0), 2),
            "volume":     getattr(info, "last_volume", 0),
            "source":     "yfinance (~15 min delayed)",
            "fetched_at": datetime.now().isoformat(),
        }
    except Exception as e:
        return {
            "symbol": symbol.upper(),
            "error":  f"Price unavailable — check NSE directly. ({e})",
            "source": "yfinance",
        }


def get_quote(symbol: str) -> dict:
    """
    Get quote with automatic fallback:
    1. Try NSE API (real-time)
    2. Fall back to yfinance (15-min delayed)
    """
    result = get_nse_quote(symbol)
    if result:
        return result
    logger.info(f"Falling back to yfinance for {symbol}")
    return get_yfinance_quote(symbol)


def reset_session():
    """Force a new NSE session (call after cookie expiry)."""
    global _session
    _session = None
    logger.info("NSE session reset.")


if __name__ == "__main__":
    q = get_quote("RELIANCE")
    print(q)
