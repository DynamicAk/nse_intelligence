"""
mcp_server/rss_tools.py
FastMCP server — exposes NSE RSS + price tools to the LLM agent.
Run: python mcp_server/rss_tools.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import feedparser
import requests
import yfinance as yf
from datetime import datetime
from fastmcp import FastMCP
from loguru import logger

# ── MCP Server init ──────────────────────────────────────────────────────────
mcp = FastMCP("NSE Intelligence Server")

# ── NSE RSS Feed URLs ─────────────────────────────────────────────────────────
RSS_FEEDS = {
    "corporate_announcement": "https://nsearchives.nseindia.com/content/RSS/corporate_announcement.xml",
    "financial_results":      "https://nsearchives.nseindia.com/content/RSS/financial_results.xml",
    "insider_trading":        "https://nsearchives.nseindia.com/content/RSS/insider_trading.xml",
    "corporate_actions":      "https://nsearchives.nseindia.com/content/RSS/corporate_actions.xml",
    "press_release":          "https://nsearchives.nseindia.com/content/RSS/press_release.xml",
    "investor_complaints":    "https://nsearchives.nseindia.com/content/RSS/investor_complaints.xml",
    "circulars":              "https://www.nseindia.com/content/RSS/circulars.xml",
}

NIFTY_INDICES = {
    "NIFTY50":   ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR",
                  "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK", "LT", "AXISBANK",
                  "ASIANPAINT", "MARUTI", "TITAN", "NESTLEIND", "WIPRO", "ULTRACEMCO",
                  "POWERGRID", "NTPC", "BAJFINANCE", "HCLTECH", "TECHM", "SUNPHARMA",
                  "ONGC", "COALINDIA", "DIVISLAB", "DRREDDY", "CIPLA", "BAJAJFINSV",
                  "GRASIM", "ADANIPORTS", "HINDALCO", "TATASTEEL", "JSWSTEEL",
                  "INDUSINDBK", "BRITANNIA", "EICHERMOT", "BPCL", "TATACONSUM",
                  "APOLLOHOSP", "HEROMOTOCO", "SBILIFE", "HDFCLIFE", "M&M",
                  "TATAMOTORS", "BAJAJ-AUTO", "UPL", "SHRIRAMFIN", "ADANIENT"],
    "NIFTYBANK": ["HDFCBANK", "ICICIBANK", "KOTAKBANK", "AXISBANK", "SBIN",
                  "INDUSINDBK", "BANDHANBNK", "FEDERALBNK", "IDFCFIRSTB", "AUBANK",
                  "PNB", "BANKBARODA"],
    "NIFTYIT":   ["TCS", "INFY", "HCLTECH", "WIPRO", "TECHM", "MPHASIS",
                  "LTTS", "COFORGE", "PERSISTENT", "OFSS"],
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _nse_headers() -> dict:
    """Standard headers to mimic a browser for NSE API calls."""
    return {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
        "Referer": "https://www.nseindia.com/",
    }


def _parse_feed(url: str, max_items: int = 10) -> list[dict]:
    """Parse an RSS feed URL and return list of item dicts."""
    try:
        feed = feedparser.parse(url)
        items = []
        for entry in feed.entries[:max_items]:
            items.append({
                "title":       entry.get("title", ""),
                "pubDate":     entry.get("published", entry.get("updated", "")),
                "description": entry.get("summary", entry.get("description", "")),
                "link":        entry.get("link", ""),
            })
        return items
    except Exception as e:
        logger.error(f"Feed parse error for {url}: {e}")
        return []


# ── MCP Tools ─────────────────────────────────────────────────────────────────

@mcp.tool()
def fetch_rss(category: str = "corporate_announcement", max_items: int = 10) -> dict:
    """
    Fetch and parse an NSE RSS feed by category.
    Categories: corporate_announcement, financial_results, insider_trading,
                corporate_actions, press_release, investor_complaints, circulars
    Returns list of items with title, pubDate, description, link.
    """
    if category not in RSS_FEEDS:
        return {"error": f"Unknown category '{category}'. Valid: {list(RSS_FEEDS.keys())}"}

    url = RSS_FEEDS[category]
    items = _parse_feed(url, max_items)
    return {
        "category":   category,
        "feed_url":   url,
        "fetched_at": datetime.now().isoformat(),
        "count":      len(items),
        "items":      items,
    }


@mcp.tool()
def get_quote(symbol: str) -> dict:
    """
    Get current price snapshot for an NSE symbol (e.g. RELIANCE, TCS, INFY).
    Returns LTP, change%, volume, 52w high/low using yfinance (15-min delayed).
    """
    symbol = symbol.upper().strip()
    yf_symbol = f"{symbol}.NS"

    try:
        ticker = yf.Ticker(yf_symbol)
        info   = ticker.fast_info

        return {
            "symbol":        symbol,
            "ltp":           round(getattr(info, "last_price", 0), 2),
            "open":          round(getattr(info, "open", 0), 2),
            "high":          round(getattr(info, "high", 0), 2),
            "low":           round(getattr(info, "low", 0), 2),
            "prev_close":    round(getattr(info, "previous_close", 0), 2),
            "volume":        getattr(info, "last_volume", 0),
            "52w_high":      round(getattr(info, "year_high", 0), 2),
            "52w_low":       round(getattr(info, "year_low", 0), 2),
            "market_cap":    getattr(info, "market_cap", 0),
            "note":          "Price data is ~15 min delayed via yfinance.",
            "fetched_at":    datetime.now().isoformat(),
        }
    except Exception as e:
        return {
            "symbol": symbol,
            "error":  f"Price unavailable — check NSE directly. ({e})",
        }


@mcp.tool()
def list_symbols(index: str = "NIFTY50") -> dict:
    """
    Return all symbols in a given index.
    Supported: NIFTY50, NIFTYBANK, NIFTYIT
    """
    index = index.upper()
    if index not in NIFTY_INDICES:
        return {"error": f"Unknown index '{index}'. Valid: {list(NIFTY_INDICES.keys())}"}
    return {
        "index":   index,
        "count":   len(NIFTY_INDICES[index]),
        "symbols": NIFTY_INDICES[index],
    }


@mcp.tool()
def search_announcements(symbol: str = "", keyword: str = "") -> dict:
    """
    Search corporate_announcement RSS for items mentioning symbol or keyword.
    Returns matching items from the live feed.
    """
    items = _parse_feed(RSS_FEEDS["corporate_announcement"], max_items=50)
    results = []
    search_term = (symbol + " " + keyword).strip().lower()

    for item in items:
        text = (item["title"] + " " + item["description"]).lower()
        if search_term and search_term in text:
            results.append(item)

    return {
        "symbol":   symbol,
        "keyword":  keyword,
        "matches":  len(results),
        "items":    results[:10],
        "searched_at": datetime.now().isoformat(),
    }


@mcp.tool()
def get_corporate_actions(symbol: str) -> dict:
    """
    Fetch upcoming/recent corporate actions (dividends, splits, bonus) for a symbol.
    Uses yfinance for reliable data.
    """
    symbol = symbol.upper().strip()
    yf_symbol = f"{symbol}.NS"

    try:
        ticker   = yf.Ticker(yf_symbol)
        actions  = ticker.actions

        if actions is not None and not actions.empty:
            recent = actions.tail(10).reset_index()
            records = []
            for _, row in recent.iterrows():
                records.append({
                    "date":      str(row.get("Date", "")),
                    "dividends": row.get("Dividends", 0),
                    "splits":    row.get("Stock Splits", 0),
                })
            return {"symbol": symbol, "actions": records}
        return {"symbol": symbol, "actions": [], "note": "No recent corporate actions found."}
    except Exception as e:
        return {"symbol": symbol, "error": str(e)}


@mcp.tool()
def get_financial_results(symbol: str) -> dict:
    """
    Fetch financial result announcements for a symbol from the NSE RSS financial_results feed.
    """
    items = _parse_feed(RSS_FEEDS["financial_results"], max_items=50)
    symbol_upper = symbol.upper()
    results = [i for i in items if symbol_upper in i["title"].upper() or symbol_upper in i["description"].upper()]

    return {
        "symbol":      symbol,
        "count":       len(results),
        "results":     results[:5],
        "feed_url":    RSS_FEEDS["financial_results"],
        "fetched_at":  datetime.now().isoformat(),
    }


@mcp.tool()
def get_all_feeds_summary() -> dict:
    """
    Fetch latest 3 items from each RSS feed category. Useful for a daily briefing.
    """
    summary = {}
    for category, url in RSS_FEEDS.items():
        items = _parse_feed(url, max_items=3)
        summary[category] = {"count": len(items), "items": items}
    return {"fetched_at": datetime.now().isoformat(), "feeds": summary}


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("Starting NSE Intelligence MCP Server...")
    mcp.run()
