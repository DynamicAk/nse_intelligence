"""
ingestion/rss_loader.py
LangChain-based RSS loader — fetches all NSE feeds and returns Document objects.
"""

import feedparser
from datetime import datetime
from langchain_core.documents import Document
from loguru import logger

RSS_FEEDS = {
    "corporate_announcement": "https://nsearchives.nseindia.com/content/RSS/corporate_announcement.xml",
    "financial_results":      "https://nsearchives.nseindia.com/content/RSS/financial_results.xml",
    "insider_trading":        "https://nsearchives.nseindia.com/content/RSS/insider_trading.xml",
    "corporate_actions":      "https://nsearchives.nseindia.com/content/RSS/corporate_actions.xml",
    "press_release":          "https://nsearchives.nseindia.com/content/RSS/press_release.xml",
    "investor_complaints":    "https://nsearchives.nseindia.com/content/RSS/investor_complaints.xml",
    "circulars":              "https://www.nseindia.com/content/RSS/circulars.xml",
}


def load_rss_feed(category: str, url: str, max_items: int = 30) -> list[Document]:
    """Parse one RSS feed and return LangChain Documents."""
    docs = []
    try:
        feed = feedparser.parse(url)
        for entry in feed.entries[:max_items]:
            title   = entry.get("title", "")
            desc    = entry.get("summary", entry.get("description", ""))
            pub     = entry.get("published", entry.get("updated", datetime.now().isoformat()))
            link    = entry.get("link", "")

            content = f"TITLE: {title}\nDATE: {pub}\nSOURCE: {link}\nDESCRIPTION: {desc}"

            docs.append(Document(
                page_content=content,
                metadata={
                    "category":   category,
                    "feed_url":   url,
                    "title":      title,
                    "pub_date":   pub,
                    "link":       link,
                    "ingested_at": datetime.now().isoformat(),
                }
            ))
        logger.info(f"Loaded {len(docs)} docs from [{category}]")
    except Exception as e:
        logger.error(f"Failed to load [{category}]: {e}")
    return docs


def load_all_feeds(max_items_per_feed: int = 30) -> list[Document]:
    """Load all NSE RSS feeds and return combined list of Documents."""
    all_docs = []
    for category, url in RSS_FEEDS.items():
        docs = load_rss_feed(category, url, max_items_per_feed)
        all_docs.extend(docs)
    logger.info(f"Total documents loaded: {len(all_docs)}")
    return all_docs


if __name__ == "__main__":
    docs = load_all_feeds()
    for d in docs[:3]:
        print(d.page_content[:200])
        print("---")
