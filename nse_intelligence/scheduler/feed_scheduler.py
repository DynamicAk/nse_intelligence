"""
scheduler/feed_scheduler.py
APScheduler — refreshes the RAG index every N minutes.
Run alongside your UI: python scheduler/feed_scheduler.py &
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from apscheduler.schedulers.blocking import BlockingScheduler
from loguru import logger
from datetime import datetime

REFRESH_INTERVAL = int(os.getenv("RAG_REFRESH_INTERVAL_MINUTES", 5))
NSE_SESSION_INTERVAL = int(os.getenv("NSE_SESSION_REFRESH_MINUTES", 15))


def refresh_rag():
    """Incremental RAG update — fetches latest RSS and re-embeds."""
    logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] Scheduler: starting RAG refresh...")
    try:
        from rag.indexer import run_update_index
        run_update_index()
        logger.success("RAG refresh complete.")
    except Exception as e:
        logger.error(f"RAG refresh failed: {e}")


def refresh_nse_session():
    """Reset the NSE requests session to rotate cookies."""
    logger.info("Scheduler: rotating NSE session cookies...")
    try:
        from ingestion.price_fetcher import reset_session
        reset_session()
        logger.success("NSE session reset.")
    except Exception as e:
        logger.error(f"NSE session reset failed: {e}")


def start_scheduler():
    scheduler = BlockingScheduler(timezone="Asia/Kolkata")

    scheduler.add_job(
        refresh_rag,
        "interval",
        minutes=REFRESH_INTERVAL,
        id="rag_refresh",
        name="NSE RAG Index Refresh",
        replace_existing=True,
    )

    scheduler.add_job(
        refresh_nse_session,
        "interval",
        minutes=NSE_SESSION_INTERVAL,
        id="nse_session_refresh",
        name="NSE Session Cookie Rotation",
        replace_existing=True,
    )

    logger.info(f"Scheduler started: RAG refresh every {REFRESH_INTERVAL} min | NSE session every {NSE_SESSION_INTERVAL} min")
    logger.info("Press Ctrl+C to stop.")

    try:
        # Run once immediately on start
        refresh_rag()
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")


if __name__ == "__main__":
    start_scheduler()
