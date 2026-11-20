"""
ui/streamlit_app.py
NSE Stock Intelligence — Streamlit Chat UI
Run: streamlit run ui/streamlit_app.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import yfinance as yf

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NSE Stock Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0d1117; }
    .stApp { background-color: #0d1117; }
    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 4px 0;
    }
    .price-up   { color: #3fb950; font-weight: bold; }
    .price-down { color: #f85149; font-weight: bold; }
    .chat-msg-user     { background: #1f2937; border-radius: 8px; padding: 10px; margin: 6px 0; }
    .chat-msg-assistant{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 10px; margin: 6px 0; }
    h1, h2, h3 { color: #e6edf3 !important; }
    .stTextInput input { background: #161b22; color: #e6edf3; border: 1px solid #30363d; }
    div[data-testid="stSidebarContent"] { background: #161b22; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)  # cache 5 min
def fetch_quote_cached(symbol: str) -> dict:
    from ingestion.price_fetcher import get_quote
    return get_quote(symbol)


@st.cache_data(ttl=900)  # cache 15 min
def fetch_ohlcv(symbol: str, period: str = "3mo") -> pd.DataFrame:
    try:
        ticker = yf.Ticker(f"{symbol}.NS")
        df = ticker.history(period=period)
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def fetch_feed_items(category: str, max_items: int = 8):
    from ingestion.rss_loader import load_rss_feed, RSS_FEEDS
    if category not in RSS_FEEDS:
        return []
    return load_rss_feed(category, RSS_FEEDS[category], max_items=max_items)


def render_price_card(symbol: str):
    q = fetch_quote_cached(symbol)
    if "error" in q:
        st.error(f"{symbol}: {q['error']}")
        return

    ltp  = q.get("ltp", 0) or 0
    chg  = q.get("change", 0) or 0
    pchg = q.get("change_pct", 0) or 0
    color = "price-up" if chg >= 0 else "price-down"
    arrow = "▲" if chg >= 0 else "▼"

    st.markdown(f"""
    <div class="metric-card">
        <b style="color:#e6edf3">{symbol}</b><br>
        <span style="font-size:1.4em;color:{'#3fb950' if chg>=0 else '#f85149'}">₹{ltp:,.2f}</span>
        <span class="{color}"> {arrow} {chg:+.2f} ({pchg:+.2f}%)</span><br>
        <small style="color:#8b949e">H: ₹{q.get('high',0)} | L: ₹{q.get('low',0)} | {q.get('source','')[:20]}</small>
    </div>
    """, unsafe_allow_html=True)


def render_candlestick(symbol: str):
    df = fetch_ohlcv(symbol)
    if df.empty:
        st.warning("No OHLCV data available.")
        return

    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"],
        low=df["Low"],   close=df["Close"],
        increasing_line_color="#3fb950",
        decreasing_line_color="#f85149",
    )])
    fig.update_layout(
        paper_bgcolor="#161b22",
        plot_bgcolor="#0d1117",
        font_color="#e6edf3",
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=30, b=10),
        height=280,
        title=f"{symbol} — 3 Month Chart",
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📈 NSE Intelligence")
    st.caption(f"🕐 {datetime.now().strftime('%d %b %Y, %H:%M IST')}")

    st.divider()

    # Watchlist
    st.subheader("⚡ Watchlist")
    watchlist_input = st.text_input(
        "Symbols (comma-separated)",
        value="RELIANCE,TCS,HDFCBANK,INFY,SBIN",
        key="watchlist",
    )
    watchlist = [s.strip().upper() for s in watchlist_input.split(",") if s.strip()]

    for sym in watchlist[:6]:
        render_price_card(sym)

    st.divider()

    # Chart
    st.subheader("📊 Chart")
    chart_symbol = st.selectbox("Select symbol", watchlist, key="chart_sym")
    if chart_symbol:
        render_candlestick(chart_symbol)

    st.divider()

    # Feed browser
    st.subheader("📰 Live Feeds")
    feed_cat = st.selectbox(
        "Category",
        ["corporate_announcement", "financial_results", "insider_trading",
         "corporate_actions", "press_release", "circulars"],
        key="feed_cat",
    )
    feed_docs = fetch_feed_items(feed_cat, max_items=5)
    for doc in feed_docs:
        meta = doc.metadata
        pub  = meta.get("pub_date", "")[:10]
        title = meta.get("title", "")[:60]
        link  = meta.get("link", "#")
        st.markdown(f"- [{title}...]({link}) `{pub}`")

    st.divider()
    st.caption("⚠️ Informational only. Not financial advice.")
    st.caption("🔒 100% local — no cloud APIs used.")


# ── Main chat area ────────────────────────────────────────────────────────────

st.title("🤖 NSE Stock Intelligence Chatbot")
st.caption("Ask me anything about NSE stocks, announcements, results, or market events.")

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "👋 Hello! I'm **NSE-GPT**, your local AI assistant for Indian equity markets.\n\n"
                "I can help you with:\n"
                "- 📈 **Stock prices** — `What is the price of Infosys?`\n"
                "- 📢 **Announcements** — `Show me latest TCS announcements`\n"
                "- 💰 **Corporate actions** — `Any upcoming dividends for HDFC Bank?`\n"
                "- 📊 **Financial results** — `Summarize Reliance quarterly results`\n"
                "- 🔍 **Market news** — `What's happening in NSE today?`\n\n"
                "All data is fetched locally from NSE RSS feeds. Let's go! 🚀"
            )
        }
    ]

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Quick prompts
st.markdown("**Quick prompts:**")
cols = st.columns(4)
quick_prompts = [
    "📈 NIFTY 50 movers today",
    "📢 Latest corporate announcements",
    "💰 Recent dividend declarations",
    "📊 Financial results this week",
]
for i, (col, prompt) in enumerate(zip(cols, quick_prompts)):
    if col.button(prompt, key=f"qp_{i}"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.rerun()

# Chat input
if user_input := st.chat_input("Ask about any NSE stock, announcement, or market event..."):
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Fetching data and reasoning..."):
            try:
                from agent.langchain_agent import chat
                response = chat(user_input)
            except Exception as e:
                response = (
                    f"⚠️ **Agent Error**: {str(e)}\n\n"
                    "Please make sure:\n"
                    "1. Ollama is running: `ollama serve`\n"
                    f"2. Model is pulled: `ollama pull {os.getenv('OLLAMA_MODEL', 'llama3.2')}`\n"
                    "3. RAG index exists: `python rag/indexer.py --full`"
                )

        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

# Clear chat button
if len(st.session_state.messages) > 1:
    if st.button("🗑️ Clear chat", key="clear"):
        st.session_state.messages = st.session_state.messages[:1]
        st.rerun()
