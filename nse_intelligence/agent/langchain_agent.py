"""
agent/langchain_agent.py
LangChain ReAct agent that binds NSE MCP tools + RAG retrieval.
Exposes a single chat(question, history) function used by the Streamlit UI.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from langchain_ollama import ChatOllama
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from loguru import logger

# Import our tools
from ingestion.price_fetcher import get_quote as _get_quote
from ingestion.rss_loader import load_rss_feed, RSS_FEEDS
from rag.retriever import similarity_search, format_rag_context

OLLAMA_URL  = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL   = os.getenv("OLLAMA_MODEL", "llama3.2")


# ── Tool definitions ──────────────────────────────────────────────────────────

@tool
def fetch_rss_tool(category: str) -> str:
    """
    Fetch latest NSE RSS announcements by category.
    Categories: corporate_announcement, financial_results, insider_trading,
    corporate_actions, press_release, investor_complaints, circulars.
    Returns formatted list of recent items.
    """
    if category not in RSS_FEEDS:
        return f"Unknown category. Valid options: {list(RSS_FEEDS.keys())}"
    docs = load_rss_feed(category, RSS_FEEDS[category], max_items=10)
    if not docs:
        return f"No items found in {category} feed."
    lines = [f"Feed: {category} ({len(docs)} items)\n"]
    for doc in docs[:5]:
        meta = doc.metadata
        lines.append(f"• [{meta.get('pub_date','')[:10]}] {meta.get('title','')}\n  {doc.page_content[100:250]}...\n")
    return "\n".join(lines)


@tool
def get_stock_quote(symbol: str) -> str:
    """
    Get current/latest price for an NSE stock symbol (e.g. RELIANCE, TCS, INFY).
    Returns LTP, change, volume, 52-week high/low.
    """
    result = _get_quote(symbol)
    if "error" in result:
        return f"Error fetching {symbol}: {result['error']}"

    chg     = result.get("change", 0) or 0
    chg_pct = result.get("change_pct", 0) or 0
    arrow   = "▲" if chg >= 0 else "▼"

    return (
        f"📈 {result['symbol']} — ₹{result.get('ltp', 'N/A')} {arrow} {chg:+.2f} ({chg_pct:+.2f}%)\n"
        f"   Open: ₹{result.get('open')} | High: ₹{result.get('high')} | Low: ₹{result.get('low')}\n"
        f"   Prev Close: ₹{result.get('prev_close')} | 52W High: ₹{result.get('52w_high')} | 52W Low: ₹{result.get('52w_low')}\n"
        f"   Source: {result.get('source')} | Time: {result.get('fetched_at','')[:19]}"
    )


@tool
def rag_search(query: str) -> str:
    """
    Search the NSE knowledge base (ChromaDB RAG) for documents relevant to the query.
    Use for historical analysis, background context, or when live feeds don't have enough detail.
    """
    chunks = similarity_search(query, top_k=5)
    return format_rag_context(chunks)


@tool
def search_symbol_announcements(input_str: str) -> str:
    """
    Search corporate announcements for a specific symbol.
    Input format: 'SYMBOL' or 'SYMBOL keyword'
    Example: 'RELIANCE dividend' or 'TCS'
    """
    parts   = input_str.strip().split(None, 1)
    symbol  = parts[0].upper() if parts else ""
    keyword = parts[1].lower() if len(parts) > 1 else ""

    docs = load_rss_feed("corporate_announcement", RSS_FEEDS["corporate_announcement"], max_items=50)
    matches = []
    for doc in docs:
        text = doc.page_content.lower()
        if symbol.lower() in text and (not keyword or keyword in text):
            matches.append(doc)

    if not matches:
        return f"No announcements found for {symbol} {keyword}."

    lines = [f"Found {len(matches)} announcement(s) for {symbol}:\n"]
    for doc in matches[:5]:
        meta = doc.metadata
        lines.append(f"• [{meta.get('pub_date','')[:10]}] {meta.get('title','')}")
    return "\n".join(lines)


TOOLS = [fetch_rss_tool, get_stock_quote, rag_search, search_symbol_announcements]


# ── System Prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are NSE-GPT, a specialized financial intelligence assistant for the National Stock Exchange of India.
You have access to live NSE data via RSS feeds, price APIs, and a RAG knowledge base.
You run fully locally — all processing is private.

TOOLS AVAILABLE:
{tools}

Tool names: {tool_names}

REASONING (use ReAct format):
Thought: What does the user want? Which tools should I call?
Action: tool_name
Action Input: the input to the tool
Observation: [tool result]
... (repeat as needed)
Thought: I now have enough to answer.
Final Answer: [structured response]

OUTPUT FORMAT for Final Answer:
**SUMMARY** — Direct 1-2 sentence answer
**LIVE DATA** — Price/quote data if relevant
**LATEST NEWS** — Top announcements from RSS
**ANALYSIS** — Your synthesis
**SOURCES** — Feed URLs and dates cited
⚠️ *This is informational only, not financial advice.*

HARD RULES:
- NEVER invent stock prices. If tool fails, say "Price unavailable — check NSE directly."
- NEVER give buy/sell recommendations.
- ALWAYS cite sources (feed category, pub date).
- NEVER hallucinate financial results.
- Always show data timestamps.

Conversation so far:
{chat_history}

Question: {input}
{agent_scratchpad}"""


# ── Agent builder ─────────────────────────────────────────────────────────────

def build_agent() -> AgentExecutor:
    llm = ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_URL,
        temperature=0.1,
    )

    prompt = PromptTemplate(
        input_variables=["input", "chat_history", "agent_scratchpad"],
        partial_variables={
            "tools":      "\n".join([f"- {t.name}: {t.description}" for t in TOOLS]),
            "tool_names": ", ".join([t.name for t in TOOLS]),
        },
        template=SYSTEM_PROMPT,
    )

    agent = create_react_agent(llm, TOOLS, prompt)

    memory = ConversationBufferWindowMemory(
        k=10,
        memory_key="chat_history",
        return_messages=False,
    )

    executor = AgentExecutor(
        agent=agent,
        tools=TOOLS,
        memory=memory,
        verbose=True,
        max_iterations=6,
        handle_parsing_errors=True,
        return_intermediate_steps=False,
    )
    return executor


# ── Public API ────────────────────────────────────────────────────────────────

_agent_executor = None


def get_agent() -> AgentExecutor:
    global _agent_executor
    if _agent_executor is None:
        logger.info("Building LangChain ReAct agent...")
        _agent_executor = build_agent()
    return _agent_executor


def chat(question: str) -> str:
    """
    Main entry point — send a question, get a structured answer.
    Maintains conversation memory across calls within the same process.
    """
    agent = get_agent()
    try:
        result = agent.invoke({"input": question})
        return result.get("output", "I could not generate a response. Please try again.")
    except Exception as e:
        logger.error(f"Agent error: {e}")
        return f"⚠️ Agent error: {str(e)}\nPlease rephrase your question or check that Ollama is running."


if __name__ == "__main__":
    print(chat("What is the current price of Reliance?"))
