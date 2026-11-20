"""
ui/gradio_app.py
Alternative Gradio chat interface for NSE Intelligence.
Run: python ui/gradio_app.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import gradio as gr
from loguru import logger


def chat_fn(message: str, history: list) -> str:
    """Gradio chat handler — wraps the LangChain agent."""
    try:
        from agent.langchain_agent import chat
        return chat(message)
    except Exception as e:
        return (
            f"⚠️ **Error**: {str(e)}\n\n"
            "Check that Ollama is running and the model is pulled."
        )


def build_ui():
    with gr.Blocks(
        title="NSE Stock Intelligence",
        theme=gr.themes.Base(
            primary_hue="green",
            secondary_hue="gray",
            neutral_hue="gray",
        ),
        css="""
            .gradio-container { background: #0d1117; }
            .chatbot { background: #161b22 !important; }
        """,
    ) as demo:
        gr.Markdown("""
        # 📈 NSE Stock Intelligence Chatbot
        > Powered by Ollama · LangChain · LlamaIndex · ChromaDB — 100% Local, No Cloud APIs

        **Ask about:** stock prices · corporate announcements · financial results · dividends · insider trading
        """)

        chatbot = gr.ChatInterface(
            fn=chat_fn,
            chatbot=gr.Chatbot(height=500, label="NSE-GPT"),
            textbox=gr.Textbox(
                placeholder="What is the current price of HDFC Bank?",
                container=False,
            ),
            examples=[
                "What is the current price of Reliance Industries?",
                "Show me latest corporate announcements from NSE",
                "Any dividend declarations this week?",
                "Summarize TCS financial results",
                "What are the top NIFTY 50 stocks by weight?",
            ],
            title="",
            retry_btn="🔄 Retry",
            undo_btn="↩️ Undo",
            clear_btn="🗑️ Clear",
        )

        gr.Markdown("""
        ---
        ⚠️ *This is informational only, not financial advice. Data sourced from NSE RSS feeds.*
        🔒 *All processing is local — no data leaves your machine.*
        """)

    return demo


if __name__ == "__main__":
    logger.info("Starting Gradio UI...")
    ui = build_ui()
    ui.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
