"""
rag/indexer.py
LlamaIndex + ChromaDB indexer.
Loads NSE RSS docs, splits, embeds with nomic-embed-text, stores in ChromaDB.

Usage:
    python rag/indexer.py --full          # fresh index
    python rag/indexer.py --update        # add new docs only
"""

import os
import sys
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import chromadb
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from loguru import logger

from ingestion.rss_loader import load_all_feeds


CHROMA_DIR   = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
OLLAMA_URL   = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL  = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
LLM_MODEL    = os.getenv("OLLAMA_MODEL", "llama3.2")
COLLECTION   = "nse_intelligence"


def setup_llama_settings():
    """Configure LlamaIndex global settings for Ollama."""
    Settings.llm = Ollama(
        model=LLM_MODEL,
        base_url=OLLAMA_URL,
        request_timeout=120.0,
    )
    Settings.embed_model = OllamaEmbedding(
        model_name=EMBED_MODEL,
        base_url=OLLAMA_URL,
    )
    Settings.node_parser = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=50,
    )
    logger.info(f"LlamaIndex configured: LLM={LLM_MODEL}, Embed={EMBED_MODEL}")


def get_chroma_store():
    """Initialize ChromaDB client and collection."""
    client     = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(COLLECTION)
    return ChromaVectorStore(chroma_collection=collection), client


def build_index(documents=None) -> VectorStoreIndex:
    """Build or load the VectorStoreIndex from ChromaDB."""
    setup_llama_settings()
    vector_store, _ = get_chroma_store()
    storage_context  = StorageContext.from_defaults(vector_store=vector_store)

    if documents:
        # Convert LangChain Documents → LlamaIndex TextNodes
        nodes = []
        for doc in documents:
            node = TextNode(
                text=doc.page_content,
                metadata=doc.metadata,
            )
            nodes.append(node)

        index = VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            show_progress=True,
        )
        logger.success(f"Index built with {len(nodes)} nodes.")
    else:
        # Load existing index
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context,
        )
        logger.info("Loaded existing ChromaDB index.")

    return index


def run_full_index():
    """Full rebuild: fetch all feeds, embed, store."""
    logger.info("Starting full RAG index build...")
    docs  = load_all_feeds(max_items_per_feed=30)
    index = build_index(documents=docs)
    logger.success(f"Full index complete. {len(docs)} documents indexed.")
    return index


def run_update_index():
    """Incremental update: fetch new docs and add to existing index."""
    logger.info("Running incremental RAG update...")
    docs  = load_all_feeds(max_items_per_feed=10)
    index = build_index(documents=docs)
    logger.success(f"Index updated with {len(docs)} fresh documents.")
    return index


def get_query_engine(similarity_top_k: int = 5):
    """Return a CitationQueryEngine for answering questions."""
    from llama_index.core.query_engine import CitationQueryEngine
    setup_llama_settings()
    index = build_index()
    engine = CitationQueryEngine.from_args(
        index,
        similarity_top_k=similarity_top_k,
        citation_chunk_size=512,
    )
    return engine


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NSE RAG Indexer")
    parser.add_argument("--full",   action="store_true", help="Full reindex")
    parser.add_argument("--update", action="store_true", help="Incremental update")
    args = parser.parse_args()

    if args.full:
        run_full_index()
    elif args.update:
        run_update_index()
    else:
        print("Usage: python rag/indexer.py --full | --update")
