"""
rag/retriever.py
Direct ChromaDB similarity search — used by the LangChain agent
when it needs RAG context without going through LlamaIndex.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import chromadb
from langchain_ollama import OllamaEmbeddings
from loguru import logger
from datetime import datetime, timedelta


CHROMA_DIR  = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
OLLAMA_URL  = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
COLLECTION  = "nse_intelligence"

_client     = None
_collection = None
_embedder   = None


def _get_collection():
    global _client, _collection
    if _collection is None:
        _client     = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = _client.get_or_create_collection(COLLECTION)
    return _collection


def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = OllamaEmbeddings(
            model=EMBED_MODEL,
            base_url=OLLAMA_URL,
        )
    return _embedder


def similarity_search(query: str, top_k: int = 5, max_age_hours: int = 24) -> list[dict]:
    """
    Search ChromaDB for documents similar to query.
    Optionally filter out documents older than max_age_hours.
    Returns list of {content, metadata, score} dicts.
    """
    try:
        coll    = _get_collection()
        embedder = _get_embedder()
        vec     = embedder.embed_query(query)

        results = coll.query(
            query_embeddings=[vec],
            n_results=min(top_k, coll.count() or 1),
            include=["documents", "metadatas", "distances"],
        )

        docs      = results.get("documents", [[]])[0]
        metas     = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        cutoff    = datetime.now() - timedelta(hours=max_age_hours)

        chunks = []
        for doc, meta, dist in zip(docs, metas, distances):
            # Optional age filter
            ingested = meta.get("ingested_at", "")
            if ingested:
                try:
                    ingested_dt = datetime.fromisoformat(ingested)
                    if ingested_dt < cutoff:
                        continue
                except ValueError:
                    pass

            chunks.append({
                "content":  doc,
                "metadata": meta,
                "score":    round(1 - dist, 4),  # convert distance → similarity
            })

        logger.info(f"RAG search returned {len(chunks)} chunks for: '{query[:60]}'")
        return chunks

    except Exception as e:
        logger.error(f"RAG retrieval error: {e}")
        return []


def format_rag_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a readable context string for the LLM."""
    if not chunks:
        return "No relevant context found in the RAG index."

    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk["metadata"]
        source = meta.get("category", "unknown")
        pub    = meta.get("pub_date", "")
        title  = meta.get("title", "")
        score  = chunk["score"]
        parts.append(
            f"[{i}] Source: {source} | Date: {pub} | Score: {score}\n"
            f"    Title: {title}\n"
            f"    {chunk['content'][:400]}...\n"
        )
    return "\n".join(parts)


if __name__ == "__main__":
    results = similarity_search("Reliance quarterly results", top_k=3)
    print(format_rag_context(results))
