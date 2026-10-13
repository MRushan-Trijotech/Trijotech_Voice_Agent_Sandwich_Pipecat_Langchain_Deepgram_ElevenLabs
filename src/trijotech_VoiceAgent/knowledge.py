"""RAG over Trijotech's documents in Supabase (pgvector) via langchain-postgres.

The ``company_documents`` table (langchain_id, content, embedding,
langchain_metadata) is exactly langchain-postgres' ``PGVectorStore`` layout, so
the store attaches to it with default column names.

IMPORTANT: the query embedding model must be the SAME model that produced the
vectors already stored in the table. Configure it in ``.env``:

    EMBEDDINGS_PROVIDER=google            # or: openai
    EMBEDDING_MODEL=gemini-embedding-001  # or e.g. text-embedding-3-small
    EMBEDDING_DIMENSIONS=                 # only if you stored shortened vectors

Check what you stored with:  select vector_dims(embedding) from company_documents limit 1;
"""

from __future__ import annotations

import os

from langchain_postgres import PGEngine, PGVectorStore

from .db import asyncpg_url

TABLE_NAME = "company_documents"

_store: PGVectorStore | None = None


def _embeddings():
    provider = os.getenv("EMBEDDINGS_PROVIDER", "google").lower()
    model = os.getenv("EMBEDDING_MODEL")
    dims = os.getenv("EMBEDDING_DIMENSIONS")

    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    kwargs = {"model": model or "gemini-embedding-001"}
    if dims:
        kwargs["output_dimensionality"] = int(dims)
    return GoogleGenerativeAIEmbeddings(**kwargs)


async def get_vector_store() -> PGVectorStore:
    """The shared vector store (created on first use)."""
    global _store
    if _store is None:
        engine = PGEngine.from_connection_string(
            url=asyncpg_url(),
            connect_args={"statement_cache_size": 0},  # safe behind Supabase poolers
        )
        _store = await PGVectorStore.create(
            engine=engine,
            table_name=TABLE_NAME,
            embedding_service=_embeddings(),
        )
    return _store