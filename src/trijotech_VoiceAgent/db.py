"""Shared Postgres (Supabase) plumbing for Trija.

One *synchronous* psycopg connection pool serves both the LangGraph checkpointer
and the appointment / handoff tools. Async code reaches it through
``asyncio.to_thread``.

Why sync: psycopg's async mode cannot run on Windows' default
``ProactorEventLoop`` (which uvicorn, and therefore Pipecat's runner, may use).
The sync driver works on every OS and every event loop, and the thread hop is
negligible next to an LLM call.

The connection string is read from ``DATABASE_URL`` (or ``SUPABASE_DB_URL``).
Use Supabase's *Session pooler* string (or the direct one); both
``postgresql://`` and ``postgresql+psycopg://`` prefixes are accepted.
"""

from __future__ import annotations

import asyncio
import os
import threading
from typing import Any

from langgraph.checkpoint.postgres import PostgresSaver
from loguru import logger
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

_URL_ENV_VARS = ("DATABASE_URL", "SUPABASE_DB_URL")

_lock = threading.RLock()
_pool: ConnectionPool | None = None
_checkpointer: "ThreadedPostgresSaver | None" = None


def _raw_url() -> str:
    for name in _URL_ENV_VARS:
        value = os.getenv(name)
        if value:
            return value
    raise RuntimeError(
        "Set DATABASE_URL in .env to your Supabase Postgres connection string "
        f"(also accepted: {', '.join(_URL_ENV_VARS[1:])})."
    )


def psycopg_url() -> str:
    """Plain ``postgresql://`` URL, as psycopg expects."""
    return "postgresql://" + _raw_url().split("://", 1)[1]


def asyncpg_url() -> str:
    """``postgresql+asyncpg://`` URL (no query string) for langchain-postgres' PGEngine."""
    rest = _raw_url().split("://", 1)[1]
    return "postgresql+asyncpg://" + rest.split("?", 1)[0]


def get_pool() -> ConnectionPool:
    """The process-wide connection pool (created on first use)."""
    global _pool
    with _lock:
        if _pool is None:
            pool = ConnectionPool(
                conninfo=psycopg_url(),
                min_size=1,
                max_size=8,
                # autocommit + dict_row are required by the LangGraph checkpointer.
                # prepare_threshold=0 keeps Supabase's pgbouncer poolers happy.
                kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
                check=ConnectionPool.check_connection,  # drop connections Supabase closed
                open=False,
            )
            pool.open(wait=True, timeout=30)
            _pool = pool
        return _pool


def query(sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Run one statement; return its rows (empty list if it returns none)."""
    with get_pool().connection() as conn:
        cur = conn.execute(sql, params)
        return cur.fetchall() if cur.description else []


class ThreadedPostgresSaver(PostgresSaver):
    """LangGraph's sync ``PostgresSaver`` with async methods that run in a thread.

    LangGraph's async engine (``astream_events`` / ``ainvoke``) calls the ``a*``
    methods. This subclass forwards each of them to the sync implementation in a
    worker thread, so the checkpointer never touches the event loop.
    """

    async def aget_tuple(self, config):
        return await asyncio.to_thread(self.get_tuple, config)

    async def alist(self, config, *, filter=None, before=None, limit=None):
        items = await asyncio.to_thread(
            lambda: list(self.list(config, filter=filter, before=before, limit=limit))
        )
        for item in items:
            yield item

    async def aput(self, config, checkpoint, metadata, new_versions):
        return await asyncio.to_thread(self.put, config, checkpoint, metadata, new_versions)

    async def aput_writes(self, config, writes, task_id, task_path=""):
        await asyncio.to_thread(self.put_writes, config, writes, task_id, task_path)

    async def adelete_thread(self, thread_id):
        await asyncio.to_thread(self.delete_thread, thread_id)


def get_checkpointer() -> ThreadedPostgresSaver:
    """The shared checkpointer. Blocking on first call, so await it via ``asyncio.to_thread``."""
    global _checkpointer
    with _lock:
        if _checkpointer is None:
            saver = ThreadedPostgresSaver(get_pool())
            try:
                saver.setup()  # idempotent; your tables already exist
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Checkpointer setup() skipped: {exc}")
            _checkpointer = saver
        return _checkpointer


def close_pool() -> None:
    global _pool, _checkpointer
    with _lock:
        if _pool is not None:
            _pool.close()
        _pool = None
        _checkpointer = None