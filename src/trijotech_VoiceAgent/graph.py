"""Trija: one tool-using LangChain agent (a compiled LangGraph graph).

The old gym graph (triage + three specialists, handoff routing) is gone. Trija is
a single ``create_agent`` loop: the model decides when to call
``search_company_knowledge``, ``talk_to_human``, ``check_appointment`` and
``schedule_appointment``, following the workflow in ``prompts.py``.

Pass a checkpointer (see ``db.get_checkpointer``) and a ``thread_id`` in the run
config to persist the conversation in Supabase Postgres.
"""

from __future__ import annotations

import os
from datetime import datetime

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

from . import prompts, tools

DEFAULT_MODEL = "google_genai:gemini-3.5-flash-lite"


def build_graph(checkpointer=None, *, model: str | None = None):
    """Compile and return Trija.

    The system prompt carries today's date, so build the graph per call session
    (voice.py does) or per request (server.py does), not once per process.
    """
    llm = init_chat_model(model or os.getenv("LLM_MODEL") or DEFAULT_MODEL)
    tz = tools.get_tz()
    return create_agent(
        llm,
        tools=[
            tools.search_company_knowledge,
            tools.talk_to_human,
            tools.check_appointment,
            tools.schedule_appointment,
        ],
        system_prompt=prompts.build_prompt(datetime.now(tz), tz.key),
        checkpointer=checkpointer,
        name="trija",
    )