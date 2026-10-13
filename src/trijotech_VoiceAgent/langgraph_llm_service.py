"""A Pipecat LLM service whose "brain" is an in-process LangGraph agent (Trija).

The graph runs *inside* Pipecat's ``llm`` span (``@traced_llm`` is re-applied), so
LangChain/LangGraph spans nest under it in LangSmith.

What changed from the stateless gym version
-------------------------------------------
* The graph now has a **Postgres checkpointer**; the conversation lives in
  Supabase under ``thread_id``. So the graph is sent only the NEW user message(s)
  each turn, not the whole Pipecat history (re-sending would duplicate messages).
* The very first turn also sends the spoken greeting (it is in Pipecat's context
  but not yet in the checkpoint), so Trija knows she already introduced herself.
* ``GoogleLLMService`` expects its subclass to push the response start/end frames
  from ``_process_context``, so we do. Without the end frame the TTS never flushes
  the last sentence.

Known trade-off: if the caller interrupts (barge-in), Pipecat trims the spoken
reply in its own context, but the checkpoint keeps the full text Trija generated.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import convert_to_messages
from loguru import logger
from pipecat.frames.frames import (
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.google.llm import GoogleLLMService
from pipecat.utils.tracing.service_decorators import traced_llm

# Graph nodes whose streamed tokens are spoken. create_agent's model node is "model".
_SPOKEN_NODES = {"model"}

FALLBACK_REPLY = "Sorry, I ran into a problem on my side. Could you please say that again?"


def _spoken_text(event: dict) -> str | None:
    """Text to voice from a stream event, or None (tool-call deltas have no text)."""
    if event.get("event") != "on_chat_model_stream":
        return None
    node = (event.get("metadata") or {}).get("langgraph_node")
    if node not in _SPOKEN_NODES:
        return None
    chunk = event.get("data", {}).get("chunk")
    # `.text` flattens string and content-block lists (Gemini/Anthropic) to a str.
    text = getattr(chunk, "text", None)
    return text if isinstance(text, str) and text else None


def _is_spoken_model_start(event: dict) -> bool:
    if event.get("event") != "on_chat_model_start":
        return False
    return (event.get("metadata") or {}).get("langgraph_node") in _SPOKEN_NODES


class LangGraphLLMService(GoogleLLMService):
    """Runs a compiled, checkpointed LangGraph agent as the Pipecat LLM stage."""

    def __init__(self, *, graph: Any, thread_id: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._graph = graph
        self._thread_id = thread_id
        self._seeded = False  # has the checkpoint seen the greeting + first turn yet?

    def _messages_for_graph(self, context: LLMContext) -> list:
        raw = [
            m for m in context.get_messages() if isinstance(m, dict) and m.get("role") != "system"
        ]
        if not self._seeded:
            return convert_to_messages(raw)  # greeting + first user turn
        trailing: list[dict] = []
        for m in reversed(raw):  # only the user message(s) since the last reply
            if m.get("role") != "user":
                break
            trailing.append(m)
        trailing.reverse()
        return convert_to_messages(trailing)

    @traced_llm  # re-applied so the `llm` span wraps the graph run (and nests it)
    async def _process_context(self, context: LLMContext) -> None:
        # GoogleLLMService normally pushes these frames inside _process_context.
        await self.push_frame(LLMFullResponseStartFrame())
        try:
            messages = self._messages_for_graph(context)
            if not messages:
                return
            config = {"configurable": {"thread_id": self._thread_id}}

            await self.start_ttfb_metrics()
            first_token = True
            need_gap = False
            async for event in self._graph.astream_events(
                {"messages": messages}, config=config, version="v2"
            ):
                if _is_spoken_model_start(event):
                    # A new model call after a tool: keep sentences from running together.
                    if need_gap:
                        await self.push_frame(LLMTextFrame(" "))
                        need_gap = False
                elif text := _spoken_text(event):
                    if first_token:
                        await self.stop_ttfb_metrics()
                        first_token = False
                    await self.push_frame(LLMTextFrame(text))
                    need_gap = True
            self._seeded = True
        except Exception as exc:  # noqa: BLE001
            logger.exception("LangGraph run failed")
            await self.push_error(error_msg=f"LangGraph run failed: {exc}", exception=exc)
            await self.push_frame(LLMTextFrame(FALLBACK_REPLY))
        finally:
            await self.push_frame(LLMFullResponseEndFrame())