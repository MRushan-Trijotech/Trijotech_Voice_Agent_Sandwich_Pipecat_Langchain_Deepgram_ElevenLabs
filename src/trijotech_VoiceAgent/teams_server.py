"""WebSocket endpoint for the Teams calling bridge.

The Windows .NET media bot connects here once per Teams call, streams the
caller's audio in, and plays back whatever audio we send. Each connection runs
the same Trija pipeline as the browser version (voice.run_bot), just over a
FastAPI WebSocket transport instead of WebRTC.

Run (from the repo root):

    uv run uvicorn gym_support.teams_server:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /health      liveness check
    WS   /teams/ws    the media bot connects here with  Authorization: Bearer <TEAMS_BRIDGE_TOKEN>
"""

from __future__ import annotations

import hmac
import os

from dotenv import load_dotenv

load_dotenv(override=True)

from fastapi import FastAPI, WebSocket  # noqa: E402
from loguru import logger  # noqa: E402
from pipecat.runner.types import WebSocketRunnerArguments  # noqa: E402
from pipecat.transports.websocket.fastapi import (  # noqa: E402
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

from .teams_serializer import TEAMS_SAMPLE_RATE, TeamsMediaSerializer  # noqa: E402
from .voice import run_bot  # noqa: E402

app = FastAPI(title="Trija - Teams bridge")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


def _authorized(websocket: WebSocket) -> bool:
    """Shared-secret check so only your media bot can open a call session."""
    expected = os.getenv("TEAMS_BRIDGE_TOKEN", "")
    header = websocket.headers.get("authorization", "")
    supplied = header[7:].strip() if header.lower().startswith("bearer ") else ""
    return bool(expected) and hmac.compare_digest(supplied, expected)


@app.websocket("/teams/ws")
async def teams_ws(websocket: WebSocket) -> None:
    if not _authorized(websocket):
        logger.warning("Rejected /teams/ws connection: bad or missing bearer token")
        await websocket.close(code=1008)
        return

    await websocket.accept()
    logger.info("Teams media bot connected")

    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=TEAMS_SAMPLE_RATE,
            audio_out_sample_rate=TEAMS_SAMPLE_RATE,
            add_wav_header=False,
            audio_out_10ms_chunks=2,       # 20 ms chunks, the Teams frame size
            fixed_audio_packet_size=640,   # 20 ms @ 16 kHz PCM16 mono
            serializer=TeamsMediaSerializer(),
        ),
    )
    await run_bot(
        transport,
        WebSocketRunnerArguments(websocket=websocket),
        audio_out_sample_rate=TEAMS_SAMPLE_RATE,
        # Microsoft forbids persisting Teams call media unless the bot first calls
        # the updateRecordingStatus API, so the WAV recording stays off here.
        # record_audio=False,
    )