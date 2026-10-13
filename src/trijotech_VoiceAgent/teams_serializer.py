"""Wire format between the Teams media bot (.NET) and Pipecat.

The Teams real-time media platform hands the bot 20 ms frames of raw PCM16,
16 kHz, mono (640 bytes) and expects the same back. So the protocol is simply:

  .NET bot  ->  Python   binary message = raw PCM16 caller audio
  Python    ->  .NET bot binary message = raw PCM16 agent audio (640-byte frames)
  Python    ->  .NET bot text message   = {"action": "clear"}  (caller interrupted:
                                          drop any agent audio still queued)

Modelled on Pipecat's own VonageFrameSerializer, which uses the same idea.
Both pipeline sample rates must be 16000 (teams_server.py sets them), so no
resampling happens here.
"""

from __future__ import annotations

import json

from loguru import logger
from pipecat.frames.frames import AudioRawFrame, Frame, InputAudioRawFrame, InterruptionFrame
from pipecat.serializers.base_serializer import FrameSerializer

TEAMS_SAMPLE_RATE = 16000


class TeamsMediaSerializer(FrameSerializer):
    """Raw PCM16 / 16 kHz / mono over a WebSocket, plus a JSON "clear" command."""

    _warned_rate = False

    async def serialize(self, frame: Frame) -> str | bytes | None:
        if isinstance(frame, InterruptionFrame):
            # Barge-in: tell the media bot to flush audio it has queued for the call.
            return json.dumps({"action": "clear"})
        if isinstance(frame, AudioRawFrame):
            if frame.sample_rate != TEAMS_SAMPLE_RATE and not self._warned_rate:
                self._warned_rate = True
                logger.warning(
                    f"Outgoing audio is {frame.sample_rate} Hz but Teams needs "
                    f"{TEAMS_SAMPLE_RATE} Hz; set audio_out_sample_rate=16000."
                )
            return frame.audio
        return None  # RTVI / other frames are not sent to the media bot

    async def deserialize(self, data: str | bytes) -> Frame | None:
        if isinstance(data, (bytes, bytearray)):
            return InputAudioRawFrame(
                audio=bytes(data), sample_rate=TEAMS_SAMPLE_RATE, num_channels=1
            )
        return None  # ignore text control messages for now