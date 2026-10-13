"""Pretend to be the Teams media bot, so you can test /teams/ws without Teams.

    uv run python -m gym_support.fake_teams_client question.wav
    uv run python -m gym_support.fake_teams_client question.wav ws://127.0.0.1:8000/teams/ws

`question.wav` must be 16 kHz, mono, 16-bit PCM (record one with Audacity, or
convert: ffmpeg -i in.mp3 -ar 16000 -ac 1 -sample_fmt s16 question.wav).

It streams the file in 20 ms frames at real-time speed, then keeps sending
silence so the bot's end-of-turn detection fires, collects the agent's audio
and writes it to reply.wav. Listen to reply.wav: if you hear Trija answering,
the whole Python side works.
"""

from __future__ import annotations

import asyncio
import os
import sys
import wave

import websockets
from dotenv import load_dotenv

load_dotenv(override=True)

FRAME_BYTES = 640  # 20 ms of 16 kHz PCM16 mono
LISTEN_SECONDS = 25  # keep the call open this long after the question


async def main(wav_path: str, url: str) -> None:
    with wave.open(wav_path, "rb") as wf:
        assert wf.getframerate() == 16000 and wf.getnchannels() == 1 and wf.getsampwidth() == 2, (
            "wav must be 16 kHz, mono, 16-bit PCM"
        )
        pcm = wf.readframes(wf.getnframes())

    headers = {"Authorization": f"Bearer {os.environ['TEAMS_BRIDGE_TOKEN']}"}
    reply = bytearray()
    clears = 0

    async with websockets.connect(url, additional_headers=headers) as ws:
        print("connected - sending your audio, then listening ...")

        async def receiver() -> None:
            nonlocal clears
            async for msg in ws:
                if isinstance(msg, bytes):
                    reply.extend(msg)
                else:
                    clears += 1  # {"action": "clear"}: Trija was interrupted

        recv_task = asyncio.create_task(receiver())

        silence = b"\x00" * FRAME_BYTES
        total_frames = LISTEN_SECONDS * 50
        for i in range(total_frames):
            chunk = pcm[i * FRAME_BYTES : (i + 1) * FRAME_BYTES]
            if len(chunk) < FRAME_BYTES:
                chunk = chunk.ljust(FRAME_BYTES, b"\x00") if chunk else silence
            await ws.send(chunk)
            await asyncio.sleep(0.02)

        recv_task.cancel()

    with wave.open("reply.wav", "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(16000)
        out.writeframes(bytes(reply))
    print(f"done: {len(reply) / 32000:.1f}s of agent audio -> reply.wav ({clears} clear commands)")


if __name__ == "__main__":
    wav = sys.argv[1] if len(sys.argv) > 1 else "question.wav"
    target = sys.argv[2] if len(sys.argv) > 2 else "ws://127.0.0.1:8000/teams/ws"
    asyncio.run(main(wav, target))