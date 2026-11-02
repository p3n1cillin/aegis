"""The CS2 ingest port only accepts telemetry from holders of AEGIS_INGEST_TOKEN."""

import asyncio

from aegis.adapters.cs2.capture import encode_frame
from aegis.service.ingest import IngestResult, _handle_client

FRAMES = [
    {"k": "hdr", "session_id": "s1", "map": "de_anubis", "seq": 1},
    {"k": "players", "players": [], "seq": 2},
]


class RecordingHub:
    def __init__(self):
        self.calls = []

    def ingest_frames(self, frames, *, gaps=None):
        self.calls.append(frames)
        return IngestResult(session_id="s1", capture_path=None, gaps=gaps or [], n_frames=len(frames))


def send(frames, *, token):
    """Send frames over a real socket; return (frames the hub received, reply bytes)."""
    hub = RecordingHub()

    async def run():
        server = await asyncio.start_server(
            lambda r, w: _handle_client(r, w, hub, token), "127.0.0.1", 0
        )
        port = server.sockets[0].getsockname()[1]
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(b"".join(encode_frame(f) for f in frames))
        await writer.drain()
        writer.write_eof()
        reply = await asyncio.wait_for(reader.read(), timeout=5)
        writer.close()
        server.close()
        await server.wait_closed()
        return reply

    reply = asyncio.run(run())
    return hub.calls, reply


def test_no_token_configured_accepts_and_ignores_auth_frames():
    calls, reply = send([{"k": "auth", "token": "anything"}, *FRAMES], token="")
    assert calls == [FRAMES]
    assert reply  # ack sent


def test_matching_token_is_accepted():
    calls, reply = send([{"k": "auth", "token": "s3cret"}, *FRAMES], token="s3cret")
    assert calls == [FRAMES]
    assert reply


def test_missing_auth_frame_is_dropped_unscored():
    calls, reply = send(FRAMES, token="s3cret")
    assert calls == []
    assert reply == b""


def test_wrong_token_is_dropped_unscored():
    calls, reply = send([{"k": "auth", "token": "guess"}, *FRAMES], token="s3cret")
    assert calls == []
    assert reply == b""


def test_auth_must_come_first():
    calls, _ = send([FRAMES[0], {"k": "auth", "token": "s3cret"}, FRAMES[1]], token="s3cret")
    assert calls == []
