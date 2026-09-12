import asyncio
import base64
import json
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from apps.api import integrated_server as server


class CaptureWebSocket:
    def __init__(self):
        self.messages = []

    async def send_text(self, raw):
        self.messages.append(json.loads(raw))


class VoiceAudioFlowTests(unittest.IsolatedAsyncioTestCase):
    async def call_audio(self, message, state=None):
        state = state or server.SessionState("voice-test")
        ws = CaptureWebSocket()
        await server._handle_audio(message, state, ws, asyncio.get_running_loop())
        self.assertFalse(state.asr_running)
        return ws.messages

    async def test_audio_result_enters_agent_and_removes_temporary_file(self):
        paths = []

        def recognize(path):
            paths.append(path)
            self.assertEqual(Path(path).read_bytes(), b"RIFF-test")
            return "你好小安"

        with (
            patch.object(server, "ASR_DEPENDENCY_AVAILABLE", True),
            patch.object(server, "_run_asr", side_effect=recognize),
            patch.object(server, "_trigger_llm", new_callable=AsyncMock) as agent,
        ):
            messages = await self.call_audio({"format": "wav", "data": base64.b64encode(b"RIFF-test").decode()})

        self.assertEqual([m["type"] for m in messages], ["asr_processing", "asr_result"])
        self.assertEqual(messages[-1]["text"], "你好小安")
        agent.assert_awaited_once()
        self.assertFalse(Path(paths[0]).exists())

    async def test_rejects_invalid_base64_and_format(self):
        for message in ({"data": "!bad", "format": "wav"}, {"data": "YWJj", "format": "../bad"}):
            messages = await self.call_audio(message)
            self.assertEqual(messages[-1]["code"], "invalid_audio")

    async def test_missing_asr_dependency_is_explicit(self):
        with patch.object(server, "ASR_DEPENDENCY_AVAILABLE", False):
            messages = await self.call_audio({"data": "YWJj", "format": "wav"})
        self.assertEqual(messages[-1]["code"], "unavailable")

    async def test_busy_session_does_not_start_another_recognition(self):
        state = server.SessionState("busy")
        state.llm_running = True
        messages = await self.call_audio({"data": "YWJj", "format": "wav"}, state)
        self.assertEqual(messages[-1]["code"], "busy")


if __name__ == "__main__":
    unittest.main()
