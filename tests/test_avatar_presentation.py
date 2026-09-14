import unittest
import base64
import json
from unittest.mock import patch
from pathlib import Path
from services.agent.workflow import DigitalXinyuWorkflow
from services.tts.providers import Qwen3TtsProvider


class AvatarPresentationTests(unittest.IsolatedAsyncioTestCase):
    async def test_stream_returns_pcm_before_completion(self):
        class Response:
            async def __aenter__(self): return self
            async def __aexit__(self, *args): pass
            def raise_for_status(self): pass
            async def aiter_lines(self):
                yield 'data: '+json.dumps({'output':{'audio':{'data':base64.b64encode(b'\x01\x00').decode()}}})
                yield 'data: [DONE]'
        class Client(Response):
            def stream(self, *args, **kwargs):
                self.body=kwargs['json']
                return Response()
        client=Client()
        with patch('services.tts.providers.httpx.AsyncClient', return_value=client):
            stream=Qwen3TtsProvider(api_key='test').stream_pcm('你好',voice='Cherry')
            self.assertEqual(await anext(stream),b'\x01\x00')
            with self.assertRaises(StopAsyncIteration): await anext(stream)
        self.assertFalse(client.body['input']['optimize_instructions'])

    async def test_content_selects_appropriate_gestures(self):
        for text,expected in (("你好", "Hello"),("点头看看", "Nod"),("摇头看看", "ShakeHead"),("我成功了，庆祝一下", "Celebrate")):
            result=await DigitalXinyuWorkflow().run(user_text=text,trace_id='avatar',session_id='avatar')
            self.assertEqual(result['avatar_command'].motion,expected)

    def test_frontend_restores_native_motion_and_keeps_stream_cancellation(self):
        html=(Path(__file__).resolve().parents[1]/'integrated.html').read_text()
        self.assertIn('if(!isPlayingMotion){',html)
        self.assertIn('tickGesture(c)',html)
        self.assertIn("fetch('/api/tts/stream'",html)
        self.assertIn('_streamSources.clear()',html)
        self.assertNotIn("document.body.className=''",html)
