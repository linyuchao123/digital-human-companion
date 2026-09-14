import asyncio
import json
import unittest

import httpx
from services.agent.providers import (
    CompanionProviderError, FakeCompanionProvider, FallbackCompanionProvider,
    OpenAICompatibleCompanionProvider, OpenAICompatibleConfig,
)
from services.agent.state import ChatMessage
from services.agent.workflow import DigitalXinyuWorkflow


def event(content=None, reasoning=None):
    return 'data: '+json.dumps({'choices': [{'delta': {
        'content': content, 'reasoning_content': reasoning}}]})+'\n\n'


class StreamingTests(unittest.IsolatedAsyncioTestCase):
    async def test_total_timeout(self):
        async def slow(request):
            await asyncio.sleep(0.1)
            return httpx.Response(200,text=event('晚到')+'data: [DONE]\n\n')
        async with httpx.AsyncClient(base_url='https://test.invalid',transport=httpx.MockTransport(slow)) as client:
            provider=OpenAICompatibleCompanionProvider(OpenAICompatibleConfig(api_key='test',timeout_seconds=0.01),client)
            with self.assertRaises(CompanionProviderError):
                _=[d async for d in provider.generate_stream([])]

    async def test_real_sse_content_only(self):
        def handle(request):
            self.assertTrue(json.loads(request.content)['stream'])
            return httpx.Response(200, text=event(reasoning='内部推理')+event('你好')+
                event('，我在。')+'data: [DONE]\n\n')
        async with httpx.AsyncClient(base_url='https://test.invalid', transport=httpx.MockTransport(handle)) as client:
            provider=OpenAICompatibleCompanionProvider(OpenAICompatibleConfig(api_key='test'),client)
            self.assertEqual([d async for d in provider.generate_stream([])],['你好','，我在。'])

    async def test_failure_before_first_token_uses_fallback(self):
        async with httpx.AsyncClient(base_url='https://test.invalid',
            transport=httpx.MockTransport(lambda r:httpx.Response(503))) as client:
            provider=FallbackCompanionProvider(OpenAICompatibleCompanionProvider(
                OpenAICompatibleConfig(api_key='test'),client),FakeCompanionProvider())
            self.assertEqual([d async for d in provider.generate_stream([])],['我在这里，你可以慢慢说。'])

    async def test_partial_failure_never_splices_fallback(self):
        async with httpx.AsyncClient(base_url='https://test.invalid',
            transport=httpx.MockTransport(lambda r:httpx.Response(200,text=event('半句话')))) as client:
            provider=FallbackCompanionProvider(OpenAICompatibleCompanionProvider(
                OpenAICompatibleConfig(api_key='test'),client),FakeCompanionProvider())
            received=[]
            with self.assertRaises(CompanionProviderError):
                async for delta in provider.generate_stream([]):received.append(delta)
            self.assertEqual(received,['半句话'])

    async def test_empty_or_malformed_stream_rejected(self):
        for body in ['data: [DONE]\n\n','data: nope\n\n']:
            async with httpx.AsyncClient(base_url='https://test.invalid',
                transport=httpx.MockTransport(lambda r:httpx.Response(200,text=body))) as client:
                provider=OpenAICompatibleCompanionProvider(OpenAICompatibleConfig(api_key='test'),client)
                with self.assertRaises(CompanionProviderError):
                    _=[d async for d in provider.generate_stream([])]

    async def test_workflow_stream_isolation_and_final_agreement(self):
        class Provider:
            async def generate_stream(self,messages):
                text=messages[-1].content
                yield text
                await asyncio.sleep(0)
                yield '回复'
        workflow=DigitalXinyuWorkflow(provider=Provider())
        async def run(text):
            parts=[]
            async def sink(delta):parts.append(delta)
            result=await workflow.run(user_text=text,trace_id=text,session_id=text,text_delta_sink=sink)
            self.assertEqual(''.join(parts),result['final_response'])
            self.assertNotIn('text_delta_sink',result)
            return parts
        self.assertEqual(await asyncio.gather(run('你好'),run('再见')),[['你好','回复'],['再见','回复']])
