import asyncio
import unittest
from unittest.mock import AsyncMock

from services.agent.providers import FakeCompanionProvider
from services.agent.workflow import DigitalXinyuWorkflow


class KnowledgeResilienceTests(unittest.IsolatedAsyncioTestCase):
    async def test_failure_continues_chat_with_sanitized_event(self):
        provider=FakeCompanionProvider()
        provider.generate=AsyncMock(return_value='我在听，你可以慢慢说。')
        retriever=AsyncMock()
        retriever.retrieve.side_effect=OSError('secret-key-private-path')
        events=[]
        async def sink(event):events.append(event)
        result=await DigitalXinyuWorkflow(provider=provider,knowledge_retriever=retriever).run(
            user_text='我睡不着',trace_id='r',session_id='r',event_sink=sink)
        self.assertEqual(result['knowledge_status'],'failed')
        self.assertEqual(result['retrieved_knowledge'],[])
        self.assertEqual(result['errors'],['knowledge_retriever:knowledge_error'])
        self.assertIn('avatar_director',result['execution_path'])
        provider.generate.assert_awaited_once()
        data=next(e.data for e in events if e.node=='knowledge_retriever')
        self.assertEqual(data['tool_status'],'failed')
        self.assertEqual(data['error_code'],'knowledge_error')
        self.assertEqual(data['result_count'],0)
        self.assertNotIn('secret-key-private-path',str(events)+str(result))
        self.assertNotIn('我睡不着',str(data))

    async def test_timeout_cancels_retrieval_and_continues(self):
        cancelled=asyncio.Event()
        async def slow(*args,**kwargs):
            try:await asyncio.Event().wait()
            finally:cancelled.set()
        retriever=AsyncMock();retriever.retrieve.side_effect=slow
        result=await DigitalXinyuWorkflow(knowledge_retriever=retriever,knowledge_timeout_seconds=0.01).run(
            user_text='我睡不着',trace_id='r',session_id='r')
        self.assertTrue(cancelled.is_set())
        self.assertEqual(result['tool_calls'][0].error_code,'knowledge_timeout')
        self.assertTrue(result['final_response'])

    async def test_empty_is_not_reported_as_failure(self):
        retriever=AsyncMock();retriever.retrieve.return_value=[]
        result=await DigitalXinyuWorkflow(knowledge_retriever=retriever).run(user_text='我睡不着',trace_id='r',session_id='r')
        self.assertEqual(result['knowledge_status'],'empty')
        self.assertEqual(result['tool_calls'][0].status,'completed')
        self.assertEqual(result['errors'],[])

    async def test_cancellation_is_not_swallowed_as_chat(self):
        started=asyncio.Event()
        async def waiting(*args,**kwargs):
            started.set()
            await asyncio.Event().wait()
        retriever=AsyncMock();retriever.retrieve.side_effect=waiting
        provider=FakeCompanionProvider();provider.generate=AsyncMock()
        task=asyncio.create_task(DigitalXinyuWorkflow(provider=provider,knowledge_retriever=retriever).run(
            user_text='我睡不着',trace_id='r',session_id='r'))
        await asyncio.wait_for(started.wait(),timeout=1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        provider.generate.assert_not_called()

    def test_invalid_timeout_is_rejected(self):
        for value in [0,-1,float('inf'),float('nan')]:
            with self.subTest(value=value),self.assertRaises(ValueError):DigitalXinyuWorkflow(knowledge_timeout_seconds=value)
