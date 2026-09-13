import unittest
from unittest.mock import AsyncMock
from services.agent.knowledge_intent import requests_knowledge
from services.agent.workflow import DigitalXinyuWorkflow
from services.agent.providers import FakeCompanionProvider
from services.agent.knowledge import BM25KnowledgeRetriever
from pathlib import Path

class KnowledgeIntentTests(unittest.IsolatedAsyncioTestCase):
    def test_positive_and_negative_matrix(self):
        for text in ['越想睡越睡不着','不会拒绝别人','我不敢拒绝','什么是CBT','怎么安慰朋友','失去亲人后怎么办','照顾家人很累','任务太多不知从哪开始','一直担心明天的面试']:
            with self.subTest(text=text):self.assertTrue(requests_knowledge(text))
        for text in ['你好','今天吃什么','我在看一本小说','推荐一首失眠的歌','不需要建议，我只想倾诉','我压力很大，不要科普']:
            with self.subTest(text=text):self.assertFalse(requests_knowledge(text))

    async def test_retrieves_sourced_sleep_knowledge_without_classifier(self):
        provider=FakeCompanionProvider();provider.select_tool=AsyncMock()
        retriever=BM25KnowledgeRetriever(Path(__file__).resolve().parents[1]/'data/knowledge/psychology.json')
        result=await DigitalXinyuWorkflow(provider=provider,knowledge_retriever=retriever).run(user_text='越想睡越睡不着',trace_id='k',session_id='k')
        self.assertIn('knowledge_retriever',result['execution_path'])
        self.assertIn('nhs-sleep-not-force',[k.document_id for k in result['retrieved_knowledge']])
        provider.select_tool.assert_not_called()

    async def test_explicit_listening_skips_knowledge_and_classifier(self):
        provider=FakeCompanionProvider();provider.select_tool=AsyncMock()
        retriever=AsyncMock()
        result=await DigitalXinyuWorkflow(provider=provider,knowledge_retriever=retriever).run(user_text='我压力很大，请问能不能只听我说，不要科普',trace_id='k',session_id='k')
        self.assertNotIn('knowledge_retriever',result['execution_path'])
        retriever.retrieve.assert_not_called();provider.select_tool.assert_not_called()

    async def test_listening_preference_reaches_reply_provider(self):
        provider=FakeCompanionProvider();provider.generate=AsyncMock(return_value='我在听。')
        await DigitalXinyuWorkflow(provider=provider).run(user_text='只想倾诉，不需要建议',trace_id='k',session_id='k')
        messages=provider.generate.call_args.args[0]
        self.assertTrue(any(m.role=='system' and '不要主动给知识讲解' in m.content for m in messages))
