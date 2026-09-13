"""Exercise the graph, not just retrieval: no real model or network calls."""
import json
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from services.agent.providers import FakeCompanionProvider
from services.agent.workflow import DigitalXinyuWorkflow


class KnowledgeRouteCoverageTests(unittest.IsolatedAsyncioTestCase):
    async def test_mixed_positive_cases_reach_knowledge_without_classifier(self):
        path=Path(__file__).resolve().parents[1]/'eval/rag/mixed_query_cases.json'
        cases=json.loads(path.read_text(encoding='utf-8'))
        for case in cases:
            if case.get('expect_no_results'):continue
            with self.subTest(query=case['query']):
                provider=FakeCompanionProvider()
                provider.select_tool=AsyncMock(return_value='{"tool":"web_search"}')
                retriever=AsyncMock();retriever.retrieve.return_value=[]
                with patch('services.agent.workflow.tavily_search',new=AsyncMock()) as search,patch('services.agent.workflow.get_weather',new=AsyncMock()) as weather:
                    result=await DigitalXinyuWorkflow(provider=provider,knowledge_retriever=retriever).run(
                        user_text=case['query'],trace_id='coverage',session_id='coverage')
                self.assertIn('knowledge_retriever',result['execution_path'])
                retriever.retrieve.assert_awaited_once_with(case['query'],top_k=3)
                provider.select_tool.assert_not_called();search.assert_not_called();weather.assert_not_called()

    async def test_precedence_and_entertainment_boundaries(self):
        cases=[('推荐一首失眠的歌','chat'),('请问上海明天会下雨吗，我最近睡不好','web_search'),
               ('一到晚上脑子像开会，只想倾诉，不需要建议','chat'),
               ('妈妈去世了，帮我忘掉这些记忆','memory_forget'),
               ('我不想活了，不要科普','safe_response')]
        for text,expected in cases:
            with self.subTest(query=text):
                retriever=AsyncMock();retriever.retrieve.return_value=[]
                with patch('services.agent.workflow.get_weather',new=AsyncMock(return_value=('测试天气',[]))):
                    result=await DigitalXinyuWorkflow(knowledge_retriever=retriever).run(user_text=text,trace_id='b',session_id='b')
                if expected=='safe_response':self.assertIn(expected,result['execution_path'])
                else:self.assertEqual(result['intent'],expected)
                retriever.retrieve.assert_not_called()
