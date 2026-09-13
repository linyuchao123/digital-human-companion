import unittest
from unittest.mock import AsyncMock
from services.agent.knowledge_relevance import clearly_unrelated, DomainFilteredKnowledgeRetriever
from services.agent.knowledge import create_knowledge_retriever
from pathlib import Path
from services.agent.rag_evaluation import load_retrieval_eval_cases

class FilterTests(unittest.IsolatedAsyncioTestCase):
    def test_weather_and_music_do_not_hide_loneliness_or_sleep(self):
        for query in ['下雨天觉得孤单，没人能聊聊天','听歌还是睡不好，越想睡越清醒','推荐一首音乐，我失眠了','查天气，我没人陪']:
            with self.subTest(query=query):self.assertFalse(clearly_unrelated(query))

    def test_mixed_requests_keep_emotional_support(self):
        for query in ['下雨让我很难过','有什么音乐可以缓解压力','食欲不好吃不下东西','担心航班取消，一直睡不着','心跳很快手心出汗','周围很热闹但我像个局外人']:
            with self.subTest(query=query):self.assertFalse(clearly_unrelated(query))
        for query in ['明天上海会下雨吗','今天中午吃什么好','推荐几首周杰伦的歌','电脑怎么连接打印机','现在几点','查一下美元汇率','航班什么时候到','帮我算一下12+35']:
            with self.subTest(query=query):self.assertTrue(clearly_unrelated(query))

    async def test_unrelated_never_calls_primary_or_fallback(self):
        inner=AsyncMock()
        filtered=DomainFilteredKnowledgeRetriever(inner)
        self.assertEqual(await filtered.retrieve('电脑怎么连接打印机'),[])
        inner.retrieve.assert_not_called()
        await filtered.retrieve('我睡不着')
        inner.retrieve.assert_awaited_once_with('我睡不着',3)

    async def test_factory_filters_bm25_and_offline_fallback(self):
        root=Path(__file__).resolve().parents[1]
        for path in [root/'data/knowledge/psychology.json',Path('/missing/corpus.json')]:
            retriever,_=create_knowledge_retriever(path)
            self.assertEqual(await retriever.retrieve('明天上海会下雨吗'),[])
            self.assertTrue(await retriever.retrieve('焦虑压力很大'))

    async def test_existing_semantic_challenge_keeps_original_candidates(self):
        root=Path(__file__).resolve().parents[1]
        retriever,_=create_knowledge_retriever(root/'data/knowledge/psychology.json')
        for case in load_retrieval_eval_cases(root/'eval/rag/semantic_challenge_cases.json'):
            with self.subTest(query=case.query):
                before=await retriever._retriever.retrieve(case.query,3)
                after=await retriever.retrieve(case.query,3)
                self.assertEqual([s.document_id for s in before],[s.document_id for s in after])
