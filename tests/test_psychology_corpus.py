import json
import unittest
from pathlib import Path
from urllib.parse import urlparse
from services.agent.knowledge import BM25KnowledgeRetriever

CORPUS=Path(__file__).resolve().parents[1]/'data/knowledge/psychology.json'
CASES=[('不完美就是失败，非黑即白','nhs-cbt-black-white'),
       ('担忧时间，一直担心，反复担忧','nhs-worry-time'),
       ('越想睡越睡不着，强迫入睡','nhs-sleep-not-force'),
       ('亲人去世，丧亲哀伤','nih-grief'),
       ('不会说不，讨好，人际边界','nih-relationship-boundaries'),
       ('照顾老人，照护压力','nih-caregiver-stress'),
       ('任务太多，优先级，学习压力','nimh-priorities'),
       ('问题解决，拆解问题，行动计划','nhs-problem-plan')]

class CorpusTests(unittest.IsolatedAsyncioTestCase):
    def test_corpus_quality(self):
        docs=json.loads(CORPUS.read_text(encoding='utf-8'))
        self.assertGreaterEqual(len(docs),30)
        self.assertEqual(len({d['id'] for d in docs}),len(docs))
        for doc in docs:
            with self.subTest(id=doc['id']):
                self.assertTrue(0<len(doc['content'])<=1200)
                self.assertTrue(doc['source'])
                self.assertTrue(doc['keywords'])
                self.assertEqual(urlparse(doc['source_url']).scheme,'https')
                self.assertIn(urlparse(doc['source_url']).hostname,{'www.who.int','www.nih.gov','www.nimh.nih.gov','www.nhs.uk'})

    async def test_topic_retrieval_top_three(self):
        retriever=BM25KnowledgeRetriever(CORPUS)
        for query,expected in CASES:
            with self.subTest(query=query):
                results=await retriever.retrieve(query,top_k=3)
                self.assertIn(expected,[r.document_id for r in results])
                self.assertTrue(all(r.source_url for r in results))
