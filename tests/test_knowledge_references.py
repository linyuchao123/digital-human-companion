import unittest
from services.agent.state import KnowledgeSnippet
from services.agent.references import knowledge_references

class ReferenceTests(unittest.TestCase):
    def test_bounds_and_no_scores_or_private_fields(self):
        items=[KnowledgeSnippet(document_id=str(i),content='知'*1000,source='来源',source_url='https://www.nhs.uk/',score=0.9) for i in range(5)]
        refs=knowledge_references(items)
        self.assertEqual(len(refs),3)
        self.assertEqual(len(refs[0]['excerpt']),400)
        self.assertEqual(set(refs[0]),{'id','source','excerpt','url'})
        self.assertEqual(knowledge_references([]),[])

    def test_unsafe_links_removed_but_excerpt_kept(self):
        for url in ['javascript:alert(1)','https://localhost','https://127.0.0.1','https://user:pass@example.org','http://example.org']:
            ref=knowledge_references([KnowledgeSnippet(content='参考知识',source='测试',source_url=url)])[0]
            self.assertIsNone(ref['url'])
            self.assertEqual(ref['excerpt'],'参考知识')
