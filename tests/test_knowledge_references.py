import unittest
from services.agent.state import KnowledgeSnippet
from services.agent.references import knowledge_references, reference_snapshot, normalize_knowledge_citations

class ReferenceTests(unittest.TestCase):
    def test_snapshot_and_invalid_citation_numbers(self):
        self.assertEqual(reference_snapshot({'source':'bad'}),[])
        snapshot=reference_snapshot([{'id':'x'*200,'source':'来源','excerpt':'知'*900,'url':'https://127.0.0.1','private':'secret'}])
        self.assertEqual(len(snapshot[0]['id']),120)
        self.assertEqual(len(snapshot[0]['excerpt']),400)
        self.assertIsNone(snapshot[0]['url'])
        self.assertNotIn('private',snapshot[0])
        snippets=[KnowledgeSnippet(content='知识',source='来源')]
        self.assertEqual(normalize_knowledge_citations('练习[1]。其他[2][99]。',snippets),'练习[1]。其他。')
        self.assertEqual(normalize_knowledge_citations('无依据[1]',[]),'无依据')
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
