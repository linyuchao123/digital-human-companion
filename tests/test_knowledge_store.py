import json
import tempfile
import unittest
from pathlib import Path

from services.agent import KnowledgeCorpusStore


class KnowledgeCorpusStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = Path(self.temp_dir.name) / "knowledge.json"
        self.path.write_text(json.dumps([
            {
                "id": "curated-one",
                "keywords": ["焦虑"],
                "content": "测试知识",
                "source": "测试来源",
            }
        ]), encoding="utf-8")
        self.store = KnowledgeCorpusStore(self.path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_add_and_delete_custom_document(self):
        document_id = self.store.add(
            content="新增知识",
            source="管理员录入",
            source_url="https://example.com/source",
            keywords=["新增"],
        )

        self.assertTrue(document_id.startswith("custom-"))
        self.assertTrue(self.store.delete_custom(document_id))
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual([item["id"] for item in payload], ["curated-one"])

    def test_builtin_document_cannot_be_deleted(self):
        with self.assertRaisesRegex(PermissionError, "不能"):
            self.store.delete_custom("curated-one")

    def test_non_https_source_url_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "https"):
            self.store.add(
                content="新增知识",
                source="管理员录入",
                source_url="http://example.com/source",
            )


if __name__ == "__main__":
    unittest.main()
