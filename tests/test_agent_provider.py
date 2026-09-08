import unittest

from services.agent import ChatMessage, FakeCompanionProvider


class FakeCompanionProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_provider_returns_predictable_empathetic_response(self):
        provider = FakeCompanionProvider()

        response = await provider.generate(
            [ChatMessage(role="user", content="今天工作特别累")]
        )

        self.assertIn("压力", response)
        self.assertTrue(response.endswith("？"))

    async def test_provider_handles_empty_history(self):
        provider = FakeCompanionProvider()

        response = await provider.generate([])

        self.assertEqual(response, "我在这里，你可以慢慢说。")


if __name__ == "__main__":
    unittest.main()
