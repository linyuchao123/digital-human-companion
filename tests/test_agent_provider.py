import json
import unittest

import httpx

from services.agent import (
    ChatMessage,
    CompanionProviderError,
    FakeCompanionProvider,
    FallbackCompanionProvider,
    OpenAICompatibleCompanionProvider,
    OpenAICompatibleConfig,
    create_companion_provider,
)


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


class OpenAICompatibleCompanionProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_provider_sends_openai_compatible_request(self):
        captured = {}

        def handle_request(request: httpx.Request) -> httpx.Response:
            captured["authorization"] = request.headers["authorization"]
            captured["payload"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": " 我愿意听你慢慢说。 "}}]},
            )

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handle_request),
            base_url="https://example.test/v1",
        ) as client:
            provider = OpenAICompatibleCompanionProvider(
                OpenAICompatibleConfig(api_key="test-key", model="qwen-test"),
                client=client,
            )
            response = await provider.generate([ChatMessage(role="user", content="我有点孤独")])

        self.assertEqual(response, "我愿意听你慢慢说。")
        self.assertEqual(captured["authorization"], "Bearer test-key")
        self.assertEqual(captured["payload"]["model"], "qwen-test")
        self.assertEqual(captured["payload"]["messages"][-1]["content"], "我有点孤独")

    async def test_provider_wraps_invalid_cloud_response(self):
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json={})),
            base_url="https://example.test/v1",
        ) as client:
            provider = OpenAICompatibleCompanionProvider(
                OpenAICompatibleConfig(api_key="test-key"),
                client=client,
            )
            with self.assertRaises(CompanionProviderError):
                await provider.generate([ChatMessage(role="user", content="你好")])

    async def test_fallback_provider_keeps_chat_available(self):
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(503)),
            base_url="https://example.test/v1",
        ) as client:
            cloud = OpenAICompatibleCompanionProvider(
                OpenAICompatibleConfig(api_key="test-key"),
                client=client,
            )
            provider = FallbackCompanionProvider(cloud, FakeCompanionProvider())
            response = await provider.generate([ChatMessage(role="user", content="今天很累")])

        self.assertIn("压力", response)

    def test_factory_uses_offline_provider_without_api_key(self):
        provider, name = create_companion_provider(api_key="")

        self.assertIsInstance(provider, FakeCompanionProvider)
        self.assertEqual(name, "offline")

    def test_factory_enables_cloud_provider_when_api_key_exists(self):
        provider, name = create_companion_provider(api_key="test-key")

        self.assertIsInstance(provider, FallbackCompanionProvider)
        self.assertEqual(name, "cloud_with_fallback")


if __name__ == "__main__":
    unittest.main()
