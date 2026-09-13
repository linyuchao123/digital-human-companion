import unittest
from unittest.mock import AsyncMock, patch
from services.agent.planning import select_tool
from services.agent.workflow import DigitalXinyuWorkflow
from services.agent.providers import FakeCompanionProvider, OpenAICompatibleCompanionProvider, OpenAICompatibleConfig
import httpx

class PlanningTests(unittest.IsolatedAsyncioTestCase):
    async def test_strict_allowlist_and_failure(self):
        for raw in ['{"tool":"shell"}', '{"tool":"weather","url":"http://localhost"}', 'not json', '[]', '{"tool":[]}']:
            provider=FakeCompanionProvider();provider.select_tool=AsyncMock(return_value=raw)
            self.assertEqual(await select_tool(provider,'请问测试问题'),('chat','invalid'))
        provider.select_tool=AsyncMock(side_effect=TimeoutError('secret'))
        self.assertEqual(await select_tool(provider,'请问测试问题'),('chat','timeout'))

    async def test_weather_model_route_and_missing_city(self):
        provider=FakeCompanionProvider();provider.select_tool=AsyncMock(return_value='{"tool":"weather"}')
        with patch('services.agent.workflow.get_weather',new=AsyncMock(return_value=('天气测试',[]))) as weather:
            result=await DigitalXinyuWorkflow(provider=provider).run(user_text='上海出门要带伞吗',trace_id='p',session_id='p')
        self.assertEqual(result['routing_source'],'model')
        self.assertEqual(result['tool_calls'][0].name,'weather_forecast')
        self.assertIn('天气',weather.call_args.args[0])
        result=await DigitalXinyuWorkflow(provider=provider).run(user_text='出门要带伞吗',trace_id='p',session_id='p')
        self.assertIn('哪个城市',result['final_response'])

    async def test_fast_chat_and_safety_never_call_selector(self):
        provider=FakeCompanionProvider();provider.select_tool=AsyncMock(return_value='{"tool":"web_search"}')
        for text in ['你好','我想自杀，请问怎么办','现在几点']:
            await DigitalXinyuWorkflow(provider=provider).run(user_text=text,trace_id='p',session_id='p')
        provider.select_tool.assert_not_called()

    async def test_knowledge_route_and_invalid_fallback(self):
        provider=FakeCompanionProvider();provider.select_tool=AsyncMock(return_value='{"tool":"knowledge"}')
        result=await DigitalXinyuWorkflow(provider=provider).run(user_text='帮我了解情绪调节',trace_id='p',session_id='p')
        self.assertIn('knowledge_retriever',result['execution_path'])
        provider.select_tool=AsyncMock(return_value='{"tool":"shell"}')
        result=await DigitalXinyuWorkflow(provider=provider).run(user_text='请问测试问题',trace_id='p',session_id='p')
        self.assertEqual(result['intent'],'chat')
        self.assertEqual(result['routing_source'],'invalid')
        self.assertNotIn('web_search',result['execution_path'])

    async def test_routing_uses_classifier_prompt_not_companion_prompt(self):
        response=httpx.Response(200,json={'choices':[{'message':{'content':'{"tool":"weather"}'}}]},request=httpx.Request('POST','https://example.org/chat/completions'))
        client=AsyncMock();client.post.return_value=response
        provider=OpenAICompatibleCompanionProvider(OpenAICompatibleConfig(api_key='test'),client)
        self.assertEqual(await select_tool(provider,'带伞吗'),('weather','model'))
        payload=client.post.call_args.kwargs['json']
        self.assertIn('工具路由分类器',payload['messages'][0]['content'])
        self.assertEqual(payload['temperature'],0)
        self.assertEqual(len(payload['messages']),2)
