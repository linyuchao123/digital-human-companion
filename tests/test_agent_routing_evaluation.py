"""Offline routing acceptance matrix: no live keys, no external requests."""
import unittest
from datetime import datetime
from zoneinfo import ZoneInfo
from unittest.mock import AsyncMock, patch
from services.agent.providers import FakeCompanionProvider
from services.agent.state import ChatMessage
from services.agent.workflow import DigitalXinyuWorkflow
from services.agent.web_search import resolve_weather_query, search_intent

class RoutingEvaluationTests(unittest.IsolatedAsyncioTestCase):
    async def test_no_external_calls_matrix_even_with_bad_model(self):
        cases=['你好','请问你是谁','我想知道你叫什么名字','今天的天气不错',
               '不要联网搜索，陪我聊天','不用上网，今天天气怎么样',
               '禁止搜索，帮我了解最近的新闻','别查天气，我只是想聊天']
        for text in cases:
            with self.subTest(text=text):
                provider=FakeCompanionProvider()
                provider.select_tool=AsyncMock(return_value='{"tool":"web_search"}')
                with patch('services.agent.workflow.tavily_search',new=AsyncMock()) as search,patch('services.agent.workflow.get_weather',new=AsyncMock()) as weather:
                    result=await DigitalXinyuWorkflow(provider=provider).run(user_text=text,trace_id='eval',session_id='eval')
                search.assert_not_called();weather.assert_not_called();provider.select_tool.assert_not_called()
                self.assertNotIn('web_search',result['execution_path'])

    async def test_city_confirmation_rejects_city_smalltalk(self):
        history=[ChatMessage(role='user',content='明天天气怎么样'),
                 ChatMessage(role='assistant',content='你想查询哪个城市的天气？请告诉我城市名称，例如上海或南京。')]
        for text in ['上海很好吃','南京是我的家乡','明天我要学习']:
            with self.subTest(text=text):
                self.assertFalse(search_intent(text,history))
        for text in ['上海','上海明天呢？','在南京']:
            with self.subTest(text=text):
                self.assertTrue(search_intent(text,history))
                self.assertIn('明天',resolve_weather_query(text,history))
        history[0]=ChatMessage(role='user',content='后天天气怎么样')
        self.assertIn('后天',resolve_weather_query('上海',history))
        self.assertNotIn('后天',resolve_weather_query('上海明天呢',history))

    def test_context_matrix_is_bounded_and_dated(self):
        now=datetime(2026,9,13,12,tzinfo=ZoneInfo('Asia/Shanghai'))
        history=[ChatMessage(role='user',content='上海天气'),ChatMessage(role='assistant',content='上海，2026-09-13预报多云。来源Open-Meteo。')]
        for text in ['那明天呢','适合出去散步吗','北京明天天气']:
            with self.subTest(text=text):self.assertIsNotNone(resolve_weather_query(text,history,now))
        self.assertTrue(resolve_weather_query('北京明天天气',history,now).startswith('北京'))
        for reply in ['上海，2026-99-99预报晴。来源Open-Meteo。','上海，2026-09-10预报晴。来源Open-Meteo。','今天过得怎么样？']:
            history[-1]=ChatMessage(role='assistant',content=reply)
            with self.subTest(reply=reply):self.assertIsNone(resolve_weather_query('那明天呢',history,now))
        self.assertIsNone(resolve_weather_query('那明天呢',[],now))

    async def test_valid_weather_followup_routes_weather_not_search(self):
        date=datetime.now(ZoneInfo('Asia/Shanghai')).date().isoformat()
        history=[ChatMessage(role='user',content='上海天气'),ChatMessage(role='assistant',content=f'上海，{date}预报多云。来源Open-Meteo。')]
        with patch('services.agent.workflow.get_weather',new=AsyncMock(return_value=('天气测试',[]))) as weather,patch('services.agent.workflow.tavily_search',new=AsyncMock()) as search:
            result=await DigitalXinyuWorkflow().run(user_text='那明天呢',messages=history,trace_id='eval',session_id='eval')
        weather.assert_awaited_once();search.assert_not_called()
        self.assertEqual(result['tool_calls'][0].name,'weather_forecast')
