import asyncio
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from services.agent.execution import execute_read
from services.agent.workflow import DigitalXinyuWorkflow
from services.agent.providers import FakeCompanionProvider
from services.agent.web_search import SearchUnavailable


class ReadRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_transient_retry_and_limit(self):
        operation = AsyncMock(side_effect=[httpx.ConnectError('private'), 'ok'])
        self.assertEqual(await execute_read(operation, timeout=1), 'ok')
        self.assertEqual(operation.await_count, 2)
        operation = AsyncMock(side_effect=httpx.ConnectError('private'))
        with self.assertRaises(httpx.ConnectError):
            await execute_read(operation, timeout=1)
        self.assertEqual(operation.await_count, 2)

    async def test_invalid_unavailable_and_cancellation_not_retried(self):
        for error in [ValueError('bad data'), SearchUnavailable('not configured'), asyncio.CancelledError()]:
            operation = AsyncMock(side_effect=error)
            with self.assertRaises(type(error)):
                await execute_read(operation, timeout=1)
            self.assertEqual(operation.await_count, 1)

    async def test_one_total_deadline(self):
        async def slow():
            await asyncio.sleep(1)
        operation = AsyncMock(side_effect=slow)
        with self.assertRaises(TimeoutError):
            await execute_read(operation, timeout=0.02)
        self.assertEqual(operation.await_count, 1)
        for budget in [float('inf'), float('nan'), 0, -1]:
            with self.assertRaises(ValueError):
                await execute_read(operation, timeout=budget)

    async def test_combined_clock_weather_and_recovery(self):
        weather = AsyncMock(side_effect=[httpx.ConnectError('secret'), ('上海天气测试', [])])
        with patch('services.agent.workflow.get_weather', weather):
            result = await DigitalXinyuWorkflow(provider=FakeCompanionProvider()).run(
                user_text='现在几点，上海天气怎么样', trace_id='combined', session_id='combined')
        self.assertIn('北京时间', result['final_response'])
        self.assertIn('上海天气测试', result['final_response'])
        self.assertEqual([t.name for t in result['tool_calls']], ['current_datetime', 'weather_forecast'])
        self.assertEqual(weather.await_count, 2)

    async def test_failure_keeps_time_and_never_guesses_weather(self):
        weather = AsyncMock(side_effect=SearchUnavailable('天气数据不可用'))
        with patch('services.agent.workflow.get_weather', weather):
            result = await DigitalXinyuWorkflow(provider=FakeCompanionProvider()).run(
                user_text='现在几点，上海天气怎么样', trace_id='failed', session_id='failed')
        self.assertIn('北京时间', result['final_response'])
        self.assertIn('不可用', result['final_response'])
        self.assertEqual(weather.await_count, 1)
        self.assertEqual(result['tool_calls'][-1].status, 'failed')

    async def test_safety_and_no_network_override_combined_plan(self):
        with patch('services.agent.workflow.get_weather', AsyncMock()) as weather:
            for text in ['我想自杀，现在几点，上海天气怎么样', '不要联网，现在几点，上海天气怎么样']:
                await DigitalXinyuWorkflow(provider=FakeCompanionProvider()).run(
                    user_text=text, trace_id='guard', session_id='guard')
        weather.assert_not_called()
