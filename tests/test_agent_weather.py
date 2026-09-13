import unittest
from datetime import datetime,timedelta
from unittest.mock import AsyncMock,patch
from services.agent.weather import format_weather,get_weather,ZONE
from services.agent.web_search import SearchUnavailable
from services.agent.workflow import DigitalXinyuWorkflow

NOW=datetime(2026,9,13,12,0,tzinfo=ZONE)
def data():
    return {'current':{'time':'2026-09-13T11:45','temperature_2m':25,'apparent_temperature':26,'weather_code':2},
            'daily':{'time':['2026-09-13','2026-09-14'],'temperature_2m_min':[20,21],
                     'temperature_2m_max':[27,28],'weather_code':[2,61],'precipitation_probability_max':[10,70]}}

class WeatherTests(unittest.IsolatedAsyncioTestCase):
    def test_current_and_tomorrow(self):
        answer=format_weather('上海',data(),'上海今天的天气',NOW)
        self.assertIn('11:45',answer);self.assertIn('25°C',answer);self.assertIn('非气象站实测',answer)
        answer=format_weather('上海',data(),'上海明天天气',NOW)
        self.assertIn('2026-09-14',answer);self.assertIn('70%',answer);self.assertNotIn('体感',answer)

    def test_stale_invalid_and_missing_forecast(self):
        with self.assertRaises(SearchUnavailable):format_weather('上海',data(),'天气',NOW+timedelta(days=1))
        invalid=data();invalid['current']['temperature_2m']=float('nan')
        with self.assertRaises(ValueError):format_weather('上海',invalid,'天气',NOW)
        invalid=data();invalid['daily']['time']=[]
        with self.assertRaises(ValueError):format_weather('上海',invalid,'天气',NOW)

    async def test_workflow_uses_dedicated_data_not_search_or_model(self):
        sources=[{'title':'Open-Meteo','url':'https://open-meteo.com/','content':''}]
        with patch('services.agent.workflow.get_weather',new=AsyncMock(return_value=('天气测试',sources))),patch('services.agent.workflow.tavily_search',new=AsyncMock()) as search:
            result=await DigitalXinyuWorkflow().run(user_text='上海天气',trace_id='weather',session_id='weather')
            search.assert_not_called()
        self.assertEqual(result['final_response'],'天气测试')
        self.assertEqual(result['tool_calls'][0].name,'weather_forecast')

    async def test_failure_does_not_fall_back_to_stale_web(self):
        with patch('services.agent.workflow.get_weather',new=AsyncMock(side_effect=SearchUnavailable('数据过旧'))),patch('services.agent.workflow.tavily_search',new=AsyncMock()) as search:
            result=await DigitalXinyuWorkflow().run(user_text='上海天气',trace_id='weather',session_id='weather')
            search.assert_not_called()
        self.assertEqual(result['final_response'],'数据过旧')
        self.assertEqual(result['tool_calls'][0].status,'failed')
