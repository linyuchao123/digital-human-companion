import unittest
from unittest.mock import AsyncMock, patch
from services.agent.workflow import DigitalXinyuWorkflow
from services.agent.web_search import search_intent, search_query, safe_url, SearchUnavailable
from services.agent.state import ChatMessage
from services.agent.web_search import tavily_search
import httpx

class Provider:
    async def generate(self, messages):
        return '上海今日天气资料见[1]，请以预报发布时间为准。'

class WebSearchTests(unittest.IsolatedAsyncioTestCase):
    async def test_city_confirmation_and_followup(self):
        with patch('services.agent.workflow.tavily_search',new=AsyncMock()) as search:
            result=await DigitalXinyuWorkflow(provider=Provider()).run(user_text='今天天气怎么样',trace_id='w',session_id='w')
            search.assert_not_called()
        self.assertIn('哪个城市',result['final_response'])
        self.assertTrue(search_intent('上海',result['messages']))
        self.assertIn('上海',search_query('上海',result['messages']))

    async def test_search_source_and_failure(self):
        sources=[{'title':'天气','url':'https://example.org/weather','content':'上海今天晴'}]
        with patch('services.agent.workflow.tavily_search',new=AsyncMock(return_value=sources)):
            result=await DigitalXinyuWorkflow(provider=Provider()).run(user_text='联网搜索上海旅游资料',trace_id='w',session_id='w')
        self.assertEqual(result['web_sources'],sources)
        self.assertIn('web_search',result['execution_path'])
        with patch('services.agent.workflow.tavily_search',new=AsyncMock(side_effect=SearchUnavailable('未配置'))):
            result=await DigitalXinyuWorkflow(provider=Provider()).run(user_text='联网查最新消息',trace_id='w',session_id='w')
        self.assertEqual(result['final_response'],'未配置')
        self.assertEqual(result['tool_calls'][0].status,'failed')

    async def test_safety_priority(self):
        with patch('services.agent.workflow.tavily_search',new=AsyncMock()) as search:
            result=await DigitalXinyuWorkflow(provider=Provider()).run(user_text='联网搜索，我想自杀',trace_id='w',session_id='w')
            search.assert_not_called()
        self.assertIn('safe_response',result['execution_path'])

    def test_safe_links_and_no_search_for_normal_chat(self):
        for url in ['javascript:alert(1)','http://localhost','http://127.0.0.1','https://x:y@example.org','file:///etc/passwd']:
            self.assertFalse(safe_url(url))
        self.assertTrue(safe_url('https://example.org'))
        self.assertFalse(search_intent('今天我有点累'))
        self.assertFalse(search_intent('今天的天气不错'))

    async def test_vendor_request_and_secret_not_in_body(self):
        response=httpx.Response(200,json={'results':[{'url':'https://example.org','title':'source','content':'data'}]},request=httpx.Request('POST','https://api.tavily.com/search'))
        client=AsyncMock();client.post.return_value=response
        with patch.dict('os.environ',{'TAVILY_API_KEY':'test-secret'}),patch('services.agent.web_search.httpx.AsyncClient') as factory:
            factory.return_value.__aenter__.return_value=client
            result=await tavily_search('上海天气')
        self.assertEqual(len(result),1)
        args=client.post.call_args
        self.assertEqual(args.args[0],'https://api.tavily.com/search')
        self.assertNotIn('test-secret',str(args.kwargs['json']))
        self.assertEqual(args.kwargs['json']['max_results'],3)

    async def test_missing_key(self):
        with patch.dict('os.environ',{'TAVILY_API_KEY':''}):
            with self.assertRaises(SearchUnavailable): await tavily_search('天气')
