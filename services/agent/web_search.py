"""Bounded Tavily search; external snippets are untrusted data, not instructions."""
import os
import re
import ipaddress
from datetime import datetime
from zoneinfo import ZoneInfo
from urllib.parse import urlparse

import httpx


CITIES = ('北京', '上海', '广州', '深圳', '南京', '杭州', '苏州', '镇江', '成都', '重庆',
          '武汉', '西安', '天津', '郑州', '长沙', '济南', '青岛', '厦门', '福州', '合肥',
          '昆明', '南昌', '南宁', '海口', '沈阳', '大连', '哈尔滨', '长春', '香港', '澳门', '台北')


def city_in(text):
    return next((city for city in CITIES if city in text), None)


def search_intent(text, messages=()):
    # 天气闲聊不是查询，不把普通聊天送到第三方。
    if re.search(r'天气(?:真|很|挺|还|太|有点|非常|特别)?(?:不错|好|差|糟糕|冷|热)', text) and not re.search(r'查询|搜索|查一下|吗|[？?]', text):
        return False
    if any(word in text for word in ('天气', '气温', '下雨', '降雨')):
        return True
    if messages and messages[-1].role == 'assistant' and '你想查询哪个城市的天气' in messages[-1].content:
        return bool(city_in(text))
    return bool(re.search(r'联网|上网查|搜索|搜一下|查一下|最新消息|最新新闻', text))


def safe_url(value):
    try:
        p = urlparse(value)
        if p.scheme not in ('http', 'https') or not p.hostname or p.username or p.password:
            return False
        host = p.hostname.lower()
        if '.' not in host or host.endswith(('.local', '.localhost')) or host == 'localhost':
            return False
        try:
            return ipaddress.ip_address(host).is_global
        except ValueError:
            return True
    except (ValueError, TypeError):
        return False


class SearchUnavailable(Exception):
    pass


async def tavily_search(query):
    key = os.environ.get('TAVILY_API_KEY', '').strip()
    if not key:
        raise SearchUnavailable('联网搜索尚未配置，请设置后端 TAVILY_API_KEY。')
    try:
        async with httpx.AsyncClient(timeout=10, follow_redirects=False) as client:
            response = await client.post('https://api.tavily.com/search',
                headers={'Authorization': f'Bearer {key}'}, json={
                    'query': query[:500], 'search_depth': 'basic', 'max_results': 3,
                    'include_answer': False, 'include_raw_content': False,
                    'include_images': False, 'auto_parameters': False,
                })
            response.raise_for_status()
            data = response.json()
        results = []
        for item in data.get('results', [])[:3]:
            if isinstance(item, dict) and safe_url(item.get('url')):
                results.append({'title': str(item.get('title', '来源'))[:160],
                    'url': item['url'][:2000], 'content': str(item.get('content', ''))[:2000]})
        if not results:
            raise SearchUnavailable('没有找到可用的联网资料，我暂时无法确认，请换个更具体的问法。')
        return results
    except SearchUnavailable:
        raise
    except (httpx.HTTPError, ValueError, TypeError, AttributeError):
        raise SearchUnavailable('联网搜索暂时失败，请稍后重试；我不能确认实时信息。') from None


def search_query(text, messages=()):
    weather = any(word in text for word in ('天气', '气温', '下雨', '降雨'))
    if messages and '你想查询哪个城市的天气' in messages[-1].content:
        weather = True
    if weather:
        city = city_in(text)
        if not city:
            return None
        date = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d')
        return f'{city} {date} {text[:200]} 天气预报'
    return text[:500]
