"""Bounded Tavily search; external snippets are untrusted data, not instructions."""
import os
import re
import ipaddress
from datetime import datetime
from zoneinfo import ZoneInfo
from urllib.parse import urlparse

import httpx
from .planning import external_tools_disabled
from .knowledge_intent import requests_knowledge


CITIES = ('北京', '上海', '广州', '深圳', '南京', '杭州', '苏州', '镇江', '成都', '重庆',
          '武汉', '西安', '天津', '郑州', '长沙', '济南', '青岛', '厦门', '福州', '合肥',
          '昆明', '南昌', '南宁', '海口', '沈阳', '大连', '哈尔滨', '长春', '香港', '澳门', '台北')


def city_in(text):
    return next((city for city in CITIES if city in text), None)


def resolve_weather_query(text, messages=(), now=None):
    """Only inherit the immediately preceding dated weather answer, never global state."""
    explicit = any(w in text for w in ('天气', '气温', '下雨', '降雨'))
    if len(messages) < 2 or messages[-1].role != 'assistant' or messages[-2].role != 'user':
        return text if explicit else None
    previous = messages[-1].content
    if '你想查询哪个城市的天气' in previous:
        if re.fullmatch(r'(?:那|在)?(?:'+'|'.join(CITIES)+r')(?:的)?(?:今天|明天|后天)?(?:呢)?[？?。！!\s]*', text):
            # Current city overrides any earlier city; only inherit the date.
            days = ('今天','明天','后天','昨天','下周','下个月')
            day = '' if any(w in text for w in days) else next((w for w in days if w in messages[-2].content), '今天')
            return f'{text} {day} 天气'
        return text if explicit else None
    match = re.match(r'(' + '|'.join(CITIES) + r')，.*?(\d{4}-\d{2}-\d{2})预报', previous)
    followup = re.fullmatch(r'(?:那|那么)?(?:今天|明天|后天)(?:呢|怎么样)?[？?。！!\s]*', text)
    outing = re.fullmatch(r'(?:那|那么)?(?:今天|明天)?(?:适合|可以|能)(?:出去|出门)?(?:散步|走走|跑步)(?:吗|么)?[？?。！!\s]*', text)
    if not match or 'Open-Meteo' not in previous or not (explicit or followup or outing):
        return text if explicit else None
    today = (now or datetime.now(ZoneInfo('Asia/Shanghai'))).date()
    try:
        age = (datetime.fromisoformat(match[2]).date() - today).days
    except ValueError:
        return text if explicit else None
    if age not in (0, 1):
        return text if explicit else None
    city = city_in(text) or match[1]
    day = '' if any(w in text for w in ('今天', '明天', '后天', '昨天', '下周', '下个月')) else ('明天' if age == 1 else '今天')
    return f'{city} {text} {day} 天气'


def search_intent(text, messages=()):
    if external_tools_disabled(text):
        return False
    # 天气闲聊不是查询，不把普通聊天送到第三方。
    if re.search(r'天气(?:真|很|挺|还|太|有点|非常|特别)?(?:不错|好|差|糟糕|冷|热)', text) and not re.search(r'查询|搜索|查一下|吗|[？?]', text):
        return False
    if any(word in text for word in ('天气', '气温', '下雨', '降雨')):
        # Personal distress containing weather words is not itself a forecast request.
        # Explicit forecast/search questions still take precedence.
        if requests_knowledge(text) and not re.search(
            r'查|搜|预报|气温|会不会|会下雨|下雨吗|天气.*(?:怎么样|如何|多少|吗|[？?])', text
        ):
            return False
        return True
    if resolve_weather_query(text, messages) is not None:
        return True
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
    except httpx.TransportError:
        raise
    except (httpx.HTTPError, ValueError, TypeError, AttributeError):
        raise SearchUnavailable('联网搜索暂时失败，请稍后重试；我不能确认实时信息。') from None


def search_query(text, messages=()):
    resolved = resolve_weather_query(text, messages)
    if resolved is not None:
        text = resolved
        city = city_in(text)
        if not city:
            return None
        date = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d')
        return f'{city} {date} {text[:200]} 天气预报'
    return text[:500]
