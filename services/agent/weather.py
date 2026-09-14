"""Dedicated forecast data with freshness checks; never fall back to old search snippets."""
import asyncio
import math
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import httpx
from .web_search import city_in, SearchUnavailable, resolve_weather_query

ZONE=ZoneInfo('Asia/Shanghai')


def weather_request(text, messages=()):
    return resolve_weather_query(text, messages) is not None


def number(value):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
        raise ValueError('invalid number')
    return value


def weather_label(code):
    if code==0: return '晴'
    if code in (1,2,3): return '少云或多云'
    if code in (45,48): return '雾'
    if code in (51,53,55,56,57): return '毛毛雨'
    if code in (61,63,65,66,67,80,81,82): return '雨'
    if code in (71,73,75,77,85,86): return '雪'
    if code in (95,96,99): return '雷暴'
    return '天气状况未明确'


def format_weather(city,data,text,now=None):
    now=(now or datetime.now(ZONE)).astimezone(ZONE)
    current=data['current']
    stamp=datetime.fromisoformat(current['time']).replace(tzinfo=ZONE)
    if not -900<=(now-stamp).total_seconds()<=10800:
        raise SearchUnavailable('天气接口数据时间过旧或异常，我暂时无法确认当前天气，请稍后重试。')
    if any(w in text for w in ('昨天','后天','下周','下个月')):
        return '目前支持当前天气及今天、明天的预报，请指定这几个时间范围。'
    daily=data['daily'];target=(now+timedelta(days=1 if '明天' in text else 0)).date().isoformat()
    index=daily['time'].index(target)
    lo=number(daily['temperature_2m_min'][index]);hi=number(daily['temperature_2m_max'][index])
    probability=number(daily['precipitation_probability_max'][index])
    if not 0<=probability<=100 or not -100<=lo<=hi<=70: raise ValueError('invalid forecast')
    forecast=f"{target}预报{weather_label(daily['weather_code'][index])}，气温{lo:g}～{hi:g}°C，最高降雨概率{probability:g}%。"
    if any(w in text for w in ('散步','走走','跑步')):
        code=daily['weather_code'][index]
        if code in (95,96,99):
            advice='预报有雷暴，建议暂缓户外活动。'
        elif probability>=50 or code in (45,48,51,53,55,56,57,61,63,65,66,67,71,73,75,77,80,81,82,85,86):
            advice='可能有降水或能见度不佳，建议优先室内活动。'
        elif hi>=32 or lo<=5:
            advice='温度偏高或偏低，建议避开不适时段，缩短户外活动时间。'
        else:
            advice='从温度和降水预报看，可以考虑短时间散步。'
        return f'{city}，{forecast}{advice}出门前仍需查看实际天气、空气质量和当地预警。来源Open-Meteo模型预报，并非实测。'
    if '明天' in text: return f'{city}，{forecast}数据来自Open-Meteo气象模型预报，并非实测。'
    temp=number(current['temperature_2m']);feels=number(current['apparent_temperature'])
    if not -100<=temp<=70 or not -150<=feels<=100: raise ValueError('invalid current')
    return f'{city}，截至北京时间{stamp:%m月%d日 %H:%M}，模型天气为{weather_label(current["weather_code"])}，气温{temp:g}°C，体感{feels:g}°C。{forecast}来源Open-Meteo，非气象站实测。'


async def get_weather(text,messages=()):
    text=resolve_weather_query(text,messages) or text
    city=city_in(text)
    if not city:
        return '你想查询哪个城市的天气？请告诉我城市名称，例如上海或南京。',[]
    async def request():
        async with httpx.AsyncClient(timeout=8,follow_redirects=False) as client:
            geo=await client.get('https://geocoding-api.open-meteo.com/v1/search',params={
                'name':city,'count':10,'language':'zh','format':'json'})
            geo.raise_for_status()
            countries={'香港':'HK','澳门':'MO','台北':'TW'}
            places=[p for p in geo.json().get('results',[]) if p.get('country_code')==countries.get(city,'CN')]
            if not places: raise SearchUnavailable('天气服务未能定位该城市，请换一个城市名称。')
            place=max(places,key=lambda p:p.get('population',0))
            lat=number(place['latitude']);lon=number(place['longitude'])
            if not -90<=lat<=90 or not -180<=lon<=180: raise ValueError('coordinates')
            result=await client.get('https://api.open-meteo.com/v1/forecast',params={
                'latitude':lat,'longitude':lon,'timezone':'Asia/Shanghai','forecast_days':2,
                'current':'temperature_2m,apparent_temperature,weather_code',
                'daily':'weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max'})
            result.raise_for_status()
            answer=format_weather(city,result.json(),text)
            return answer,[{'title':'Open-Meteo 天气模型数据','url':'https://open-meteo.com/','content':''}]
    try:
        return await asyncio.wait_for(request(),timeout=12)
    except SearchUnavailable: raise
    except httpx.TransportError: raise
    except (httpx.HTTPError,TimeoutError,ValueError,KeyError,TypeError,IndexError,AttributeError):
        raise SearchUnavailable('专用天气服务暂时不可用或数据不完整，我不能确认当前天气，请稍后重试。') from None
