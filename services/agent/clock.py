"""确定性时间查询，按显式城市时区或默认北京时间回答。"""
import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo


def is_clock_query(text: str) -> bool:
    text = re.sub(r"[\s，。！？?,.!]", "", text)
    text = re.sub(r"^(?:你好|小安)", "", text)
    text = re.sub(r"^(?:你知道|你能告诉我|能告诉我|请问)", "", text)
    text = re.sub(r"(?:吗|么|呀|啊)$", "", text)
    return bool(re.fullmatch(
        r"(?:请|帮我|告诉我)?(?:一下)?(?:现在|今天|明天|北京时间|上海|北京|纽约|伦敦|东京)?"
        r"(?:现在)?(?:是)?(?:几点(?:了|钟)?|什么时间|几月几号|几号|什么日期|星期几|周几|日期)(?:了|呢|是多少)?",
        text,
    ))

def clock_answer(text: str, now: datetime | None = None) -> str:
    zone = next((zone for city, zone in (
        ("纽约", "America/New_York"), ("伦敦", "Europe/London"),
        ("东京", "Asia/Tokyo")
    ) if city in text), "Asia/Shanghai")
    label = {"America/New_York": "纽约当地时间", "Europe/London": "伦敦当地时间",
             "Asia/Tokyo": "东京当地时间", "Asia/Shanghai": "北京时间"}[zone]
    current = (now or datetime.now(ZoneInfo("UTC"))).astimezone(ZoneInfo(zone))
    if "明天" in text:
        current += timedelta(days=1)
    weekday = "一二三四五六日"[current.weekday()]
    if any(word in text for word in ("日期", "几号", "几月", "星期", "周几", "明天")):
        return f"按{label}，{'明天' if '明天' in text else '今天'}是{current.year}年{current.month}月{current.day}日，星期{weekday}。"
    return f"现在是{label} {current:%H:%M}。"
