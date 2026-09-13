"""Resolve only short, explicit follow-ups within recent user/assistant pairs."""
import re
from typing import Sequence

from .activities import requests_activities
from .clock import is_clock_query
from .knowledge_intent import EMOTIONAL_SUPPORT_KEYWORDS, declines_knowledge, requests_knowledge
from .safety import SafetyTriage
from .state import ChatMessage
from .web_search import search_intent

MAX_CONTEXT_PAIRS = 3
MAX_ANCHOR_CHARS = 320
MAX_FOLLOWUP_CHARS = 160


def is_knowledge_followup(text: str) -> bool:
    if len(text) > MAX_FOLLOWUP_CHARS or declines_knowledge(text):
        return False
    return bool(re.fullmatch(
        r'(?:那|那么)?(?:请|能|可以|能不能|可不可以)?(?:再)?'
        r'(?:详细(?:解释|说|讲)(?:一下|一点)?|解释一下|具体(?:怎么做|怎么用|如何做)|'
        r'(?:这个|那个|刚才的|上面的)(?:练习|方法|技巧)(?:具体)?(?:怎么做|怎么用|如何做)|'
        r'(?:给我)?举(?:个|一个)例子)(?:吗|呢|好吗)?[？?。！!，,\s]*', text.strip()))


def resolve_knowledge_followup(text: str, messages: Sequence[ChatMessage]) -> str | None:
    if not is_knowledge_followup(text):
        return None
    # Never search across arbitrary intervening topics, assistant claims or session notes.
    recent = list(messages)[-MAX_CONTEXT_PAIRS * 2:]
    for end in range(len(recent), 1, -2):
        user, assistant = recent[end - 2:end]
        if user.role != "user" or assistant.role != "assistant":
            return None
        anchor = user.content.strip()
        if is_knowledge_followup(anchor):
            continue
        if (
            len(anchor) > MAX_ANCHOR_CHARS
            or declines_knowledge(anchor)
            or SafetyTriage().evaluate(anchor).requires_safe_response
            or is_clock_query(anchor)
            or search_intent(anchor)
            or requests_activities(anchor)
            or any(word in anchor for word in ("忘掉", "忘记", "不要再记得", "删除关于"))
            or ("清空" in anchor and "记忆" in anchor)
        ):
            return None
        if requests_knowledge(anchor) or any(word in anchor for word in EMOTIONAL_SUPPORT_KEYWORDS):
            return f"{anchor}\n当前追问：{text.strip()}"
        return None
    return None
