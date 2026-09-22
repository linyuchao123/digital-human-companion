from __future__ import annotations

from dataclasses import dataclass, field
import asyncio
from contextvars import ContextVar, Token
import json
from typing import AsyncIterator
from typing import Any, Protocol, Sequence

import httpx

from .state import ChatMessage
from .planning import ROUTING_PROMPT
from .usage import estimate_tokens, record_usage, usage_from_payload


DAILY_SYSTEM_PROMPT = """你是“数字心屿”中的 AI 情绪陪伴助手小安，当前是日常对话模式。
像一位熟悉用户、温暖但有边界的朋友那样交流：先理解这句话里最具体的细节、语气和真正需要，再决定是接住情绪、自然延展话题、给一个小建议，还是追问一个必要问题。
回应要贴合当前对话和已有上下文，可以有自然口语与节奏变化；不要机械复述原话，不要反复使用“听起来你……”或“愿意和我说说……”等固定开头，也不要每轮都总结、建议和提问。
用户只是闲聊时就轻松聊天；表达困扰时再温和梳理。用户明确感受和更正优先于你的推断，不把推测说成事实。
不要进行医疗诊断，不提供药物建议，也不要声称可以替代专业帮助。
通常回复 80～220 个汉字，简单问候可以更短；不要输出动作标签或内部推理过程。"""

EMOTIONAL_SYSTEM_PROMPT = """你是“数字心屿”中的 AI 情绪陪伴助手小安，当前是用户主动选择的情感对话模式。
先抓住用户经历中的具体人物、事件、矛盾和措辞，再表达更深入但克制的共鸣，分析可能的情绪、关系矛盾与需要；不要泛泛安慰，也不要把推测说成事实。
用户想解决问题时给 1～2 个具体建议或可直接使用的沟通话术；只想倾诉时少给建议。
表达方式随上下文变化，像真正听懂之后再回应；避免模板式复述、连续追问、固定安慰开头和空泛大道理，不要求每条以问题结尾。用户当前表达和更正优先。
不要医疗诊断或药物建议，不声称替代专业帮助，不设定恋爱关系、不虚构真人感情、不鼓励排他依赖。
通常回复 200～350 个汉字，也可根据上下文短答；不要输出动作标签或内部推理过程。"""

COMPANION_SYSTEM_PROMPT = DAILY_SYSTEM_PROMPT
_generation_profile: ContextVar[tuple[str, str]] = ContextVar(
    "companion_generation_profile", default=("daily", "confidant")
)


def set_generation_profile(mode: str, style: str) -> Token:
    safe_mode = mode if mode in {"daily", "emotional"} else "daily"
    safe_style = style if style in {"confidant", "gentle"} else "confidant"
    return _generation_profile.set((safe_mode, safe_style))


def reset_generation_profile(token: Token) -> None:
    _generation_profile.reset(token)


def generation_settings() -> tuple[str, float, int]:
    mode, style = _generation_profile.get()
    if mode == "emotional":
        style_prompt = (
            "语气采用温柔亲昵风格：只使用用户认可过的称呼，亲昵仅限表达方式，保持清晰边界。"
            if style == "gentle" else "语气采用知心风格：真诚、平等、具体，不居高临下。"
        )
        return f"{EMOTIONAL_SYSTEM_PROMPT}\n{style_prompt}", 0.82, 8192
    return DAILY_SYSTEM_PROMPT, 0.68, 8192


class CompanionProviderError(RuntimeError):
    """云端 Provider 请求失败或返回内容无效。"""


@dataclass(frozen=True)
class OpenAICompatibleConfig:
    api_key: str
    provider_name: str = "cloud"
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model: str = "qwen-plus"
    temperature: float = 0.75
    max_tokens: int = 250
    timeout_seconds: float = 30.0
    extra_body: dict[str, Any] = field(default_factory=dict)


class CompanionProvider(Protocol):
    async def generate(self, messages: Sequence[ChatMessage]) -> str:
        """根据已审查的上下文生成回复。"""
        ...


class FakeCompanionProvider:
    """无密钥、无网络时的可预测开发与 CI Provider。"""

    async def generate(self, messages: Sequence[ChatMessage]) -> str:
        user_text = next(
            (message.content for message in reversed(messages) if message.role == "user"),
            "",
        )
        if not user_text:
            return "我在这里，你可以慢慢说。"
        mode, _ = _generation_profile.get()
        if mode == "emotional" and any(keyword in user_text for keyword in ("累", "疲惫", "压力", "焦虑", "睡不着")):
            return "你像是在一边扛着具体的压力，一边又责怪自己为什么还没调整好。先不用急着证明自己没事；如果你想解决，我们可以只挑眼前最消耗你的那一件事，看看它背后是需要休息、被理解，还是需要更清楚的边界。"
        if any(keyword in user_text for keyword in ("累", "疲惫", "压力", "焦虑", "睡不着")):
            return "听起来你今天承受了不少压力。愿意和我说说，哪一件事最让你疲惫吗？"
        return f"我听到你说“{user_text}”。此刻你最希望我陪你聊哪一部分？"


class OpenAICompatibleCompanionProvider:
    """适配 DashScope 等 OpenAI Chat Completions 兼容服务。"""

    def __init__(
        self,
        config: OpenAICompatibleConfig,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not config.api_key.strip():
            raise ValueError("api_key 不能为空")
        self.config = config
        self._client = client

    async def generate(self, messages: Sequence[ChatMessage]) -> str:
        prompt, temperature, max_tokens = generation_settings()
        return await self._generate(messages, prompt, temperature=temperature, max_tokens=max_tokens)

    async def generate_stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        prompt, temperature, max_tokens = generation_settings()
        payload = {"model": self.config.model, "messages": [
            {"role": "system", "content": prompt},
            *(message.model_dump() for message in messages)],
            "temperature": temperature, "max_tokens": max_tokens,
            **self.config.extra_body, "stream": True,
            "stream_options": {"include_usage": True}}
        if payload.get("thinking", {}).get("type") == "enabled":
            # DeepSeek 思考模式不使用 temperature；移除它以保持请求语义明确。
            payload.pop("temperature", None)
        client = self._client or httpx.AsyncClient(base_url=self.config.base_url.rstrip('/'),
            timeout=self.config.timeout_seconds)
        seen = False; recorded = False; exact_usage = None; chunks = []
        input_estimate = estimate_tokens([prompt, *(message.content for message in messages)])
        try:
            async with asyncio.timeout(self.config.timeout_seconds):
                async with client.stream('POST', '/chat/completions', json=payload,
                    headers={"Authorization": f"Bearer {self.config.api_key}"}) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if len(line) > 65536:
                            raise CompanionProviderError('模型流式事件超限')
                        if not line.startswith('data:'):
                            continue
                        data = line[5:].strip()
                        if data == '[DONE]':
                            if not seen:
                                raise CompanionProviderError('模型返回空回复')
                            usage = exact_usage or (input_estimate, estimate_tokens(chunks), 0)
                            record_usage(provider=self.config.provider_name, model=self.config.model,
                                request_kind="chat", input_tokens=usage[0], output_tokens=usage[1],
                                cached_input_tokens=usage[2], estimated=exact_usage is None)
                            recorded = True
                            return
                        event = json.loads(data)
                        if usage_from_payload(event) is not None:
                            exact_usage = usage_from_payload(event)
                        choices = event.get('choices', [])
                        if not choices:
                            continue
                        content = choices[0].get('delta', {}).get('content')
                        if content is None or content == '':
                            continue
                        if not isinstance(content, str):
                            raise CompanionProviderError('模型增量格式错误')
                        seen = True
                        chunks.append(content)
                        yield content
                    raise CompanionProviderError('模型回复流未正常结束')
        except (httpx.HTTPError, TimeoutError, ValueError, KeyError, IndexError, TypeError,
                AttributeError, CompanionProviderError) as exc:
            if not recorded:
                record_usage(provider=self.config.provider_name, model=self.config.model,
                    request_kind="chat", input_tokens=input_estimate, output_tokens=0,
                    estimated=True, status="failed")
            raise CompanionProviderError('模型流式请求失败或超时') from exc
        finally:
            if self._client is None:
                await client.aclose()

    async def select_tool(self, text: str) -> str:
        return await self._generate([ChatMessage(role='user',content=text)], ROUTING_PROMPT, routing=True)

    async def _generate(self, messages: Sequence[ChatMessage], system_prompt: str, routing: bool = False,
                        temperature: float | None = None, max_tokens: int | None = None) -> str:
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                *(message.model_dump() for message in messages),
            ],
            "temperature": self.config.temperature if temperature is None else temperature,
            "max_tokens": self.config.max_tokens if max_tokens is None else max_tokens,
            **self.config.extra_body,
        }
        if routing:
            # 工具路由只做固定 JSON 分类，不需要高成本思考，也避免增加首句延迟。
            payload.update(temperature=0, max_tokens=80, thinking={"type": "disabled"})
            payload.pop("reasoning_effort", None)
        elif payload.get("thinking", {}).get("type") == "enabled":
            payload.pop("temperature", None)
        headers = {"Authorization": f"Bearer {self.config.api_key}"}
        input_estimate = estimate_tokens([system_prompt, *(message.content for message in messages)])
        try:
            if self._client is not None:
                response = await self._client.post(
                    "/chat/completions", json=payload, headers=headers
                )
            else:
                async with httpx.AsyncClient(
                    base_url=self.config.base_url.rstrip("/"),
                    headers=headers,
                    timeout=self.config.timeout_seconds,
                ) as client:
                    response = await client.post("/chat/completions", json=payload)
            response.raise_for_status()
            body = response.json()
            content = body["choices"][0]["message"]["content"].strip()
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
            record_usage(provider=self.config.provider_name, model=self.config.model,
                request_kind="routing" if routing else "chat", input_tokens=input_estimate,
                output_tokens=0, estimated=True, status="failed")
            raise CompanionProviderError("云端陪伴模型调用失败") from exc
        if not content:
            record_usage(provider=self.config.provider_name, model=self.config.model,
                request_kind="routing" if routing else "chat", input_tokens=input_estimate,
                output_tokens=0, estimated=True, status="failed")
            raise CompanionProviderError("云端陪伴模型返回了空内容")
        usage = usage_from_payload(body) or (input_estimate, estimate_tokens([content]), 0)
        record_usage(provider=self.config.provider_name, model=self.config.model,
            request_kind="routing" if routing else "chat", input_tokens=usage[0],
            output_tokens=usage[1], cached_input_tokens=usage[2],
            estimated=usage_from_payload(body) is None)
        return content


class FallbackCompanionProvider:
    """云端不可用时自动回退，保证体验链路仍可运行。"""

    def __init__(self, primary: CompanionProvider, fallback: CompanionProvider) -> None:
        self._primary = primary
        self._fallback = fallback

    async def generate(self, messages: Sequence[ChatMessage]) -> str:
        try:
            return await self._primary.generate(messages)
        except CompanionProviderError:
            return await self._fallback.generate(messages)

    async def generate_stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        started = False
        try:
            async for delta in stream_companion(self._primary, messages):
                started = True
                yield delta
        except CompanionProviderError:
            # 已展示的半句话不能拼接另一模型的全新回答。
            if started:
                raise
            async for delta in stream_companion(self._fallback, messages):
                yield delta

    async def select_tool(self, text: str) -> str:
        # Routing has a short total deadline; do not invoke a second cloud model.
        selector = getattr(self._primary, 'select_tool', None)
        if selector is None:
            return '{"tool":"chat"}'
        return await selector(text)


async def stream_companion(provider: CompanionProvider, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
    stream = getattr(provider, 'generate_stream', None)
    if stream is None:
        yield await provider.generate(messages)
    else:
        async for delta in stream(messages):
            yield delta


def create_companion_provider(
    *,
    api_key: str,
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    model: str = "qwen-plus",
    provider_name: str = "cloud",
    fallback_api_key: str = "",
    fallback_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    fallback_model: str = "qwen-plus",
    fallback_name: str = "cloud",
    extra_body: dict[str, Any] | None = None,
) -> tuple[CompanionProvider, str]:
    """创建主模型、备用模型、离线实现组成的有序降级链。"""
    if not api_key.strip():
        if not fallback_api_key.strip():
            return FakeCompanionProvider(), "offline"
        fallback_cloud = OpenAICompatibleCompanionProvider(
            OpenAICompatibleConfig(
                api_key=fallback_api_key,
                provider_name=fallback_name,
                base_url=fallback_base_url,
                model=fallback_model,
            )
        )
        return (
            FallbackCompanionProvider(fallback_cloud, FakeCompanionProvider()),
            f"{fallback_name}_with_offline_fallback",
        )
    primary_cloud = OpenAICompatibleCompanionProvider(
        OpenAICompatibleConfig(
            api_key=api_key,
            provider_name=provider_name,
            base_url=base_url,
            model=model,
            extra_body=extra_body or {},
        )
    )
    if not fallback_api_key.strip():
        return (
            FallbackCompanionProvider(primary_cloud, FakeCompanionProvider()),
            f"{provider_name}_with_offline_fallback" if provider_name != "cloud" else "cloud_with_fallback",
        )
    fallback_cloud = OpenAICompatibleCompanionProvider(
        OpenAICompatibleConfig(
            api_key=fallback_api_key,
            provider_name=fallback_name,
            base_url=fallback_base_url,
            model=fallback_model,
        )
    )
    return (
        FallbackCompanionProvider(
            primary_cloud,
            FallbackCompanionProvider(fallback_cloud, FakeCompanionProvider()),
        ),
        f"{provider_name}_with_{fallback_name}_fallback",
    )
