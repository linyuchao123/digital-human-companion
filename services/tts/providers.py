"""可替换的服务端 TTS 提供者实现。"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence
from urllib.parse import urlparse

import httpx


class TtsProviderError(RuntimeError):
    """服务端语音提供者不可用或合成失败。"""


@dataclass(frozen=True)
class TtsVoice:
    id: str
    name: str
    locale: str
    provider: str


@dataclass(frozen=True)
class SynthesizedAudio:
    content: bytes
    media_type: str
    provider: str
    voice: str


class MacOSSayProvider:
    """通过 macOS 原生 `say` 命令生成标准 PCM WAV。"""

    provider_id = "macos_say"
    _VOICE_PATTERN = re.compile(r"^(.+?)\s+([a-z]{2}_[A-Z]{2})\s+#")

    def __init__(
        self,
        *,
        executable: str | None = None,
        voices: Sequence[TtsVoice] | None = None,
    ) -> None:
        self.executable = executable or shutil.which("say") or ""
        if not self.executable:
            raise TtsProviderError("macOS say 命令不可用")
        self._voices = tuple(voices) if voices is not None else self._discover_voices()
        if not self._voices:
            raise TtsProviderError("未发现可用的中文系统音色")

    def _discover_voices(self) -> tuple[TtsVoice, ...]:
        try:
            completed = subprocess.run(
                [self.executable, "-v", "?"],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise TtsProviderError(f"读取系统音色失败: {type(exc).__name__}") from exc
        voices = []
        for line in completed.stdout.splitlines():
            match = self._VOICE_PATTERN.match(line)
            if match is None:
                continue
            name, locale = match.groups()
            if not locale.startswith("zh_"):
                continue
            cleaned_name = name.strip()
            voices.append(TtsVoice(
                id=cleaned_name,
                name=cleaned_name,
                locale=locale,
                provider=self.provider_id,
            ))
        return tuple(voices)

    def list_voices(self) -> tuple[TtsVoice, ...]:
        return self._voices

    def synthesize(self, text: str, *, voice: str, rate: int = 185) -> SynthesizedAudio:
        normalized_text = text.strip()
        if not normalized_text:
            raise TtsProviderError("语音文本不能为空")
        allowed_voices = {item.id for item in self._voices}
        if voice not in allowed_voices:
            raise TtsProviderError("请求的系统音色不在允许列表中")
        safe_rate = min(max(int(rate), 120), 260)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as output:
                temporary_path = Path(output.name)
            subprocess.run(
                [
                    self.executable,
                    "-v",
                    voice,
                    "-r",
                    str(safe_rate),
                    "-o",
                    str(temporary_path),
                    "--file-format=WAVE",
                    "--data-format=LEI16@24000",
                    normalized_text,
                ],
                check=True,
                capture_output=True,
                timeout=30,
            )
            content = temporary_path.read_bytes()
            if not content.startswith(b"RIFF") or len(content) < 44:
                raise TtsProviderError("系统语音未生成有效 WAV")
            return SynthesizedAudio(
                content=content,
                media_type="audio/wav",
                provider=self.provider_id,
                voice=voice,
            )
        except TtsProviderError:
            raise
        except (OSError, subprocess.SubprocessError) as exc:
            raise TtsProviderError(f"系统语音合成失败: {type(exc).__name__}") from exc
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)


class Qwen3TtsProvider:
    """通过 DashScope HTTP API 生成带角色表现力的中文语音。"""

    provider_id = "qwen3_tts"
    default_endpoint = (
        "https://dashscope.aliyuncs.com/api/v1/services/"
        "aigc/multimodal-generation/generation"
    )
    _voices = (
        TtsVoice("Chelsie", "Chelsie · 二次元少女", "zh-CN", provider_id),
        TtsVoice("Momo", "Momo · 活泼俏皮", "zh-CN", provider_id),
        TtsVoice("Cherry", "Cherry · 阳光自然", "zh-CN", provider_id),
    )
    _instructions = {
        "Chelsie": "使用可爱俏皮、亲切自然的年轻女性声线，语气灵动，语速稍快但吐字清晰。",
        "Momo": "使用活泼调皮、元气十足的年轻女性声线，带一点开心的上扬语调。",
        "Cherry": "使用阳光温柔、自然亲切的年轻女性声线，像朋友一样真诚交流。",
    }

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "qwen3-tts-instruct-flash",
        endpoint: str | None = None,
        client_factory: Callable[..., Any] = httpx.Client,
    ) -> None:
        if not api_key.strip():
            raise TtsProviderError("DashScope API Key 未配置")
        self.api_key = api_key.strip()
        self.model = model.strip() or "qwen3-tts-instruct-flash"
        self.endpoint = endpoint or self.default_endpoint
        self.client_factory = client_factory

    def list_voices(self) -> tuple[TtsVoice, ...]:
        return self._voices

    @staticmethod
    def _validate_audio_url(audio_url: str) -> None:
        parsed = urlparse(audio_url)
        hostname = (parsed.hostname or "").lower()
        if parsed.scheme not in {"http", "https"}:
            raise TtsProviderError("Qwen3-TTS 返回了不安全的音频地址")
        if hostname != "aliyuncs.com" and not hostname.endswith(".aliyuncs.com"):
            raise TtsProviderError("Qwen3-TTS 音频地址不属于阿里云域名")

    def synthesize(self, text: str, *, voice: str, rate: int = 185) -> SynthesizedAudio:
        del rate  # Qwen3-TTS 使用自然语言指令控制语速与表现力。
        normalized_text = text.strip()
        if not normalized_text:
            raise TtsProviderError("语音文本不能为空")
        allowed_voices = {item.id for item in self._voices}
        if voice not in allowed_voices:
            raise TtsProviderError("请求的 Qwen3 音色不在允许列表中")
        request_body = {
            "model": self.model,
            "input": {
                "text": normalized_text,
                "voice": voice,
                "language_type": "Chinese",
                "instructions": self._instructions[voice],
                "optimize_instructions": True,
            },
        }
        try:
            with self.client_factory(timeout=45.0, follow_redirects=True) as client:
                response = client.post(
                    self.endpoint,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=request_body,
                )
                response.raise_for_status()
                payload = response.json()
                audio_url = payload["output"]["audio"]["url"]
                self._validate_audio_url(audio_url)
                audio_response = client.get(audio_url)
                audio_response.raise_for_status()
                content = audio_response.content
                if not content or len(content) > 10 * 1024 * 1024:
                    raise TtsProviderError("Qwen3-TTS 返回的音频为空或超过大小限制")
                media_type = audio_response.headers.get("content-type", "audio/wav")
                if not media_type.startswith("audio/"):
                    media_type = "audio/wav"
        except TtsProviderError:
            raise
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
            raise TtsProviderError(f"Qwen3-TTS 合成失败: {type(exc).__name__}") from exc
        return SynthesizedAudio(
            content=content,
            media_type=media_type.split(";", 1)[0],
            provider=self.provider_id,
            voice=voice,
        )
