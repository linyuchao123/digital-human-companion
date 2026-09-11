"""可替换的服务端 TTS 提供者实现。"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


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
