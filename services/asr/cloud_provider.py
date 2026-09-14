"""百炼句级 ASR。密钥、音频与厂商响应体不写日志。"""
import base64
import os
from pathlib import Path
from urllib.parse import urlparse

import httpx


class CloudAsrError(RuntimeError):
    pass


def api_key():
    return next((os.environ.get(name, "").strip() for name in
                 ("ASR_API_KEY", "DASHSCOPE_API_KEY", "TTS_API_KEY", "QWEN_API_KEY")
                 if os.environ.get(name, "").strip()), "")


def endpoint():
    url = os.environ.get("ASR_CLOUD_URL", "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation").strip()
    host = urlparse(url)
    if host.scheme != "https" or not host.hostname or not (
        host.hostname == "dashscope.aliyuncs.com" or
        host.hostname == "dashscope-intl.aliyuncs.com" or
        host.hostname.endswith(".maas.aliyuncs.com")
    ) or host.username or host.password:
        raise CloudAsrError("invalid_endpoint")
    return url


def transcribe(audio_path, hotwords=""):
    key = api_key()
    if not key:
        raise CloudAsrError("key_missing")
    path = Path(audio_path)
    if path.suffix.lower() != ".wav":
        raise CloudAsrError("wav_required")
    data = path.read_bytes()
    if not data or len(data) > 2 * 1024 * 1024:
        raise CloudAsrError("audio_size")
    model = os.environ.get("ASR_CLOUD_MODEL", "qwen-audio-3.0-asr-flash")
    parameters = {"format": "wav", "language_hints": ["zh"]}
    if model == "qwen-audio-3.0-asr-flash" and hotwords:
        parameters["vocabulary"] = {word: 2 for word in hotwords.split()[:40]}
    payload = {
        "model": model,
        "input": {"messages": [{"role": "user", "content": [{
            "type": "input_audio", "input_audio": {
                "data": "data:audio/wav;base64," + base64.b64encode(data).decode()
            }
        }]}]},
        "parameters": parameters,
    }
    try:
        # 禁止重定向，避免 Authorization 被发送到其他服务。
        with httpx.Client(timeout=30, follow_redirects=False) as client:
            response = client.post(endpoint(), headers={"Authorization": f"Bearer {key}"}, json=payload)
        if response.status_code != 200:
            raise CloudAsrError(f"http_{response.status_code}")
        result = response.json()
        text = result.get("output", {}).get("text")
        if not isinstance(text, str) or not text.strip():
            raise CloudAsrError("empty_result")
        return text.strip()
    except CloudAsrError:
        raise
    except Exception as exc:
        raise CloudAsrError(type(exc).__name__) from None
