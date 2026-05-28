# Protocols（模块协作 JSON 协议）

本文定义两条关键接口：

- 感知模块 → LLM：`PerceptionToLLM`
- LLM → 驱动模块：`LLMToDriver`

目标：

- 支持 10 轮以上对话记忆（显式携带 `memory` 与 `turn` 元数据）
- 便于端到端评测（强制携带 `trace_id/session_id/turn_id`）
- 便于延迟控制（携带 `deadline_ms/time_budget_ms`，允许分段输出 `streaming`）

说明：

- 所有时间戳使用 ISO-8601（UTC）或 `epoch_ms` 二选一，建议两者都给。
- 所有字段默认可扩展：允许出现未定义的 `x_*` 字段，不影响解析。
- 字段命名统一 `snake_case`。

---

## 1. 通用信封（Envelope）

两条协议都使用统一信封。

```json
{
  "protocol": "perception_to_llm",
  "version": "1.0",
  "trace_id": "2a4c7d9f0f7a4c43b7d4d66f8a0b2b7e",
  "session_id": "s_20260402_000001",
  "turn_id": 12,
  "timestamp": {
    "iso_utc": "2026-04-02T10:30:45.123Z",
    "epoch_ms": 1775135445123
  },
  "locale": "zh-CN",
  "user": {
    "user_id": "u_001",
    "display_name": "用户",
    "age_range": "adult",
    "x_profile": {}
  },
  "client": {
    "device_id": "dev_001",
    "app_id": "web",
    "ip": "x.x.x.x",
    "x_meta": {}
  },
  "constraints": {
    "deadline_ms": 60000,
    "time_budget_ms": 55000,
    "safety_level": "normal"
  }
}
```

字段约束：

- `trace_id`：一次端到端链路的唯一 ID（贯穿 ASR/LLM/TTS/驱动/评测）
- `session_id`：一次用户会话 ID（用于 10+ 轮记忆）
- `turn_id`：会话内第几轮（从 1 递增）
- `constraints.deadline_ms`：本轮总截止时间（满足 LLM 响应 < 60s）

---

## 2. 感知模块 → LLM（PerceptionToLLM）

### 2.1 协议结构

```json
{
  "protocol": "perception_to_llm",
  "version": "1.0",
  "trace_id": "…",
  "session_id": "…",
  "turn_id": 12,
  "timestamp": { "iso_utc": "…", "epoch_ms": 0 },
  "locale": "zh-CN",
  "user": { "user_id": "…", "display_name": "…", "age_range": "adult", "x_profile": {} },
  "client": { "device_id": "…", "app_id": "…", "ip": "…", "x_meta": {} },
  "constraints": { "deadline_ms": 60000, "time_budget_ms": 55000, "safety_level": "normal" },

  "turn": {
    "utterance_id": "utt_20260402_000012",
    "input_mode": "voice",
    "barge_in": false,
    "vad": {
      "speech_start_ms": 120,
      "speech_end_ms": 5340
    }
  },

  "asr": {
    "text": "我最近总是睡不好，心里很烦。",
    "language": "zh-CN",
    "confidence": 0.92,
    "words": [
      { "w": "我", "start_ms": 140, "end_ms": 220, "conf": 0.98 },
      { "w": "最近", "start_ms": 221, "end_ms": 480, "conf": 0.95 }
    ],
    "wer_hint": {
      "domain": "daily",
      "hotwords": ["失眠", "焦虑", "压力"]
    }
  },

  "emotion": {
    "primary": "sad",
    "valence": -0.55,
    "arousal": 0.35,
    "confidence": 0.78,
    "signals": {
      "voice": { "enabled": true, "x_features": {} },
      "text": { "enabled": true, "x_features": {} },
      "vision": { "enabled": true, "x_features": {} }
    }
  },

  "vision": {
    "enabled": true,
    "provider": "mediapipe",
    "model": {
      "name": "face_landmarker.task",
      "with_blendshapes": true
    },
    "mode": "live_stream",
    "frame_rate_fps": 15,
    "window_ms": 1000,
    "face_count": 1,
    "features": {
      "face": {
        "landmarks_478": [
          { "i": 0, "x": 0.501, "y": 0.412, "z": -0.031, "presence": 0.99, "visibility": 0.99 }
        ],
        "blendshapes_52": [
          { "name": "mouthSmileLeft", "score": 0.12 },
          { "name": "mouthSmileRight", "score": 0.10 }
        ],
        "au_15": [
          { "name": "AU1", "intensity": 0.05 },
          { "name": "AU4", "intensity": 0.18 }
        ],
        "head_pose": { "pitch": -3.2, "yaw": 8.6, "roll": 1.1 },
        "gaze": { "x": 0.02, "y": -0.11, "z": 0.99 }
      }
    },
    "processing": {
      "au_mapping": {
        "name": "name2auweight",
        "version": "v1",
        "au_schema": {
          "source": "dataset",
          "schema_id": "dataset_au_v1",
          "au_names": ["AU1", "AU2", "AU4", "AU6", "AU7", "AU9", "AU10", "AU12", "AU14", "AU15", "AU17", "AU23", "AU24", "AU25", "AU26"],
          "va_names": ["valence", "arousal"],
          "expression_8_names": ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger", "Contempt"]
        }
      },
      "symmetry": { "enabled": true, "method": "left_right_mean" },
      "smoothing": { "enabled": true, "type": "iir_1st_order", "alpha": 0.7 }
    },
    "summary": {
      "va": { "valence": -0.3, "arousal": 0.25, "confidence": 0.6 },
      "expression_8": { "label": "Sad", "probs": { "Neutral": 0.22, "Happy": 0.02, "Sad": 0.58, "Surprise": 0.04, "Fear": 0.03, "Disgust": 0.03, "Anger": 0.04, "Contempt": 0.04 } }
    },
    "quality": {
      "tracking_state": "tracked",
      "confidence": 0.85,
      "dropped_frames": 0
    },
    "x_ext": {}
  },

  "context": {
    "environment": {
      "noise_level": "medium",
      "location": "unknown",
      "x_env": {}
    },
    "conversation_brief": {
      "last_user_utterance": "…",
      "last_assistant_utterance": "…"
    }
  },

  "memory": {
    "read": {
      "enabled": true,
      "top_k": 6,
      "query": "睡不好 心烦 失眠",
      "results": [
        {
          "memory_id": "m_000128",
          "type": "preference",
          "content": "用户不喜欢被强行建议去医院，更希望先听共情与可执行的小步骤。",
          "score": 0.81,
          "timestamp_iso_utc": "2026-03-25T09:12:03.000Z"
        }
      ]
    },
    "state": {
      "summary": "用户近期压力大，睡眠质量差，倾向先共情后建议。",
      "facts": [
        { "key": "sleep_issue", "value": true, "confidence": 0.9 },
        { "key": "topic", "value": "insomnia", "confidence": 0.7 }
      ],
      "x_memory_state": {}
    }
  },

  "attachments": [
    {
      "type": "image",
      "content_ref": "file://…/frame_00012.jpg",
      "sha256": "…",
      "caption": "可选：图像描述或 OCR 文本",
      "x_meta": {}
    }
  ],

  "x_ext": {}
}
```

### 2.2 字段说明（关键字段）

- `turn.input_mode`：`voice | text | multimodal`
- `asr.text`：最终识别文本（LLM 主输入）；如是纯文本输入则仍填 `asr.text`
- `asr.words[]`：可选，用于对齐字幕与打断（barge-in）及 WER 评测定位
- `emotion.*`：情绪理解结果；`valence`/`arousal` 建议范围 `[-1, 1]`
- `vision.*`：视觉面部特征（可选但推荐）。为了避免 payload 过大，`landmarks_478` 可仅在调试/评测时开启；生产默认建议仅传 `blendshapes_52`、`au_15`、`head_pose`、`gaze` 与 `summary`
- `vision.processing.au_mapping.au_schema`：数据集对齐口径。`au_15[].name` 必须来自 `au_names`；缺失的 AU 必须按 `intensity=0` 补齐，避免下游特征维度漂移
- `memory.read.results[]`：记忆检索结果（可由记忆服务填充，也可为空）
- `memory.state.summary`：会话级短摘要（保障 10+ 轮可控上下文）

---

## 3. LLM → 驱动模块（LLMToDriver）

驱动模块负责把 LLM 的“意图与多模态控制”落到具体执行：TTS、数字人表情/动作、UI 卡片、外部工具等。

### 3.1 协议结构

```json
{
  "protocol": "llm_to_driver",
  "version": "1.0",
  "trace_id": "…",
  "session_id": "…",
  "turn_id": 12,
  "timestamp": { "iso_utc": "…", "epoch_ms": 0 },
  "locale": "zh-CN",
  "constraints": { "deadline_ms": 60000, "time_budget_ms": 55000, "safety_level": "normal" },

  "llm": {
    "model": "your-llm-name",
    "response_id": "resp_20260402_000012",
    "latency_ms": 23850,
    "usage": { "input_tokens": 0, "output_tokens": 0 }
  },

  "assistant": {
    "text": "听起来你最近真的很辛苦，睡不好会让人更容易烦躁。你愿意说说，最近让你最累的事情是什么吗？",
    "text_ssml": null,
    "language": "zh-CN"
  },

  "policy": {
    "style": {
      "persona": "empathetic_companion",
      "tone": "warm",
      "verbosity": "medium"
    },
    "safety": {
      "risk_level": "low",
      "notices": []
    }
  },

  "render": {
    "avatar": {
      "expression": {
        "name": "concerned",
        "intensity": 0.6,
        "duration_ms": 2200
      },
      "gesture": {
        "name": "nod_slow",
        "intensity": 0.5,
        "duration_ms": 1800
      }
    },
    "voice": {
      "voice_id": "female_zh_001",
      "speed": 1.0,
      "pitch": 0.0,
      "volume": 0.0,
      "emotion": {
        "primary": "calm",
        "intensity": 0.55
      }
    },
    "ui": {
      "cards": [
        {
          "type": "quick_replies",
          "title": "你更像哪一种？",
          "items": [
            { "id": "q1", "text": "压力大睡不着" },
            { "id": "q2", "text": "总是醒很多次" },
            { "id": "q3", "text": "做噩梦影响睡眠" }
          ]
        }
      ]
    }
  },

  "actions": [
    {
      "type": "memory_write",
      "payload": {
        "items": [
          {
            "type": "episode",
            "content": "用户反馈最近睡不好，心烦。",
            "importance": 0.55,
            "ttl_days": 30,
            "privacy": "session"
          }
        ]
      }
    },
    {
      "type": "tts_speak",
      "payload": {
        "text_ref": "assistant.text",
        "interruptible": true,
        "chunking": "auto"
      }
    }
  ],

  "streaming": {
    "enabled": false,
    "stage": "final",
    "partial_index": 0
  },

  "x_ext": {}
}
```

### 3.2 动作（actions）枚举

`actions[].type` 取值建议：

- `tts_speak`：驱动 TTS 播报（可 `interruptible` 支持打断）
- `avatar_expression`：单独控制表情（也可统一走 `render.avatar`）
- `avatar_gesture`：单独控制动作
- `ui_show`：展示 UI 卡片（也可统一走 `render.ui`）
- `memory_write`：写入长期/会话记忆（支撑 10+ 轮）
- `tool_call`：调用外部工具/检索/日历/天气等（若启用）
- `handoff`：转人工/转热线/转其他模块（风险场景）
- `end_turn`：明确本轮结束（用于驱动端状态机）

动作通用字段：

```json
{
  "type": "tts_speak",
  "id": "act_001",
  "depends_on": [],
  "payload": {},
  "timeout_ms": 12000,
  "on_error": "continue"
}
```

- `depends_on`：用于编排（例如先写记忆再播报）
- `timeout_ms`：单动作超时；用于满足 `< 60s` 的总约束
- `on_error`：`continue | abort_turn`

---

## 4. 记忆写入格式（memory_write.payload.items）

```json
{
  "type": "episode",
  "content": "用户最近睡不好，心烦。",
  "importance": 0.55,
  "ttl_days": 30,
  "privacy": "session",
  "tags": ["sleep", "emotion"],
  "source": { "trace_id": "…", "turn_id": 12 }
}
```

约束：

- `importance`：`0~1`，驱动记忆服务决定是否落库/合并
- `privacy`：`session | long_term | private`
- `ttl_days`：可选；不填表示由记忆策略决定

---

## 5. 最小可用子集（MVP）

为了尽快联调，允许最小字段集：

### 5.1 感知 → LLM（MVP）

```json
{
  "protocol": "perception_to_llm",
  "version": "1.0",
  "trace_id": "…",
  "session_id": "…",
  "turn_id": 1,
  "timestamp": { "iso_utc": "…", "epoch_ms": 0 },
  "locale": "zh-CN",
  "constraints": { "deadline_ms": 60000, "time_budget_ms": 55000, "safety_level": "normal" },
  "asr": { "text": "你好，我有点难过。", "language": "zh-CN", "confidence": 0.9 },
  "memory": { "read": { "enabled": false, "top_k": 0, "query": "", "results": [] }, "state": { "summary": "", "facts": [] } }
}
```

### 5.2 LLM → 驱动（MVP）

```json
{
  "protocol": "llm_to_driver",
  "version": "1.0",
  "trace_id": "…",
  "session_id": "…",
  "turn_id": 1,
  "timestamp": { "iso_utc": "…", "epoch_ms": 0 },
  "locale": "zh-CN",
  "constraints": { "deadline_ms": 60000, "time_budget_ms": 55000, "safety_level": "normal" },
  "assistant": { "text": "我在呢。愿意和我说说发生了什么吗？", "text_ssml": null, "language": "zh-CN" },
  "actions": [{ "type": "tts_speak", "payload": { "text_ref": "assistant.text", "interruptible": true, "chunking": "auto" } }]
}
```

---

## 6. 版本与兼容

- `version` 采用语义化：`主版本.次版本`
- 解析端必须：
  - 忽略未知字段
  - 对可选字段缺省使用默认值
  - 仅在 `version` 主版本不一致时拒绝解析

