#!/usr/bin/env python3
"""
AI数字人情感陪护 全功能整合后端
端口: 8800
接口:
  WS   /ws/main           ← 主双向通道（帧/音频/驱动参数/LLM回复）
  POST /api/upload_video  ← 上传MP4文件
  GET  /api/status        ← 系统状态
  GET  /                  ← 返回 integrated.html
  GET  /shizuku/{path}    ← Live2D资产静态文件
"""
from __future__ import annotations

import asyncio
import base64
import hmac
import importlib.util
import io
import json
import os
import re
import sqlite3
import sys
import tempfile
import threading
import time
import traceback
import uuid
from datetime import date
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from urllib.parse import quote, urlsplit

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, File, Form, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from dotenv import load_dotenv

import numpy as np

from services.tts import MacOSSayProvider, Qwen3TtsProvider, TtsProviderError
from services.asr import cloud_provider as cloud_asr

load_dotenv(ROOT / ".env", override=False)

# ── 线程池（CPU密集型推理用）──────────────────────────────────
_executor = ThreadPoolExecutor(max_workers=4)

# ══════════════════════════════════════════════════════════════
# 用户认证与历史会话数据库（SQLite）
# ══════════════════════════════════════════════════════════════
DB_PATH = ROOT / "data" / "users.db"

try:
    import bcrypt as _bcrypt
    HAS_BCRYPT = True
except ImportError:
    HAS_BCRYPT = False
    print("[Auth] bcrypt 未安装，已禁用密码认证与注册")

def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def _init_db():
    """初始化数据库表结构"""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = _get_db()
    try:
        c = conn.cursor()
        c.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS auth_tokens (
                token TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                expires_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                title TEXT DEFAULT '新对话',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                emotion_label TEXT DEFAULT '',
                ts TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS user_memory_settings (
                user_id INTEGER PRIMARY KEY,
                enabled INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS user_memories (
                id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                content TEXT NOT NULL,
                category TEXT NOT NULL DEFAULT 'context',
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS agent_runs (
                trace_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                user_id INTEGER NOT NULL,
                provider TEXT NOT NULL,
                risk_level TEXT NOT NULL,
                emotion TEXT NOT NULL,
                execution_path TEXT NOT NULL,
                tool_calls TEXT NOT NULL,
                node_timings_ms TEXT NOT NULL,
                total_latency_ms REAL NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS activity_tasks (
                id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                client_id TEXT NOT NULL,
                template_id TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'started',
                created_at TEXT NOT NULL,
                completed_at TEXT,
                removed_at TEXT,
                UNIQUE(user_id,client_id)
            );
            CREATE INDEX IF NOT EXISTS idx_activity_tasks_user ON activity_tasks(user_id,created_at);
            CREATE INDEX IF NOT EXISTS idx_sessions_user ON chat_sessions(user_id);
            CREATE INDEX IF NOT EXISTS idx_messages_session ON chat_messages(session_id);
            CREATE INDEX IF NOT EXISTS idx_tokens_user ON auth_tokens(user_id);
            CREATE INDEX IF NOT EXISTS idx_memories_user ON user_memories(user_id);
            CREATE INDEX IF NOT EXISTS idx_agent_runs_user ON agent_runs(user_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_agent_runs_session ON agent_runs(session_id, created_at);
        """)
        columns = {row[1] for row in conn.execute("PRAGMA table_info(activity_tasks)")}
        if "removed_at" not in columns:
            conn.execute("ALTER TABLE activity_tasks ADD COLUMN removed_at TEXT")
        user_columns = {row[1] for row in conn.execute("PRAGMA table_info(users)")}
        message_columns = {row[1] for row in conn.execute("PRAGMA table_info(chat_messages)")}
        session_columns = {row[1] for row in conn.execute("PRAGMA table_info(chat_sessions)")}
        session_additions = {
            "is_main": "INTEGER NOT NULL DEFAULT 0",
            "dialogue_mode": "TEXT NOT NULL DEFAULT 'daily'",
            "emotion_style": "TEXT NOT NULL DEFAULT 'confidant'",
            "title_manual": "INTEGER NOT NULL DEFAULT 0",
        }
        for name, definition in session_additions.items():
            if name not in session_columns:
                conn.execute(f"ALTER TABLE chat_sessions ADD COLUMN {name} {definition}")
        # 可重复迁移：每个账号只保留一个主对话。优先选择最近更新且有消息的历史。
        user_ids = [row[0] for row in conn.execute("SELECT id FROM users")]
        for user_id in user_ids:
            mains = conn.execute(
                "SELECT id FROM chat_sessions WHERE user_id=? AND is_main=1 ORDER BY updated_at DESC, rowid DESC",
                (user_id,),
            ).fetchall()
            if mains:
                keep = mains[0][0]
                conn.execute("UPDATE chat_sessions SET is_main=0 WHERE user_id=? AND id<>?", (user_id, keep))
                continue
            candidate = conn.execute(
                """SELECT s.id FROM chat_sessions s
                   WHERE s.user_id=? AND EXISTS(SELECT 1 FROM chat_messages m WHERE m.session_id=s.id)
                   ORDER BY s.updated_at DESC, s.rowid DESC LIMIT 1""",
                (user_id,),
            ).fetchone()
            if candidate:
                conn.execute("UPDATE chat_sessions SET is_main=1 WHERE id=?", (candidate[0],))
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_sessions_one_main ON chat_sessions(user_id) WHERE is_main=1")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_page ON chat_sessions(user_id,is_main DESC,updated_at DESC,id DESC)")
        if 'knowledge_sources' not in message_columns:
            conn.execute("ALTER TABLE chat_messages ADD COLUMN knowledge_sources TEXT NOT NULL DEFAULT '[]'")
        for name in ("display_name", "birthday", "avatar"):
            if name not in user_columns:
                conn.execute(f"ALTER TABLE users ADD COLUMN {name} TEXT NOT NULL DEFAULT ''")
        from apps.api.external_auth import init_auth_tables
        init_auth_tables(conn)
        from apps.api.admin_dashboard import init_admin_tables
        init_admin_tables(conn)
        from services.agent.session_notes import init_session_notes
        init_session_notes(conn)
        from apps.api.account_lifecycle import install_owner_guards
        install_owner_guards(conn)
        conn.commit()
        print("[DB] 数据库初始化完成")
    finally:
        conn.close()

def _hash_password(pwd: str) -> str:
    if HAS_BCRYPT:
        return _bcrypt.hashpw(pwd.encode(), _bcrypt.gensalt()).decode()
    raise RuntimeError("bcrypt 不可用，禁止不安全密码存储")

def _check_password(pwd: str, hashed: str) -> bool:
    if HAS_BCRYPT:
        try:
            return _bcrypt.checkpw(pwd.encode(), hashed.encode())
        except Exception:
            return False
    return False

def _verify_auth_token(token: str) -> Optional[int]:
    """验证 token，返回 user_id 或 None"""
    if not token:
        return None
    conn = _get_db()
    try:
        row = conn.execute(
            "SELECT user_id, expires_at FROM auth_tokens WHERE token=?", (token,)
        ).fetchone()
        if not row:
            return None
        if row["expires_at"] < time.strftime("%Y-%m-%dT%H:%M:%S"):
            conn.execute("DELETE FROM auth_tokens WHERE token=?", (token,))
            conn.commit()
            return None
        return row["user_id"]
    finally:
        conn.close()

def _db_save_message(session_id: str, role: str, content: str, emotion_label: str = "", knowledge_sources=None, memory_revision=None):
    """持久化一条消息到数据库"""
    conn = None
    try:
        conn = _get_db()
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        from services.agent.references import reference_snapshot
        references=reference_snapshot(knowledge_sources) if role=='assistant' else []
        conn.execute(
            "INSERT INTO chat_messages(session_id,role,content,emotion_label,ts,knowledge_sources,memory_revision) VALUES(?,?,?,?,?,?,?)",
            (session_id, role, content, emotion_label, now, json.dumps(references,ensure_ascii=False),memory_revision)
        )
        conn.execute("UPDATE chat_sessions SET updated_at=? WHERE id=?", (now, session_id))
        conn.commit()
    except Exception as e:
        print(f"[DB] 保存消息失败: {e}")
    finally:
        if conn is not None:
            conn.close()


def _db_load_message_context(session_id: str, limit: int = 40) -> list[dict[str, str]]:
    """按时间顺序加载最近的对话上下文。调用前必须完成会话归属校验。"""
    conn = _get_db()
    try:
        rows = conn.execute(
            """SELECT role, content FROM (
                   SELECT id, role, content FROM chat_messages
                   WHERE session_id=? AND role IN ('user', 'assistant')
                   ORDER BY id DESC LIMIT ?
               ) ORDER BY id ASC""",
            (session_id, limit),
        ).fetchall()
        return [{"role": row["role"], "content": row["content"]} for row in rows]
    finally:
        conn.close()


def _normalize_session_title(value: str) -> str:
    title = re.sub(r"[\s\r\n\t，。！？、,:：;；'\"“”‘’《》【】]+", "", value or "")[:16]
    if len(title) < 8:
        title = (title + "相关话题记录")[:16]
    return title or "新的对话记录"


def _memory_enabled_for_user(user_id: int) -> bool:
    conn = _get_db()
    try:
        row = conn.execute(
            "SELECT enabled FROM user_memory_settings WHERE user_id=?", (user_id,)
        ).fetchone()
        return bool(row["enabled"]) if row else False
    finally:
        conn.close()


def _db_save_agent_run(result: dict[str, Any], user_id: int, provider: str) -> None:
    """保存脱敏运行摘要；不记录用户原文和模型回复。"""
    try:
        timings = result.get("node_timings_ms", {})
        conn = _get_db()
        try:
            conn.execute(
                """INSERT OR REPLACE INTO agent_runs(
                       trace_id,session_id,user_id,provider,risk_level,emotion,
                       execution_path,tool_calls,node_timings_ms,total_latency_ms,created_at
                   ) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    result["trace_id"],
                    result["session_id"],
                    user_id,
                    provider,
                    result["safety"].risk_level.value,
                    result["emotion_context"].emotion,
                    json.dumps(result.get("execution_path", []), ensure_ascii=False),
                    json.dumps(
                        [item.model_dump(mode="json") for item in result.get("tool_calls", [])],
                        ensure_ascii=False,
                    ),
                    json.dumps(timings, ensure_ascii=False),
                    round(sum(float(value) for value in timings.values()), 3),
                    time.strftime("%Y-%m-%dT%H:%M:%S"),
                ),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception as exc:
        print(f"[DB] 保存智能体运行摘要失败: {type(exc).__name__}")

# 初始化数据库
_init_db()

# ── 导入 emotion_to_live2d 映射（数字人表情驱动）────────────────
HAS_EMOTION_MAP = False
try:
    _emo_map_path = str(ROOT / "digital_human_engine" / "train.jioaben")
    if _emo_map_path not in sys.path:
        sys.path.insert(0, _emo_map_path)
    from emotion_to_live2d import (
        emotion25_to_live2d as _emo25_to_live2d,
        live2d_to_dict as _live2d_to_dict,
        PARAM_NAMES as _PARAM_NAMES_24,
        EXP_INDICES, VA_VALENCE, VA_AROUSAL, AU_INDICES,
    )
    HAS_EMOTION_MAP = True
    print("[IntegratedServer] emotion_to_live2d 映射加载成功")
except Exception as e:
    print(f"[IntegratedServer] emotion_to_live2d 不可用: {e}")

# ══════════════════════════════════════════════════════════════
# 模块懒加载（允许部分模块缺失时降级运行）
# ══════════════════════════════════════════════════════════════

# 1. 正式情感反应 Transformer（数字人面部行为驱动）
_face_driver = None
_face_driver_lock = asyncio.Lock()
HAS_DRIVER = False
try:
    from services.avatar.emotion_reaction_model import EmotionReactionModel, torch
    HAS_DRIVER = True
    if torch is None:
        HAS_DRIVER = False
        raise ImportError("PyTorch 未安装")
    print("[IntegratedServer] 正式情感反应模型运行时可用")
except Exception as e:
    print(f"[IntegratedServer] torch不可用，驱动模型降级: {e}")

LIVE2D_PARAMS = {
    'PARAM_ANGLE_X':     (-30, 30),
    'PARAM_ANGLE_Y':     (-30, 30),
    'PARAM_ANGLE_Z':     (-30, 30),
    'PARAM_EYE_L_OPEN':  (0, 1),
    'PARAM_EYE_R_OPEN':  (0, 1),
    'PARAM_EYE_BALL_X':  (-1, 1),
    'PARAM_EYE_BALL_Y':  (-1, 1),
    'PARAM_BROW_L_Y':    (-1, 1),
    'PARAM_BROW_R_Y':    (-1, 1),
    'PARAM_BROW_L_X':    (-1, 1),
    'PARAM_BROW_R_X':    (-1, 1),
    'PARAM_BROW_L_ANGLE':(-1, 1),
    'PARAM_BROW_R_ANGLE':(-1, 1),
    'PARAM_MOUTH_OPEN_Y':(0, 1),
    'PARAM_MOUTH_FORM':  (-1, 1),
    'PARAM_BODY_ANGLE_X':(-10, 10),
    'PARAM_BODY_ANGLE_Y':(-10, 10),
    'PARAM_BODY_ANGLE_Z':(-10, 10),
    'PARAM_BREATH':      (0, 1),
}

def _load_face_driver():
    """同步加载驱动模型，在线程池中执行"""
    global _face_driver
    if not HAS_DRIVER:
        return None
    if _face_driver is not None:
        return _face_driver
    try:
        _face_driver = EmotionReactionModel()
        metadata = _face_driver.metadata
        print(
            f"[Driver] 正式模型加载成功，epoch={metadata.epoch}, "
            f"val_loss={metadata.val_loss:.4f}, 设备={metadata.device}"
        )
        return _face_driver
    except Exception as e:
        print(f"[Driver] 模型加载失败: {e}")
        return None


def _driver_runtime_payload(reaction_model) -> Dict[str, Any]:
    """构建前端可展示的正式模型运行状态。"""
    if reaction_model is None:
        return {
            "ready": False,
            "mode": "fallback",
            "epoch": None,
            "device": None,
            "message": "正式模型未加载，使用规则驱动",
        }
    metadata = reaction_model.metadata
    fallback_reason = getattr(metadata, "device_fallback_reason", None)
    return {
        "ready": True,
        "mode": "official",
        "epoch": metadata.epoch,
        "device": metadata.device,
        "device_fallback": bool(fallback_reason),
        "message": (
            f"正式模型已加载（第 {metadata.epoch} 轮，MPS 不兼容算子已切换 CPU）"
            if fallback_reason
            else f"正式模型已加载（第 {metadata.epoch} 轮）"
        ),
    }


def _infer_live2d_params(
    reaction_model, emotion_seq: np.ndarray, intensity: float
) -> Optional[Dict[str, float]]:
    """用正式模型预测倾听者情绪，并映射为 Live2D 参数。"""
    try:
        prediction = reaction_model.predict(emotion_seq, num_candidates=1, seed=42)
        emotion_25 = prediction[0, -1]
        params = _emo25_to_live2d(emotion_25, intensity=intensity)
        return _live2d_to_dict(params)
    except Exception as e:
        print(f"[Driver] 正式模型推理失败: {type(e).__name__}: {e}")
        return None

def _default_live2d_params() -> Dict[str, float]:
    """返回默认静止姿态参数"""
    defaults = {'PARAM_EYE_L_OPEN': 1.0, 'PARAM_EYE_R_OPEN': 1.0}
    return {k: defaults.get(k, 0.0) for k in LIVE2D_PARAMS}

# 2. MediaPipe 视觉提取（简化版，直接用 face_mesh）
HAS_MEDIAPIPE = False
try:
    import mediapipe as mp
    _mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )
    HAS_MEDIAPIPE = True
    print("[IntegratedServer] MediaPipe 加载成功")
except Exception as e:
    print(f"[IntegratedServer] MediaPipe不可用: {e}")
    _mp_face_mesh = None

def _extract_features_from_bgr(frame_bgr: np.ndarray) -> np.ndarray:
    """从BGR帧提取25维特征向量（与 FaceBehaviorModel 输入对齐）"""
    if not HAS_MEDIAPIPE or _mp_face_mesh is None:
        return np.zeros(25, dtype=np.float32)
    try:
        import cv2
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = _mp_face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return np.zeros(25, dtype=np.float32)
        lm = results.multi_face_landmarks[0].landmark
        features = np.zeros(25, dtype=np.float32)

        def dist(a, b):
            return float(np.sqrt((lm[a].x - lm[b].x)**2 + (lm[a].y - lm[b].y)**2))

        def eye_aspect(upper_ids, lower_ids, corner_l, corner_r):
            v = sum(dist(u, l) for u, l in zip(upper_ids, lower_ids)) / len(upper_ids)
            h = dist(corner_l, corner_r) + 1e-6
            return min(max(v / h * 2.5, 0), 1)

        # 0: 左眼开合
        features[0] = eye_aspect([159, 160, 161], [145, 144, 153], 33, 133)
        # 1: 右眼开合
        features[1] = eye_aspect([386, 385, 384], [374, 373, 380], 362, 263)
        # 2: 左眉高度
        features[2] = min(max((lm[105].y - lm[70].y) * 12 + 0.5, 0), 1)
        # 3: 右眉高度
        features[3] = min(max((lm[334].y - lm[300].y) * 12 + 0.5, 0), 1)
        # 4: 微笑程度
        features[4] = blendshape_smile_approx(lm)
        # 5: 左眼球X方向
        features[5] = float(np.clip((lm[468].x - lm[33].x) / (dist(33, 133) + 1e-6) * 2 - 1, -1, 1)) if len(lm) > 468 else 0
        # 6: 左眼球Y方向
        features[6] = float(np.clip((lm[468].y - lm[159].y) / (dist(159, 145) + 1e-6) * 2 - 1, -1, 1)) if len(lm) > 468 else 0
        # 7: 右眼球X方向
        features[7] = float(np.clip((lm[473].x - lm[362].x) / (dist(362, 263) + 1e-6) * 2 - 1, -1, 1)) if len(lm) > 473 else 0
        # 8: 右眼球Y方向
        features[8] = float(np.clip((lm[473].y - lm[386].y) / (dist(386, 374) + 1e-6) * 2 - 1, -1, 1)) if len(lm) > 473 else 0
        # 9-12: 眉毛角度近似
        features[9] = float(np.clip((lm[70].y - lm[107].y) * 15, -1, 1))   # 左眉内侧
        features[10] = float(np.clip((lm[300].y - lm[336].y) * 15, -1, 1)) # 右眉内侧
        features[11] = float(np.clip((lm[46].y - lm[70].y) * 15, -1, 1))   # 左眉外侧
        features[12] = float(np.clip((lm[276].y - lm[300].y) * 15, -1, 1)) # 右眉外侧
        # 13: 嘴开合
        mouth_h = dist(13, 14)
        mouth_w = dist(61, 291) + 1e-6
        features[13] = min(max(mouth_h / mouth_w * 2.5, 0), 1)
        # 14: 嘴型（微笑vs噘嘴）
        features[14] = float(np.clip(features[4] * 2 - 0.5, -1, 1))
        # 15: pitch（头部上下）
        features[15] = float(np.clip((lm[10].y - lm[152].y) * 3 - 0.5, -1, 1))
        # 16: yaw（头部左右）
        features[16] = float(np.clip((lm[454].x - lm[234].x) * 3 - 0.5, -1, 1))
        # 17: roll（头部侧倾）
        features[17] = float(np.clip((lm[234].y - lm[454].y) * 5, -1, 1))
        # 18-20: 鼻尖位置（身体角度近似）
        features[18] = features[16] * 0.3  # 身体跟随头部X
        features[19] = features[15] * 0.3  # 身体跟随头部Y
        features[20] = features[17] * 0.2  # 身体跟随头部Z
        # 21-24: 预留
        features[21] = 0  # 呼吸（由前端处理）
        features[22] = float(np.clip(dist(61, 291) * 5, 0, 1))  # 嘴宽
        features[23] = float(np.clip((lm[17].y - lm[0].y) * 8, -1, 1))  # 下巴
        features[24] = 0
        return features
    except Exception:
        return np.zeros(25, dtype=np.float32)

def blendshape_smile_approx(lm) -> float:
    """近似计算微笑程度"""
    try:
        left_mouth = lm[61]
        right_mouth = lm[291]
        top_lip = lm[0]
        smile = abs(left_mouth.y - top_lip.y) + abs(right_mouth.y - top_lip.y)
        return float(min(smile * 3, 1.0))
    except Exception:
        return 0.0

# 3. FunASR
_asr_model = None
HAS_ASR = False
ASR_DEPENDENCY_AVAILABLE = importlib.util.find_spec("funasr") is not None
ASR_LAST_ERROR: str | None = None
_asr_inference_lock = threading.Lock()

def _get_asr_model():
    global _asr_model, HAS_ASR, ASR_LAST_ERROR
    if not ASR_DEPENDENCY_AVAILABLE:
        ASR_LAST_ERROR = "dependency_missing"
        return None
    if _asr_model is None:
        try:
            os.environ.setdefault("MODELSCOPE_CACHE", str(ROOT / "models" / "asr"))
            from funasr import AutoModel
            _asr_model = AutoModel(
                model=os.environ.get("ASR_MODEL", "paraformer-zh"),
                vad_model="fsmn-vad",
                punc_model="ct-punc",
                device=os.environ.get("ASR_DEVICE", "cpu"),
                disable_update=True,
            )
            HAS_ASR = True
            ASR_LAST_ERROR = None
            print("[IntegratedServer] FunASR AutoModel 初始化成功")
        except Exception as e:
            print(f"[IntegratedServer] FunASR不可用: {e}")
            _asr_model = None
            HAS_ASR = False
            ASR_LAST_ERROR = type(e).__name__
    return _asr_model


def _asr_status_payload() -> Dict[str, Any]:
    if os.environ.get("ASR_PROVIDER", "funasr") == "qwen":
        configured = bool(cloud_asr.api_key())
        return {"available": configured, "ready": False, "provider": "qwen_cloud",
                "reason": None if configured else "key_missing",
                "message": "百炼云端ASR已配置，录音将发送到阿里云" if configured else "未配置ASR密钥"}
    if not ASR_DEPENDENCY_AVAILABLE:
        return {
            "available": False,
            "ready": False,
            "provider": "browser_speech_recognition",
            "reason": "dependency_missing",
            "message": "FunASR 依赖未安装，使用浏览器语音识别",
        }
    if HAS_ASR and _asr_model is not None:
        return {
            "available": True,
            "ready": True,
            "provider": "funasr_paraformer",
            "reason": None,
            "message": "FunASR 服务端语音识别已就绪",
        }
    return {
        "available": True,
        "ready": False,
        "provider": "funasr_paraformer",
        "reason": ASR_LAST_ERROR,
        "message": (
            f"FunASR 初始化失败：{ASR_LAST_ERROR}"
            if ASR_LAST_ERROR
            else "FunASR 将在首次录音时加载"
        ),
    }

def _recognize_audio(audio_path):
    if os.environ.get("ASR_PROVIDER", "funasr") != "qwen":
        return _run_asr(audio_path), "funasr_paraformer", None
    try:
        return cloud_asr.transcribe(audio_path, _asr_hotwords()), "qwen_cloud", None
    except cloud_asr.CloudAsrError as exc:
        reason = str(exc)
        if os.environ.get("ASR_FALLBACK_LOCAL", "true").lower() == "true" and ASR_DEPENDENCY_AVAILABLE:
            return _run_asr(audio_path), "funasr_paraformer", reason
        raise


def _asr_hotwords() -> str:
    """只传递有限的词语，避免 FunASR 将配置误解释为路径或 URL。"""
    words = re.split(r"[,，;；\s]+", os.environ.get("ASR_HOTWORDS", "小安 数字心屿"))
    return " ".join(dict.fromkeys(
        word for word in words if re.fullmatch(r"[\w\u4e00-\u9fff]{1,20}", word)
    ))[:400]


def _run_asr(audio_path: str) -> str:
    """同步运行ASR，在线程池中执行"""
    with _asr_inference_lock:
        model = _get_asr_model()
        if model is None:
            return ""
        try:
            # 合并短停顿的 VAD 片段，为识别保留句内上下文；热词只影响声学解码，
            # 不用 LLM 改写转录，以免改变用户的否定词、情绪和风险表达。
            res = model.generate(
                input=audio_path, batch_size_s=30,
                merge_vad=True, merge_length_s=15,
                hotword=_asr_hotwords(),
            )
            if res and isinstance(res, list):
                return "".join(str(item.get("text", "")) for item in res if isinstance(item, dict)).strip()
            return ""
        except Exception as e:
            print(f"[ASR] 识别失败: {type(e).__name__}")
            return ""

# 4. 大模型与语音 API。专用密钥为空时回退到共用 DashScope 密钥。
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "").strip()
DEEPSEEK_API_KEY = (
    os.environ.get("DEEPSEEK_API_KEY", "").strip()
    or os.environ.get("LLM_API_KEY", "").strip()
)
QWEN_API_KEY = os.environ.get("QWEN_API_KEY", "").strip() or DASHSCOPE_API_KEY
TTS_API_KEY = os.environ.get("TTS_API_KEY", "").strip() or DASHSCOPE_API_KEY
TTS_PROVIDER = os.environ.get("TTS_PROVIDER", "auto").strip().lower()
TTS_DEFAULT_VOICE = os.environ.get("TTS_DEFAULT_VOICE", "").strip()
TTS_QWEN3_MODEL = os.environ.get("TTS_QWEN3_MODEL", "qwen3-tts-instruct-flash").strip()
try:
    TTS_RATE = min(max(int(os.environ.get("TTS_RATE", "185")), 120), 260)
except ValueError:
    TTS_RATE = 185
DEEPSEEK_BASE_URL = os.environ.get(
    "DEEPSEEK_BASE_URL", "https://api.deepseek.com"
).rstrip("/")
DEEPSEEK_MODEL = (
    os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash").strip()
    or "deepseek-v4-flash"
)
QWEN_BASE_URL = os.environ.get(
    "QWEN_BASE_URL",
    os.environ.get(
        "LLM_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ),
).rstrip("/")
QWEN_MODEL = (
    os.environ.get("QWEN_MODEL", "").strip()
    or os.environ.get("LLM_MODEL", "").strip()
    or "qwen-plus"
)
RAG_EMBEDDING_MODEL_PATH = os.environ.get("RAG_EMBEDDING_MODEL_PATH", "").strip()
LIVETALKING_BASE_URL = os.environ.get("LIVETALKING_BASE_URL", "").strip().rstrip("/")
LIVETALKING_AVATAR_ID = os.environ.get("LIVETALKING_AVATAR_ID", "wav2lip256_avatar1").strip()


def _validated_livetalking_base_url() -> str:
    """只接受由部署者配置的 HTTP(S) 服务源，避免把接口变成任意 URL 代理。"""
    if not LIVETALKING_BASE_URL:
        return ""
    parsed = urlsplit(LIVETALKING_BASE_URL)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        return ""
    return LIVETALKING_BASE_URL

# ══════════════════════════════════════════════════════════════
# FastAPI 应用
# ══════════════════════════════════════════════════════════════
app = FastAPI(title="AI数字人情感陪护全功能整合", version="2.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=[x.strip() for x in os.getenv("ALLOWED_ORIGINS", "http://127.0.0.1:8801,http://localhost:8801").split(",") if x.strip()],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)
from apps.api.security import install_http_security, allow_request
from services.vision.observation import CameraObservation
def _public_deployment():
    return os.getenv("PUBLIC_DEPLOYMENT","false").lower()=="true"
install_http_security(app,_verify_auth_token,_public_deployment)

# 托管 MediaPipe 本地文件（避免 CDN 访问不稳定）
MEDIAPIPE_STATIC_DIR = ROOT / "static" / "mediapipe"
@app.get('/api/vision/face-landmarker-model')
async def browser_face_landmarker_model():
    return FileResponse(ROOT / 'models' / 'face_landmarker.task', media_type='application/octet-stream')
if MEDIAPIPE_STATIC_DIR.exists():
    app.mount("/static/mediapipe", StaticFiles(directory=str(MEDIAPIPE_STATIC_DIR)), name="mediapipe-static")
    print(f"[IntegratedServer] MediaPipe 本地静态资源挂载: {MEDIAPIPE_STATIC_DIR}")

# 项目介绍页使用的作者头像等轻量品牌资源。
CREATOR_STATIC_DIR = ROOT / "static" / "creator"
if CREATOR_STATIC_DIR.exists():
    app.mount("/static/creator", StaticFiles(directory=str(CREATOR_STATIC_DIR)), name="creator-static")

# 托管 Live2D SDK JS 文件（本地加载，无需CDN）
LIVE2D_JS_DIR = ROOT / "digital_human_engine" / "live2d_web" / "js"
if LIVE2D_JS_DIR.exists():
    app.mount("/live2d-js", StaticFiles(directory=str(LIVE2D_JS_DIR)), name="live2d-js")
    print(f"[IntegratedServer] Live2D SDK JS 挂载: {LIVE2D_JS_DIR}")

# 托管 Live2D 模型资产（来自 digital_human_engine）
SHIZUKU_DIR = ROOT / "digital_human_engine" / "live2d_web" / "shizuku"
if not SHIZUKU_DIR.exists():
    SHIZUKU_DIR = ROOT / "shizuku"  # 兼容旧路径
if SHIZUKU_DIR.exists():
    app.mount("/shizuku", StaticFiles(directory=str(SHIZUKU_DIR)), name="shizuku")
    print(f"[IntegratedServer] Live2D资产挂载: {SHIZUKU_DIR}")
else:
    print("[IntegratedServer] 警告: shizuku 目录未找到")

INTEGRATED_HTML = ROOT / "integrated.html"
VISION_DEMO_HTML = ROOT / "vision_demo.html"

@app.get("/")
async def root():
    if INTEGRATED_HTML.exists():
        return FileResponse(
            str(INTEGRATED_HTML),
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            }
        )
    return HTMLResponse("<h1>integrated.html 未找到</h1>", status_code=404)

@app.get("/vision")
async def vision_demo():
    """视觉模块实时监控页面"""
    if VISION_DEMO_HTML.exists():
        return FileResponse(str(VISION_DEMO_HTML))
    return HTMLResponse("<h1>vision_demo.html 未找到</h1>", status_code=404)

@app.get("/api/status")
async def api_status(request: Request):
    from services.avatar.model_assets import inspect_face_driver_checkpoint

    checkpoint_status = inspect_face_driver_checkpoint()
    return JSONResponse({
        "status": "running",
        "modules": {
            "vision_mediapipe": HAS_MEDIAPIPE,
            "asr_funasr": ASR_DEPENDENCY_AVAILABLE,
            "driver_model": HAS_DRIVER and checkpoint_status.ready,
            "deepseek_api": bool(DEEPSEEK_API_KEY),
            "qwen_api": bool(QWEN_API_KEY),
            "tts_cosyvoice": HAS_TTS and bool(TTS_API_KEY),
            "tts_qwen3": bool(TTS_API_KEY),
            "agent_provider": _configured_agent_provider_name(),
        },
        "tts": _tts_status_payload(),
        "asr": _asr_status_payload(),
        "model_assets": {
            "face_driver": checkpoint_status.to_public_dict(),
        },
        "port": request.scope.get("server", (None, None))[1],
    })


class LiveTalkingOfferRequest(BaseModel):
    sdp: str = Field(min_length=1, max_length=200_000)
    type: str = Field(pattern="^offer$")
    avatar: str | None = Field(default=None, max_length=128)


class LiveTalkingSessionRequest(BaseModel):
    sessionid: str = Field(min_length=1, max_length=128)


def _livetalking_unavailable():
    return JSONResponse({
        "error": "livetalking_unavailable",
        "message": "尚未配置 LiveTalking 服务；请设置 LIVETALKING_BASE_URL 并重启后端。",
    }, status_code=503)


@app.get("/api/avatar/catalog")
async def api_avatar_catalog():
    base_url = _validated_livetalking_base_url()
    return JSONResponse({"avatars": [
        {"id": "live2d", "name": "小安 · Live2D", "available": True, "transport": "canvas"},
        {"id": "wav2lip", "name": "播报员 · Wav2Lip", "available": bool(base_url),
         "transport": "webrtc", "avatar_id": LIVETALKING_AVATAR_ID or "wav2lip256_avatar1",
         "reason": None if base_url else "service_not_configured"},
    ]})


@app.post("/api/avatar/livetalking/offer")
async def api_livetalking_offer(payload: LiveTalkingOfferRequest):
    base_url = _validated_livetalking_base_url()
    if not base_url:
        return _livetalking_unavailable()
    import httpx
    upstream = {"sdp": payload.sdp, "type": payload.type,
                "avatar": payload.avatar or LIVETALKING_AVATAR_ID or "wav2lip256_avatar1"}
    try:
        async with httpx.AsyncClient(timeout=35.0, follow_redirects=False) as client:
            response = await client.post(f"{base_url}/offer", json=upstream)
            response.raise_for_status()
            answer = response.json()
        if not isinstance(answer, dict) or not answer.get("sdp") or not answer.get("sessionid"):
            raise ValueError("invalid offer response")
        return JSONResponse({"sdp": answer["sdp"], "type": answer.get("type", "answer"),
                             "sessionid": str(answer["sessionid"])})
    except (httpx.HTTPError, ValueError, json.JSONDecodeError) as error:
        print(f"[LiveTalking] WebRTC 协商失败: {type(error).__name__}")
        return JSONResponse({"error": "livetalking_offer_failed", "message": "Wav2Lip 服务连接失败"}, status_code=502)


@app.post("/api/avatar/livetalking/audio")
async def api_livetalking_audio(sessionid: str = Form(...), file: UploadFile = File(...)):
    base_url = _validated_livetalking_base_url()
    if not base_url:
        return _livetalking_unavailable()
    if not sessionid or len(sessionid) > 128:
        return JSONResponse({"error": "invalid_session"}, status_code=400)
    audio = await file.read(20 * 1024 * 1024 + 1)
    if not audio or len(audio) > 20 * 1024 * 1024:
        return JSONResponse({"error": "invalid_audio"}, status_code=400)
    import httpx
    try:
        async with httpx.AsyncClient(timeout=45.0, follow_redirects=False) as client:
            response = await client.post(
                f"{base_url}/humanaudio",
                data={"sessionid": sessionid},
                files={"file": (file.filename or "speech.wav", audio, file.content_type or "audio/wav")},
            )
            response.raise_for_status()
            result = response.json()
        return JSONResponse(result if isinstance(result, dict) else {"code": 0, "msg": "ok"})
    except (httpx.HTTPError, ValueError, json.JSONDecodeError) as error:
        print(f"[LiveTalking] 音频驱动失败: {type(error).__name__}")
        return JSONResponse({"error": "livetalking_audio_failed", "message": "Wav2Lip 音频驱动失败"}, status_code=502)


async def _livetalking_session_command(path: str, sessionid: str):
    base_url = _validated_livetalking_base_url()
    if not base_url:
        return _livetalking_unavailable()
    import httpx
    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=False) as client:
            response = await client.post(f"{base_url}/{path}", json={"sessionid": sessionid})
            response.raise_for_status()
            result = response.json()
        return JSONResponse(result if isinstance(result, dict) else {"code": 0, "msg": "ok"})
    except (httpx.HTTPError, ValueError, json.JSONDecodeError):
        return JSONResponse({"error": f"livetalking_{path}_failed"}, status_code=502)


@app.post("/api/avatar/livetalking/interrupt")
async def api_livetalking_interrupt(payload: LiveTalkingSessionRequest):
    return await _livetalking_session_command("interrupt_talk", payload.sessionid)


@app.post("/api/avatar/livetalking/speaking")
async def api_livetalking_speaking(payload: LiveTalkingSessionRequest):
    return await _livetalking_session_command("is_speaking", payload.sessionid)


@app.get('/api/health/ready')
async def readiness():
    if not HAS_BCRYPT:
        return JSONResponse({'ready':False,'reason':'password_dependency_unavailable'},status_code=503)
    try:
        with _get_db() as conn: conn.execute('SELECT 1 FROM users LIMIT 1')
    except sqlite3.Error:
        return JSONResponse({'ready':False,'reason':'database_unavailable'},status_code=503)
    return JSONResponse({'ready':True})


class AgentChatRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    session_id: str | None = None


class MemorySettingsRequest(BaseModel):
    enabled: bool


_agent_workflow = None
_agent_provider_name = "offline"
_agent_knowledge_provider_name = "uninitialized"


def _configured_agent_provider_name() -> str:
    if DEEPSEEK_API_KEY and QWEN_API_KEY:
        return "deepseek_with_qwen_fallback"
    if DEEPSEEK_API_KEY:
        return "deepseek_with_offline_fallback"
    if QWEN_API_KEY:
        return "qwen_with_offline_fallback"
    return "offline"


def _get_agent_workflow():
    global _agent_knowledge_provider_name, _agent_provider_name, _agent_workflow
    if _agent_workflow is None:
        from services.agent import (
            DigitalXinyuWorkflow,
            SQLiteMemoryStore,
            create_companion_provider,
        )

        provider, _agent_provider_name = create_companion_provider(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
            model=DEEPSEEK_MODEL,
            provider_name="deepseek",
            fallback_api_key=QWEN_API_KEY,
            fallback_base_url=QWEN_BASE_URL,
            fallback_model=QWEN_MODEL,
            fallback_name="qwen",
            extra_body={"thinking": {"type": "disabled"}},
        )
        knowledge_retriever, _agent_knowledge_provider_name = (
            _get_rag_search_retriever()
        )
        _agent_workflow = DigitalXinyuWorkflow(
            provider=provider,
            knowledge_retriever=knowledge_retriever,
            memory_store=SQLiteMemoryStore(_get_db),
        )
    return _agent_workflow


@app.post("/api/agent/chat")
async def agent_chat(payload: AgentChatRequest, request: Request):
    """使用离线 Provider 运行一次可观测的智能体工作流。"""
    from services.agent.usage import begin_usage_collection, finish_usage_collection
    usage_token = begin_usage_collection()
    usage_saved = False
    trace_id = str(uuid.uuid4())
    session_id = payload.session_id or str(uuid.uuid4())
    user_id = _get_user_id_from_request(request)
    try:
        result = await _get_agent_workflow().run(
            user_text=payload.text,
            trace_id=trace_id,
            session_id=session_id,
        )
        usage_rows = finish_usage_collection(usage_token)
        usage_token = None
        if user_id is not None:
            from apps.api.admin_dashboard import save_model_usage
            save_model_usage(_get_db, usage_rows, user_id=user_id,
                             session_id=session_id, trace_id=trace_id)
        usage_saved = True
        return JSONResponse({
            "trace_id": trace_id,
            "session_id": session_id,
            "provider": _agent_provider_name,
            "knowledge_provider": _agent_knowledge_provider_name,
            "response": result["final_response"],
            "activities": [card.model_dump() for card in result.get("activities", [])],
            "safety": result["safety"].model_dump(mode="json"),
            "emotion": result["emotion_context"].model_dump(mode="json"),
            "knowledge": [item.model_dump(mode="json") for item in result["retrieved_knowledge"]],
            "avatar": result["avatar_command"].model_dump(mode="json"),
            "execution_path": result["execution_path"],
            "node_timings_ms": result["node_timings_ms"],
        })
    except ImportError as exc:
        return JSONResponse(
            {"error": "agent_dependencies_unavailable", "detail": str(exc)},
            status_code=503,
        )
    finally:
        if usage_token is not None and not usage_saved:
            usage_rows = finish_usage_collection(usage_token)
            if user_id is not None:
                from apps.api.admin_dashboard import save_model_usage
                save_model_usage(_get_db, usage_rows, user_id=user_id,
                                 session_id=session_id, trace_id=trace_id)


# ══════════════════════════════════════════════════════════════
# 用户认证接口
# ══════════════════════════════════════════════════════════════
@app.post("/api/auth/register")
async def auth_register(request: Request):
    try: body = await request.json()
    except ValueError: return JSONResponse({'error':'请求格式不正确'},status_code=400)
    if not isinstance(body,dict) or not isinstance(body.get("username"),str) or not isinstance(body.get("password"),str):
        return JSONResponse({"error":"请填写有效用户名和密码"},status_code=400)
    username = (body.get("username") or "").strip()
    password = body.get("password") or ""
    email_value = body.get("email", "")
    email_code = body.get("email_code", "")
    if not isinstance(email_value,str) or not isinstance(email_code,str):
        return JSONResponse({"error":"邮箱或验证码格式不正确"},status_code=400)
    if not HAS_BCRYPT:
        return JSONResponse({"error":"密码服务不可用"}, status_code=503)
    if not username or not password:
        return JSONResponse({"error": "用户名和密码不能为空"}, status_code=400)
    if len(username) < 2 or len(username) > 20:
        return JSONResponse({"error": "用户名长度须2-20位"}, status_code=400)
    if not 8 <= len(password) <= 64 or len(password.encode()) > 72:
        return JSONResponse({"error": "密码须8-64位，UTF-8长度不超过72字节"}, status_code=400)
    if not re.fullmatch(r"[\w-]{2,20}",username):
        return JSONResponse({"error":"用户名仅支持文字、数字、下划线和连字符"},status_code=400)
    from apps.api.external_auth import email_ready, normalize_email, registration_digest
    email_required = os.getenv('PUBLIC_DEPLOYMENT','false').lower() == 'true' and email_ready()
    if email_required and (not email_value or not email_code):
        return JSONResponse({"error":"请先完成邮箱验证码验证"},status_code=400)
    address = ''
    if email_value or email_code:
        if not email_ready():
            return JSONResponse({"error":"邮件服务尚未配置，暂不能使用邮箱注册"},status_code=503)
        try: address = normalize_email(email_value)
        except ValueError: return JSONResponse({"error":"邮箱格式不正确"},status_code=400)
        if not re.fullmatch(r'\d{6}',email_code):
            return JSONResponse({"error":"请输入六位邮箱验证码"},status_code=400)
    conn = _get_db()
    try:
        conn.execute('BEGIN IMMEDIATE')
        exists = conn.execute("SELECT id FROM users WHERE username=?", (username,)).fetchone()
        if exists:
            return JSONResponse({"error": "用户名已存在"}, status_code=409)
        if address:
            if conn.execute("SELECT id FROM users WHERE email=? AND email_verified_at<>''",(address,)).fetchone():
                return JSONResponse({"error":"该邮箱已注册，可以直接登录"},status_code=409)
            row=conn.execute('SELECT * FROM registration_email_codes WHERE email=?',(address,)).fetchone()
            if not row or row['expires']<=time.time() or row['attempts']>=5:
                return JSONResponse({"error":"验证码不正确或已失效，请重新发送"},status_code=400)
            conn.execute('UPDATE registration_email_codes SET attempts=attempts+1 WHERE email=?',(address,))
            if not hmac.compare_digest(row['digest'],registration_digest(address,row['salt'],email_code)):
                conn.commit()
                return JSONResponse({"error":"验证码不正确或已失效，请重新发送"},status_code=400)
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        from apps.api.admin_dashboard import configured_role
        role = configured_role(username)
        if address:
            cursor=conn.execute(
                "INSERT INTO users(username,password_hash,created_at,email,email_verified_at,role) VALUES(?,?,?,?,?,?)",
                (username,_hash_password(password),now,address,now,role))
            conn.execute('DELETE FROM registration_email_codes WHERE email=?',(address,))
        else:
            cursor=conn.execute(
                "INSERT INTO users(username,password_hash,created_at,role) VALUES(?,?,?,?)",
                (username, _hash_password(password), now, role))
        user_id = cursor.lastrowid
        token = str(uuid.uuid4())
        expires = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(time.time() + 7*86400))
        conn.execute("INSERT INTO auth_tokens(token,user_id,expires_at) VALUES(?,?,?)", (token, user_id, expires))
        conn.commit()
        return JSONResponse({"token": token, "username": username, "user_id": user_id})
    except sqlite3.IntegrityError:
        conn.rollback()
        return JSONResponse({"error":"用户名或邮箱已被使用"},status_code=409)
    finally:
        conn.close()

@app.post("/api/auth/login")
async def auth_login(request: Request):
    try: body = await request.json()
    except ValueError: return JSONResponse({'error':'请求格式不正确'},status_code=400)
    if not isinstance(body,dict) or not isinstance(body.get("username"),str) or not isinstance(body.get("password"),str):
        return JSONResponse({"error":"用户名或密码错误"},status_code=401)
    username = (body.get("username") or "").strip()
    password = body.get("password") or ""
    if not isinstance(password,str) or len(password.encode())>72:
        return JSONResponse({"error":"用户名或密码错误"},status_code=401)
    if not username or not password:
        return JSONResponse({"error": "用户名和密码不能为空"}, status_code=400)
    conn = _get_db()
    try:
        row = conn.execute("SELECT id,password_hash,username FROM users WHERE username=? OR (email=? AND email_verified_at<>'')", (username,username.lower())).fetchone()
        if not row or not _check_password(password, row["password_hash"]):
            return JSONResponse({"error": "用户名或密码错误"}, status_code=401)
        token = str(uuid.uuid4())
        expires = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(time.time() + 7*86400))
        conn.execute("INSERT OR REPLACE INTO auth_tokens(token,user_id,expires_at) VALUES(?,?,?)",
                     (token, row["id"], expires))
        conn.commit()
        return JSONResponse({"token": token, "username": row['username'], "user_id": row["id"]})
    finally:
        conn.close()

@app.post("/api/auth/logout")
async def auth_logout(request: Request):
    token = request.headers.get("X-Auth-Token", "")
    if token:
        conn = _get_db()
        conn.execute("DELETE FROM auth_tokens WHERE token=?", (token,))
        conn.commit()
        conn.close()
    return JSONResponse({"ok": True})

# ══════════════════════════════════════════════════════════════
# 会话管理接口
# ══════════════════════════════════════════════════════════════
def _get_user_id_from_request(request: Request) -> Optional[int]:
    token = request.headers.get("X-Auth-Token", "")
    return _verify_auth_token(token)


class ProfileUpdate(BaseModel):
    display_name: str = Field(default="",max_length=40)
    username: str = Field(min_length=2,max_length=20)
    birthday: str = Field(default="",max_length=10)
    avatar: str = Field(default="",max_length=180000)
    current_password: str = Field(default="",max_length=64)


class PasswordUpdate(BaseModel):
    current_password: str = Field(default="",max_length=64)
    new_password: str = Field(min_length=8,max_length=64)

class AccountDelete(BaseModel):
    current_password: str = Field(min_length=1,max_length=64)
    confirmation: str = Field(max_length=20)


@app.post('/api/profile/account/delete')
async def delete_account(payload: AccountDelete,request: Request):
    user_id=_get_user_id_from_request(request)
    if user_id is None:return JSONResponse({'error':'请先登录'},status_code=401)
    if payload.confirmation!='注销我的账号':return JSONResponse({'error':'请输入完整确认文字：注销我的账号'},status_code=400)
    if not HAS_BCRYPT:return JSONResponse({'error':'密码服务不可用'},status_code=503)
    if len(payload.current_password.encode())>72:return JSONResponse({'error':'密码格式不正确'},status_code=400)
    from apps.api.account_lifecycle import delete_owned_account
    conn=_get_db()
    try:
        conn.execute('BEGIN IMMEDIATE')
        row=conn.execute('SELECT password_hash,password_set FROM users WHERE id=?',(user_id,)).fetchone()
        if not row or not row['password_set'] or not _check_password(payload.current_password,row['password_hash']):
            return JSONResponse({'error':'当前密码不正确；未设置密码的第三方账号请先设置密码'},status_code=403)
        delete_owned_account(conn,user_id)
        conn.commit()
        return JSONResponse({'ok':True,'reauthenticate':True,'message':'账号及本项目内关联数据已删除，无法从界面恢复。'})
    finally:
        conn.close()


@app.get("/api/profile")
async def get_profile(request: Request):
    user_id=_get_user_id_from_request(request)
    if user_id is None:
        return JSONResponse({"error":"请先登录"},status_code=401)
    with _get_db() as conn:
        row=conn.execute("SELECT id,username,display_name,birthday,avatar,created_at,email,email_verified_at,password_set,role FROM users WHERE id=?",(user_id,)).fetchone()
        profile=dict(row)
        from apps.api.admin_dashboard import user_role
        profile["role"]=user_role(conn,user_id)
    return JSONResponse(profile)


@app.patch("/api/profile")
async def update_profile(payload: ProfileUpdate,request: Request):
    user_id=_get_user_id_from_request(request)
    if user_id is None:
        return JSONResponse({"error":"请先登录"},status_code=401)
    if not re.fullmatch(r"[\w-]{2,20}",payload.username):
        return JSONResponse({"error":"用户名仅支持文字、数字、下划线和连字符"},status_code=400)
    if payload.birthday:
        try:
            born=date.fromisoformat(payload.birthday)
            if born>date.today() or born.year<1900: raise ValueError()
        except ValueError:
            return JSONResponse({"error":"生日须为1900年至今的有效日期"},status_code=400)
    if payload.avatar:
        try:
            prefix,encoded=payload.avatar.split(',',1)
            if prefix not in ('data:image/jpeg;base64','data:image/png;base64'): raise ValueError()
            raw=base64.b64decode(encoded,validate=True)
            if not (raw.startswith(b'\xff\xd8\xff') or raw.startswith(b'\x89PNG\r\n\x1a\n')): raise ValueError()
        except (ValueError,TypeError):
            return JSONResponse({"error":"头像仅支持有效 PNG/JPEG 图片"},status_code=400)
    with _get_db() as conn:
        row=conn.execute("SELECT username,password_hash FROM users WHERE id=?",(user_id,)).fetchone()
        if payload.username!=row['username'] and not _check_password(payload.current_password,row['password_hash']):
            return JSONResponse({"error":"修改用户名需要验证当前密码"},status_code=403)
        try:
            conn.execute("UPDATE users SET username=?,display_name=?,birthday=?,avatar=? WHERE id=?",
                         (payload.username,payload.display_name.strip(),payload.birthday,payload.avatar,user_id))
            conn.commit()
        except sqlite3.IntegrityError:
            return JSONResponse({"error":"用户名已被使用"},status_code=409)
    return JSONResponse({"ok":True})


@app.post("/api/profile/password")
async def update_password(payload: PasswordUpdate,request: Request):
    user_id=_get_user_id_from_request(request)
    if user_id is None: return JSONResponse({"error":"请先登录"},status_code=401)
    if not HAS_BCRYPT: return JSONResponse({"error":"密码服务不可用"},status_code=503)
    if len(payload.new_password.encode())>72:
        return JSONResponse({"error":"密码UTF-8长度不能超过72字节"},status_code=400)
    with _get_db() as conn:
        row=conn.execute("SELECT password_hash,password_set FROM users WHERE id=?",(user_id,)).fetchone()
        session=conn.execute("SELECT source,issued_at FROM auth_tokens WHERE token=?",(request.headers.get('X-Auth-Token',''),)).fetchone()
        fresh_social=not row['password_set'] and session and session['source']=='oauth' and session['issued_at']>time.time()-600
        if not fresh_social and not _check_password(payload.current_password,row['password_hash']):
            return JSONResponse({"error":"当前密码不正确"},status_code=403)
        conn.execute("UPDATE users SET password_hash=?,password_set=1 WHERE id=?",(_hash_password(payload.new_password),user_id))
        conn.execute("DELETE FROM password_reset_codes WHERE user_id=?",(user_id,))
        conn.execute("DELETE FROM auth_tokens WHERE user_id=?",(user_id,))
        conn.commit()
    return JSONResponse({"ok":True,"reauthenticate":True})


from apps.api.external_auth import install_external_auth
install_external_auth(app,_get_db,_verify_auth_token,_hash_password,_check_password)
from apps.api.admin_dashboard import install_admin_routes
install_admin_routes(app,_get_db,_verify_auth_token)


def _session_belongs_to_user(session_id: str, user_id: int) -> bool:
    if not session_id:
        return False
    conn = _get_db()
    try:
        row = conn.execute(
            "SELECT 1 FROM chat_sessions WHERE id=? AND user_id=?",
            (session_id, user_id),
        ).fetchone()
        return row is not None
    finally:
        conn.close()


class ActivityStartRequest(BaseModel):
    template_id: str = Field(min_length=1, max_length=40)
    client_id: str = Field(min_length=1, max_length=80)


def _activity_payload(row):
    from services.agent.activities import activity_cards
    template = next((card for card in activity_cards() if card.id == row["template_id"]), None)
    return {**(template.model_dump() if template else {}),
            "task_id": row["id"], "status": row["status"],
            "created_at": row["created_at"], "completed_at": row["completed_at"]}


@app.post("/api/activities")
async def start_activity(payload: ActivityStartRequest, request: Request):
    user_id = _get_user_id_from_request(request)
    if not user_id:
        return JSONResponse({"error": "未登录"}, status_code=401)
    from services.agent.activities import activity_cards
    if payload.template_id not in {card.id for card in activity_cards()}:
        return JSONResponse({"error": "未知活动模板"}, status_code=422)
    conn = _get_db()
    try:
        conn.execute("INSERT OR IGNORE INTO activity_tasks(id,user_id,client_id,template_id,created_at) VALUES(?,?,?,?,?)",
                     (str(uuid.uuid4()), user_id, payload.client_id, payload.template_id,
                      time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
        conn.commit()
        row = conn.execute("SELECT * FROM activity_tasks WHERE user_id=? AND client_id=?", (user_id,payload.client_id)).fetchone()
        if row["template_id"] != payload.template_id:
            return JSONResponse({"error": "请求标识已用于其他活动"}, status_code=409)
        if row["removed_at"]:
            return JSONResponse({"error": "该活动记录已移除，请重新选择活动"}, status_code=410)
        return JSONResponse(_activity_payload(row))
    finally:
        conn.close()


@app.get("/api/activities")
async def list_activities(request: Request):
    user_id = _get_user_id_from_request(request)
    if not user_id:
        return JSONResponse({"error": "未登录"}, status_code=401)
    conn = _get_db()
    try:
        rows = conn.execute("SELECT * FROM activity_tasks WHERE user_id=? AND removed_at IS NULL ORDER BY created_at DESC,rowid DESC LIMIT 50", (user_id,)).fetchall()
        return JSONResponse({"activities": [_activity_payload(row) for row in rows]})
    finally:
        conn.close()


@app.get("/api/activities/summary")
async def activity_summary(request: Request):
    user_id = _get_user_id_from_request(request)
    if not user_id:
        return JSONResponse({"error": "未登录"}, status_code=401)
    conn = _get_db()
    try:
        row = conn.execute("SELECT COUNT(*) AS total,COALESCE(SUM(status='completed'),0) AS completed FROM activity_tasks WHERE user_id=? AND removed_at IS NULL", (user_id,)).fetchone()
        return JSONResponse({"total": row["total"], "completed": row["completed"],
                             "started": row["total"]-row["completed"]})
    finally:
        conn.close()


@app.delete("/api/activities/{task_id}")
async def remove_activity(task_id: str, request: Request):
    user_id = _get_user_id_from_request(request)
    if not user_id:
        return JSONResponse({"error": "未登录"}, status_code=401)
    conn = _get_db()
    try:
        cursor = conn.execute("UPDATE activity_tasks SET removed_at=COALESCE(removed_at,?) WHERE id=? AND user_id=?",
                              (time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),task_id,user_id))
        conn.commit()
        if cursor.rowcount == 0:
            return JSONResponse({"error": "活动不存在"}, status_code=404)
        return JSONResponse({"removed": True})
    finally:
        conn.close()


@app.post("/api/activities/{task_id}/complete")
async def complete_activity(task_id: str, request: Request):
    user_id = _get_user_id_from_request(request)
    if not user_id:
        return JSONResponse({"error": "未登录"}, status_code=401)
    conn = _get_db()
    try:
        conn.execute("UPDATE activity_tasks SET status='completed',completed_at=COALESCE(completed_at,?) WHERE id=? AND user_id=? AND removed_at IS NULL",
                     (time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),task_id,user_id))
        conn.commit()
        row = conn.execute("SELECT * FROM activity_tasks WHERE id=? AND user_id=? AND removed_at IS NULL", (task_id,user_id)).fetchone()
        if not row:
            return JSONResponse({"error": "活动不存在"}, status_code=404)
        return JSONResponse(_activity_payload(row))
    finally:
        conn.close()


@app.get("/api/memory/settings")
async def get_memory_settings(request: Request):
    user_id = _get_user_id_from_request(request)
    if not user_id:
        return JSONResponse({"error": "未登录"}, status_code=401)
    conn = _get_db()
    try:
        row = conn.execute(
            "SELECT enabled FROM user_memory_settings WHERE user_id=?", (user_id,)
        ).fetchone()
        count = conn.execute(
            "SELECT COUNT(*) FROM user_memories WHERE user_id=?", (user_id,)
        ).fetchone()[0]
        summary_count=conn.execute("SELECT COUNT(*) FROM session_notes WHERE user_id=? AND notes NOT IN ('[]','')",(user_id,)).fetchone()[0]
        return JSONResponse({"enabled": bool(row["enabled"]) if row else False, "count": count,"summary_count":summary_count})
    finally:
        conn.close()


@app.put("/api/memory/settings")
async def update_memory_settings(payload: MemorySettingsRequest, request: Request):
    user_id = _get_user_id_from_request(request)
    if not user_id:
        return JSONResponse({"error": "未登录"}, status_code=401)
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    conn = _get_db()
    try:
        conn.execute('BEGIN IMMEDIATE')
        previous=conn.execute('SELECT enabled FROM user_memory_settings WHERE user_id=?',(user_id,)).fetchone()
        conn.execute(
            """INSERT INTO user_memory_settings(user_id, enabled, updated_at)
               VALUES(?,?,?)
               ON CONFLICT(user_id) DO UPDATE SET enabled=excluded.enabled,
               updated_at=excluded.updated_at""",
            (user_id, int(payload.enabled), now),
        )
        from services.agent.session_notes import reset_session_notes
        if not previous or bool(previous['enabled'])!=payload.enabled:
            reset_session_notes(conn,user_id)
        conn.commit()
        return JSONResponse({"enabled": payload.enabled})
    finally:
        conn.close()


@app.get("/api/memories")
async def get_memories(request: Request):
    user_id = _get_user_id_from_request(request)
    if not user_id:
        return JSONResponse({"error": "未登录"}, status_code=401)
    conn = _get_db()
    try:
        rows = conn.execute(
            """SELECT id, content, category, created_at FROM user_memories
               WHERE user_id=? ORDER BY created_at DESC LIMIT 100""",
            (user_id,),
        ).fetchall()
        return JSONResponse({"memories": [dict(row) for row in rows]})
    finally:
        conn.close()


@app.delete("/api/memories")
async def delete_memories(request: Request):
    user_id = _get_user_id_from_request(request)
    if not user_id:
        return JSONResponse({"error": "未登录"}, status_code=401)
    conn = _get_db()
    try:
        cursor = conn.execute("DELETE FROM user_memories WHERE user_id=?", (user_id,))
        from services.agent.session_notes import reset_session_notes
        reset_session_notes(conn,user_id)
        conn.commit()
        return JSONResponse({"ok": True, "deleted": cursor.rowcount})
    finally:
        conn.close()


@app.delete("/api/memories/{memory_id}")
async def delete_memory(memory_id: str, request: Request):
    user_id = _get_user_id_from_request(request)
    if not user_id:
        return JSONResponse({"error": "未登录"}, status_code=401)
    conn = _get_db()
    try:
        cursor = conn.execute(
            "DELETE FROM user_memories WHERE id=? AND user_id=?",
            (memory_id, user_id),
        )
        if cursor.rowcount == 0:
            return JSONResponse({"error": "记忆不存在"}, status_code=404)
        from services.agent.session_notes import reset_session_notes
        reset_session_notes(conn,user_id)
        conn.commit()
        return JSONResponse({"ok": True, "deleted": 1})
    finally:
        conn.close()


@app.get("/api/agent/runs")
async def get_agent_runs(request: Request, limit: int = 20):
    user_id = _get_user_id_from_request(request)
    if not user_id:
        return JSONResponse({"error": "未登录"}, status_code=401)
    safe_limit = min(max(limit, 1), 100)
    conn = _get_db()
    try:
        rows = conn.execute(
            """SELECT trace_id,session_id,provider,risk_level,emotion,
                      execution_path,tool_calls,node_timings_ms,total_latency_ms,created_at
               FROM agent_runs WHERE user_id=? ORDER BY created_at DESC, rowid DESC LIMIT ?""",
            (user_id, safe_limit),
        ).fetchall()
        runs = []
        for row in rows:
            item = dict(row)
            for field in ("execution_path", "tool_calls", "node_timings_ms"):
                item[field] = json.loads(item[field])
            runs.append(item)
        return JSONResponse({"runs": runs})
    finally:
        conn.close()

@app.get("/api/sessions")
async def get_sessions(request: Request, limit: int = 30, cursor: str = ""):
    user_id = _get_user_id_from_request(request)
    if not user_id:
        return JSONResponse({"error": "未登录"}, status_code=401)
    conn = _get_db()
    try:
        safe_limit = min(max(limit, 1), 100)
        params: list[Any] = [user_id]
        where = "user_id=? AND is_main=0"
        if cursor:
            try:
                updated_at, session_id = base64.urlsafe_b64decode(cursor.encode()).decode().split("|", 1)
            except (ValueError, UnicodeDecodeError):
                return JSONResponse({"error": "分页游标无效"}, status_code=400)
            where += " AND (updated_at < ? OR (updated_at=? AND id<?))"
            params.extend((updated_at, updated_at, session_id))
        rows = conn.execute(
            f"""SELECT id,title,created_at,updated_at,is_main,dialogue_mode,emotion_style,title_manual
                FROM chat_sessions WHERE {where}
                ORDER BY updated_at DESC,id DESC LIMIT ?""",
            (*params, safe_limit + 1),
        ).fetchall()
        page = [dict(row) for row in rows[:safe_limit]]
        if not cursor:
            main = conn.execute(
                """SELECT id,title,created_at,updated_at,is_main,dialogue_mode,emotion_style,title_manual
                   FROM chat_sessions WHERE user_id=? AND is_main=1""", (user_id,),
            ).fetchone()
            if main:
                page.insert(0, dict(main))
        next_cursor = None
        if len(rows) > safe_limit and page:
            last = page[-1]
            next_cursor = base64.urlsafe_b64encode(f"{last['updated_at']}|{last['id']}".encode()).decode()
        return JSONResponse({"sessions": page, "next_cursor": next_cursor})
    finally:
        conn.close()

@app.post("/api/sessions")
async def create_session(request: Request):
    user_id = _get_user_id_from_request(request)
    if not user_id:
        return JSONResponse({"error": "未登录"}, status_code=401)
    session_id = str(uuid.uuid4())
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    conn = _get_db()
    try:
        blank = conn.execute(
            """SELECT s.* FROM chat_sessions s WHERE s.user_id=? AND s.is_main=0
               AND NOT EXISTS(SELECT 1 FROM chat_messages m WHERE m.session_id=s.id)
               ORDER BY s.created_at DESC LIMIT 1""", (user_id,),
        ).fetchone()
        if blank:
            return JSONResponse({**dict(blank), "session_id": blank["id"], "reused": True})
        conn.execute(
            "INSERT INTO chat_sessions(id,user_id,title,created_at,updated_at) VALUES(?,?,?,?,?)",
            (session_id, user_id, "新对话", now, now)
        )
        conn.commit()
        return JSONResponse({"session_id": session_id, "title": "新对话", "created_at": now})
    finally:
        conn.close()

@app.get("/api/sessions/{session_id}/messages")
async def get_session_messages(session_id: str, request: Request, limit: int = 50, before_id: int | None = None):
    user_id = _get_user_id_from_request(request)
    if not user_id:
        return JSONResponse({"error": "未登录"}, status_code=401)
    conn = _get_db()
    try:
        sess = conn.execute(
            "SELECT id FROM chat_sessions WHERE id=? AND user_id=?", (session_id, user_id)
        ).fetchone()
        if not sess:
            return JSONResponse({"error": "会话不存在"}, status_code=404)
        safe_limit = min(max(limit, 1), 100)
        before_clause = " AND id<?" if before_id is not None else ""
        params = (session_id, before_id, safe_limit + 1) if before_id is not None else (session_id, safe_limit + 1)
        msgs = conn.execute(
            f"""SELECT id,role,content,emotion_label,ts,knowledge_sources FROM (
                 SELECT id,role,content,emotion_label,ts,knowledge_sources FROM chat_messages
                 WHERE session_id=?{before_clause} ORDER BY id DESC LIMIT ?)
                 ORDER BY id ASC""", params,
        ).fetchall()
        from services.agent.references import reference_snapshot
        messages=[]
        for row in msgs:
            item=dict(row)
            try:
                item['knowledge_sources']=reference_snapshot(json.loads(item['knowledge_sources'])) if item['role']=='assistant' else []
            except (ValueError,TypeError):
                item['knowledge_sources']=[]
            messages.append(item)
        has_more = len(messages) > safe_limit
        if has_more:
            messages = messages[1:]
        return JSONResponse({"messages": messages, "next_before_id": messages[0]["id"] if has_more and messages else None})
    finally:
        conn.close()

@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str, request: Request):
    user_id = _get_user_id_from_request(request)
    if not user_id:
        return JSONResponse({"error": "未登录"}, status_code=401)
    conn = _get_db()
    try:
        session = conn.execute(
            "SELECT id,is_main FROM chat_sessions WHERE id=? AND user_id=?",
            (session_id, user_id),
        ).fetchone()
        if not session:
            return JSONResponse({"error": "会话不存在"}, status_code=404)
        if session["is_main"]:
            return JSONResponse({"error": "主对话不可删除，请使用清空操作"}, status_code=409)
        conn.execute("DELETE FROM session_notes WHERE session_id=? AND user_id=?", (session_id,user_id))
        conn.execute("DELETE FROM chat_messages WHERE session_id=?", (session_id,))
        conn.execute("DELETE FROM chat_sessions WHERE id=?", (session_id,))
        conn.commit()
        return JSONResponse({"ok": True})
    finally:
        conn.close()

class SessionSettingsRequest(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=32)
    dialogue_mode: str | None = None
    emotion_style: str | None = None

@app.post("/api/sessions/main")
async def ensure_main_session(request: Request):
    user_id = _get_user_id_from_request(request)
    if not user_id:
        return JSONResponse({"error": "未登录"}, status_code=401)
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    conn = _get_db()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT * FROM chat_sessions WHERE user_id=? AND is_main=1", (user_id,)).fetchone()
        if not row:
            row = conn.execute(
                """SELECT s.* FROM chat_sessions s WHERE s.user_id=?
                   ORDER BY EXISTS(SELECT 1 FROM chat_messages m WHERE m.session_id=s.id) DESC,
                            s.updated_at DESC,s.rowid DESC LIMIT 1""", (user_id,),
            ).fetchone()
            if row:
                conn.execute("UPDATE chat_sessions SET is_main=1 WHERE id=?", (row["id"],))
            else:
                sid = str(uuid.uuid4())
                conn.execute("""INSERT INTO chat_sessions
                    (id,user_id,title,created_at,updated_at,is_main,dialogue_mode,emotion_style,title_manual)
                    VALUES(?,?,?,?,?,1,'daily','confidant',0)""", (sid,user_id,"主对话",now,now))
                row = conn.execute("SELECT * FROM chat_sessions WHERE id=?", (sid,)).fetchone()
        conn.commit()
        return JSONResponse({**dict(row), "session_id": row["id"]})
    finally:
        conn.close()

@app.patch("/api/sessions/{session_id}")
async def update_session(session_id: str, payload: SessionSettingsRequest, request: Request):
    user_id = _get_user_id_from_request(request)
    if not user_id:
        return JSONResponse({"error": "未登录"}, status_code=401)
    if payload.dialogue_mode not in (None, "daily", "emotional"):
        return JSONResponse({"error": "对话模式无效"}, status_code=422)
    if payload.emotion_style not in (None, "confidant", "gentle"):
        return JSONResponse({"error": "情感风格无效"}, status_code=422)
    fields: list[str] = []; values: list[Any] = []
    if payload.title is not None:
        title = re.sub(r"[\r\n\t]+", " ", payload.title).strip()
        if not title:
            return JSONResponse({"error": "标题不能为空"}, status_code=422)
        fields += ["title=?", "title_manual=1"]; values.append(title)
    if payload.dialogue_mode is not None:
        fields.append("dialogue_mode=?"); values.append(payload.dialogue_mode)
    if payload.emotion_style is not None:
        fields.append("emotion_style=?"); values.append(payload.emotion_style)
    if not fields:
        return JSONResponse({"error": "没有可更新的设置"}, status_code=422)
    conn = _get_db()
    try:
        cursor = conn.execute(f"UPDATE chat_sessions SET {','.join(fields)} WHERE id=? AND user_id=?", (*values, session_id, user_id))
        if not cursor.rowcount:
            return JSONResponse({"error": "会话不存在"}, status_code=404)
        conn.commit()
        return JSONResponse(dict(conn.execute("SELECT * FROM chat_sessions WHERE id=?", (session_id,)).fetchone()))
    finally:
        conn.close()

@app.post("/api/sessions/{session_id}/clear")
async def clear_session(session_id: str, request: Request):
    user_id = _get_user_id_from_request(request)
    if not user_id:
        return JSONResponse({"error": "未登录"}, status_code=401)
    conn = _get_db()
    try:
        session = conn.execute("SELECT is_main FROM chat_sessions WHERE id=? AND user_id=?", (session_id,user_id)).fetchone()
        if not session:
            return JSONResponse({"error": "会话不存在"}, status_code=404)
        conn.execute("DELETE FROM chat_messages WHERE session_id=?", (session_id,))
        conn.execute("DELETE FROM session_notes WHERE session_id=? AND user_id=?", (session_id,user_id))
        conn.execute("UPDATE chat_sessions SET title=CASE WHEN is_main=1 THEN '主对话' ELSE '新对话' END,title_manual=0,updated_at=? WHERE id=?", (time.strftime("%Y-%m-%dT%H:%M:%S"),session_id))
        conn.commit()
        return JSONResponse({"ok": True})
    finally:
        conn.close()

@app.post("/api/sessions/{session_id}/generate_title")
async def generate_session_title(session_id: str, request: Request):
    user_id = _get_user_id_from_request(request)
    if not user_id:
        return JSONResponse({"error": "未登录"}, status_code=401)
    conn = _get_db()
    try:
        session = conn.execute(
            "SELECT id,title_manual FROM chat_sessions WHERE id=? AND user_id=?",
            (session_id, user_id),
        ).fetchone()
        if not session:
            return JSONResponse({"error": "会话不存在"}, status_code=404)
        if session["title_manual"]:
            current = conn.execute("SELECT title FROM chat_sessions WHERE id=?", (session_id,)).fetchone()[0]
            return JSONResponse({"title": current, "session_id": session_id, "manual": True})
        msgs = conn.execute(
            "SELECT role,content FROM chat_messages WHERE session_id=? ORDER BY id ASC LIMIT 6",
            (session_id,)
        ).fetchall()
        if not msgs:
            return JSONResponse({"title": "新对话"})
        summary = "\n".join(f"{'用户' if m['role']=='user' else '小安'}: {m['content'][:40]}" for m in msgs)
        title = "新对话"
        if QWEN_API_KEY:
            try:
                import httpx
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.post(
                        f"{QWEN_BASE_URL}/chat/completions",
                        headers={"Authorization": f"Bearer {QWEN_API_KEY}"},
                        json={"model": QWEN_MODEL, "messages": [
                            {"role": "system", "content": "你是文本摘要助手，只输出结果。"},
                            {"role": "user", "content": f"请用8-16个汉字准确概括以下对话主题，只输出标题，不要标点：\n{summary}"}
                        ], "max_tokens": 20, "temperature": 0.3}
                    )
                    data = resp.json()
                    title = _normalize_session_title(data["choices"][0]["message"]["content"])
            except Exception as e:
                print(f"[Title] 生成失败: {e}")
                title = _normalize_session_title(msgs[0]["content"]) if msgs else "新对话"
        else:
            title = _normalize_session_title(msgs[0]["content"]) if msgs else "新对话"
        conn.execute(
            "UPDATE chat_sessions SET title=? WHERE id=? AND user_id=? AND title_manual=0",
            (title, session_id, user_id),
        )
        conn.commit()
        return JSONResponse({"title": title, "session_id": session_id})
    finally:
        conn.close()



# 多提供者 TTS：CosyVoice 云端 + macOS 服务端系统音色 + 浏览器最终降级
try:
    import dashscope
    from dashscope.audio.tts_v2 import SpeechSynthesizer as _TtsSynthesizer
    dashscope.api_key = TTS_API_KEY
    HAS_TTS = True
    print(f"[IntegratedServer] CosyVoice TTS 已加载")
except ImportError:
    HAS_TTS = False
    print("[IntegratedServer] 提示: dashscope 未安装，将尝试本地系统语音")


_COSYVOICE_VOICES = (
    {"id": "cosyvoice:longxiaochun", "name": "龙小淳", "locale": "zh-CN", "provider": "cosyvoice"},
)
_system_tts_provider: MacOSSayProvider | None = None
_system_tts_checked = False
_qwen3_tts_provider: Qwen3TtsProvider | None = None
_qwen3_tts_config: tuple[str, str] | None = None


def _get_system_tts_provider() -> MacOSSayProvider | None:
    global _system_tts_checked, _system_tts_provider
    if not _system_tts_checked:
        _system_tts_checked = True
        try:
            _system_tts_provider = MacOSSayProvider()
            voice_count = len(_system_tts_provider.list_voices())
            print(f"[IntegratedServer] macOS 系统 TTS 已加载，共 {voice_count} 个中文音色")
        except TtsProviderError as exc:
            _system_tts_provider = None
            print(f"[IntegratedServer] 系统 TTS 不可用: {exc}")
    return _system_tts_provider


def _cloud_tts_available() -> bool:
    return HAS_TTS and bool(TTS_API_KEY)


def _get_qwen3_tts_provider() -> Qwen3TtsProvider | None:
    global _qwen3_tts_config, _qwen3_tts_provider
    if not TTS_API_KEY:
        return None
    config = (TTS_API_KEY, TTS_QWEN3_MODEL)
    if _qwen3_tts_provider is None or _qwen3_tts_config != config:
        _qwen3_tts_provider = Qwen3TtsProvider(
            api_key=TTS_API_KEY,
            model=TTS_QWEN3_MODEL,
        )
        _qwen3_tts_config = config
    return _qwen3_tts_provider


def _tts_voice_catalog() -> List[Dict[str, str]]:
    voices: List[Dict[str, str]] = []
    if TTS_PROVIDER in {"auto", "qwen3_tts"}:
        provider = _get_qwen3_tts_provider()
        if provider is not None:
            voices.extend({
                "id": f"qwen3_tts:{item.id}",
                "name": item.name,
                "locale": item.locale,
                "provider": item.provider,
            } for item in provider.list_voices())
    if TTS_PROVIDER in {"auto", "cosyvoice"} and _cloud_tts_available():
        voices.extend(dict(item) for item in _COSYVOICE_VOICES)
    if TTS_PROVIDER in {"auto", "macos_say"}:
        provider = _get_system_tts_provider()
        if provider is not None:
            voices.extend({
                "id": f"macos_say:{item.id}",
                "name": item.name,
                "locale": item.locale.replace("_", "-"),
                "provider": item.provider,
            } for item in provider.list_voices())
    return voices


def _default_tts_voice(voices: List[Dict[str, str]]) -> str | None:
    ids = {item["id"] for item in voices}
    if TTS_DEFAULT_VOICE:
        candidates = (TTS_DEFAULT_VOICE, f"macos_say:{TTS_DEFAULT_VOICE}")
        for candidate in candidates:
            if candidate in ids:
                return candidate
    for preferred in (
        "qwen3_tts:Chelsie",
        "qwen3_tts:Momo",
        "cosyvoice:longxiaochun",
        "macos_say:Tingting",
    ):
        if preferred in ids:
            return preferred
    return voices[0]["id"] if voices else None


def _tts_status_payload() -> Dict[str, Any]:
    voices = _tts_voice_catalog()
    default_voice = _default_tts_voice(voices)
    if voices:
        providers = sorted({item["provider"] for item in voices})
        return {
            "available": True,
            "provider": " + ".join(providers),
            "reason": None,
            "default_voice": default_voice,
            "voice_count": len(voices),
            "message": f"服务端语音可用：{len(voices)} 个音色",
        }
    if TTS_PROVIDER == "qwen3_tts" and not TTS_API_KEY:
        reason = "credential_missing"
        message = "未配置 DASHSCOPE_API_KEY，使用浏览器语音"
    elif TTS_PROVIDER == "cosyvoice" and not HAS_TTS:
        reason = "dependency_missing"
        message = "未安装 dashscope，使用浏览器语音"
    elif TTS_PROVIDER == "cosyvoice" and not TTS_API_KEY:
        reason = "credential_missing"
        message = "未配置 DASHSCOPE_API_KEY，使用浏览器语音"
    elif TTS_PROVIDER not in {"auto", "qwen3_tts", "cosyvoice", "macos_say", "browser"}:
        reason = "invalid_provider"
        message = "TTS_PROVIDER 配置无效，使用浏览器语音"
    else:
        reason = "provider_unavailable"
        message = "未发现可用的服务端语音，使用浏览器语音"
    return {
        "available": False,
        "provider": "browser_fallback",
        "reason": reason,
        "default_voice": None,
        "voice_count": 0,
        "message": message,
    }

class TtsRequest(BaseModel):
    text: str = Field(min_length=1, max_length=500)
    voice: str | None = Field(default=None, max_length=100)
    rate: int = Field(default=TTS_RATE, ge=120, le=260)


@app.get("/api/tts/voices")
async def api_tts_voices():
    voices = _tts_voice_catalog()
    return JSONResponse({
        "available": bool(voices),
        "default_voice": _default_tts_voice(voices),
        "voices": voices,
    }, headers={"Cache-Control": "no-store"})


@app.post("/api/tts/stream")
async def api_tts_stream(payload: TtsRequest):
    voices = _tts_voice_catalog()
    voice_id = payload.voice or _default_tts_voice(voices)
    voice = next((item for item in voices if item["id"] == voice_id), None)
    if not voice or voice["provider"] != "qwen3_tts":
        return JSONResponse({"error": "stream_unavailable"}, status_code=400)
    provider = _get_qwen3_tts_provider()
    iterator = provider.stream_pcm(payload.text, voice=voice_id.split(":",1)[1])
    try:
        first = await anext(iterator)
    except (TtsProviderError, StopAsyncIteration):
        await iterator.aclose()
        return JSONResponse({"error": "stream_failed"}, status_code=502)
    async def output():
        try:
            yield first
            async for chunk in iterator:
                yield chunk
        finally:
            await iterator.aclose()
    return StreamingResponse(output(), media_type="application/octet-stream",
                             headers={"Cache-Control":"no-store", "X-Accel-Buffering":"no",
                                      "X-Audio-Sample-Rate":"24000"})


@app.post("/api/tts")
async def api_tts(payload: TtsRequest):
    """使用选定的安全白名单音色生成服务端音频。"""
    text = payload.text.strip()
    if not text:
        return JSONResponse(
            {"error": "empty_text", "message": "语音文本不能为空"},
            status_code=400,
        )
    voices = _tts_voice_catalog()
    voice_id = payload.voice or _default_tts_voice(voices)
    voice = next((item for item in voices if item["id"] == voice_id), None)
    if not voices:
        status = _tts_status_payload()
        return JSONResponse(
            {
                "error": "tts_unavailable",
                "reason": status["reason"],
                "message": status["message"],
                "fallback": "browser_speech_synthesis",
            },
            status_code=503,
            headers={"Cache-Control": "no-store"},
        )
    if voice is None:
        return JSONResponse(
            {"error": "invalid_voice", "message": "请求的音色不在可用目录中"},
            status_code=400,
            headers={"Cache-Control": "no-store"},
        )
    try:
        loop = asyncio.get_event_loop()
        provider_name = voice["provider"]
        raw_voice = voice["id"].split(":", 1)[1]
        if provider_name == "macos_say":
            provider = _get_system_tts_provider()
            if provider is None:
                raise TtsProviderError("系统语音提供者已不可用")
            result = await loop.run_in_executor(
                _executor,
                lambda: provider.synthesize(text, voice=raw_voice, rate=payload.rate),
            )
            audio_bytes = result.content
            media_type = result.media_type
        elif provider_name == "qwen3_tts":
            provider = _get_qwen3_tts_provider()
            if provider is None:
                raise TtsProviderError("Qwen3-TTS 提供者已不可用")
            result = await loop.run_in_executor(
                _executor,
                lambda: provider.synthesize(text, voice=raw_voice, rate=payload.rate),
            )
            audio_bytes = result.content
            media_type = result.media_type
        else:
            def _synthesize_cloud():
                synth = _TtsSynthesizer(model="cosyvoice-v1", voice=raw_voice)
                return synth.call(text)
            audio_bytes = await loop.run_in_executor(_executor, _synthesize_cloud)
            media_type = "audio/mpeg"
        if not audio_bytes or len(audio_bytes) == 0:
            print(f"[TTS] {provider_name} 返回空音频")
            return JSONResponse({"error": "empty audio"}, status_code=502)
        print(f"[TTS] {provider_name}/{raw_voice} 成功，{len(audio_bytes)} 字节")
        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type=media_type,
            headers={
                "Content-Length": str(len(audio_bytes)),
                "Cache-Control": "no-store",
                "X-TTS-Provider": provider_name,
                "X-TTS-Voice": quote(raw_voice, safe=" ()"),
            },
        )
    except Exception as e:
        print(f"[TTS] 异常: {e}")
        traceback.print_exc()
        return JSONResponse(
            {"error": "tts_failed", "message": "服务端语音合成失败"},
            status_code=502,
            headers={"Cache-Control": "no-store"},
        )


# ══════════════════════════════════════════════════════════════
# 每个 WebSocket 会话的状态
# ══════════════════════════════════════════════════════════════
class SessionState:
    def __init__(self, session_id: str):
        self.camera_observation = CameraObservation()
        self.session_id = session_id
        self.user_id: Optional[int] = None  # 登录用户ID（未登录时为None）
        self.db_session_id: Optional[str] = None  # 对应数据库的 chat_sessions.id
        self.msg_count: int = 0  # 本次会话消息数（用于触发标题生成）
        self.feature_buffer: List[np.ndarray] = []
        self.seq_len = 8
        self.model_device = None
        self.asr_text_buffer = ""
        self.last_asr_trigger = 0.0
        self.llm_running = False
        self.asr_running = False
        self.au_latest: Dict[str, float] = {}
        # 情感状态（LLM返回后更新，用于表情叠加）
        self.current_emotion: str = "Neutral"
        self.current_valence: float = 0.0
        self.current_arousal: float = 0.0
        self.emotion_intensity: float = 0.3   # 表情强度（0~1）
        self.emotion_decay: float = 0.0        # 情感衰减计时
        self.model_emotion_signature: Optional[tuple] = None
        self.model_emotion_params: Optional[Dict[str, float]] = None
        # 待触发的动作
        self.pending_motion: Optional[str] = None
        # 智能体短期上下文；登录用户在安全校验后从数据库恢复。
        self.agent_messages: list[dict[str, str]] = []
        self.conversation_summary = ''
        self.memory_consent: bool = False
        self.dialogue_mode: str = "daily"
        self.emotion_style: str = "confidant"

# 驱动 WebSocket 客户端集合（供 /ws/drive 广播）
_drive_clients: set = set()


# ══════════════════════════════════════════════════════════════
# WebSocket 驱动通道（供 integrated.html 的 8767 兼容层）
# ══════════════════════════════════════════════════════════════
@app.websocket("/ws/drive")
async def ws_drive(websocket: WebSocket):
    """独立驱动 WebSocket，以 {type:'params', data:{...}} 格式推送 Live2D 参数"""
    await websocket.accept()
    _drive_clients.add(websocket)
    # 发送欢迎消息
    try:
        await websocket.send_json({"type": "info", "msg": "drive channel ready"})
        while True:
            # 主动接收断开事件，避免服务关停时等待下一次心跳。
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=30)
            except TimeoutError:
                await websocket.send_json({"type": "ping"})
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        _drive_clients.discard(websocket)


# ══════════════════════════════════════════════════════════════
# WebSocket 主通道
# ══════════════════════════════════════════════════════════════
@app.websocket("/ws/main")
async def ws_main(websocket: WebSocket):
    origin=websocket.headers.get('origin')
    allowed={x.strip() for x in os.getenv(
        'ALLOWED_ORIGINS',
        'http://127.0.0.1:8800,http://localhost:8800,http://127.0.0.1:8801,http://localhost:8801',
    ).split(',')}
    if origin and origin not in allowed:
        await websocket.close(code=1008)
        return
    await websocket.accept()
    session_id = str(uuid.uuid4())
    state = SessionState(session_id)
    loop = asyncio.get_event_loop()

    print(f"[WS] 新连接: {session_id}")

    # 预加载驱动模型（异步，不阻塞握手）
    async def preload_driver():
        md = None
        if HAS_DRIVER:
            async with _face_driver_lock:
                md = await loop.run_in_executor(_executor, _load_face_driver)
        state.model_device = md
        await _send(websocket, {"type": "status",
            "modules": {
                "vision": HAS_MEDIAPIPE, "asr": _asr_status_payload()["available"],
                "browser_vision": {"available": (ROOT / 'models' / 'face_landmarker.task').exists(), "enabled": False, "processing": "local_browser"},
                "driver": md is not None,
                "llm": bool(DEEPSEEK_API_KEY or QWEN_API_KEY),
            },
            "driver_model": _driver_runtime_payload(md),
        })

    preload_task=asyncio.create_task(preload_driver())
    pending_tasks=set()
    def spawn_session_task(coroutine):
        if len(pending_tasks)>=4:
            coroutine.close()
            return
        task=asyncio.create_task(coroutine)
        pending_tasks.add(task)
        task.add_done_callback(pending_tasks.discard)

    # 驱动参数推送任务（30fps）
    drive_task = asyncio.create_task(_drive_loop(websocket, state, loop))

    try:
        while True:
            raw = await websocket.receive_text()
            if len(raw)>3*1024*1024:
                await websocket.close(code=1009)
                break
            try:
                msg = json.loads(raw)
            except Exception:
                continue

            if not isinstance(msg, dict):
                continue
            msg_type = msg.get("type", "")
            if not isinstance(msg_type, str):
                continue
            if msg_type in {'audio','text_input','frame','vision_control','vision_features'}:
                if getattr(state,'auth_token',None) and _verify_auth_token(state.auth_token)!=state.user_id:
                    await _send(websocket,{'type':'error','code':'unauthorized','message':'登录状态已失效'})
                    await websocket.close(code=1008)
                    break
                if _public_deployment() and state.user_id is None:
                    await _send(websocket,{'type':'error','code':'unauthorized','message':'请先登录'})
                    continue
                limit = 240 if msg_type == 'vision_features' else 1800 if msg_type == 'frame' else 60
                if not allow_request(('ws',websocket.client.host,state.user_id,msg_type), limit):
                    await _send(websocket,{'type':'error','code':'rate_limited','message':'请求过于频繁'})
                    continue

            if msg_type in {'vision_control', 'vision_features'}:
                try:
                    if len(raw.encode('utf-8')) > 4096:
                        raise ValueError('payload_too_large')
                    status = (state.camera_observation.control(msg) if msg_type == 'vision_control'
                              else state.camera_observation.update(msg))
                    await _send(websocket, {'type': 'vision_status', 'status': status,
                        'stream_id': msg.get('stream_id'), 'seq': state.camera_observation.seq,
                        'observation': state.camera_observation.summary()})
                except ValueError as error:
                    await _send(websocket, {'type': 'vision_status', 'status': '观察暂停', 'code': str(error),
                        'stream_id': msg.get('stream_id') if isinstance(msg.get('stream_id'),str) else None})
                continue

            if msg_type == "init":
                # 前端登录后绑定 user_id 和 db_session_id
                token = msg.get("token", "")
                db_sid = msg.get("session_id", "")
                user_id = _verify_auth_token(token)
                if not user_id:
                    await _send(websocket, {
                        "type": "error",
                        "code": "unauthorized",
                        "message": "登录状态已失效",
                    })
                    continue
                if db_sid and not _session_belongs_to_user(db_sid, user_id):
                    await _send(websocket, {
                        "type": "error",
                        "code": "session_forbidden",
                        "message": "无权绑定该会话",
                    })
                    continue
                # 换绑会话前取消旧会话任务，避免迟到回复写入新会话。
                for task in list(pending_tasks): task.cancel()
                await asyncio.gather(*pending_tasks,return_exceptions=True)
                pending_tasks.clear()
                state.feature_buffer.clear()
                state.camera_observation.clear()
                state.user_id = user_id
                state.auth_token = token
                state.db_session_id = db_sid if db_sid else None
                state.agent_messages = (
                    _db_load_message_context(db_sid) if db_sid else []
                )
                if db_sid:
                    with _get_db() as conn:
                        settings = conn.execute(
                            "SELECT dialogue_mode,emotion_style FROM chat_sessions WHERE id=? AND user_id=?",
                            (db_sid, user_id),
                        ).fetchone()
                    state.dialogue_mode = settings["dialogue_mode"] if settings else "daily"
                    state.emotion_style = settings["emotion_style"] if settings else "confidant"
                else:
                    state.dialogue_mode = "daily"
                    state.emotion_style = "confidant"
                state.conversation_summary = ''
                state.memory_consent = _memory_enabled_for_user(user_id)
                print(f"[WS] 用户 {user_id} 绑定会话 {db_sid}")
                await _send(websocket, {
                    "type": "session_bound",
                    "session_id": state.db_session_id,
                })
                continue

            elif msg_type == "frame":
                # 解码图像帧 → 提取特征 → 缓冲
                spawn_session_task(_handle_frame(msg, state, websocket, loop))

            elif msg_type == "audio":
                # 解码音频 → ASR → 触发LLM
                spawn_session_task(_handle_audio(msg, state, websocket, loop))

            elif msg_type == "text_input":
                # 手动文字输入（ASR不可用时的降级）
                text = msg.get("text", "").strip()
                if text:
                    spawn_session_task(_trigger_llm(text, state, websocket, _request_id(msg)))

            elif msg_type == "control":
                action = msg.get("action", "")
                if action == "reset":
                    state.feature_buffer.clear()
                    state.asr_text_buffer = ""
                    await _send(websocket, {"type": "reset_ack"})

    except WebSocketDisconnect:
        print(f"[WS] 断开: {session_id}")
    except Exception as e:
        print(f"[WS] 异常: {e}")
        traceback.print_exc()
    finally:
        preload_task.cancel()
        state.camera_observation.clear()
        for task in list(pending_tasks): task.cancel()
        await asyncio.gather(preload_task,*pending_tasks,return_exceptions=True)
        drive_task.cancel()
        with suppress(asyncio.CancelledError):
            await drive_task
        print(f"[WS] 会话结束: {session_id}")


async def _send(ws: WebSocket, data: dict):
    try:
        await ws.send_text(json.dumps(data, ensure_ascii=False))
    except Exception:
        pass


async def _handle_frame(msg: dict, state: SessionState, ws: WebSocket, loop):
    """处理视频帧：解码→特征提取→缓冲"""
    try:
        img_b64 = msg.get("image", "")
        if not img_b64:
            return
        img_bytes = base64.b64decode(img_b64)
        import cv2
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return

        # 特征提取（线程池，不阻塞事件循环）
        features = await loop.run_in_executor(
            _executor, _extract_features_from_bgr, frame
        )
        state.feature_buffer.append(features)
        if len(state.feature_buffer) > state.seq_len:
            state.feature_buffer.pop(0)

        # 发送AU特征到前端（15维，与训练集FEATURE_ORDER对齐：AU1/2/4/6/7/9/10/12/14/15/17/23/24/25/26）
        # 特征索引映射：0=左眼开合(AU7), 1=右眼(AU7avg), 2=左眉(AU1近似), 3=右焉(AU2近似)
        # 4=微笑(AU12), 9=左焉内(AU4近似), 13=嘴开合(AU25)
        brow_inner = max(0, (features[9] + features[10]) / 2)  # AU4 皱眉近似
        brow_outer = max(0, (features[11] + features[12]) / 2)  # AU1/AU2 近似
        eye_open = (features[0] + features[1]) / 2  # AU7 眼睑收紧反面
        au_data = {
            "AU1":  round(float(max(0, brow_outer * 0.8)), 3),
            "AU2":  round(float(max(0, brow_outer * 0.6)), 3),
            "AU4":  round(float(max(0, brow_inner)), 3),
            "AU6":  round(float(features[4]), 3),    # 微笑→脸颊上扬
            "AU7":  round(float(1.0 - eye_open), 3), # 眼睑收紧 = 1-眼开合
            "AU9":  round(float(max(0, brow_inner * 0.3)), 3),
            "AU10": round(float(max(0, features[14] * 0.5 + 0.1) if features[14] > 0 else 0), 3),
            "AU12": round(float(features[4]), 3),    # 唇角上扬 = 微笑
            "AU14": round(float(features[4] * 0.4), 3),
            "AU15": round(float(max(0, -features[14] * 0.5) if features[14] < 0 else 0), 3),
            "AU17": round(float(max(0, features[23] * 0.5)), 3),  # 下巴上扬近似
            "AU23": round(float(max(0, 0.3 - features[13]) * 0.5), 3),
            "AU24": round(float(max(0, 0.2 - features[13]) * 0.3), 3),
            "AU25": round(float(features[13]), 3),   # 嘴开合
            "AU26": round(float(max(0, features[13] * 0.6)), 3),
        }
        state.au_latest = au_data
        await _send(ws, {"type": "au_features", "aus": au_data})

    except Exception as e:
        print(f"[Frame] 处理失败: {e}")  # 不再静默，便于排查


async def _handle_audio(msg: dict, state: SessionState, ws: WebSocket, loop):
    """处理音频片段：保存临时文件→ASR→触发LLM"""
    if state.asr_running or state.llm_running:
        await _send(ws, {"type": "asr_error", "code": "busy", "message": "正在处理上一句话，请稍后再录音"})
        return
    state.asr_running = True
    try:
        audio_b64 = msg.get("data", "")
        if not isinstance(audio_b64, str) or not audio_b64 or len(audio_b64) > 2_800_000:
            raise ValueError("音频为空或超过 2MB 限制")
        audio_bytes = base64.b64decode(audio_b64, validate=True)
        suffix = msg.get("format", "webm")
        if suffix not in {"wav", "webm", "ogg", "mp4"}:
            raise ValueError("不支持的音频格式")
        if len(audio_bytes) > 2 * 1024 * 1024:
            raise ValueError("音频超过 2MB 限制")
        if not _asr_status_payload()["available"]:
            await _send(ws, {"type": "asr_error", "code": "unavailable", "message": "服务端语音识别不可用，请使用浏览器语音或文字输入"})
            return
        await _send(ws, {"type": "asr_processing", "message": "正在识别录音，首次使用需要加载模型…"})

        with tempfile.NamedTemporaryFile(suffix=f".{suffix}", delete=False) as f:
            f.write(audio_bytes)
            tmp_path = f.name

        try:
            text, provider, fallback_reason = await loop.run_in_executor(_executor, _recognize_audio, tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

        if text:
            await _send(ws, {"type": "asr_result", "text": text, "is_final": True,
                             "provider": provider, "fallback_reason": fallback_reason})
            await _trigger_llm(text, state, ws, _request_id(msg))
        else:
            await _send(ws, {"type": "asr_error", "code": "empty_result", "message": "未识别到文字或模型加载失败，请重试或使用文字输入"})

    except cloud_asr.CloudAsrError:
        await _send(ws, {"type": "asr_error", "code": "cloud_failed", "message": "云端识别失败，请检查ASR密钥、地域与模型权限，或切换本地识别"})
    except Exception as e:
        print(f"[Audio] 处理失败: {type(e).__name__}")
        await _send(ws, {"type": "asr_error", "code": "invalid_audio", "message": "录音处理失败，请检查格式与大小后重试"})
    finally:
        state.asr_running = False


_MOTION_PATTERN = re.compile(r'\[MOTION:(FlickUp|Tap|Flick3|Idle)\]', re.IGNORECASE)

def _parse_motion_and_clean(reply: str):
    """
    从LLM回复中提取动作标签，返回(清洁文本, 动作名称或None)
    """
    match = _MOTION_PATTERN.search(reply)
    motion = match.group(1) if match else None
    clean_text = _MOTION_PATTERN.sub('', reply).strip()
    return clean_text, motion


def _request_id(msg):
    value=msg.get('request_id')
    pattern=r'[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}'
    return value if isinstance(value,str) and re.fullmatch(pattern,value,re.IGNORECASE) else None


async def _trigger_llm(text: str, state: SessionState, ws: WebSocket, request_id=None):
    """通过数字心屿智能体工作流生成回复并同步数字人状态。"""
    if not text or state.llm_running:
        return
    state.llm_running = True
    trace_id = str(uuid.uuid4())
    slow_notice = None
    from services.agent.usage import begin_usage_collection, finish_usage_collection
    usage_token = begin_usage_collection()
    usage_saved = False
    try:
        await _send(ws, {"type": "llm_thinking", "trace_id": trace_id, "request_id": request_id, "text": "小安正在思考..."})
        from services.agent import ChatMessage

        session_id = state.db_session_id or state.session_id
        notes_revision=None
        if state.user_id is not None:
            state.memory_consent=_memory_enabled_for_user(state.user_id)
            if state.db_session_id:
                from services.agent.session_notes import load_session_notes
                conn=_get_db()
                try:
                    state.conversation_summary,notes_revision=load_session_notes(conn,session_id,state.user_id)
                except Exception:
                    state.conversation_summary='';notes_revision=None
                finally:conn.close()
        history = [ChatMessage(**message) for message in state.agent_messages]

        async def send_agent_event(event):
            await _send(ws, {
                "type": "agent_event",
                "event": event.model_dump(mode="json"),
            })

        async def send_delta(delta):
            await _send(ws, {"type": "llm_delta", "trace_id": trace_id, "text": delta})

        async def notify_slow():
            await asyncio.sleep(8)
            await _send(ws, {"type": "llm_waiting", "trace_id": trace_id,
                "text": "模型或工具响应较慢，仍在处理，请稍候…"})

        slow_notice = asyncio.create_task(notify_slow())
        result = await asyncio.wait_for(_get_agent_workflow().run(
            user_text=text,
            trace_id=trace_id,
            session_id=session_id,
            messages=history,
            conversation_summary=state.conversation_summary,
            visual_observation=state.camera_observation.summary(),
            user_id=state.user_id,
            memory_consent=state.memory_consent,
            dialogue_mode=state.dialogue_mode,
            emotion_style=state.emotion_style,
            event_sink=send_agent_event,
            text_delta_sink=send_delta,
        ), timeout=60)
        usage_rows = finish_usage_collection(usage_token)
        usage_token = None
        if state.user_id is not None:
            from apps.api.admin_dashboard import save_model_usage
            save_model_usage(_get_db, usage_rows, user_id=state.user_id,
                session_id=session_id, trace_id=trace_id)
        usage_saved = True
        result["trace_id"] = trace_id
        result["session_id"] = session_id
        state.agent_messages = [
            message.model_dump() for message in result.get("messages", [])
        ]
        state.conversation_summary = result.get('conversation_summary', '')
        if state.user_id is not None:
            _db_save_agent_run(result, state.user_id, _agent_provider_name)

        reply_text = result["final_response"]
        safety = result["safety"]
        avatar = result["avatar_command"]
        emotion = result["emotion_context"]
        emo_result = {
            "emotion": emotion.emotion,
            "valence": emotion.valence,
            "arousal": emotion.arousal,
            "risk_level": safety.risk_level.value,
            "emotion_label": emotion.label,
        }
        motion_name = avatar.motion

        # 更新会话情感状态（用于表情叠加）
        state.current_emotion = emo_result["emotion"]
        state.current_valence = float(emo_result["valence"])
        state.current_arousal = float(emo_result["arousal"])
        # 根据情感强度动态调整表情强度
        base_intensity = abs(state.current_valence) * 0.5 + abs(state.current_arousal) * 0.3
        state.emotion_intensity = max(0.25, min(0.65, 0.3 + base_intensity))
        state.emotion_decay = time.time() + 12.0  # 情感状态保持12秒后衰减

        # 设置待触发动作
        if motion_name and motion_name != 'Idle':
            state.pending_motion = motion_name

        await _send(ws, {
            "type": "agent_trace",
            "trace_id": trace_id,
            "session_id": session_id,
            "provider": _agent_provider_name,
            "knowledge_provider": _agent_knowledge_provider_name,
            "safety": safety.model_dump(mode="json"),
            "emotion": emotion.model_dump(mode="json"),
            "knowledge": [
                item.model_dump(mode="json") for item in result["retrieved_knowledge"]
            ],
            "memories": [
                item.model_dump(mode="json") for item in result["retrieved_memories"]
            ],
            "tool_calls": [item.model_dump(mode="json") for item in result["tool_calls"]],
            "avatar": avatar.model_dump(mode="json"),
            "execution_path": result["execution_path"],
            "node_timings_ms": result["node_timings_ms"],
        })

        msg = {
            "type": "llm_reply",
            "trace_id": trace_id,
            "text": reply_text,
            "emotion": emo_result["emotion"],
            "valence": emo_result["valence"],
            "arousal": emo_result["arousal"],
            "risk_level": emo_result["risk_level"],
            "emotion_label": emo_result["emotion_label"],
        }
        msg["activities"] = [card.model_dump() for card in result.get("activities", [])]
        msg["sources"] = result.get("web_sources", [])
        from services.agent.references import knowledge_references
        msg['knowledge_sources'] = knowledge_references(result.get('retrieved_knowledge', []))
        if motion_name:
            msg["motion"] = motion_name
        await _send(ws, msg)

        if state.db_session_id:
            _db_save_message(state.db_session_id, "user", text,memory_revision=notes_revision)
            _db_save_message(state.db_session_id, "assistant", reply_text, emo_result["emotion_label"], msg['knowledge_sources'])
            if state.user_id is not None:
                from services.agent.session_notes import update_session_notes
                conn=_get_db()
                try:
                    if result.get('intent')=='memory_forget':
                        from services.agent.session_notes import reset_session_notes
                        reset_session_notes(conn,state.user_id);conn.commit()
                    else:update_session_notes(conn,state.db_session_id,state.user_id,notes_revision)
                except Exception:
                    print('[DB] 会话摘录更新未完成')
                finally:conn.close()
            state.msg_count += 1
            # 首轮形成初始标题，之后每四轮适度重新概括；手动标题在生成函数内受保护。
            if state.msg_count == 1 or state.msg_count % 4 == 0:
                asyncio.create_task(_auto_generate_title(state.db_session_id, ws))

    except Exception as e:
        print(f"[Agent] 失败: {e}")
        await _send(ws, {"type": "llm_reply", "text": "抱歉，我暂时无法回应，请稍后再试。",
                         "trace_id": trace_id, "stream_failed": True,
                         "emotion": "Neutral", "valence": 0, "arousal": 0,
                         "risk_level": "low", "emotion_label": "平静"})
    finally:
        if usage_token is not None and not usage_saved:
            usage_rows = finish_usage_collection(usage_token)
            if state.user_id is not None:
                from apps.api.admin_dashboard import save_model_usage
                save_model_usage(_get_db, usage_rows, user_id=state.user_id,
                    session_id=state.db_session_id or state.session_id, trace_id=trace_id)
        if slow_notice:
            slow_notice.cancel()
        state.llm_running = False


async def _auto_generate_title(db_session_id: str, ws: WebSocket):
    """后台自动为会话生成标题，并通过 WebSocket 推送更新"""
    try:
        conn = _get_db()
        session = conn.execute("SELECT title_manual,user_id FROM chat_sessions WHERE id=?", (db_session_id,)).fetchone()
        if not session or session["title_manual"]:
            conn.close()
            return
        msgs = conn.execute(
            "SELECT role,content FROM chat_messages WHERE session_id=? ORDER BY id ASC LIMIT 6",
            (db_session_id,)
        ).fetchall()
        conn.close()
        if not msgs:
            return
        summary = "\n".join(f"{'用户' if m['role']=='user' else '小安'}: {m['content'][:40]}" for m in msgs)
        title = _normalize_session_title(msgs[0]["content"]) if msgs else "新对话"
        if QWEN_API_KEY:
            try:
                import httpx
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.post(
                        f"{QWEN_BASE_URL}/chat/completions",
                        headers={"Authorization": f"Bearer {QWEN_API_KEY}"},
                        json={"model": QWEN_MODEL, "messages": [
                            {"role": "system", "content": "你是文本摘要助手，只输出结果。"},
                            {"role": "user", "content": f"请用8-16个汉字准确概括以下对话主题，只输出标题，不要标点：\n{summary}"}
                        ], "max_tokens": 20, "temperature": 0.3}
                    )
                    data = resp.json()
                    title = _normalize_session_title(data["choices"][0]["message"]["content"])
                    from services.agent.usage import estimate_tokens, usage_from_payload
                    usage = usage_from_payload(data)
                    estimated = usage is None
                    if usage is None:
                        usage = (estimate_tokens([summary]), estimate_tokens([title]), 0)
                    from apps.api.admin_dashboard import save_model_usage
                    save_model_usage(_get_db, [{
                        "provider": "qwen", "model": QWEN_MODEL,
                        "request_kind": "title", "input_tokens": usage[0],
                        "output_tokens": usage[1], "cached_input_tokens": usage[2],
                        "estimated": estimated, "status": "success",
                    }], user_id=session["user_id"], session_id=db_session_id,
                        trace_id=f"title-{uuid.uuid4()}")
            except Exception:
                pass
        conn2 = _get_db()
        cursor = conn2.execute("UPDATE chat_sessions SET title=? WHERE id=? AND title_manual=0", (title, db_session_id))
        conn2.commit()
        conn2.close()
        # 推送标题更新给前端
        if cursor.rowcount:
            await _send(ws, {"type": "session_title", "session_id": db_session_id, "title": title})
    except Exception as e:
        print(f"[Title] 自动生成标题失败: {e}")


async def _drive_loop(ws: WebSocket, state: SessionState, loop):
    """30fps 推送 Live2D 驱动参数 + 视觉监控数据"""
    interval = 1.0 / 30
    while True:
        try:
            await asyncio.sleep(interval)
            params = await _compute_live2d_params(state, loop)
            await _send(ws, {"type": "live2d_params", "params": params})

            # 不再向独立 /ws/drive 广播，防止不同账号的驱动互相串扰。

            # 额外推送视觉监控数据（供 vision_demo.html 展示）
            if state.feature_buffer:
                feats = state.feature_buffer[-1].tolist()
                # 从 au_latest 推算简单表情分布
                au = state.au_latest
                smile = au.get('smile', 0)
                mouth = au.get('mouth', 0)
                eye = (au.get('eye_l',0.7)+au.get('eye_r',0.7))/2
                expr = {
                    'Neutral': max(0, 1-smile*1.5-mouth*0.5),
                    'Happy':   min(1, smile*2),
                    'Sad':     max(0, (0.5-eye)*0.8),
                    'Surprise': max(0, mouth*1.5-0.2),
                    'Fear':    0.02, 'Disgust':0.02,
                    'Anger':   0.02, 'Contempt':0.01
                }
                total = sum(expr.values()) or 1
                expr = {k: round(v/total, 3) for k,v in expr.items()}
                va = {
                    'valence': round(smile - max(0,(0.5-eye)*0.8), 3),
                    'arousal': round(mouth + smile*0.5 - 0.2, 3)
                }
                await _send(ws, {
                    "type": "drive",
                    "features": feats,
                    "expression": expr,
                    "va": va,
                    "face_detected": bool(state.au_latest)
                })
        except asyncio.CancelledError:
            break
        except Exception:
            pass


async def _compute_live2d_params(state: SessionState, loop) -> Dict[str, float]:
    """计算当前帧的Live2D参数"""
    base_params = _features_to_params_simple(state)
    if state.model_device is None or not HAS_EMOTION_MAP:
        return base_params

    signature = (
        state.current_emotion,
        round(state.current_valence, 3),
        round(state.current_arousal, 3),
        round(state.emotion_intensity, 3),
    )
    if signature != state.model_emotion_signature:
        emotion_25 = _build_emotion_25d(
            state.current_emotion,
            state.current_valence,
            state.current_arousal,
        )
        sequence = np.repeat(emotion_25[None, :], state.seq_len, axis=0)
        predicted = await loop.run_in_executor(
            _executor,
            _infer_live2d_params,
            state.model_device,
            sequence,
            state.emotion_intensity,
        )
        if predicted is not None:
            state.model_emotion_params = predicted
            state.model_emotion_signature = signature
        else:
            # 不可恢复的推理异常只尝试一次，当前会话改用规则驱动，避免 30fps 刷屏。
            state.model_device = None
            state.model_emotion_params = None

    # 正式模型负责倾听表情；摄像头跟踪、口型和呼吸仍保留实时规则驱动。
    blend_keys = {
        'PARAM_BROW_L_Y', 'PARAM_BROW_R_Y',
        'PARAM_BROW_L_ANGLE', 'PARAM_BROW_R_ANGLE',
        'PARAM_EYE_BALL_FORM', 'PARAM_MOUTH_FORM', 'PARAM_TERE',
    }
    if state.model_emotion_params:
        for key in blend_keys:
            if key in base_params and key in state.model_emotion_params:
                alpha = 0.55
                base_params[key] = round(
                    base_params[key] * (1 - alpha)
                    + state.model_emotion_params[key] * alpha,
                    3,
                )
    return base_params


def _build_emotion_25d(emotion: str, valence: float, arousal: float) -> np.ndarray:
    """
    将 LLM 情感分析结果映射为 25 维情感特征向量
    用于通过 emotion_to_live2d 驱动数字人表情
    """
    e = np.zeros(25, dtype=np.float32)
    # VA 维度
    e[VA_VALENCE] = float(np.clip(valence, -1, 1))
    e[VA_AROUSAL] = float(np.clip(arousal, -1, 1))
    # EXP 分布
    exp_map = {
        'Happy':   (EXP_INDICES['Happy'],   0.8,  'AU12', 0.7, 'AU6', 0.5),
        'Sad':     (EXP_INDICES['Sad'],     0.8,  'AU15', 0.6, 'AU4', 0.4),
        'Anxiety': (EXP_INDICES['Fear'],    0.6,  'AU4',  0.5, 'AU7', 0.4),
        'Fear':    (EXP_INDICES['Fear'],    0.9,  'AU1',  0.7, 'AU4', 0.6),
        'Neutral': (EXP_INDICES['Neutral'], 0.8,  None,   0.0, None,  0.0),
    }
    if emotion in exp_map:
        idx, exp_v, au1_name, au1_v, au2_name, au2_v = exp_map[emotion]
        e[idx] = exp_v
        if au1_name:
            e[AU_INDICES[au1_name]] = au1_v
        if au2_name:
            e[AU_INDICES[au2_name]] = au2_v
    else:
        e[EXP_INDICES['Neutral']] = 0.6
    return e


def _features_to_params_simple(state: SessionState) -> Dict[str, float]:
    """无模型时的丰富特征→参数映射（降级，含自然动画 + LLM情感叠加）"""
    import math
    t = time.time()
    params = _default_live2d_params()

    # 自然呼吸（始终）
    params['PARAM_BREATH'] = round(0.5 + 0.5 * math.sin(t * 2.0), 3)

    if state.feature_buffer:
        f = state.feature_buffer[-1]
        params['PARAM_EYE_L_OPEN'] = round(float(np.clip(f[0], 0, 1)), 3)
        params['PARAM_EYE_R_OPEN'] = round(float(np.clip(f[1], 0, 1)), 3)
        mouth = float(np.clip(f[13] * 1.5, 0, 1))
        params['PARAM_MOUTH_OPEN_Y'] = round(mouth, 3)
        params['PARAM_MOUTH_FORM'] = round(float(np.clip(f[4] * 2 - 0.3, -1, 1)), 3)
        params['PARAM_ANGLE_X'] = round(float(np.clip(f[16] * 45, -30, 30)), 3)
        params['PARAM_ANGLE_Y'] = round(float(np.clip(f[15] * 45, -30, 30)), 3)
        params['PARAM_BODY_ANGLE_X'] = round(float(np.clip(f[16] * 10, -10, 10)), 3)
        params['PARAM_BODY_ANGLE_Y'] = round(float(np.clip(f[15] * 10, -10, 10)), 3)
        params['PARAM_BROW_L_Y'] = round(float(np.clip(f[2] * 2 - 1, -1, 1)), 3)
        params['PARAM_BROW_R_Y'] = round(float(np.clip(f[3] * 2 - 1, -1, 1)), 3)
        params['PARAM_EYE_BALL_X'] = round(float(np.clip(f[16] * 0.6, -1, 1)), 3)
        params['PARAM_EYE_BALL_Y'] = round(float(np.clip(f[15] * 0.4, -1, 1)), 3)
    else:
        # 无摄像头时空闲动画
        params['PARAM_ANGLE_X'] = round(math.sin(t * 0.31) * 3, 3)
        params['PARAM_ANGLE_Y'] = round(math.sin(t * 0.23) * 2, 3)
        params['PARAM_ANGLE_Z'] = round(math.sin(t * 0.17) * 1.5, 3)
        params['PARAM_BODY_ANGLE_X'] = round(math.sin(t * 0.31) * 1, 3)
        blink_phase = t % 4.0
        if blink_phase < 0.15:
            eye_v = max(0, 1.0 - blink_phase / 0.075) if blink_phase < 0.075 else min(1, (blink_phase - 0.075) / 0.075)
            params['PARAM_EYE_L_OPEN'] = round(eye_v, 3)
            params['PARAM_EYE_R_OPEN'] = round(eye_v, 3)

    # ── LLM 情感表情叠加 ──────────────────────────────────────
    if HAS_EMOTION_MAP and state.current_emotion != "Neutral":
        # 情感时间衰减（12秒后渐变回平静）
        remain = state.emotion_decay - t
        if remain > 0:
            decay_factor = min(1.0, remain / 5.0)  # 最后5秒线性衰减
            intensity = state.emotion_intensity * decay_factor
            try:
                e25 = _build_emotion_25d(
                    state.current_emotion,
                    state.current_valence,
                    state.current_arousal
                )
                emo_params_arr = _emo25_to_live2d(e25, intensity=intensity)
                emo_dict = _live2d_to_dict(emo_params_arr)
                # 叠加表情参数（口型不覆盖，保留TTS驱动）
                BLEND_KEYS = {
                    'PARAM_BROW_L_Y', 'PARAM_BROW_R_Y',
                    'PARAM_BROW_L_ANGLE', 'PARAM_BROW_R_ANGLE',
                    'PARAM_EYE_BALL_FORM', 'PARAM_MOUTH_FORM',
                    'PARAM_ANGLE_X', 'PARAM_ANGLE_Y',
                }
                for k in BLEND_KEYS:
                    if k in emo_dict and k in params:
                        alpha = 0.4  # 叠加权重
                        params[k] = round(params[k] * (1 - alpha) + emo_dict[k] * alpha, 3)
            except Exception:
                pass
        else:
            # 情感衰减完毕，重置为平静
            state.current_emotion = "Neutral"
            state.emotion_intensity = 0.3

    return params


# ══════════════════════════════════════════════════════════════
# 视频文件上传处理
# ══════════════════════════════════════════════════════════════
@app.post("/api/upload_video")
async def upload_video(request: Request,file: UploadFile = File(...)):
    """上传MP4文件，返回task_id，通过WS推送处理进度"""
    suffix = Path(file.filename).suffix.lower() if file.filename else ".mp4"
    if suffix not in {'.mp4','.webm','.mov'}:
        return JSONResponse({'error':'仅支持MP4、WebM或MOV视频'},status_code=400)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        content = await file.read()
        f.write(content)
        tmp_path = f.name

    task_id = str(uuid.uuid4())
    _video_owners[task_id]=_get_user_id_from_request(request)
    # 后台处理任务
    asyncio.create_task(_process_video_file(tmp_path, task_id))
    return JSONResponse({"task_id": task_id, "status": "processing",
                         "message": f"视频已接收（{len(content)//1024}KB），开始处理"})


# 视频任务状态存储
_video_tasks: Dict[str, Dict] = {}
_video_owners: Dict[str, Optional[int]] = {}

async def _process_video_file(video_path: str, task_id: str):
    """后台处理视频文件"""
    loop = asyncio.get_event_loop()
    _video_tasks[task_id] = {"status": "processing", "progress": 0}
    try:
        import cv2

        # 提取音频（用ffmpeg或moviepy）
        audio_path = video_path.replace(Path(video_path).suffix, "_audio.wav")
        audio_ok = await loop.run_in_executor(
            _executor, _extract_audio_from_video, video_path, audio_path
        )

        # ASR处理音频
        asr_text = ""
        if audio_ok and os.path.exists(audio_path):
            asr_text = await loop.run_in_executor(_executor, _run_asr, audio_path)
            try:
                os.unlink(audio_path)
            except Exception:
                pass

        # 处理视频帧（抽帧）
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_every = max(1, int(fps / 5))  # 每秒5帧
        feature_seq = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % sample_every == 0:
                features = await loop.run_in_executor(
                    _executor, _extract_features_from_bgr, frame
                )
                feature_seq.append(features)
            frame_idx += 1

        cap.release()

        _video_tasks[task_id] = {
            "status": "done",
            "progress": 100,
            "asr_text": asr_text,
            "frame_count": len(feature_seq),
            "total_frames": total_frames,
        }
        print(f"[VideoTask] {task_id} 完成，帧数:{len(feature_seq)}，ASR:{asr_text[:50]}")

    except Exception as e:
        _video_tasks[task_id] = {"status": "error", "message": str(e)}
        print(f"[VideoTask] {task_id} 失败: {e}")
    finally:
        try:
            os.unlink(video_path)
        except Exception:
            pass
        if 'audio_path' in locals():
            with suppress(OSError): os.unlink(audio_path)


def _extract_audio_from_video(video_path: str, audio_path: str) -> bool:
    """从视频提取音频（依赖ffmpeg）"""
    try:
        import subprocess
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-ac", "1", "-ar", "16000",
             "-vn", audio_path],
            capture_output=True, timeout=120
        )
        return result.returncode == 0 and os.path.exists(audio_path)
    except Exception as e:
        print(f"[FFmpeg] 音频提取失败: {e}")
        return False


@app.get("/api/video_task/{task_id}")
async def get_video_task(task_id: str,request: Request):
    if _public_deployment() and (_get_user_id_from_request(request) is None or _video_owners.get(task_id)!=_get_user_id_from_request(request)):
        return JSONResponse({'error':'无权访问该视频任务'},status_code=403)
    result = _video_tasks.get(task_id, {"status": "not_found"})
    return JSONResponse(result)


# ══════════════════════════════════════════════════════════════
# RAG 知识库管理 API
# ══════════════════════════════════════════════════════════════

KNOWLEDGE_CORPUS_PATH = ROOT / "data" / "knowledge" / "psychology.json"
RAG_ADMIN_TOKEN = os.environ.get("RAG_ADMIN_TOKEN", "").strip()
_rag_search_retriever = None
_rag_search_provider = "uninitialized"


def _load_bm25_knowledge():
    from services.agent import BM25KnowledgeRetriever

    return BM25KnowledgeRetriever(KNOWLEDGE_CORPUS_PATH)


def _get_rag_search_retriever():
    global _rag_search_provider, _rag_search_retriever
    if _rag_search_retriever is None:
        from services.agent import create_knowledge_retriever

        _rag_search_retriever, _rag_search_provider = create_knowledge_retriever(
            KNOWLEDGE_CORPUS_PATH,
            embedding_model_path=(
                Path(RAG_EMBEDDING_MODEL_PATH).expanduser()
                if RAG_EMBEDDING_MODEL_PATH
                else None
            ),
        )
    return _rag_search_retriever, _rag_search_provider


def _invalidate_rag_caches() -> None:
    global _agent_workflow, _rag_search_provider, _rag_search_retriever
    _agent_workflow = None
    _rag_search_retriever = None
    _rag_search_provider = "uninitialized"


def _knowledge_corpus_label() -> str:
    """返回不暴露项目外绝对路径的语料标识。"""
    try:
        return str(KNOWLEDGE_CORPUS_PATH.relative_to(ROOT))
    except ValueError:
        return KNOWLEDGE_CORPUS_PATH.name


def _verify_rag_admin(request: Request) -> JSONResponse | None:
    if not RAG_ADMIN_TOKEN:
        return JSONResponse(
            {"success": False, "message": "服务端未启用知识库写入"},
            status_code=503,
        )
    supplied = request.headers.get("X-RAG-Admin-Token", "")
    if not supplied or not hmac.compare_digest(supplied, RAG_ADMIN_TOKEN):
        return JSONResponse(
            {"success": False, "message": "管理员凭证无效"},
            status_code=403,
        )
    return None

@app.get("/api/rag/stats")
async def rag_stats():
    """获取知识库统计信息"""
    try:
        retriever = _load_bm25_knowledge()
        _, search_provider = _get_rag_search_retriever()
        return JSONResponse({
            "status": "ready",
            "count": retriever.document_count,
            "db_path": _knowledge_corpus_label(),
            "embedding_model": search_provider,
            "mutable": bool(RAG_ADMIN_TOKEN),
        })
    except (OSError, ValueError) as exc:
        return JSONResponse({
            "status": "invalid",
            "count": 0,
            "db_path": _knowledge_corpus_label(),
            "embedding_model": "BM25 中文二元分词（离线）",
            "mutable": bool(RAG_ADMIN_TOKEN),
            "error": type(exc).__name__,
        }, status_code=503)


@app.get("/api/rag/list")
async def rag_list(limit: int = 50):
    """列出知识库所有文档"""
    try:
        retriever = _load_bm25_knowledge()
        documents = [
            {
                "id": item.document_id,
                "content": item.content,
                "metadata": {
                    "category": "curated",
                    "source": item.source,
                    "source_url": item.source_url,
                },
            }
            for item in retriever.list_documents(limit)
        ]
        return JSONResponse({
            "documents": documents,
            "total": retriever.document_count,
            "mutable": bool(RAG_ADMIN_TOKEN),
        })
    except (OSError, ValueError) as exc:
        return JSONResponse(
            {"documents": [], "total": 0, "error": type(exc).__name__},
            status_code=503,
        )


@app.post("/api/rag/add")
async def rag_add(request: Request):
    """新增知识条目"""
    denied = _verify_rag_admin(request)
    if denied is not None:
        return denied
    try:
        from services.agent import KnowledgeCorpusStore

        body = await request.json()
        content = body.get("content", "")
        category = body.get("category", "custom")
        source = body.get("source", "手动录入")
        source_url = body.get("source_url")
        keywords = body.get("keywords", [])
        if (
            not isinstance(content, str)
            or not isinstance(category, str)
            or not isinstance(source, str)
            or (source_url is not None and not isinstance(source_url, str))
            or not isinstance(keywords, list)
            or not all(isinstance(item, str) for item in keywords)
        ):
            return JSONResponse(
                {"success": False, "message": "知识条目字段类型不正确"},
                status_code=400,
            )
        store = KnowledgeCorpusStore(KNOWLEDGE_CORPUS_PATH)
        document_id = store.add(
            content=content,
            category=category,
            source=source,
            source_url=source_url,
            keywords=keywords,
        )
        _invalidate_rag_caches()
        total = _load_bm25_knowledge().document_count
        return JSONResponse({
            "success": True,
            "message": "添加成功",
            "id": document_id,
            "total": total,
        })
    except ValueError as exc:
        return JSONResponse(
            {"success": False, "message": str(exc)},
            status_code=400,
        )
    except OSError as exc:
        return JSONResponse(
            {"success": False, "message": type(exc).__name__},
            status_code=500,
        )


@app.delete("/api/rag/delete/{doc_id}")
async def rag_delete(doc_id: str, request: Request):
    """删除知识条目"""
    denied = _verify_rag_admin(request)
    if denied is not None:
        return denied
    try:
        from services.agent import KnowledgeCorpusStore

        deleted = KnowledgeCorpusStore(KNOWLEDGE_CORPUS_PATH).delete_custom(doc_id)
        if not deleted:
            return JSONResponse(
                {"success": False, "message": "知识条目不存在"},
                status_code=404,
            )
        _invalidate_rag_caches()
        return JSONResponse({"success": True, "message": "删除成功"})
    except PermissionError as exc:
        return JSONResponse(
            {"success": False, "message": str(exc)},
            status_code=403,
        )
    except (OSError, ValueError) as exc:
        return JSONResponse(
            {"success": False, "message": type(exc).__name__},
            status_code=500,
        )


@app.post("/api/rag/search")
async def rag_search(request: Request):
    """检索测试"""
    try:
        body = await request.json()
        query = body.get("query", "").strip()
        top_k = min(max(int(body.get("top_k", 3)), 1), 20)
        if not query:
            return JSONResponse(
                {"results": [], "message": "查询不能为空"},
                status_code=400,
            )
        retriever, provider = _get_rag_search_retriever()
        matches = await retriever.retrieve(query, top_k=top_k)
        results = [
            {
                "id": item.document_id,
                "document": item.content,
                "metadata": {
                    "source": item.source,
                    "source_url": item.source_url,
                    "category": "curated",
                },
                "similarity": item.score,
            }
            for item in matches
        ]
        return JSONResponse({
            "results": results,
            "query": query,
            "provider": provider,
        })
    except (OSError, ValueError) as exc:
        return JSONResponse(
            {"results": [], "message": type(exc).__name__},
            status_code=503,
        )


@app.get("/rag")
async def rag_page():
    """RAG知识库管理页面"""
    rag_html = ROOT / "rag_kb.html"
    if rag_html.exists():
        return FileResponse(str(rag_html))
    return HTMLResponse("<h1>rag_kb.html 未找到</h1>", status_code=404)


# ══════════════════════════════════════════════════════════════
# 启动入口
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("AI数字人情感陪护 全功能整合后端")
    print(f"访问地址: http://localhost:8800")
    print(f"WebSocket: ws://localhost:8800/ws/main")
    print("=" * 60)

    uvicorn.run(
        "integrated_server:app",
        host="0.0.0.0",
        port=8800,
        reload=False,
        app_dir=str(Path(__file__).parent),
    )
