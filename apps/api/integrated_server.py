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
import io
import json
import os
import re
import sqlite3
import sys
import tempfile
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from urllib.parse import quote

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, File, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from dotenv import load_dotenv

import numpy as np

from services.tts import MacOSSayProvider, Qwen3TtsProvider, TtsProviderError

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
    print("[Auth] 警告: bcrypt 未安装，密码将使用明文存储（不安全）")

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
            CREATE INDEX IF NOT EXISTS idx_sessions_user ON chat_sessions(user_id);
            CREATE INDEX IF NOT EXISTS idx_messages_session ON chat_messages(session_id);
            CREATE INDEX IF NOT EXISTS idx_tokens_user ON auth_tokens(user_id);
            CREATE INDEX IF NOT EXISTS idx_memories_user ON user_memories(user_id);
            CREATE INDEX IF NOT EXISTS idx_agent_runs_user ON agent_runs(user_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_agent_runs_session ON agent_runs(session_id, created_at);
        """)
        conn.commit()
        print("[DB] 数据库初始化完成")
    finally:
        conn.close()

def _hash_password(pwd: str) -> str:
    if HAS_BCRYPT:
        return _bcrypt.hashpw(pwd.encode(), _bcrypt.gensalt()).decode()
    return pwd  # 降级明文

def _check_password(pwd: str, hashed: str) -> bool:
    if HAS_BCRYPT:
        try:
            return _bcrypt.checkpw(pwd.encode(), hashed.encode())
        except Exception:
            return False
    return pwd == hashed

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

def _db_save_message(session_id: str, role: str, content: str, emotion_label: str = ""):
    """持久化一条消息到数据库"""
    try:
        conn = _get_db()
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        conn.execute(
            "INSERT INTO chat_messages(session_id,role,content,emotion_label,ts) VALUES(?,?,?,?,?)",
            (session_id, role, content, emotion_label, now)
        )
        conn.execute("UPDATE chat_sessions SET updated_at=? WHERE id=?", (now, session_id))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[DB] 保存消息失败: {e}")


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

def _get_asr_model():
    global _asr_model, HAS_ASR
    if _asr_model is None:
        try:
            from funasr import AutoModel
            _asr_model = AutoModel(
                model="paraformer-zh",
                vad_model="fsmn-vad",
                punc_model="ct-punc",
            )
            HAS_ASR = True
            print("[IntegratedServer] FunASR AutoModel 初始化成功")
        except Exception as e:
            print(f"[IntegratedServer] FunASR不可用: {e}")
            _asr_model = None
    return _asr_model

def _run_asr(audio_path: str) -> str:
    """同步运行ASR，在线程池中执行"""
    model = _get_asr_model()
    if model is None:
        return ""
    try:
        res = model.generate(input=audio_path, batch_size_s=300)
        if res and isinstance(res, list):
            return "".join(str(item.get("text", "")) for item in res if isinstance(item, dict)).strip()
        return ""
    except Exception as e:
        print(f"[ASR] 识别失败: {e}")
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

# ══════════════════════════════════════════════════════════════
# FastAPI 应用
# ══════════════════════════════════════════════════════════════
app = FastAPI(title="AI数字人情感陪护全功能整合", version="2.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

# 托管 MediaPipe 本地文件（避免 CDN 访问不稳定）
MEDIAPIPE_STATIC_DIR = ROOT / "static" / "mediapipe"
if MEDIAPIPE_STATIC_DIR.exists():
    app.mount("/static/mediapipe", StaticFiles(directory=str(MEDIAPIPE_STATIC_DIR)), name="mediapipe-static")
    print(f"[IntegratedServer] MediaPipe 本地静态资源挂载: {MEDIAPIPE_STATIC_DIR}")

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
            "asr_funasr": HAS_ASR,
            "driver_model": HAS_DRIVER and checkpoint_status.ready,
            "deepseek_api": bool(DEEPSEEK_API_KEY),
            "qwen_api": bool(QWEN_API_KEY),
            "tts_cosyvoice": HAS_TTS and bool(TTS_API_KEY),
            "tts_qwen3": bool(TTS_API_KEY),
            "agent_provider": _configured_agent_provider_name(),
        },
        "tts": _tts_status_payload(),
        "model_assets": {
            "face_driver": checkpoint_status.to_public_dict(),
        },
        "port": request.scope.get("server", (None, None))[1],
    })


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
async def agent_chat(payload: AgentChatRequest):
    """使用离线 Provider 运行一次可观测的智能体工作流。"""
    try:
        trace_id = str(uuid.uuid4())
        session_id = payload.session_id or str(uuid.uuid4())
        result = await _get_agent_workflow().run(
            user_text=payload.text,
            trace_id=trace_id,
            session_id=session_id,
        )
        return JSONResponse({
            "trace_id": trace_id,
            "session_id": session_id,
            "provider": _agent_provider_name,
            "knowledge_provider": _agent_knowledge_provider_name,
            "response": result["final_response"],
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


# ══════════════════════════════════════════════════════════════
# 用户认证接口
# ══════════════════════════════════════════════════════════════
@app.post("/api/auth/register")
async def auth_register(request: Request):
    body = await request.json()
    username = (body.get("username") or "").strip()
    password = (body.get("password") or "").strip()
    if not username or not password:
        return JSONResponse({"error": "用户名和密码不能为空"}, status_code=400)
    if len(username) < 2 or len(username) > 20:
        return JSONResponse({"error": "用户名长度须2-20位"}, status_code=400)
    if len(password) < 4:
        return JSONResponse({"error": "密码至少4位"}, status_code=400)
    conn = _get_db()
    try:
        exists = conn.execute("SELECT id FROM users WHERE username=?", (username,)).fetchone()
        if exists:
            return JSONResponse({"error": "用户名已存在"}, status_code=409)
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        conn.execute(
            "INSERT INTO users(username,password_hash,created_at) VALUES(?,?,?)",
            (username, _hash_password(password), now)
        )
        conn.commit()
        user_id = conn.execute("SELECT id FROM users WHERE username=?", (username,)).fetchone()["id"]
        token = str(uuid.uuid4())
        expires = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(time.time() + 7*86400))
        conn.execute("INSERT INTO auth_tokens(token,user_id,expires_at) VALUES(?,?,?)", (token, user_id, expires))
        conn.commit()
        return JSONResponse({"token": token, "username": username, "user_id": user_id})
    finally:
        conn.close()

@app.post("/api/auth/login")
async def auth_login(request: Request):
    body = await request.json()
    username = (body.get("username") or "").strip()
    password = (body.get("password") or "").strip()
    if not username or not password:
        return JSONResponse({"error": "用户名和密码不能为空"}, status_code=400)
    conn = _get_db()
    try:
        row = conn.execute("SELECT id,password_hash FROM users WHERE username=?", (username,)).fetchone()
        if not row or not _check_password(password, row["password_hash"]):
            return JSONResponse({"error": "用户名或密码错误"}, status_code=401)
        token = str(uuid.uuid4())
        expires = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(time.time() + 7*86400))
        conn.execute("INSERT OR REPLACE INTO auth_tokens(token,user_id,expires_at) VALUES(?,?,?)",
                     (token, row["id"], expires))
        conn.commit()
        return JSONResponse({"token": token, "username": username, "user_id": row["id"]})
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
        return JSONResponse({"enabled": bool(row["enabled"]) if row else False, "count": count})
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
        conn.execute(
            """INSERT INTO user_memory_settings(user_id, enabled, updated_at)
               VALUES(?,?,?)
               ON CONFLICT(user_id) DO UPDATE SET enabled=excluded.enabled,
               updated_at=excluded.updated_at""",
            (user_id, int(payload.enabled), now),
        )
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
        conn.commit()
        if cursor.rowcount == 0:
            return JSONResponse({"error": "记忆不存在"}, status_code=404)
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
async def get_sessions(request: Request):
    user_id = _get_user_id_from_request(request)
    if not user_id:
        return JSONResponse({"error": "未登录"}, status_code=401)
    conn = _get_db()
    try:
        rows = conn.execute(
            "SELECT id,title,created_at,updated_at FROM chat_sessions WHERE user_id=? ORDER BY updated_at DESC LIMIT 100",
            (user_id,)
        ).fetchall()
        return JSONResponse({"sessions": [dict(r) for r in rows]})
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
        conn.execute(
            "INSERT INTO chat_sessions(id,user_id,title,created_at,updated_at) VALUES(?,?,?,?,?)",
            (session_id, user_id, "新对话", now, now)
        )
        conn.commit()
        return JSONResponse({"session_id": session_id, "title": "新对话", "created_at": now})
    finally:
        conn.close()

@app.get("/api/sessions/{session_id}/messages")
async def get_session_messages(session_id: str, request: Request):
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
        msgs = conn.execute(
            "SELECT role,content,emotion_label,ts FROM chat_messages WHERE session_id=? ORDER BY id ASC",
            (session_id,)
        ).fetchall()
        return JSONResponse({"messages": [dict(m) for m in msgs]})
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
            "SELECT id FROM chat_sessions WHERE id=? AND user_id=?",
            (session_id, user_id),
        ).fetchone()
        if not session:
            return JSONResponse({"error": "会话不存在"}, status_code=404)
        conn.execute("DELETE FROM chat_messages WHERE session_id=?", (session_id,))
        conn.execute("DELETE FROM chat_sessions WHERE id=?", (session_id,))
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
            "SELECT id FROM chat_sessions WHERE id=? AND user_id=?",
            (session_id, user_id),
        ).fetchone()
        if not session:
            return JSONResponse({"error": "会话不存在"}, status_code=404)
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
                            {"role": "user", "content": f"请用5-8个汉字总结以下对话的主题，只输出词语，不要标点：\n{summary}"}
                        ], "max_tokens": 20, "temperature": 0.3}
                    )
                    data = resp.json()
                    title = data["choices"][0]["message"]["content"].strip()[:15]
            except Exception as e:
                print(f"[Title] 生成失败: {e}")
                title = msgs[0]["content"][:10] if msgs else "新对话"
        else:
            title = msgs[0]["content"][:10] if msgs else "新对话"
        conn.execute(
            "UPDATE chat_sessions SET title=? WHERE id=? AND user_id=?",
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
        self.memory_consent: bool = False

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
                "vision": HAS_MEDIAPIPE, "asr": HAS_ASR,
                "driver": md is not None,
                "llm": bool(DEEPSEEK_API_KEY or QWEN_API_KEY),
            },
            "driver_model": _driver_runtime_payload(md),
        })

    asyncio.create_task(preload_driver())

    # 驱动参数推送任务（30fps）
    drive_task = asyncio.create_task(_drive_loop(websocket, state, loop))

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except Exception:
                continue

            msg_type = msg.get("type", "")

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
                state.user_id = user_id
                state.db_session_id = db_sid if db_sid else None
                state.agent_messages = (
                    _db_load_message_context(db_sid) if db_sid else []
                )
                state.memory_consent = _memory_enabled_for_user(user_id)
                print(f"[WS] 用户 {user_id} 绑定会话 {db_sid}")
                await _send(websocket, {
                    "type": "session_bound",
                    "session_id": state.db_session_id,
                })
                continue

            elif msg_type == "frame":
                # 解码图像帧 → 提取特征 → 缓冲
                asyncio.create_task(_handle_frame(msg, state, websocket, loop))

            elif msg_type == "audio":
                # 解码音频 → ASR → 触发LLM
                asyncio.create_task(_handle_audio(msg, state, websocket, loop))

            elif msg_type == "text_input":
                # 手动文字输入（ASR不可用时的降级）
                text = msg.get("text", "").strip()
                if text:
                    asyncio.create_task(_trigger_llm(text, state, websocket))

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
    try:
        audio_b64 = msg.get("data", "")
        if not audio_b64:
            return
        audio_bytes = base64.b64decode(audio_b64)
        suffix = msg.get("format", "webm")

        with tempfile.NamedTemporaryFile(suffix=f".{suffix}", delete=False) as f:
            f.write(audio_bytes)
            tmp_path = f.name

        try:
            text = await loop.run_in_executor(_executor, _run_asr, tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

        if text:
            await _send(ws, {"type": "asr_result", "text": text, "is_final": True})
            state.asr_text_buffer += text
            # 触发LLM（节流：距上次触发 > 1s）
            now = time.time()
            if not state.llm_running and (now - state.last_asr_trigger) > 1.0:
                state.last_asr_trigger = now
                full_text = state.asr_text_buffer.strip()
                state.asr_text_buffer = ""
                asyncio.create_task(_trigger_llm(full_text, state, ws))

    except Exception as e:
        print(f"[Audio] 处理失败: {e}")


_MOTION_PATTERN = re.compile(r'\[MOTION:(FlickUp|Tap|Flick3|Idle)\]', re.IGNORECASE)

def _parse_motion_and_clean(reply: str):
    """
    从LLM回复中提取动作标签，返回(清洁文本, 动作名称或None)
    """
    match = _MOTION_PATTERN.search(reply)
    motion = match.group(1) if match else None
    clean_text = _MOTION_PATTERN.sub('', reply).strip()
    return clean_text, motion


async def _trigger_llm(text: str, state: SessionState, ws: WebSocket):
    """通过数字心屿智能体工作流生成回复并同步数字人状态。"""
    if not text or state.llm_running:
        return
    state.llm_running = True
    try:
        await _send(ws, {"type": "llm_thinking", "text": "小安正在思考..."})
        from services.agent import ChatMessage

        trace_id = str(uuid.uuid4())
        session_id = state.db_session_id or state.session_id
        history = [ChatMessage(**message) for message in state.agent_messages]

        async def send_agent_event(event):
            await _send(ws, {
                "type": "agent_event",
                "event": event.model_dump(mode="json"),
            })

        result = await _get_agent_workflow().run(
            user_text=text,
            trace_id=trace_id,
            session_id=session_id,
            messages=history,
            user_id=state.user_id,
            memory_consent=state.memory_consent,
            event_sink=send_agent_event,
        )
        result["trace_id"] = trace_id
        result["session_id"] = session_id
        state.agent_messages = [
            message.model_dump() for message in result.get("messages", [])
        ]
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
        legacy_motion_map = {"Comfort": "FlickUp", "Listen": "Flick3"}
        motion_name = legacy_motion_map.get(avatar.motion, "Idle")

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
            "text": reply_text,
            "emotion": emo_result["emotion"],
            "valence": emo_result["valence"],
            "arousal": emo_result["arousal"],
            "risk_level": emo_result["risk_level"],
            "emotion_label": emo_result["emotion_label"],
        }
        if motion_name:
            msg["motion"] = motion_name
        await _send(ws, msg)

        if state.db_session_id:
            _db_save_message(state.db_session_id, "user", text)
            _db_save_message(state.db_session_id, "assistant", reply_text, emo_result["emotion_label"])
            state.msg_count += 1
            # 第2条消息后触发标题生成（后台异步）
            if state.msg_count == 2:
                asyncio.create_task(_auto_generate_title(state.db_session_id, ws))

    except Exception as e:
        print(f"[Agent] 失败: {e}")
        await _send(ws, {"type": "llm_reply", "text": "抱歉，我暂时无法回应，请稍后再试。",
                         "emotion": "Neutral", "valence": 0, "arousal": 0,
                         "risk_level": "low", "emotion_label": "平静"})
    finally:
        state.llm_running = False


async def _auto_generate_title(db_session_id: str, ws: WebSocket):
    """后台自动为会话生成标题，并通过 WebSocket 推送更新"""
    try:
        conn = _get_db()
        msgs = conn.execute(
            "SELECT role,content FROM chat_messages WHERE session_id=? ORDER BY id ASC LIMIT 6",
            (db_session_id,)
        ).fetchall()
        conn.close()
        if not msgs:
            return
        summary = "\n".join(f"{'用户' if m['role']=='user' else '小安'}: {m['content'][:40]}" for m in msgs)
        title = msgs[0]["content"][:10] if msgs else "新对话"
        if QWEN_API_KEY:
            try:
                import httpx
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.post(
                        f"{QWEN_BASE_URL}/chat/completions",
                        headers={"Authorization": f"Bearer {QWEN_API_KEY}"},
                        json={"model": QWEN_MODEL, "messages": [
                            {"role": "system", "content": "你是文本摘要助手，只输出结果。"},
                            {"role": "user", "content": f"请用5-8个汉字总结以下对话的主题，只输出词语，不要标点：\n{summary}"}
                        ], "max_tokens": 20, "temperature": 0.3}
                    )
                    data = resp.json()
                    title = data["choices"][0]["message"]["content"].strip()[:15]
            except Exception:
                pass
        conn2 = _get_db()
        conn2.execute("UPDATE chat_sessions SET title=? WHERE id=?", (title, db_session_id))
        conn2.commit()
        conn2.close()
        # 推送标题更新给前端
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

            # 同步广播给 /ws/drive 的客户端（integrated.html 驱动通道）
            if _drive_clients:
                drive_msg = {"type": "params", "data": params}
                if state.pending_motion:
                    drive_msg["motion"] = state.pending_motion
                    state.pending_motion = None
                dead = set()
                for dc in list(_drive_clients):
                    try:
                        await dc.send_json(drive_msg)
                    except Exception:
                        dead.add(dc)
                _drive_clients.difference_update(dead)

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
async def upload_video(file: UploadFile = File(...)):
    """上传MP4文件，返回task_id，通过WS推送处理进度"""
    suffix = Path(file.filename).suffix.lower() if file.filename else ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        content = await file.read()
        f.write(content)
        tmp_path = f.name

    task_id = str(uuid.uuid4())
    # 后台处理任务
    asyncio.create_task(_process_video_file(tmp_path, task_id))
    return JSONResponse({"task_id": task_id, "status": "processing",
                         "message": f"视频已接收（{len(content)//1024}KB），开始处理"})


# 视频任务状态存储
_video_tasks: Dict[str, Dict] = {}

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
async def get_video_task(task_id: str):
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
