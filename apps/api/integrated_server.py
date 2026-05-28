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
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, File, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

import numpy as np

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
            CREATE INDEX IF NOT EXISTS idx_sessions_user ON chat_sessions(user_id);
            CREATE INDEX IF NOT EXISTS idx_messages_session ON chat_messages(session_id);
            CREATE INDEX IF NOT EXISTS idx_tokens_user ON auth_tokens(user_id);
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

# ── RAG 心理学知识库 ──────────────────────────────────────────
_rag_engine = None
HAS_RAG = False

def _init_rag():
    """延迟初始化 RAG 引擎 + 心理学知识库"""
    global _rag_engine, HAS_RAG
    try:
        from services.llm.rag_engine import RAGEngine, PsychologyKnowledgeBase
        _rag_engine = RAGEngine()
        PsychologyKnowledgeBase.initialize_kb(_rag_engine)
        HAS_RAG = True
        stats = _rag_engine.get_stats()
        print(f"[RAG] 心理学知识库就绪，文档数: {stats.get('count', 0)}")
    except Exception as e:
        print(f"[RAG] 初始化失败（降级运行）: {e}")
        _rag_engine = None

# ══════════════════════════════════════════════════════════════
# 模块懒加载（允许部分模块缺失时降级运行）
# ══════════════════════════════════════════════════════════════

# 1. FaceBehaviorModel（数字人面部行为驱动）
_face_driver = None
_face_driver_lock = asyncio.Lock()
HAS_DRIVER = False
try:
    import torch
    import torch.nn as nn

    class FaceBehaviorModel(nn.Module):
        def __init__(self, input_dim=25, hidden_dim=512, num_layers=6,
                     num_candidates=10, dropout=0.2):
            super().__init__()
            self.num_candidates = num_candidates
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim),
                nn.ReLU(), nn.Dropout(dropout)
            )
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim * 4,
                dropout=dropout, batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.decoders = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2),
                    nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim // 2, input_dim)
                ) for _ in range(num_candidates)
            ])

        def forward(self, x):
            h = self.encoder(x)
            h = self.transformer(h)
            outputs = [dec(h) for dec in self.decoders]
            return torch.stack(outputs, dim=1)

    HAS_DRIVER = True
    print("[IntegratedServer] FaceBehaviorModel 定义成功")
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
    if not HAS_DRIVER:
        return None
    model_path = ROOT / "digital_human_engine" / "checkpoints_v2" / "best_model.pt"
    if not model_path.exists():
        print(f"[Driver] 模型文件不存在: {model_path}")
        return None
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = FaceBehaviorModel().to(device)
        ckpt = torch.load(str(model_path), map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        print(f"[Driver] 模型加载成功，设备: {device}")
        return (model, device)
    except Exception as e:
        print(f"[Driver] 模型加载失败: {e}")
        return None

def _infer_live2d_params(model_device, feature_seq: np.ndarray) -> Dict[str, float]:
    """推理Live2D参数，在线程池中执行"""
    model, device = model_device
    try:
        with torch.no_grad():
            t = torch.FloatTensor(feature_seq).unsqueeze(0).to(device)  # [1,T,25]
            pred = model(t)  # [1,K,T,25]
            p = pred[0, 0, -1].cpu().numpy()  # [25]
        param_names = list(LIVE2D_PARAMS.keys())
        result = {}
        for i, name in enumerate(param_names):
            lo, hi = LIVE2D_PARAMS[name]
            raw = float(np.clip(p[i % len(p)], -1, 1))
            result[name] = round((raw + 1) / 2 * (hi - lo) + lo, 3)
        return result
    except Exception as e:
        return _default_live2d_params()

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

# 4. Qwen API
QWEN_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "sk-81e2a139a4ed42c3a004fd2d67f5de7f")
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_MODEL = "qwen-plus"
_session_histories: Dict[str, list] = {}

_SYSTEM_PROMPT = """你是一位专业的心理陪护助手，名叫小安，外表是温柔的动漫女孩形象。

【核心职责】
- 共情倾听：感受用户情绪，给予真诚回应
- 情感支持：用温暖、积极的语言帮助用户舒缓情绪
- 识别危机：当用户出现自伤、轻生等信号时，立即提供危机热线

【安全红线】
- 不进行医疗诊断，不提供药物建议
- 不做出无法兑现的承诺

【回复格式要求】
每次回复须包含两部分（严格按此格式）：
1. 正文：温暖共情的回复，2-3句话，不超过80字，用中文
2. 动作标签：在正文末尾另起一行，根据对话情感选择最合适的一个动作标签：
   - [MOTION:FlickUp] —— 用户表达悲伤、感动、哭泣时
   - [MOTION:Tap] —— 用户表达惊喜、害羞、意外时
   - [MOTION:Flick3] —— 用户自我否定、需要鼓励或摇头安慰时
   - [MOTION:Idle] —— 正常倾听、对话平稳时（默认）

【示例】
用户说"我最近很难过，总是哭"
回复：
我听到你了，最近一定承受了很多。哭出来没什么不好，这是你在释放情绪。我陪着你。
[MOTION:FlickUp]"""

EMOTION_RULES = [
    (r'压力|焦虑|紧张|烦躁|烦恼', 'Anxiety', -0.3, 0.5, 'medium', '关切'),
    (r'难过|伤心|悲伤|哭|失落|痛苦', 'Sad', -0.5, -0.2, 'medium', '温柔'),
    (r'开心|高兴|快乐|棒|好消息|很好|很棒', 'Happy', 0.7, 0.4, 'low', '喜悦'),
    (r'睡不着|失眠|睡眠|睡不好', 'Anxiety', -0.2, 0.2, 'low', '关心'),
    (r'孤独|孤单|没人|一个人', 'Sad', -0.3, -0.3, 'low', '陪伴'),
    (r'想死|不想活|轻生|自杀|放弃生命', 'Fear', -0.9, 0.3, 'high', '紧急'),
    (r'抑郁|抑郁症|双相|躁郁', 'Sad', -0.5, -0.1, 'medium', '专注'),
]

def _local_analyze(text: str) -> Dict[str, Any]:
    for pattern, emotion, valence, arousal, risk, label in EMOTION_RULES:
        if re.search(pattern, text):
            return {"emotion": emotion, "valence": valence,
                    "arousal": arousal, "risk_level": risk, "emotion_label": label}
    return {"emotion": "Neutral", "valence": 0.05,
            "arousal": 0.0, "risk_level": "low", "emotion_label": "平静"}

def _get_rag_context(text: str) -> str:
    """从RAG知识库检索相关心理学知识，构建上下文"""
    if not HAS_RAG or _rag_engine is None:
        return ""
    try:
        context = _rag_engine.build_context(text, top_k=3)
        return context
    except Exception:
        return ""

_SAFETY_KEYWORDS = re.compile(r'想死|不想活|轻生|自杀|结束生命|活不下去|去死|死了算了')

def _check_crisis(text: str) -> bool:
    """危机信号检测"""
    return bool(_SAFETY_KEYWORDS.search(text))

async def _qwen_reply(text: str, session_id: str) -> Optional[str]:
    if not QWEN_API_KEY:
        return None
    try:
        import httpx
        # 危机信号优先处理
        if _check_crisis(text):
            crisis_reply = ("我非常担心你现在的状态，你说的话让我很揪心。"
                           "请立即拨打心理援助热线：400-161-9995 或 12320。"
                           "你不是一个人在承受这些，我陪着你。\n[MOTION:FlickUp]")
            history = _session_histories.setdefault(session_id, [])
            history.append({"role": "user", "content": text})
            history.append({"role": "assistant", "content": crisis_reply})
            return crisis_reply

        # RAG 检索心理学知识
        rag_context = _get_rag_context(text)

        # 构建增强系统提示词
        if rag_context:
            system_content = _SYSTEM_PROMPT + f"\n\n{rag_context}\n\n请结合以上知识给出更专业的回应。"
            enable_search = False   # RAG 已有上下文，无需联网
        else:
            system_content = _SYSTEM_PROMPT
            enable_search = True    # 知识库无相关内容，启用联网搜索补充

        history = _session_histories.setdefault(session_id, [])
        history.append({"role": "user", "content": text})
        if len(history) > 20:
            history[:] = history[-20:]
        messages = [{"role": "system", "content": system_content}] + history
        # 构建请求体，RAG无结果时启用联网搜索
        request_body = {
            "model": QWEN_MODEL,
            "messages": messages,
            "max_tokens": 250,
            "temperature": 0.75,
        }
        if enable_search:
            request_body["enable_search"] = True
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{QWEN_BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {QWEN_API_KEY}"},
                json=request_body
            )
            data = resp.json()
            reply = data["choices"][0]["message"]["content"].strip()
            history.append({"role": "assistant", "content": reply})
            return reply
    except Exception as e:
        print(f"[Qwen API] 失败: {e}")
        return None

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

@app.on_event("startup")
async def _on_startup():
    """应用启动时在后台线程初始化 RAG"""
    loop = asyncio.get_event_loop()
    loop.run_in_executor(_executor, _init_rag)
    print("[IntegratedServer] 后台 RAG 初始化已启动")

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
async def api_status():
    return JSONResponse({
        "status": "running",
        "modules": {
            "vision_mediapipe": HAS_MEDIAPIPE,
            "asr_funasr": HAS_ASR,
            "driver_model": HAS_DRIVER,
            "qwen_api": bool(QWEN_API_KEY),
        },
        "port": 8800,
    })


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
        conn.execute("DELETE FROM chat_messages WHERE session_id=?", (session_id,))
        conn.execute("DELETE FROM chat_sessions WHERE id=? AND user_id=?", (session_id, user_id))
        conn.commit()
        _session_histories.pop(session_id, None)
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
        conn.execute("UPDATE chat_sessions SET title=? WHERE id=?", (title, session_id))
        conn.commit()
        return JSONResponse({"title": title, "session_id": session_id})
    finally:
        conn.close()



# CosyVoice TTS 接口 —— 使用 dashscope SDK
try:
    import dashscope
    from dashscope.audio.tts_v2 import SpeechSynthesizer as _TtsSynthesizer
    dashscope.api_key = QWEN_API_KEY
    HAS_TTS = True
    print(f"[IntegratedServer] CosyVoice TTS 已加载")
except ImportError:
    HAS_TTS = False
    print("[IntegratedServer] 警告: dashscope 未安装，TTS不可用")

@app.get("/api/tts")
async def api_tts(text: str):
    """调用阿里云 CosyVoice TTS 生成音频，返回 audio/mpeg"""
    if not HAS_TTS or not QWEN_API_KEY or not text:
        return JSONResponse({"error": "TTS not available"}, status_code=400)
    try:
        loop = asyncio.get_event_loop()
        # SDK 是同步调用，放到线程池运行避免阻塞
        def _synthesize():
            synth = _TtsSynthesizer(model="cosyvoice-v1", voice="longxiaochun")
            return synth.call(text)
        audio_bytes = await loop.run_in_executor(_executor, _synthesize)
        if not audio_bytes or len(audio_bytes) == 0:
            print("[TTS] CosyVoice 返回空音频")
            return JSONResponse({"error": "empty audio"}, status_code=502)
        print(f"[TTS] CosyVoice 成功，{len(audio_bytes)} 字节")
        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type="audio/mpeg",
            headers={"Content-Length": str(len(audio_bytes))}
        )
    except Exception as e:
        import traceback
        print(f"[TTS] 异常: {e}")
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


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
        # 待触发的动作
        self.pending_motion: Optional[str] = None

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
            # 保持连接，等待断开
            await asyncio.sleep(30)
            await websocket.send_json({"type": "ping"})
    except Exception:
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
        if HAS_DRIVER:
            md = await loop.run_in_executor(_executor, _load_face_driver)
            state.model_device = md
            await _send(websocket, {"type": "status",
                "modules": {
                    "vision": HAS_MEDIAPIPE, "asr": HAS_ASR,
                    "driver": md is not None, "llm": bool(QWEN_API_KEY)
                }
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
                if user_id:
                    state.user_id = user_id
                    state.db_session_id = db_sid if db_sid else None
                    print(f"[WS] 用户 {user_id} 绑定会话 {db_sid}")
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
                    _session_histories.pop(session_id, None)
                    await _send(websocket, {"type": "reset_ack"})

    except WebSocketDisconnect:
        print(f"[WS] 断开: {session_id}")
    except Exception as e:
        print(f"[WS] 异常: {e}")
        traceback.print_exc()
    finally:
        drive_task.cancel()
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
    """调用Qwen API生成回复，解析动作标签，并发送到前端"""
    if not text or state.llm_running:
        return
    state.llm_running = True
    try:
        # 先发"思考中"状态
        await _send(ws, {"type": "llm_thinking", "text": "小安正在思考..."})

        # 调用Qwen API（含RAG + 危机检测）
        reply = await _qwen_reply(text, state.session_id)

        # 降级到规则
        if not reply:
            emo = _local_analyze(text)
            fallback_map = {
                'Anxiety': ('我听到你了，这种感受很正常。能和我多说说吗？', 'Flick3'),
                'Sad':     ('谢谢你愿意告诉我这些。我在这里陪着你。', 'FlickUp'),
                'Happy':   ('听到你这么说我也很开心！', 'Tap'),
                'Fear':    ('我非常担心你。请立即拨打心理援助热线 400-161-9995。', 'FlickUp'),
            }
            fb = fallback_map.get(emo['emotion'],
                                  ('谢谢你的分享，我在认真倾听。你现在最想聊的是什么？', 'Idle'))
            reply_text, motion_name = fb[0], fb[1]
            emo_result = emo
        else:
            # 解析动作标签
            reply_text, motion_name = _parse_motion_and_clean(reply)
            if not motion_name:
                motion_name = 'Idle'
            emo_result = _local_analyze(text)

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

        # 发送回复（含动作触发）
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

        # 消息持久化：保存到数据库
        if state.db_session_id:
            _db_save_message(state.db_session_id, "user", text)
            _db_save_message(state.db_session_id, "assistant", reply_text, emo_result["emotion_label"])
            state.msg_count += 1
            # 第2条消息后触发标题生成（后台异步）
            if state.msg_count == 2:
                asyncio.create_task(_auto_generate_title(state.db_session_id, ws))

    except Exception as e:
        print(f"[LLM] 失败: {e}")
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
    if state.model_device is None or len(state.feature_buffer) < state.seq_len:
        # 模型未就绪时：使用简单规则映射
        return _features_to_params_simple(state)

    seq = np.array(state.feature_buffer[-state.seq_len:], dtype=np.float32)
    params = await loop.run_in_executor(
        _executor, _infer_live2d_params, state.model_device, seq
    )
    return params


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

@app.get("/api/rag/stats")
async def rag_stats():
    """获取知识库统计信息"""
    if not HAS_RAG or _rag_engine is None:
        return JSONResponse({"status": "uninitialized", "count": 0,
                             "db_path": "", "embedding_model": "TF-IDF本地"})
    stats = _rag_engine.get_stats()
    stats["embedding_model"] = "TF-IDF本地（离线）"
    return JSONResponse(stats)


@app.get("/api/rag/list")
async def rag_list(limit: int = 50):
    """列出知识库所有文档"""
    if not HAS_RAG or _rag_engine is None:
        return JSONResponse({"documents": [], "total": 0})
    try:
        coll = _rag_engine._collection
        if coll is None:
            return JSONResponse({"documents": [], "total": 0})
        total = coll.count()
        result = coll.get(limit=limit, include=["documents", "metadatas"])
        docs = []
        for i, doc_id in enumerate(result.get("ids", [])):
            docs.append({
                "id": doc_id,
                "content": result["documents"][i] if result.get("documents") else "",
                "metadata": result["metadatas"][i] if result.get("metadatas") else {},
            })
        return JSONResponse({"documents": docs, "total": total})
    except Exception as e:
        return JSONResponse({"documents": [], "total": 0, "error": str(e)})


@app.post("/api/rag/add")
async def rag_add(request: Request):
    """新增知识条目"""
    if not HAS_RAG or _rag_engine is None:
        return JSONResponse({"success": False, "message": "RAG未初始化"}, status_code=503)
    try:
        body = await request.json()
        content = body.get("content", "").strip()
        category = body.get("category", "custom")
        source = body.get("source", "手动录入")
        if not content:
            return JSONResponse({"success": False, "message": "内容不能为空"}, status_code=400)
        _rag_engine.add_documents(
            documents=[content],
            metadatas=[{"category": category, "source": source}]
        )
        stats = _rag_engine.get_stats()
        return JSONResponse({"success": True, "message": "添加成功",
                             "total": stats.get("count", 0)})
    except Exception as e:
        return JSONResponse({"success": False, "message": str(e)}, status_code=500)


@app.delete("/api/rag/delete/{doc_id}")
async def rag_delete(doc_id: str):
    """删除知识条目"""
    if not HAS_RAG or _rag_engine is None:
        return JSONResponse({"success": False, "message": "RAG未初始化"}, status_code=503)
    try:
        _rag_engine._collection.delete(ids=[doc_id])
        return JSONResponse({"success": True, "message": "删除成功"})
    except Exception as e:
        return JSONResponse({"success": False, "message": str(e)}, status_code=500)


@app.post("/api/rag/search")
async def rag_search(request: Request):
    """检索测试"""
    if not HAS_RAG or _rag_engine is None:
        return JSONResponse({"results": [], "message": "RAG未初始化"})
    try:
        body = await request.json()
        query = body.get("query", "").strip()
        top_k = int(body.get("top_k", 3))
        if not query:
            return JSONResponse({"results": [], "message": "查询不能为空"})
        results = _rag_engine.retrieve(query, top_k=top_k)
        return JSONResponse({"results": results, "query": query})
    except Exception as e:
        return JSONResponse({"results": [], "message": str(e)})


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
