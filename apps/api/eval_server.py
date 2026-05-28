#!/usr/bin/env python3
"""
视觉听觉评测服务
支持:
  POST /asr                 ← 官方评测接口：Body为Binary mp3，返回{"result":"..."}
  POST /api/eval/asr        ← 上传MP4/音频 → ASR转文字
  POST /api/eval/wer        ← 提交(hypothesis, reference) → 计算WER
  POST /api/eval/ser        ← 上传MP4 → 视觉情感识别SER
  POST /api/eval/full       ← 上传MP4 + 参考文本 → 完整评测(ASR+WER+SER)
  GET  /eval                ← 返回 eval.html 前端页面
"""
from __future__ import annotations

import io
import os
import sys
import json
import math
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, File, Form, Request, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import asyncio
from concurrent.futures import ThreadPoolExecutor

_thread_pool = ThreadPoolExecutor(max_workers=2)

import numpy as np

# ── 可选依赖（降级处理）─────────────────────────────────────────
try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# ASR模块（懒加载）
_asr_model = None   # 直接用 FunASR AutoModel，内置分句，不经过自研VAD

def get_asr_model():
    """直接调 FunASR AutoModel，支持长音频+内置VAD分句"""
    global _asr_model
    if _asr_model is None:
        try:
            from funasr import AutoModel
            # paraformer-zh-streaming 或 paraformer-zh 均可
            # vad_model 让 FunASR 内置 VAD 自动切句，避免漏识别
            _asr_model = AutoModel(
                model="paraformer-zh",
                vad_model="fsmn-vad",
                punc_model="ct-punc",   # 标点恢复
            )
            print("[EvalServer] FunASR AutoModel (paraformer-zh + fsmn-vad + ct-punc) 初始化成功")
        except Exception as e:
            print(f"[EvalServer] FunASR AutoModel 初始化失败: {e}")
            _asr_model = None
    return _asr_model


# 保留旧 pipeline 用于兼容（此处不再使用）
_asr_pipeline = None

def get_asr():
    return None   # 已改用 get_asr_model()


# ── FastAPI App ─────────────────────────────────────────────────
app = FastAPI(title="视觉听觉评测服务", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

EVAL_HTML = ROOT / "eval.html"


@app.get("/eval", response_class=HTMLResponse)
async def serve_eval():
    if EVAL_HTML.exists():
        return HTMLResponse(EVAL_HTML.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>eval.html not found</h1>", status_code=404)


# ════════════════════════════════════════════════════════
# 官方评测接口
# POST /asr
# Body: Binary mp3 文件内容（Content-Type: application/octet-stream）
# 返回: {"result": "识别出的文字"}
# ════════════════════════════════════════════════════════
@app.post("/asr")
async def official_asr(request: Request):
    """官方ASR评测接口：Body直接是mp3二进制，返回{\"result\": \"...\"}\n"""
    audio_bytes = await request.body()
    if not audio_bytes:
        return JSONResponse({"result": ""}, status_code=400)

    # 写入临时文件（保留.mp3后缀，让FunASR/librosa正确解码）
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        # 在线程池里运行 FunASR，避免阻塞事件循环
        loop = asyncio.get_event_loop()
        asr_result = await loop.run_in_executor(_thread_pool, run_funasr, tmp_path)
        text = asr_result.get("text", "").strip()
        print(f"[/asr] 识别结果: {text[:50]}..." if len(text) > 50 else f"[/asr] 识别结果: {text}")
        return JSONResponse({"result": text})
    except Exception as e:
        print(f"[/asr] 识别失败: {e}")
        return JSONResponse({"result": "", "error": str(e)}, status_code=500)
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ── 工具函数 ─────────────────────────────────────────────────────

def extract_audio_from_mp4(video_path: str, target_sr: int = 16000) -> np.ndarray:
    """从MP4中提取音频，返回16kHz单声道int16数组"""
    if not HAS_LIBROSA:
        raise RuntimeError("librosa未安装，请执行: pip install librosa")
    audio, sr = librosa.load(video_path, sr=target_sr, mono=True)
    pcm16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
    return pcm16


def compute_wer(reference: str, hypothesis: str) -> Dict[str, Any]:
    """
    计算词错误率 WER（字级别，适合中文）
    WER = (S + D + I) / N
    S=替换, D=删除, I=插入, N=参考字数
    """
    # 中文按字拆分，英文按空格拆分
    def tokenize(text: str) -> List[str]:
        text = text.strip()
        # 判断是否主要是中文
        cn_count = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        if cn_count > len(text) * 0.3:
            # 中文：按字符分割，过滤空格和标点
            return [c for c in text if c.strip() and c not in '，。！？、；：""''（）【】…']
        else:
            return text.split()

    ref_tokens = tokenize(reference)
    hyp_tokens = tokenize(hypothesis)

    N = len(ref_tokens)
    if N == 0:
        return {"wer": 0.0, "substitutions": 0, "deletions": 0, "insertions": 0,
                "ref_len": 0, "hyp_len": len(hyp_tokens), "detail": "参考文本为空"}

    # 动态规划编辑距离
    r, h = len(ref_tokens), len(hyp_tokens)
    dp = [[0] * (h + 1) for _ in range(r + 1)]
    for i in range(r + 1):
        dp[i][0] = i
    for j in range(h + 1):
        dp[0][j] = j
    for i in range(1, r + 1):
        for j in range(1, h + 1):
            if ref_tokens[i-1] == hyp_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j-1],  # 替换
                                   dp[i-1][j],     # 删除
                                   dp[i][j-1])     # 插入

    # 回溯计算S/D/I
    s, d, ins = 0, 0, 0
    i, j = r, h
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_tokens[i-1] == hyp_tokens[j-1]:
            i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            s += 1; i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            d += 1; i -= 1
        else:
            ins += 1; j -= 1

    wer = (s + d + ins) / N
    return {
        "wer": round(wer, 4),
        "wer_pct": f"{wer*100:.2f}%",
        "substitutions": s,
        "deletions": d,
        "insertions": ins,
        "ref_len": N,
        "hyp_len": h,
        "edit_distance": dp[r][h],
    }


def compute_ser_from_video(video_path: str) -> Dict[str, Any]:
    """
    从视频中提取帧并进行情感识别（SER）
    输出：主导情绪、AU强度分布、情感维度(Valence/Arousal)
    """
    if not HAS_CV2:
        return {"error": "opencv-python未安装", "emotion": "N/A", "confidence": 0}

    try:
        from services.vision.inference.mediapipe_face import MediaPipeFaceProcessor, VisionConfig

        # 找MediaPipe模型
        model_candidates = [
            str(ROOT / "models" / "face_landmarker.task"),
            str(ROOT / "models" / "face_landmarker_v2_with_blendshapes.task"),
            "face_landmarker.task",
        ]
        model_path = next((p for p in model_candidates if Path(p).exists()), None)
        if not model_path:
            return {"error": "MediaPipe模型文件未找到，请下载 face_landmarker.task 到 models/", 
                    "emotion": "N/A", "confidence": 0}

        cfg = VisionConfig(model_path=model_path, frame_rate_fps=5, enable_gaze=False)
        processor = MediaPipeFaceProcessor(cfg)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_step = max(1, int(fps / 5))  # 每秒采5帧

        emotion_votes: Dict[str, int] = {}
        au_accum: Dict[str, List[float]] = {}
        va_list: List[Dict] = []
        processed = 0

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % sample_step == 0:
                result = processor.process_frame(frame)
                if result and result.get("faces"):
                    face = result["faces"][0]
                    # 情绪
                    emo = face.get("expression", "Neutral")
                    if isinstance(emo, dict):
                        emo = max(emo, key=emo.get)
                    emotion_votes[emo] = emotion_votes.get(emo, 0) + 1
                    # AU
                    for k, v in face.get("au", {}).items():
                        au_accum.setdefault(k, []).append(float(v))
                    # VA
                    va = face.get("va", {})
                    if va:
                        va_list.append(va)
                    processed += 1
            frame_idx += 1

        cap.release()
        processor.close()

        if processed == 0:
            return {"error": "未检测到人脸", "emotion": "N/A", "frames_processed": 0}

        dominant_emotion = max(emotion_votes, key=emotion_votes.get) if emotion_votes else "Neutral"
        emotion_dist = {k: round(v/processed, 3) for k, v in emotion_votes.items()}
        au_mean = {k: round(float(np.mean(v)), 4) for k, v in au_accum.items()}
        va_mean = {}
        if va_list:
            va_mean = {
                "valence": round(float(np.mean([x.get("valence", 0) for x in va_list])), 4),
                "arousal": round(float(np.mean([x.get("arousal", 0) for x in va_list])), 4),
            }

        # 简单SER准确率估算（有参考时才准确，这里给出置信度）
        top_conf = emotion_votes.get(dominant_emotion, 0) / processed if processed else 0

        return {
            "dominant_emotion": dominant_emotion,
            "confidence": round(top_conf, 3),
            "emotion_distribution": emotion_dist,
            "au_mean": au_mean,
            "va_mean": va_mean,
            "frames_sampled": processed,
            "total_frames": total_frames,
        }

    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc(), "emotion": "N/A"}


def compute_ser_accuracy(predictions: List[str], references: List[str]) -> Dict[str, Any]:
    """计算SER准确率（需要有标注参考时使用）"""
    if len(predictions) != len(references):
        return {"error": "预测数量与参考数量不匹配"}
    correct = sum(1 for p, r in zip(predictions, references) if p.strip() == r.strip())
    total = len(predictions)
    return {
        "ser_accuracy": round(correct / total, 4) if total > 0 else 0,
        "ser_pct": f"{correct/total*100:.2f}%" if total > 0 else "0%",
        "correct": correct,
        "total": total,
    }


def split_sentences(text: str) -> List[str]:
    """按标点切句，返回非空句子列表"""
    import re
    # 按中文句末标点、换行切分
    parts = re.split(r'[。！？\n\.!?]+', text)
    return [p.strip() for p in parts if p.strip()]


def _clean_text(s: str) -> str:
    """去标点空格，保留纯汉字/字母/数字用于匹配"""
    import re
    return re.sub(r'[^\u4e00-\u9fffA-Za-z0-9]', '', s)


def _lcs_length(a: str, b: str) -> int:
    """计算两个字符串的LCS（最长公共子序列）长度"""
    if not a or not b:
        return 0
    la, lb = len(a), len(b)
    # 空间优化：滚动数组
    prev = [0] * (lb + 1)
    curr = [0] * (lb + 1)
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            if a[i-1] == b[j-1]:
                curr[j] = prev[j-1] + 1
            else:
                curr[j] = max(prev[j], curr[j-1])
        prev, curr = curr, [0] * (lb + 1)
    return prev[lb]


def _find_best_window(ref_clean: str, hyp_clean: str, search_start: int = 0) -> tuple:
    """
    在 hyp_clean 中从 search_start 开始，用滑动窗口找到与 ref_clean 最匹配的子串。
    返回 (best_start, best_end, recall)：
    - recall = LCS(ref, hyp_window) / len(ref)，越大越好
    """
    ref_len = len(ref_clean)
    hyp_len = len(hyp_clean)
    if ref_len == 0 or hyp_len == 0:
        return (0, 0, 0.0)

    best_recall = 0.0
    best_start = search_start
    best_end = search_start

    # 窗口大小：从 0.5x 到 3x 参考句长度
    min_win = max(ref_len // 2, 1)
    max_win = min(ref_len * 3, hyp_len - search_start)

    # 粗搜索：以 ref_len 为基础窗口，在搜索范围内滑动
    for win_size in [ref_len, int(ref_len * 1.5), ref_len * 2, min_win]:
        if win_size > hyp_len - search_start:
            win_size = hyp_len - search_start
        step = max(1, ref_len // 4)
        for start in range(search_start, hyp_len - min_win + 1, step):
            end = min(start + win_size, hyp_len)
            window = hyp_clean[start:end]
            lcs = _lcs_length(ref_clean, window)
            recall = lcs / ref_len
            if recall > best_recall:
                best_recall = recall
                best_start = start
                best_end = end

    return (best_start, best_end, best_recall)


def compute_text_ser(reference: str, hypothesis: str) -> Dict[str, Any]:
    """
    句错率 SER（纯文本，不依赖视觉）
    SER = 错误句数 / 总句数

    核心策略：不切分识别文本！
    1. 只对参考文本按句切分
    2. 对每句参考文本，直接在识别全文中用 LCS 查找其内容
    3. 若 recall(LCS长度/参考句长度) >= 阈值，则该句被正确识别
    """
    ref_sents = split_sentences(reference)
    total = len(ref_sents)
    if total == 0:
        return {"ser": 0.0, "ser_pct": "0.00%", "total_sentences": 0,
                "error_sentences": 0, "sentences": []}

    # 清洗识别全文（不切分！保持完整）
    hyp_clean = _clean_text(hypothesis)
    hyp_full = hypothesis.strip()

    sentences = []
    error_count = 0
    search_pos = 0  # 顺序搜索指针，保持前后顺序

    RECALL_THRESHOLD = 0.5  # recall >= 50% 判为正确

    for idx, ref_s in enumerate(ref_sents):
        ref_clean = _clean_text(ref_s)
        ref_clen = len(ref_clean)

        if ref_clen == 0:
            sentences.append({
                "index": idx + 1, "ref": ref_s, "hyp": "",
                "wer": "0.00%", "substitutions": 0, "deletions": 0,
                "insertions": 0, "correct": True,
            })
            continue

        # 在识别全文中找最佳匹配区域
        best_start, best_end, recall = _find_best_window(ref_clean, hyp_clean, search_pos)

        # 从匹配区域反推对应的原始 hyp 文本片段（用于展示）
        # 简单方式：按比例映射回原始文本
        if len(hyp_clean) > 0 and best_end > best_start:
            ratio_s = best_start / len(hyp_clean)
            ratio_e = best_end / len(hyp_clean)
            orig_s = int(ratio_s * len(hyp_full))
            orig_e = int(ratio_e * len(hyp_full))
            hyp_segment = hyp_full[orig_s:orig_e].strip()
            # 前进搜索指针（保证后续句子在后面找）
            search_pos = max(search_pos, best_start + 1)
        else:
            hyp_segment = ""

        # 计算该句的 WER（用于展示）
        wer_result = compute_wer(ref_s, hyp_segment) if hyp_segment else {
            "wer": 1.0, "wer_pct": "100.00%",
            "substitutions": 0, "deletions": len(ref_clean), "insertions": 0,
            "ref_len": ref_clen, "hyp_len": 0, "edit_distance": ref_clen
        }

        # 用 LCS recall 判断正确/错误（核心！不用 WER）
        is_correct = recall >= RECALL_THRESHOLD
        if not is_correct:
            error_count += 1

        sentences.append({
            "index": idx + 1,
            "ref": ref_s,
            "hyp": hyp_segment,
            "wer": wer_result["wer_pct"],
            "recall": f"{recall*100:.1f}%",
            "substitutions": wer_result.get("substitutions", 0),
            "deletions": wer_result.get("deletions", 0),
            "insertions": wer_result.get("insertions", 0),
            "correct": is_correct,
        })

    ser = error_count / total if total > 0 else 0.0
    return {
        "ser": round(ser, 4),
        "ser_pct": f"{ser * 100:.2f}%",
        "total_sentences": total,
        "error_sentences": error_count,
        "sentences": sentences,
    }


def run_funasr(file_path: str) -> Dict[str, Any]:
    """
    直接用 FunASR AutoModel 处理音频/视频文件
    内置 VAD 分句，支持长音频，返回统一格式
    """
    model = get_asr_model()
    if model is None:
        return {"text": "[ASR引擎未就绪，请检查FunASR安装]",
                "confidence": None, "segments": [], "words": [], "status": "asr_unavailable"}

    res = model.generate(input=file_path, batch_size_s=300)

    # 解析 FunASR 的返回格式
    full_text = ""
    segments = []
    confidence = None
    words = []

    if res and isinstance(res, list):
        for item in res:
            if not isinstance(item, dict):
                continue
            t = str(item.get("text", "")).strip()
            if t:
                full_text += t
            if "confidence" in item:
                try:
                    confidence = float(item["confidence"])
                except Exception:
                    pass
            # 时间戳（key 可能是 timestamp 或 sentence_info）
            ts = item.get("timestamp") or item.get("sentence_info") or []
            if isinstance(ts, list):
                for s in ts:
                    if isinstance(s, (list, tuple)) and len(s) >= 2:
                        seg_text = str(s[0]) if len(s) >= 3 else ""
                        start_ms = int(float(s[-2])) if len(s) >= 2 else 0
                        end_ms   = int(float(s[-1])) if len(s) >= 1 else 0
                        segments.append({"start_ms": start_ms, "end_ms": end_ms, "text": seg_text, "confidence": None})
                    elif isinstance(s, dict):
                        segments.append({
                            "start_ms": int(s.get("start", s.get("start_ms", 0))),
                            "end_ms":   int(s.get("end",   s.get("end_ms",   0))),
                            "text":     str(s.get("text", "")),
                            "confidence": s.get("confidence"),
                        })

    return {
        "text": full_text,
        "confidence": confidence,
        "segments": segments,
        "words": words,
        "status": "ok",
    }


# ── API 接口 ─────────────────────────────────────────────────────

@app.post("/api/eval/asr")
async def eval_asr(file: UploadFile = File(...)):
    """上传MP4或音频文件 → ASR转文字"""
    suffix = Path(file.filename or "audio.mp4").suffix.lower()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        if not HAS_LIBROSA:
            return JSONResponse({"error": "librosa未安装，请执行: pip install librosa"}, status_code=500)

        # 计算时长用（librosa就可）
        pcm16 = extract_audio_from_mp4(tmp_path, target_sr=16000)
        duration_s = len(pcm16) / 16000

        # 直接把文件路径传给 FunASR，内置 VAD+分句
        loop = asyncio.get_event_loop()
        asr_result = await loop.run_in_executor(_thread_pool, run_funasr, tmp_path)
        asr_result["duration_s"] = round(duration_s, 2)
        return JSONResponse(asr_result)
    except Exception as e:
        return JSONResponse({"error": str(e), "traceback": traceback.format_exc()}, status_code=500)
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


class WerRequest(BaseModel):
    reference: str
    hypothesis: str


@app.post("/api/eval/wer")
async def eval_wer(req: WerRequest):
    """计算WER（字错误率）"""
    result = compute_wer(req.reference, req.hypothesis)
    return JSONResponse(result)


class TextSerRequest(BaseModel):
    reference: str
    hypothesis: str


@app.post("/api/eval/text_ser")
async def eval_text_ser(req: TextSerRequest):
    """计算SER句错率（纯文本对比，不依赖MediaPipe）"""
    result = compute_text_ser(req.reference, req.hypothesis)
    return JSONResponse(result)


@app.post("/api/eval/ser")
async def eval_ser(file: UploadFile = File(...)):
    """上传MP4 → 视觉情感识别"""
    suffix = Path(file.filename or "video.mp4").suffix.lower()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = compute_ser_from_video(tmp_path)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


@app.post("/api/eval/full")
async def eval_full(
    file: UploadFile = File(...),
    reference: str = Form(default=""),
    enable_ser: str = Form(default="false"),
):
    """
    完整评测：ASR + WER（如有参考文本） + SER（可选）
    """
    suffix = Path(file.filename or "video.mp4").suffix.lower()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    result: Dict[str, Any] = {"file": file.filename, "status": "ok"}

    try:
        if not HAS_LIBROSA:
            result["error"] = "librosa未安装"
            return JSONResponse(result, status_code=500)
    
        # 1. 时长计算
        pcm16 = extract_audio_from_mp4(tmp_path, target_sr=16000)
        result["duration_s"] = round(len(pcm16) / 16000, 2)
    
        # 2. ASR（直接传文件路径给 FunASR，内置 VAD）
        loop = asyncio.get_event_loop()
        asr_result = await loop.run_in_executor(_thread_pool, run_funasr, tmp_path)
        result["asr"] = {
            "text":       asr_result.get("text", ""),
            "confidence": asr_result.get("confidence"),
            "segments":   asr_result.get("segments", []),
            "words":      asr_result.get("words", []),
        }

        # 2. WER（有参考文本时）
        hyp_text = result["asr"]["text"]
        if reference.strip():
            result["wer"] = compute_wer(reference.strip(), hyp_text)
        else:
            result["wer"] = None

        # 3. 文本SER句错率（有参考文本时自动计算，不依赖MediaPipe）
        if reference.strip():
            result["text_ser"] = compute_text_ser(reference.strip(), hyp_text)
        else:
            result["text_ser"] = None

        # 4. 视觉SER（可选，依赖MediaPipe）
        if enable_ser.lower() in ("true", "1", "yes"):
            result["ser"] = compute_ser_from_video(tmp_path)
        else:
            result["ser"] = None

        return JSONResponse(result)

    except Exception as e:
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
        return JSONResponse(result, status_code=500)
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8766, log_level="info")
