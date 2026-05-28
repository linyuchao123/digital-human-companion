"""
25维 Emotion → 雫人物 Live2D 真实参数映射

基于 FACS (面部动作编码系统) 到 Live2D 参数的对应关系,
将模型输出的 25维面部行为特征映射为雫人物的 24 个 Live2D 驱动参数.

输入 25维:
  AU (0-14): AU1, AU2, AU4, AU6, AU7, AU9, AU10, AU12, AU14, AU15, AU17, AU23, AU24, AU25, AU26
  VA (15-16): Valence, Arousal
  EXP (17-24): Neutral, Happy, Sad, Surprise, Fear, Disgust, Anger, Contempt

输出 24个 Live2D 参数 (雫人物 shizuku.cdi3.json 真实参数定义):
  头部: PARAM_ANGLE_X, PARAM_ANGLE_Y, PARAM_ANGLE_Z
  眼睛: PARAM_EYE_L_OPEN, PARAM_EYE_R_OPEN, PARAM_EYE_BALL_X, PARAM_EYE_BALL_Y, PARAM_EYE_BALL_FORM
  眉毛: PARAM_BROW_L_Y, PARAM_BROW_R_Y, PARAM_BROW_L_X, PARAM_BROW_R_X,
         PARAM_BROW_L_ANGLE, PARAM_BROW_R_ANGLE, PARAM_BROW_L_FORM, PARAM_BROW_R_FORM
  嘴巴: PARAM_MOUTH_OPEN_Y, PARAM_MOUTH_FORM, PARAM_MOUTH_SIZE
  表情: PARAM_TERE (害羞/脸红)
  身体: PARAM_BODY_X, PARAM_BODY_Y, PARAM_BODY_Z, PARAM_BREATH
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

# ============================================================================
# Live2D 参数定义及范围 (来自 shizuku.cdi3.json)
# ============================================================================

LIVE2D_PARAMS: List[Dict[str, object]] = [
    # 头部
    {"name": "PARAM_ANGLE_X",       "index": 0,  "min": -30.0, "max": 30.0,  "default": 0.0},
    {"name": "PARAM_ANGLE_Y",       "index": 1,  "min": -30.0, "max": 30.0,  "default": 0.0},
    {"name": "PARAM_ANGLE_Z",       "index": 2,  "min": -30.0, "max": 30.0,  "default": 0.0},
    # 眼睛
    {"name": "PARAM_EYE_L_OPEN",    "index": 3,  "min": 0.0,   "max": 1.0,   "default": 1.0},
    {"name": "PARAM_EYE_R_OPEN",    "index": 4,  "min": 0.0,   "max": 1.0,   "default": 1.0},
    {"name": "PARAM_EYE_BALL_X",    "index": 5,  "min": -1.0,  "max": 1.0,   "default": 0.0},
    {"name": "PARAM_EYE_BALL_Y",    "index": 6,  "min": -1.0,  "max": 1.0,   "default": 0.0},
    {"name": "PARAM_EYE_BALL_FORM", "index": 7,  "min": -1.0,  "max": 1.0,   "default": 0.0},
    # 眉毛
    {"name": "PARAM_BROW_L_Y",      "index": 8,  "min": -1.0,  "max": 1.0,   "default": 0.0},
    {"name": "PARAM_BROW_R_Y",      "index": 9,  "min": -1.0,  "max": 1.0,   "default": 0.0},
    {"name": "PARAM_BROW_L_X",      "index": 10, "min": -1.0,  "max": 1.0,   "default": 0.0},
    {"name": "PARAM_BROW_R_X",      "index": 11, "min": -1.0,  "max": 1.0,   "default": 0.0},
    {"name": "PARAM_BROW_L_ANGLE",  "index": 12, "min": -1.0,  "max": 1.0,   "default": 0.0},
    {"name": "PARAM_BROW_R_ANGLE",  "index": 13, "min": -1.0,  "max": 1.0,   "default": 0.0},
    {"name": "PARAM_BROW_L_FORM",   "index": 14, "min": 0.0,   "max": 1.0,   "default": 0.0},
    {"name": "PARAM_BROW_R_FORM",   "index": 15, "min": 0.0,   "max": 1.0,   "default": 0.0},
    # 嘴巴
    {"name": "PARAM_MOUTH_OPEN_Y",  "index": 16, "min": 0.0,   "max": 1.0,   "default": 0.0},
    {"name": "PARAM_MOUTH_FORM",    "index": 17, "min": -1.0,  "max": 1.0,   "default": 0.0},
    {"name": "PARAM_MOUTH_SIZE",    "index": 18, "min": -1.0,  "max": 1.0,   "default": 0.0},
    # 表情
    {"name": "PARAM_TERE",          "index": 19, "min": 0.0,   "max": 1.0,   "default": 0.0},
    # 身体
    {"name": "PARAM_BODY_X",        "index": 20, "min": -10.0, "max": 10.0,  "default": 0.0},
    {"name": "PARAM_BODY_Y",        "index": 21, "min": -10.0, "max": 10.0,  "default": 0.0},
    {"name": "PARAM_BODY_Z",        "index": 22, "min": -10.0, "max": 10.0,  "default": 0.0},
    {"name": "PARAM_BREATH",        "index": 23, "min": 0.0,   "max": 1.0,   "default": 0.0},
]

PARAM_NAMES = [p["name"] for p in LIVE2D_PARAMS]
NUM_LIVE2D_PARAMS = len(LIVE2D_PARAMS)  # 24

# AU 索引 (在25维输出中的位置)
AU_INDICES = {
    "AU1":  0,   # Inner Brow Raiser (内眉上扬)
    "AU2":  1,   # Outer Brow Raiser (外眉上扬)
    "AU4":  2,   # Brow Lowerer (皱眉)
    "AU6":  3,   # Cheek Raiser (颧肌收缩/眯眼笑)
    "AU7":  4,   # Lid Tightener (眼睑收紧)
    "AU9":  5,   # Nose Wrinkler (鼻子皱起)
    "AU10": 6,   # Upper Lip Raiser (上唇上提)
    "AU12": 7,   # Lip Corner Puller (嘴角上扬/微笑)
    "AU14": 8,   # Dimpler (酒窝)
    "AU15": 9,   # Lip Corner Depressor (嘴角下垂)
    "AU17": 10,  # Chin Raiser (下巴上提)
    "AU23": 11,  # Lip Tightener (嘴唇收紧)
    "AU24": 12,  # Lip Presser (嘴唇紧闭)
    "AU25": 13,  # Lips Part (嘴唇分开)
    "AU26": 14,  # Jaw Drop (下巴下降)
}

# VA 索引
VA_VALENCE = 15
VA_AROUSAL = 16

# EXP 索引
EXP_INDICES = {
    "Neutral":  17,
    "Happy":    18,
    "Sad":      19,
    "Surprise": 20,
    "Fear":     21,
    "Disgust":  22,
    "Anger":    23,
    "Contempt": 24,
}


def _clamp(value: float, vmin: float, vmax: float) -> float:
    return max(vmin, min(vmax, value))


def emotion25_to_live2d(emotion_25: np.ndarray, intensity: float = 0.3) -> np.ndarray:
    """
    将单帧 25维 emotion 特征映射为 24维 Live2D 参数

    情感陪护数字人场景：安静/倾听状态下面部变化应当微妙自然，
    intensity 控制表情反应的整体强度 (0.0=完全静止, 1.0=最大反应)。
    默认 0.3 适合倾听/安静状态，对话中可适当提高到 0.5-0.7。

    Args:
        emotion_25: [25] 单帧特征向量
        intensity: 表情反应强度 (0.0~1.0), 默认0.3

    Returns:
        [24] Live2D 参数向量
    """
    e = emotion_25
    params = np.zeros(NUM_LIVE2D_PARAMS, dtype=np.float32)
    s = float(intensity)  # 强度缩放因子

    # --- 提取 AU 值 ---
    au1  = e[AU_INDICES["AU1"]]
    au2  = e[AU_INDICES["AU2"]]
    au4  = e[AU_INDICES["AU4"]]
    au6  = e[AU_INDICES["AU6"]]
    au7  = e[AU_INDICES["AU7"]]
    au9  = e[AU_INDICES["AU9"]]
    au10 = e[AU_INDICES["AU10"]]
    au12 = e[AU_INDICES["AU12"]]
    au14 = e[AU_INDICES["AU14"]]
    au15 = e[AU_INDICES["AU15"]]
    au17 = e[AU_INDICES["AU17"]]
    au23 = e[AU_INDICES["AU23"]]
    au24 = e[AU_INDICES["AU24"]]
    au25 = e[AU_INDICES["AU25"]]
    au26 = e[AU_INDICES["AU26"]]

    valence = e[VA_VALENCE]
    arousal = e[VA_AROUSAL]

    # EXP 概率
    exp_happy    = e[EXP_INDICES["Happy"]]
    exp_sad      = e[EXP_INDICES["Sad"]]
    exp_surprise = e[EXP_INDICES["Surprise"]]
    exp_fear     = e[EXP_INDICES["Fear"]]
    exp_disgust  = e[EXP_INDICES["Disgust"]]
    exp_anger    = e[EXP_INDICES["Anger"]]

    # ========== 映射规则 (系数已大幅降低，适合陪护场景) ==========

    # --- 头部角度 ---
    # 安静状态下头部仅有极微小偏转 (±2~3度)
    params[0] = _clamp(valence * 2.0 * s, -30.0, 30.0)       # PARAM_ANGLE_X
    params[1] = _clamp(arousal * 1.5 * s, -30.0, 30.0)       # PARAM_ANGLE_Y
    params[2] = _clamp(valence * 0.8 * s, -30.0, 30.0)       # PARAM_ANGLE_Z

    # --- 眼睛 ---
    # 基础状态: 眼睛自然睁开=1.0, 仅在明显表情时微调
    eye_open_base = 1.0
    eye_close = max(au6 * 0.15 * s, au7 * 0.2 * s)  # 微笑时轻微眯眼
    eye_open_wide = (au1 + au2) * 0.05 * s            # 惊讶时微微睁大
    eye_open = eye_open_base - eye_close + eye_open_wide
    params[3] = _clamp(eye_open, 0.0, 1.0)   # PARAM_EYE_L_OPEN
    params[4] = _clamp(eye_open, 0.0, 1.0)   # PARAM_EYE_R_OPEN

    # 眼球微动 (极轻微)
    params[5] = _clamp(valence * 0.03 * s, -1.0, 1.0)  # PARAM_EYE_BALL_X
    params[6] = _clamp(arousal * 0.02 * s, -1.0, 1.0)  # PARAM_EYE_BALL_Y

    # 眼睛形状: 开心时微微眯眼笑
    params[7] = _clamp(au6 * 0.15 * s - exp_sad * 0.1 * s, -1.0, 1.0)  # PARAM_EYE_BALL_FORM

    # --- 眉毛 ---
    brow_up = (au1 + au2) * 0.08 * s
    brow_down = au4 * 0.12 * s
    brow_y = brow_up - brow_down
    params[8]  = _clamp(brow_y, -1.0, 1.0)   # PARAM_BROW_L_Y
    params[9]  = _clamp(brow_y, -1.0, 1.0)   # PARAM_BROW_R_Y
    brow_x = -au4 * 0.08 * s
    params[10] = _clamp(brow_x, -1.0, 1.0)   # PARAM_BROW_L_X
    params[11] = _clamp(-brow_x, -1.0, 1.0)  # PARAM_BROW_R_X (对称)
    brow_angle = (exp_anger * 0.15 - exp_sad * 0.12 - au4 * 0.1) * s
    params[12] = _clamp(brow_angle, -1.0, 1.0)   # PARAM_BROW_L_ANGLE
    params[13] = _clamp(brow_angle, -1.0, 1.0)   # PARAM_BROW_R_ANGLE
    brow_form = (exp_surprise + exp_fear) * 0.08 * s
    params[14] = _clamp(brow_form, 0.0, 1.0)     # PARAM_BROW_L_FORM
    params[15] = _clamp(brow_form, 0.0, 1.0)     # PARAM_BROW_R_FORM

    # --- 嘴巴 ---
    # 安静/倾听时嘴巴几乎不动, 仅微笑
    mouth_open = (au25 * 0.08 + au26 * 0.15 + exp_surprise * 0.1) * s
    params[16] = _clamp(mouth_open, 0.0, 1.0)    # PARAM_MOUTH_OPEN_Y

    # 微笑 (核心表情, 保留较高灵敏度)
    mouth_form = (au12 * 0.25 - au15 * 0.2 + exp_happy * 0.15 - exp_sad * 0.1) * s
    params[17] = _clamp(mouth_form, -1.0, 1.0)   # PARAM_MOUTH_FORM

    mouth_size = (-au23 * 0.08 - au24 * 0.08 + au12 * 0.05) * s
    params[18] = _clamp(mouth_size, -1.0, 1.0)   # PARAM_MOUTH_SIZE

    # --- 害羞/脸红 ---
    tere = (exp_happy * arousal * 0.15 + max(0, valence) * 0.05) * s
    params[19] = _clamp(tere, 0.0, 1.0)          # PARAM_TERE

    # --- 身体 ---
    # 安静时身体几乎不动 (仅极微小晃动)
    params[20] = _clamp(valence * 0.4 * s, -10.0, 10.0)   # PARAM_BODY_X
    params[21] = _clamp(arousal * 0.3 * s, -10.0, 10.0)   # PARAM_BODY_Y
    params[22] = _clamp(valence * 0.2 * s, -10.0, 10.0)   # PARAM_BODY_Z

    # 呼吸: 保持自然节奏, 不受情感过大影响
    breath = 0.5 + arousal * 0.08 * s
    params[23] = _clamp(breath, 0.0, 1.0)             # PARAM_BREATH

    return params


def _generate_natural_blink(T: int, fps: int = 25) -> np.ndarray:
    """
    生成自然眨眼序列
    
    真人平均每3-5秒眨一次眼, 每次眨眼约0.15-0.4秒:
    - 闭眼: 约0.1秒 (快速闭合)
    - 睁眼: 约0.15-0.3秒 (稍慢恢复)
    
    Returns:
        [T] 眨眼因子 (1.0=完全睁开, 0.0=完全闭合)
    """
    blink = np.ones(T, dtype=np.float32)
    t = 0
    while t < T:
        # 下次眨眼间隔: 2.5~5秒 (随机)
        interval = int(fps * (2.5 + np.random.random() * 2.5))
        t += interval
        if t >= T:
            break
        # 眨眼动画: 快闭(3帧) + 全闭(1-2帧) + 慢开(4帧)
        close_frames = min(3, T - t)
        hold_frames = min(int(1 + np.random.random() * 1), T - t - close_frames)
        open_frames = min(4, T - t - close_frames - hold_frames)
        
        for i in range(close_frames):
            if t + i < T:
                blink[t + i] = 1.0 - (i + 1) / close_frames  # 快速闭合
        pos = t + close_frames
        for i in range(hold_frames):
            if pos + i < T:
                blink[pos + i] = 0.05  # 几乎完全闭合
        pos += hold_frames
        for i in range(open_frames):
            if pos + i < T:
                blink[pos + i] = (i + 1) / open_frames  # 缓慢睁开
        
        t = pos + open_frames
        
        # 偶尔连续眨两次 (约20%概率)
        if np.random.random() < 0.2 and t + fps < T:
            t += int(fps * 0.3)  # 短间隔后再眨一次
            close_frames = min(2, T - t)
            open_frames = min(3, T - t - close_frames)
            for i in range(close_frames):
                if t + i < T:
                    blink[t + i] = 1.0 - (i + 1) / close_frames
            pos = t + close_frames
            for i in range(open_frames):
                if pos + i < T:
                    blink[pos + i] = (i + 1) / open_frames
            t = pos + open_frames
    
    return blink


def _generate_natural_breath(T: int, fps: int = 25) -> np.ndarray:
    """
    生成自然呼吸曲线
    
    正常安静呼吸: 每分钟12-20次, 取15次 → 每4秒一个周期
    使用正弦波模拟, 加上微小随机扰动使其不机械
    
    Returns:
        [T] 呼吸值 (0.3~0.7, 正弦波形)
    """
    breath_period = fps * 4.0  # 4秒一个周期
    t = np.arange(T, dtype=np.float32)
    # 基础正弦呼吸
    breath = 0.5 + 0.2 * np.sin(2 * np.pi * t / breath_period)
    # 加入轻微随机性 (每个周期略有不同)
    noise = np.random.normal(0, 0.02, T).astype(np.float32)
    # 对噪声做平滑
    kernel = np.ones(fps) / fps
    noise = np.convolve(noise, kernel, mode='same').astype(np.float32)
    breath = np.clip(breath + noise, 0.2, 0.8)
    return breath


def _generate_eye_micro_motion(T: int, fps: int = 25) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成眼球微动序列 (saccade-like)
    
    真人眼球在注视时会有微小跳动 (微扫视), 每隔1-3秒
    幅度极小: ±0.05 左右
    
    Returns:
        (eye_x, eye_y): 各 [T] 微动偏移
    """
    eye_x = np.zeros(T, dtype=np.float32)
    eye_y = np.zeros(T, dtype=np.float32)
    
    t = 0
    target_x, target_y = 0.0, 0.0
    while t < T:
        # 每1-3秒一次微跳
        interval = int(fps * (1.0 + np.random.random() * 2.0))
        new_target_x = np.random.normal(0, 0.03)
        new_target_y = np.random.normal(0, 0.02)
        new_target_x = np.clip(new_target_x, -0.08, 0.08)
        new_target_y = np.clip(new_target_y, -0.06, 0.06)
        
        # 快速跳到新位置(2帧) + 缓慢稳定
        move_frames = min(3, T - t)
        for i in range(min(interval, T - t)):
            alpha = min(1.0, (i + 1) / move_frames)
            eye_x[t + i] = target_x + (new_target_x - target_x) * alpha
            eye_y[t + i] = target_y + (new_target_y - target_y) * alpha
        
        target_x = new_target_x
        target_y = new_target_y
        t += interval
    
    return eye_x, eye_y


def _generate_head_micro_motion(T: int, fps: int = 25) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    生成头部微动 (人在安静状态下头部会有极轻微摆动)
    
    使用超低频正弦波叠加, 幅度 ±0.5~1度
    """
    t = np.arange(T, dtype=np.float32) / fps
    # 多个低频正弦波叠加, 产生自然不规则晃动
    head_x = (0.3 * np.sin(0.3 * t + 0.5) + 0.2 * np.sin(0.7 * t + 1.2)).astype(np.float32)
    head_y = (0.2 * np.sin(0.25 * t + 0.8) + 0.15 * np.sin(0.55 * t + 2.1)).astype(np.float32)
    head_z = (0.15 * np.sin(0.2 * t + 1.5) + 0.1 * np.sin(0.45 * t + 0.3)).astype(np.float32)
    return head_x, head_y, head_z


def emotion25_sequence_to_live2d(
    emotion_seq: np.ndarray,
    smooth_window: int = 15,
    fps: int = 25,
    intensity: float = 0.3,
    add_idle_behaviors: bool = True,
) -> np.ndarray:
    """
    将 [T, 25] 序列转换为 [T, 24] Live2D 参数序列
    
    添加自然生理行为 (眨眼、呼吸、眼球微动、头部微动),
    使数字人在安静/倾听状态下表现如真人般自然。

    Args:
        emotion_seq: [T, 25] 情感特征序列
        smooth_window: 平滑窗口大小 (增大可减少抖动)
        fps: 帧率
        intensity: 表情反应强度 (0.0~1.0)
        add_idle_behaviors: 是否叠加自然生理行为

    Returns:
        [T, 24] Live2D 参数序列
    """
    T = emotion_seq.shape[0]
    live2d_seq = np.zeros((T, NUM_LIVE2D_PARAMS), dtype=np.float32)

    for t in range(T):
        live2d_seq[t] = emotion25_to_live2d(emotion_seq[t], intensity=intensity)

    # 时序平滑 (高斯加权移动平均, 比简单均值更自然)
    if smooth_window > 1 and T > smooth_window:
        # 高斯核: 中心权重大, 边缘权重小
        x = np.arange(smooth_window) - smooth_window // 2
        sigma = smooth_window / 4.0
        kernel = np.exp(-x ** 2 / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        
        smoothed = np.zeros_like(live2d_seq)
        for d in range(NUM_LIVE2D_PARAMS):
            smoothed[:, d] = np.convolve(live2d_seq[:, d], kernel, mode="same")
        live2d_seq = smoothed

    # 叠加自然生理行为
    if add_idle_behaviors:
        # 自然眨眼
        blink = _generate_natural_blink(T, fps)
        live2d_seq[:, 3] *= blink   # PARAM_EYE_L_OPEN
        live2d_seq[:, 4] *= blink   # PARAM_EYE_R_OPEN
        
        # 自然呼吸 (替换为更真实的呼吸曲线)
        breath = _generate_natural_breath(T, fps)
        live2d_seq[:, 23] = breath  # PARAM_BREATH
        # 呼吸带动身体微动
        live2d_seq[:, 21] += (breath - 0.5) * 0.3  # 身体Y随呼吸起伏
        
        # 眼球微动
        eye_x, eye_y = _generate_eye_micro_motion(T, fps)
        live2d_seq[:, 5] += eye_x   # PARAM_EYE_BALL_X
        live2d_seq[:, 6] += eye_y   # PARAM_EYE_BALL_Y
        
        # 头部微动
        head_x, head_y, head_z = _generate_head_micro_motion(T, fps)
        live2d_seq[:, 0] += head_x  # PARAM_ANGLE_X
        live2d_seq[:, 1] += head_y  # PARAM_ANGLE_Y
        live2d_seq[:, 2] += head_z  # PARAM_ANGLE_Z

    # 最终 clamp 确保在合法范围内
    for i, p in enumerate(LIVE2D_PARAMS):
        live2d_seq[:, i] = np.clip(live2d_seq[:, i], p["min"], p["max"])

    return live2d_seq


def live2d_to_dict(params: np.ndarray) -> Dict[str, float]:
    """将 [24] 参数向量转为 {参数名: 值} 字典"""
    return {PARAM_NAMES[i]: float(params[i]) for i in range(NUM_LIVE2D_PARAMS)}


def live2d_sequence_to_dicts(params_seq: np.ndarray) -> List[Dict[str, float]]:
    """将 [T, 24] 参数序列转为字典列表, 可直接用于 Live2D SDK 驱动"""
    return [live2d_to_dict(params_seq[t]) for t in range(params_seq.shape[0])]


# ============================================================================
# 示例用法
# ============================================================================

if __name__ == "__main__":
    # 模拟一个25维输入 (开心微笑表情)
    emotion = np.zeros(25, dtype=np.float32)
    emotion[AU_INDICES["AU6"]] = 0.7    # 颧肌收缩 (笑)
    emotion[AU_INDICES["AU12"]] = 0.8   # 嘴角上扬
    emotion[AU_INDICES["AU25"]] = 0.3   # 嘴唇微开
    emotion[VA_VALENCE] = 0.6           # 正向情感
    emotion[VA_AROUSAL] = 0.4           # 中等唤醒
    emotion[EXP_INDICES["Happy"]] = 0.8 # 开心概率高

    live2d_params = emotion25_to_live2d(emotion)
    param_dict = live2d_to_dict(live2d_params)

    print("=== 开心微笑表情 → Live2D 参数 ===")
    for name, value in param_dict.items():
        if abs(value) > 0.01:
            print(f"  {name}: {value:.3f}")

    # 模拟悲伤表情
    emotion2 = np.zeros(25, dtype=np.float32)
    emotion2[AU_INDICES["AU1"]] = 0.5    # 内眉上扬
    emotion2[AU_INDICES["AU4"]] = 0.6    # 皱眉
    emotion2[AU_INDICES["AU15"]] = 0.7   # 嘴角下垂
    emotion2[VA_VALENCE] = -0.5          # 负向情感
    emotion2[VA_AROUSAL] = -0.2          # 低唤醒
    emotion2[EXP_INDICES["Sad"]] = 0.7   # 悲伤概率高

    live2d_params2 = emotion25_to_live2d(emotion2)
    param_dict2 = live2d_to_dict(live2d_params2)

    print("\n=== 悲伤表情 → Live2D 参数 ===")
    for name, value in param_dict2.items():
        if abs(value) > 0.01:
            print(f"  {name}: {value:.3f}")
