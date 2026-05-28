#!/usr/bin/env python3
"""
数字人面部行为驱动模型 - 映射层
58D面部参数 → VRM Blendshape / Live2D 参数
2个数字人形象统一驱动接口
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


# ── 58D参数语义定义（与训练数据对齐）──────────────────────────
# 参考 ARKit 52个blendshape + 头部姿态6维
FACE_PARAM_NAMES = [
    # 眼部 (0-13)
    "eyeBlinkLeft", "eyeBlinkRight",
    "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight",
    "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight",
    "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight",
    # 嘴部口型 (14-35)
    "mouthClose", "mouthFunnel", "mouthPucker",
    "mouthLeft", "mouthRight",
    "mouthSmileLeft", "mouthSmileRight",
    "mouthFrownLeft", "mouthFrownRight",
    "mouthDimpleLeft", "mouthDimpleRight",
    "mouthStretchLeft", "mouthStretchRight",
    "mouthRollLower", "mouthRollUpper",
    "mouthShrugLower", "mouthShrugUpper",
    "mouthPressLeft", "mouthPressRight",
    "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthUpperUpLeft", "mouthUpperUpRight",
    # 眉毛 (36-41)
    "browDownLeft", "browDownRight",
    "browInnerUp",
    "browOuterUpLeft", "browOuterUpRight",
    # 脸颊/鼻 (41-47)
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "noseSneerLeft", "noseSneerRight",
    # 颌 (47-51)
    "jawOpen", "jawForward", "jawLeft", "jawRight",
    # 其他 (51-57)
    "tongueOut",
    "headPitch", "headYaw", "headRoll",   # 头部姿态 (欧拉角，归一化)
    "headX", "headY", "headZ",            # 头部位移
]


@dataclass
class AvatarMapping:
    """数字人驱动参数映射配置"""
    avatar_id: str
    avatar_type: str  # "vrm" or "live2d"
    mapping_version: str = "v1"


class VRMMapper:
    """
    58D面部参数 → VRM Blendshape
    支持 VRoid / UniVRM 标准blendshape名称
    """
    MAPPING_VERSION = "v1"

    # 58D索引 → VRM blendshape名称 与 权重
    PARAM_TO_VRM: Dict[int, List] = {
        0:  [("Blink_L", 1.0)],
        1:  [("Blink_R", 1.0)],
        10: [("Blink_L", 0.5)],   # eyeSquintLeft → 叠加眨眼
        11: [("Blink_R", 0.5)],
        14: [("A", 0.3)],          # mouthClose → 口型A权重
        15: [("O", 0.8)],          # mouthFunnel → O口型
        16: [("O", 0.6), ("U", 0.4)],  # mouthPucker
        19: [("Joy", 0.7), ("A", 0.2)],  # mouthSmileLeft
        20: [("Joy", 0.7), ("A", 0.2)],  # mouthSmileRight
        21: [("Sorrow", 0.5)],     # mouthFrownLeft
        22: [("Sorrow", 0.5)],     # mouthFrownRight
        36: [("Angry", 0.5)],      # browDownLeft
        37: [("Angry", 0.5)],      # browDownRight
        38: [("Sorrow", 0.4)],     # browInnerUp
        47: [("A", 1.0), ("O", 0.3)],  # jawOpen → 嘴部开合
    }

    # 情绪 → VRM表情blendshape（来自LLM情绪意图）
    EMOTION_TO_PRESET: Dict[str, str] = {
        "Neutral":  "Neutral",
        "Happy":    "Joy",
        "Sad":      "Sorrow",
        "Surprise": "Fun",
        "Fear":     "Sorrow",
        "Disgust":  "Angry",
        "Anger":    "Angry",
        "Contempt": "Angry",
    }

    def map(
        self,
        face_params: np.ndarray,     # (58,) 单帧
        emotion_25: np.ndarray,      # (25,) 情绪特征
        emotion_intent: Optional[str] = None,  # LLM情绪意图
        emotion_intensity: float = 0.5,
    ) -> Dict[str, float]:
        """
        转换为VRM blendshape字典
        """
        blendshapes: Dict[str, float] = {}

        # 1. 从58D参数映射blendshape
        for param_idx, targets in self.PARAM_TO_VRM.items():
            if param_idx >= len(face_params):
                continue
            val = float(face_params[param_idx])
            val = max(0.0, min(1.0, val))  # 限制[0,1]
            for bs_name, weight in targets:
                blendshapes[bs_name] = blendshapes.get(bs_name, 0.0) + val * weight

        # 2. 叠加情绪表情（来自25D的EXP部分）
        exp_8 = emotion_25[17:]  # 8维表情概率
        exp_names = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger", "Contempt"]
        dominant_emotion = exp_names[int(np.argmax(exp_8))]
        dominant_prob = float(exp_8.max())

        if dominant_emotion in self.EMOTION_TO_PRESET:
            vrm_preset = self.EMOTION_TO_PRESET[dominant_emotion]
            strength = dominant_prob * 0.6  # 不超过60%，留给口型空间
            blendshapes[vrm_preset] = blendshapes.get(vrm_preset, 0.0) + strength

        # 3. LLM情绪意图叠加（优先级最高）
        if emotion_intent and emotion_intent in self.EMOTION_TO_PRESET:
            vrm_preset = self.EMOTION_TO_PRESET[emotion_intent]
            blendshapes[vrm_preset] = max(
                blendshapes.get(vrm_preset, 0.0),
                emotion_intensity * 0.8
            )

        # 4. 限幅
        for k in blendshapes:
            blendshapes[k] = max(0.0, min(1.0, blendshapes[k]))

        # 5. 头部姿态（度数→归一化）
        if len(face_params) >= 55:
            blendshapes["HeadPitch"] = float(np.clip(face_params[52], -1, 1))
            blendshapes["HeadYaw"] = float(np.clip(face_params[53], -1, 1))
            blendshapes["HeadRoll"] = float(np.clip(face_params[54], -1, 1))

        return blendshapes


class Live2DMapper:
    """
    58D面部参数 → Live2D Cubism 参数
    支持标准 Live2D 控制参数
    """

    def map(
        self,
        face_params: np.ndarray,
        emotion_25: np.ndarray,
        emotion_intent: Optional[str] = None,
        emotion_intensity: float = 0.5,
    ) -> Dict[str, float]:
        """
        转换为Live2D参数字典
        """
        params: Dict[str, float] = {}

        # 眼睛开合（眨眼）: 1=全开, 0=闭眼
        blink_l = 1.0 - float(np.clip(face_params[0], 0, 1)) if len(face_params) > 0 else 1.0
        blink_r = 1.0 - float(np.clip(face_params[1], 0, 1)) if len(face_params) > 1 else 1.0
        params["ParamEyeLOpen"] = blink_l
        params["ParamEyeROpen"] = blink_r

        # 嘴型
        jaw_open = float(np.clip(face_params[47], 0, 1)) if len(face_params) > 47 else 0.0
        params["ParamMouthOpenY"] = jaw_open

        smile_l = float(np.clip(face_params[19], 0, 1)) if len(face_params) > 19 else 0.0
        smile_r = float(np.clip(face_params[20], 0, 1)) if len(face_params) > 20 else 0.0
        frown_l = float(np.clip(face_params[21], 0, 1)) if len(face_params) > 21 else 0.0
        mouth_form = (smile_l + smile_r) / 2 - (frown_l * 0.5)
        params["ParamMouthForm"] = float(np.clip(mouth_form, -1, 1))

        # 眉毛
        brow_down = float(np.clip(face_params[36], 0, 1)) if len(face_params) > 36 else 0.0
        brow_up = float(np.clip(face_params[38], 0, 1)) if len(face_params) > 38 else 0.0
        params["ParamBrowLY"] = brow_up - brow_down
        params["ParamBrowRY"] = brow_up - brow_down

        # 情绪叠加（表情概率）
        exp_8 = emotion_25[17:]
        exp_names = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger", "Contempt"]
        dominant = exp_names[int(np.argmax(exp_8))]

        if dominant == "Happy":
            params["ParamMouthForm"] = max(params.get("ParamMouthForm", 0.0), 0.7)
        elif dominant == "Sad":
            params["ParamMouthForm"] = min(params.get("ParamMouthForm", 0.0), -0.5)
        elif dominant == "Surprise":
            params["ParamEyeLOpen"] = 1.0
            params["ParamEyeROpen"] = 1.0
            params["ParamBrowLY"] = max(params.get("ParamBrowLY", 0.0), 0.5)

        # LLM情绪意图覆盖
        if emotion_intent == "happy":
            params["ParamMouthForm"] = max(params.get("ParamMouthForm", 0.0), emotion_intensity)
        elif emotion_intent in ("sad", "gentle"):
            params["ParamMouthForm"] = min(params.get("ParamMouthForm", 0.0), -0.3 * emotion_intensity)

        # 头部姿态
        if len(face_params) >= 55:
            params["ParamAngleY"] = float(np.clip(face_params[53] * 30, -30, 30))  # 偏航→角度
            params["ParamAngleX"] = float(np.clip(face_params[52] * 30, -30, 30))  # 俯仰
            params["ParamAngleZ"] = float(np.clip(face_params[54] * 30, -30, 30))  # 横滚

        # 呼吸动画（缓慢自动）
        params["ParamBreath"] = 0.5

        # 限幅
        for k, v in params.items():
            if "Open" in k or k in ("ParamBreath",):
                params[k] = float(np.clip(v, 0, 1))
            else:
                params[k] = float(np.clip(v, -1, 1)) if "Angle" not in k else float(np.clip(v, -30, 30))

        return params


def get_mapper(avatar_type: str):
    """工厂函数：根据数字人类型返回对应映射器"""
    if avatar_type == "vrm":
        return VRMMapper()
    elif avatar_type == "live2d":
        return Live2DMapper()
    else:
        raise ValueError(f"未知数字人类型: {avatar_type}，支持 vrm / live2d")
