from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from packages.common.constants import (
    DATASET_AU_NAMES,
    DATASET_EXPRESSION_8_NAMES,
    DATASET_VA_NAMES,
)


@dataclass(frozen=True)
class VisionConfig:
    model_path: str
    frame_rate_fps: int = 15
    window_ms: int = 1000
    face_count: int = 1
    au_mapping_path: Optional[str] = None
    smoothing_alpha: float = 0.7
    enable_landmarks: bool = False
    enable_gaze: bool = True


def _unit_vector(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= 1e-9:
        return v
    return v / n


def _rotation_matrix_to_euler_xyz_degrees(r: np.ndarray) -> Tuple[float, float, float]:
    sy = math.sqrt(r[0, 0] * r[0, 0] + r[1, 0] * r[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(r[2, 1], r[2, 2])
        y = math.atan2(-r[2, 0], sy)
        z = math.atan2(r[1, 0], r[0, 0])
    else:
        x = math.atan2(-r[1, 2], r[1, 1])
        y = math.atan2(-r[2, 0], sy)
        z = 0.0
    return (math.degrees(x), math.degrees(y), math.degrees(z))


def _estimate_head_pose_from_landmarks(
    image_shape: Tuple[int, int],
    face_landmarks_xy: np.ndarray,
) -> Optional[Dict[str, float]]:
    h, w = image_shape
    idx = {
        "nose_tip": 1,
        "chin": 152,
        "left_eye_outer": 33,
        "right_eye_outer": 263,
        "left_mouth": 61,
        "right_mouth": 291,
    }
    if face_landmarks_xy.shape[0] <= max(idx.values()):
        return None

    image_points = np.array(
        [
            face_landmarks_xy[idx["nose_tip"]],
            face_landmarks_xy[idx["chin"]],
            face_landmarks_xy[idx["left_eye_outer"]],
            face_landmarks_xy[idx["right_eye_outer"]],
            face_landmarks_xy[idx["left_mouth"]],
            face_landmarks_xy[idx["right_mouth"]],
        ],
        dtype=np.float64,
    )

    model_points = np.array(
        [
            (0.0, 0.0, 0.0),
            (0.0, -63.6, -12.5),
            (-43.3, 32.7, -26.0),
            (43.3, 32.7, -26.0),
            (-28.9, -28.9, -24.1),
            (28.9, -28.9, -24.1),
        ],
        dtype=np.float64,
    )

    focal_length = w
    center = (w / 2.0, h / 2.0)
    camera_matrix = np.array(
        [
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    ok, rvec, _tvec = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None

    rmat, _ = cv2.Rodrigues(rvec)
    pitch, yaw, roll = _rotation_matrix_to_euler_xyz_degrees(rmat)
    return {"pitch": float(pitch), "yaw": float(yaw), "roll": float(roll)}


class AuMapper:
    def __init__(self, mapping: Dict[str, Dict[str, float]]):
        self._mapping = mapping

    @staticmethod
    def load_from_json(path: str) -> "AuMapper":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("au mapping json must be an object")
        mapping: Dict[str, Dict[str, float]] = {}
        for au_name, weights in data.items():
            if not isinstance(weights, dict):
                continue
            mapping[str(au_name)] = {str(k): float(v) for k, v in weights.items()}
        return AuMapper(mapping)

    def blendshapes_to_au(self, blendshape_scores: Dict[str, float]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for au in DATASET_AU_NAMES:
            weights = self._mapping.get(au, {})
            s = 0.0
            for name, w in weights.items():
                s += float(blendshape_scores.get(name, 0.0)) * float(w)
            out[au] = float(max(0.0, min(1.0, s)))
        return out


class IIRFilter:
    def __init__(self, alpha: float):
        self.alpha = float(alpha)
        self._state: Dict[str, float] = {}

    def update(self, x: Dict[str, float]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for k, v in x.items():
            prev = self._state.get(k, v)
            y = self.alpha * prev + (1.0 - self.alpha) * float(v)
            self._state[k] = y
            out[k] = y
        return out


class VisionExtractor:
    def __init__(self, config: VisionConfig):
        self.config = config
        self._au_mapper = (
            AuMapper.load_from_json(config.au_mapping_path)
            if config.au_mapping_path
            else AuMapper(mapping={})
        )
        self._au_filter = IIRFilter(alpha=config.smoothing_alpha)
        self._lock = threading.Lock()
        self._latest: Optional[Dict[str, Any]] = None
        self._latest_ts: Optional[int] = None
        self._event = threading.Event()
        self._dropped_frames = 0

        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
        import mediapipe as _mp_root
        
        # 处理路径，确保 MediaPipe 能正确读取
        model_path = str(config.model_path).replace("\\", "/")
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.LIVE_STREAM,
            num_faces=config.face_count,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            result_callback=self._on_result,
        )
        self._mp_vision = mp_vision
        # mediapipe 0.10.x: MPImage 已移至 mp.Image / mp.ImageFormat
        self._MPImage = _mp_root.Image
        self._ImageFormat = _mp_root.ImageFormat
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    def close(self) -> None:
        self._landmarker.close()

    def _on_result(self, result: Any, output_image: Any, timestamp_ms: int) -> None:
        payload = self._result_to_payload(result, output_image)
        with self._lock:
            self._latest = payload
            self._latest_ts = int(timestamp_ms)
            self._event.set()

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        timestamp_ms: int,
        wait_result: bool = True,
        timeout_ms: int = 200,
    ) -> Optional[Dict[str, Any]]:
        mp_image = self._MPImage(image_format=self._ImageFormat.SRGB, data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        self._event.clear()
        self._landmarker.detect_async(mp_image, int(timestamp_ms))

        if not wait_result:
            return None

        ok = self._event.wait(timeout=timeout_ms / 1000.0)
        if not ok:
            with self._lock:
                self._dropped_frames += 1
            return None

        with self._lock:
            return self._latest

    def _result_to_payload(self, result: Any, output_image: Any) -> Dict[str, Any]:
        image_h = int(output_image.height) if output_image is not None else 0
        image_w = int(output_image.width) if output_image is not None else 0

        face_landmarks = result.face_landmarks[0] if result.face_landmarks else []
        face_blendshapes = result.face_blendshapes[0] if result.face_blendshapes else []

        blendshape_scores: Dict[str, float] = {}
        blendshapes_list: List[Dict[str, Any]] = []
        for c in face_blendshapes:
            name = str(c.category_name)
            score = float(c.score)
            blendshape_scores[name] = score
            blendshapes_list.append({"name": name, "score": score})

        au = self._au_mapper.blendshapes_to_au(blendshape_scores)
        au = self._au_filter.update(au)

        au_list = [{"name": k, "intensity": float(v)} for k, v in au.items()]

        landmarks_list: List[Dict[str, Any]] = []
        if self.config.enable_landmarks and face_landmarks:
            for i, lm in enumerate(face_landmarks):
                landmarks_list.append(
                    {
                        "i": int(i),
                        "x": float(lm.x),
                        "y": float(lm.y),
                        "z": float(lm.z),
                        "presence": float(getattr(lm, "presence", 0.0)) if hasattr(lm, "presence") else None,
                        "visibility": float(getattr(lm, "visibility", 0.0)) if hasattr(lm, "visibility") else None,
                    }
                )

        head_pose = None
        gaze = None
        if face_landmarks and image_h > 0 and image_w > 0:
            xy = np.array([(float(lm.x) * image_w, float(lm.y) * image_h) for lm in face_landmarks], dtype=np.float64)
            hp = _estimate_head_pose_from_landmarks((image_h, image_w), xy)
            if hp:
                head_pose = hp

        if self.config.enable_gaze and getattr(result, "facial_transformation_matrixes", None):
            m = result.facial_transformation_matrixes[0]
            m_np = np.array(m.data, dtype=np.float64).reshape((4, 4))
            rot = m_np[:3, :3]
            forward = _unit_vector(rot @ np.array([0.0, 0.0, 1.0], dtype=np.float64))
            gaze = {"x": float(forward[0]), "y": float(forward[1]), "z": float(forward[2])}

        # 简单的表情分类和VA估计（基于blendshape和AU）
        def estimate_expression_and_va(blendshape_scores: Dict[str, float], au: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, float]]:
            # 基于blendshape和AU的简单规则估计
            expression_probs = {
                "Neutral": 0.5,
                "Happy": 0.1,
                "Sad": 0.1,
                "Surprise": 0.1,
                "Fear": 0.05,
                "Disgust": 0.05,
                "Anger": 0.05,
                "Contempt": 0.05
            }
            
            # 基于smile相关blendshape调整Happy概率
            smile_score = blendshape_scores.get("mouthSmileLeft", 0.0) + blendshape_scores.get("mouthSmileRight", 0.0)
            expression_probs["Happy"] = min(1.0, smile_score * 2.0)
            expression_probs["Neutral"] = max(0.0, 1.0 - sum(expression_probs.values()) + expression_probs["Neutral"])
            
            # 基于AU调整其他表情概率
            if au.get("AU12", 0.0) > 0.3:  # 微笑
                expression_probs["Happy"] = max(expression_probs["Happy"], au["AU12"])
            if au.get("AU1", 0.0) > 0.3 or au.get("AU2", 0.0) > 0.3:  # 眉毛提升
                expression_probs["Surprise"] = max(expression_probs["Surprise"], (au.get("AU1", 0.0) + au.get("AU2", 0.0)) / 2.0)
            if au.get("AU4", 0.0) > 0.3:  # 眉毛降低
                expression_probs["Sad"] = max(expression_probs["Sad"], au["AU4"])
            if au.get("AU7", 0.0) > 0.3:  # 眼睛紧张
                expression_probs["Fear"] = max(expression_probs["Fear"], au["AU7"])
            
            # 归一化概率
            total = sum(expression_probs.values())
            if total > 0:
                expression_probs = {k: v / total for k, v in expression_probs.items()}
            
            # 简单的VA估计
            valence = (expression_probs["Happy"] - expression_probs["Sad"] - expression_probs["Anger"] - expression_probs["Fear"]) * 0.8
            arousal = (expression_probs["Surprise"] + expression_probs["Fear"] + expression_probs["Anger"] - expression_probs["Neutral"]) * 0.8
            
            return expression_probs, {"valence": valence, "arousal": arousal}
        
        expression_probs, va = estimate_expression_and_va(blendshape_scores, au)
        
        # 确定主要表情
        primary_expression = max(expression_probs, key=expression_probs.get)
        
        payload: Dict[str, Any] = {
            "enabled": True,
            "provider": "mediapipe",
            "model": {"name": Path(self.config.model_path).name, "with_blendshapes": True},
            "mode": "live_stream",
            "frame_rate_fps": int(self.config.frame_rate_fps),
            "window_ms": int(self.config.window_ms),
            "face_count": int(self.config.face_count),
            "features": {
                "face": {
                    "landmarks_478": landmarks_list,
                    "blendshapes_52": blendshapes_list,
                    "au_15": au_list,
                    "head_pose": head_pose,
                    "gaze": gaze,
                }
            },
            "processing": {
                "au_mapping": {
                    "name": "name2auweight",
                    "version": "v1",
                    "au_schema": {
                        "source": "dataset",
                        "schema_id": "dataset_au_v1",
                        "au_names": list(DATASET_AU_NAMES),
                        "va_names": list(DATASET_VA_NAMES),
                        "expression_8_names": list(DATASET_EXPRESSION_8_NAMES),
                    },
                },
                "symmetry": {"enabled": True, "method": "left_right_mean"},
                "smoothing": {"enabled": True, "type": "iir_1st_order", "alpha": float(self.config.smoothing_alpha)},
            },
            "summary": {
                "va": {
                    "valence": va["valence"],
                    "arousal": va["arousal"],
                    "confidence": 0.7  # 固定置信度，实际应用中可根据模型性能调整
                },
                "expression_8": {
                    "label": primary_expression,
                    "probs": expression_probs
                }
            },
            "quality": {
                "tracking_state": "tracked" if face_landmarks else "no_face",
                "confidence": 1.0 if face_landmarks else 0.0,
                "dropped_frames": int(self._dropped_frames),
            },
            "x_ext": {},
        }
        return payload

