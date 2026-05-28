#!/usr/bin/env python3
"""
MP4视频处理器 - 用于官方评测
整合视觉、听觉模块处理MP4视频文件，输出融合结果
"""

from __future__ import annotations

import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class MP4ProcessorConfig:
    """MP4处理器配置"""
    video_sample_rate: int = 15  # 视频采样率fps
    audio_sample_rate: int = 16000  # 音频采样率
    window_ms: int = 1000  # 融合窗口大小
    hop_ms: int = 200  # 窗口滑动步长
    enable_vision: bool = True
    enable_audio: bool = True
    enable_asr: bool = True


class MP4Processor:
    """
    MP4视频处理器
    
    处理流程:
    1. 提取视频帧 -> 视觉模块处理
    2. 提取音频 -> 听觉模块处理
    3. 时空对齐 -> 多模态融合
    4. 输出融合结果
    """
    
    def __init__(
        self,
        config: Optional[MP4ProcessorConfig] = None,
        vision_extractor=None,
        audio_pipeline=None,
        fusion_engine=None,
    ):
        self.config = config or MP4ProcessorConfig()
        self._vision_extractor = vision_extractor
        self._audio_pipeline = audio_pipeline
        self._fusion_engine = fusion_engine
    
    def process_mp4(
        self,
        mp4_path: str,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        处理MP4视频文件
        
        Args:
            mp4_path: MP4文件路径
            output_dir: 输出目录（可选）
            
        Returns:
            融合结果字典
        """
        mp4_path = Path(mp4_path)
        if not mp4_path.exists():
            raise FileNotFoundError(f"MP4文件不存在: {mp4_path}")
        
        print(f"处理MP4文件: {mp4_path}")
        
        # 提取音频
        audio_data = self._extract_audio(str(mp4_path))
        print(f"  音频: {len(audio_data)/self.config.audio_sample_rate:.2f}秒")
        
        # 提取视频帧
        video_frames = self._extract_video_frames(str(mp4_path))
        print(f"  视频: {len(video_frames)}帧")
        
        # 处理音频（ASR + 声学特征）
        audio_result = None
        if self._audio_pipeline and self.config.enable_audio:
            print("  处理音频...")
            audio_result = self._audio_pipeline.process_audio(
                audio_data,
                self.config.audio_sample_rate
            )
            if audio_result.get('asr', {}).get('text'):
                print(f"    ASR: '{audio_result['asr']['text']}'")
        
        # 处理视频帧
        vision_results = []
        if self._vision_extractor and self.config.enable_vision:
            print("  处理视频帧...")
            for i, (frame, timestamp_ms) in enumerate(video_frames):
                result = self._vision_extractor.process_frame(
                    frame,
                    timestamp_ms,
                    wait_result=True,
                    timeout_ms=100
                )
                if result:
                    vision_results.append({
                        'timestamp_ms': timestamp_ms,
                        'result': result
                    })
                if (i + 1) % 30 == 0:
                    print(f"    已处理 {i+1}/{len(video_frames)} 帧")
            print(f"    成功处理 {len(vision_results)} 帧")
        
        # 时空对齐与融合
        if self._fusion_engine:
            print("  多模态融合...")
            fusion_result = self._fuse_modalities(
                audio_result,
                vision_results,
                audio_data
            )
        else:
            fusion_result = self._simple_fusion(audio_result, vision_results)
        
        # 保存结果（可选）
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            self._save_results(fusion_result, output_dir, mp4_path.stem)
        
        return fusion_result
    
    def _extract_audio(self, mp4_path: str) -> np.ndarray:
        """从MP4提取音频为PCM16格式"""
        import subprocess
        
        # 使用ffmpeg提取音频到临时wav文件
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # ffmpeg命令: 提取音频，转换为16kHz单声道16bit PCM
            cmd = [
                'ffmpeg',
                '-i', mp4_path,
                '-vn',  # 不处理视频
                '-acodec', 'pcm_s16le',  # 16bit PCM
                '-ar', str(self.config.audio_sample_rate),  # 采样率
                '-ac', '1',  # 单声道
                '-y',  # 覆盖输出
                tmp_path
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            # 读取wav文件
            with wave.open(tmp_path, 'rb') as wf:
                n_frames = wf.getnframes()
                raw_data = wf.readframes(n_frames)
                pcm16 = np.frombuffer(raw_data, dtype=np.int16)
                
            return pcm16
            
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    
    def _extract_video_frames(self, mp4_path: str) -> List[Tuple[np.ndarray, int]]:
        """提取视频帧及时间戳"""
        cap = cv2.VideoCapture(mp4_path)
        
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {mp4_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frames = []
        frame_interval = max(1, int(fps / self.config.video_sample_rate))
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 按目标采样率抽取帧
            if frame_idx % frame_interval == 0:
                timestamp_ms = int(frame_idx / fps * 1000)
                frames.append((frame, timestamp_ms))
            
            frame_idx += 1
        
        cap.release()
        return frames
    
    def _fuse_modalities(
        self,
        audio_result: Optional[Dict],
        vision_results: List[Dict],
        audio_data: np.ndarray
    ) -> Dict[str, Any]:
        """多模态融合"""
        # 构建模拟的PerceptionToLLM输入
        from packages.common.protocols import (
            PerceptionToLLM, Turn, AsrResult, EmotionSignals,
            VoiceSignal, VisionResult, VisionFeatures, VisionFace,
            VisionSummary, VA, Expression8, VisionProcessing
        )
        
        # 计算VAD信息
        duration_ms = len(audio_data) * 1000 // self.config.audio_sample_rate
        
        # 构建turn
        turn = Turn(
            utterance_id=f"utt_{int(np.random.randint(0, 1000000))}",
            input_mode="voice",
            barge_in=False,
            vad={"speech_start_ms": 0, "speech_end_ms": duration_ms} if audio_result else None
        )
        
        # 构建ASR结果
        asr = None
        if audio_result and audio_result.get('asr'):
            asr_data = audio_result['asr']
            asr = AsrResult(
                text=asr_data.get('text', ''),
                language=asr_data.get('language', 'zh-CN'),
                confidence=asr_data.get('confidence'),
                words=asr_data.get('words', []),
                wer_hint=asr_data.get('wer_hint', {})
            )
        
        # 构建emotion signals
        voice_signal = None
        if audio_result and audio_result.get('emotion', {}).get('signals', {}).get('voice'):
            voice_data = audio_result['emotion']['signals']['voice']
            voice_signal = VoiceSignal(
                enabled=voice_data.get('enabled', False),
                x_features=voice_data.get('x_features', {})
            )
        
        emotion_signals = EmotionSignals(
            voice=voice_signal,
            text=None,
            vision=None
        )
        
        # 构建vision结果（使用最后一帧）
        vision = None
        if vision_results:
            last_vision = vision_results[-1]['result']
            vision = self._convert_vision_result(last_vision)
        
        # 构建PerceptionToLLM
        perception = PerceptionToLLM(
            turn=turn,
            asr=asr,
            emotion=emotion_signals,
            vision=vision
        )
        
        # 调用融合引擎
        if self._fusion_engine:
            fusion_output = self._fusion_engine.fuse(perception)
        else:
            fusion_output = self._simple_fusion_output(
                audio_result, vision_results, duration_ms
            )
        
        # 添加处理元信息
        fusion_output['meta'] = {
            'audio_duration_ms': duration_ms,
            'vision_frames_processed': len(vision_results),
            'window_ms': self.config.window_ms,
            'hop_ms': self.config.hop_ms,
        }
        
        return fusion_output
    
    def _convert_vision_result(self, vision_data: Dict) -> VisionResult:
        """转换视觉结果为协议格式"""
        from packages.common.protocols import (
            VisionResult, VisionFeatures, VisionFace, VisionSummary,
            VA, Expression8, VisionProcessing
        )
        
        features_data = vision_data.get('features', {}).get('face', {})
        summary_data = vision_data.get('summary', {})
        
        # 构建AU列表
        au_15 = []
        for au_item in features_data.get('au_15', []):
            au_15.append({
                'name': au_item['name'],
                'intensity': au_item['intensity']
            })
        
        # 构建VA
        va_data = summary_data.get('va', {})
        va = VA(
            valence=va_data.get('valence', 0.0),
            arousal=va_data.get('arousal', 0.0),
            confidence=va_data.get('confidence', 0.0)
        )
        
        # 构建Expression8
        expr_data = summary_data.get('expression_8', {})
        expression_8 = Expression8(
            label=expr_data.get('label', 'Neutral'),
            probs=expr_data.get('probs', {})
        )
        
        # 构建VisionResult
        vision = VisionResult(
            enabled=True,
            provider=vision_data.get('provider', 'mediapipe'),
            model=vision_data.get('model', {}),
            mode=vision_data.get('mode', 'live_stream'),
            frame_rate_fps=vision_data.get('frame_rate_fps', 15),
            window_ms=vision_data.get('window_ms', 1000),
            face_count=vision_data.get('face_count', 1),
            features=VisionFeatures(
                face=VisionFace(
                    landmarks_478=[],
                    blendshapes_52=[],
                    au_15=au_15,
                    head_pose=features_data.get('head_pose'),
                    gaze=features_data.get('gaze')
                )
            ),
            processing=VisionProcessing(
                au_mapping=vision_data.get('processing', {}).get('au_mapping', {}),
                symmetry=vision_data.get('processing', {}).get('symmetry', {}),
                smoothing=vision_data.get('processing', {}).get('smoothing', {})
            ),
            summary=VisionSummary(
                va=va,
                expression_8=expression_8
            ),
            quality=vision_data.get('quality', {}),
            x_ext=vision_data.get('x_ext', {})
        )
        
        return vision
    
    def _simple_fusion(
        self,
        audio_result: Optional[Dict],
        vision_results: List[Dict]
    ) -> Dict[str, Any]:
        """简单融合（无融合引擎时使用）"""
        # 提取视觉特征的平均值
        vision_au_avg = {}
        vision_va_avg = {'valence': 0.0, 'arousal': 0.0}
        
        if vision_results:
            au_sums = {}
            va_sums = {'valence': 0.0, 'arousal': 0.0}
            count = 0
            
            for vr in vision_results:
                result = vr['result']
                summary = result.get('summary', {})
                
                # 累加AU
                for au_item in result.get('features', {}).get('face', {}).get('au_15', []):
                    name = au_item['name']
                    intensity = au_item['intensity']
                    au_sums[name] = au_sums.get(name, 0.0) + intensity
                
                # 累加VA
                va = summary.get('va', {})
                va_sums['valence'] += va.get('valence', 0.0)
                va_sums['arousal'] += va.get('arousal', 0.0)
                count += 1
            
            if count > 0:
                vision_au_avg = {k: v / count for k, v in au_sums.items()}
                vision_va_avg = {k: v / count for k, v in va_sums.items()}
        
        # 构建输出
        text = ''
        if audio_result and audio_result.get('asr'):
            text = audio_result['asr'].get('text', '')
        
        return {
            'text': text,
            'vision_au_avg': vision_au_avg,
            'vision_va_avg': vision_va_avg,
            'vision_frames': len(vision_results),
        }
    
    def _simple_fusion_output(
        self,
        audio_result: Optional[Dict],
        vision_results: List[Dict],
        duration_ms: int
    ) -> Dict[str, Any]:
        """简单融合输出格式"""
        result = self._simple_fusion(audio_result, vision_results)
        
        return {
            'emotion': {
                'primary': 'neutral',
                'valence': result['vision_va_avg'].get('valence', 0.0),
                'arousal': result['vision_va_avg'].get('arousal', 0.0),
                'confidence': 0.5,
            },
            'x_ext': {
                'fusion': {
                    'version': 'simple_v1',
                    'text': result['text'],
                    'vision_au_avg': result['vision_au_avg'],
                    'vision_va_avg': result['vision_va_avg'],
                }
            }
        }
    
    def _save_results(
        self,
        result: Dict[str, Any],
        output_dir: Path,
        base_name: str
    ):
        """保存处理结果"""
        import json
        
        # 保存JSON结果
        output_path = output_dir / f"{base_name}_fusion.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"  结果已保存: {output_path}")


def process_mp4_for_evaluation(
    mp4_path: str,
    model_path: str = "d:/face_landmarker.task",
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    评测用MP4处理函数
    
    简化接口，自动初始化所有组件
    
    Args:
        mp4_path: MP4文件路径
        model_path: MediaPipe模型路径
        output_dir: 输出目录
        
    Returns:
        融合结果
    """
    from services.vision.inference.mediapipe_face import VisionExtractor, VisionConfig
    from services.asr.audio_pipeline import AudioPipeline, AudioPipelineConfig
    from services.multimodal_fusion.fusion_engine import FusionEngine, FusionConfig
    
    # 初始化组件
    vision_config = VisionConfig(
        model_path=model_path,
        frame_rate_fps=15,
        au_mapping_path="d:/AI数字人情感陪护项目/services/vision/config/au_mapping_v1.json",
    )
    vision_extractor = VisionExtractor(vision_config)
    
    audio_config = AudioPipelineConfig(
        sample_rate_hz=16000,
        enable_vad=True,
        enable_asr=True,
        enable_acoustic_features=True,
    )
    audio_pipeline = AudioPipeline(audio_config)
    
    fusion_config = FusionConfig(
        window_ms=1000,
        hop_ms=200,
        fused_embedding_dim=256,
    )
    fusion_engine = FusionEngine(fusion_config)
    
    # 创建处理器
    processor_config = MP4ProcessorConfig(
        video_sample_rate=15,
        audio_sample_rate=16000,
        enable_vision=True,
        enable_audio=True,
        enable_asr=True,
    )
    processor = MP4Processor(
        config=processor_config,
        vision_extractor=vision_extractor,
        audio_pipeline=audio_pipeline,
        fusion_engine=fusion_engine,
    )
    
    try:
        result = processor.process_mp4(mp4_path, output_dir)
        return result
    finally:
        vision_extractor.close()
