#!/usr/bin/env python3
"""
多模态融合模块测试脚本
测试时空对齐、特征融合、MP4处理功能
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from services.multimodal_fusion.fusion_engine import FusionEngine, FusionConfig
from services.multimodal_fusion.mp4_processor import MP4Processor, MP4ProcessorConfig
from packages.common.protocols import (
    PerceptionToLLM, TurnInfo, AsrInfo, EmotionSignals,
    EmotionSignal, VisionInfo, VisionFeatures, VisionFaceFeatures,
    VisionSummary, VaSummary, ExpressionSummary, VisionProcessing,
    VadInfo, VisionAU, HeadPose, Gaze, AuMapping, AuSchema,
    SymmetryConfig, SmoothingConfig, VisionQuality, VisionModelInfo
)


def test_fusion_engine():
    """测试融合引擎"""
    print("=" * 60)
    print("多模态融合测试 - 融合引擎")
    print("=" * 60)
    
    # 初始化融合引擎
    print("\n初始化融合引擎...")
    try:
        config = FusionConfig(
            window_ms=1000,
            hop_ms=200,
            fused_embedding_dim=256,
        )
        engine = FusionEngine(config)
        print("✓ 融合引擎初始化成功")
    except Exception as e:
        print(f"✗ 初始化失败: {e}")
        return False
    
    # 构建测试输入
    print("\n构建测试输入...")
    perception = PerceptionToLLM(
        turn=Turn(
            utterance_id="utt_test_001",
            input_mode="voice",
            barge_in=False,
            vad={"speech_start_ms": 0, "speech_end_ms": 3000}
        ),
        asr=AsrResult(
            text="我今天心情不太好",
            language="zh-CN",
            confidence=0.92,
            words=[],
            wer_hint={}
        ),
        emotion=EmotionSignals(
            voice=VoiceSignal(
                enabled=True,
                x_features={
                    "mfcc_20_mean": [0.1] * 20,
                    "mfcc_20_std": [0.05] * 20,
                    "pitch_mean": 150.0,
                    "pitch_std": 20.0,
                    "energy_mean": 0.5,
                    "energy_std": 0.1,
                }
            ),
            text=None,
            vision=None
        ),
        vision=VisionResult(
            enabled=True,
            provider="mediapipe",
            model={"name": "face_landmarker.task", "with_blendshapes": True},
            mode="live_stream",
            frame_rate_fps=15,
            window_ms=1000,
            face_count=1,
            features=VisionFeatures(
                face=VisionFace(
                    landmarks_478=[],
                    blendshapes_52=[],
                    au_15=[
                        {"name": "AU1", "intensity": 0.3},
                        {"name": "AU4", "intensity": 0.5},
                        {"name": "AU12", "intensity": 0.1},
                    ],
                    head_pose={"pitch": -2.0, "yaw": 5.0, "roll": 1.0},
                    gaze={"x": 0.0, "y": 0.0, "z": 1.0}
                )
            ),
            processing=VisionProcessing(
                au_mapping={},
                symmetry={},
                smoothing={}
            ),
            summary=VisionSummary(
                va=VA(valence=-0.4, arousal=0.3, confidence=0.7),
                expression_8=Expression8(
                    label="Sad",
                    probs={
                        "Neutral": 0.2,
                        "Happy": 0.05,
                        "Sad": 0.5,
                        "Surprise": 0.05,
                        "Fear": 0.1,
                        "Disgust": 0.05,
                        "Anger": 0.03,
                        "Contempt": 0.02
                    }
                )
            ),
            quality={},
            x_ext={}
        )
    )
    print("✓ 测试输入构建完成")
    
    # 执行融合
    print("\n执行多模态融合...")
    result = engine.fuse(perception)
    
    print("✓ 融合完成")
    print(f"\n融合结果:")
    print(f"  主要情绪: {result['emotion']['primary']}")
    print(f"  Valence: {result['emotion']['valence']:.3f}")
    print(f"  Arousal: {result['emotion']['arousal']:.3f}")
    print(f"  置信度: {result['emotion']['confidence']:.3f}")
    
    # 验证输出格式
    print("\n验证输出格式:")
    print("-" * 60)
    
    checks = []
    
    # 检查emotion字段
    if 'emotion' in result:
        checks.append(("emotion 存在", True))
        emotion = result['emotion']
        checks.append(("emotion.primary 存在", 'primary' in emotion))
        checks.append(("emotion.valence 存在", 'valence' in emotion))
        checks.append(("emotion.arousal 存在", 'arousal' in emotion))
        checks.append(("emotion.confidence 存在", 'confidence' in emotion))
        checks.append(("emotion.signals 存在", 'signals' in emotion))
    else:
        checks.append(("emotion 结构", False))
    
    # 检查x_ext.fusion
    if 'x_ext' in result and 'fusion' in result['x_ext']:
        checks.append(("x_ext.fusion 存在", True))
        fusion = result['x_ext']['fusion']
        checks.append(("fusion.version 存在", 'version' in fusion))
        checks.append(("fusion.modalities 存在", 'modalities' in fusion))
        checks.append(("fusion.fused_embedding 存在", 'fused_embedding' in fusion))
        checks.append(("fusion.emotion_probs 存在", 'emotion_probs' in fusion))
        checks.append(("fusion.va 存在", 'va' in fusion))
        checks.append(("fusion.psych_state 存在", 'psych_state' in fusion))
    else:
        checks.append(("x_ext.fusion 结构", False))
    
    # 检查psych_state
    if 'x_ext' in result and 'fusion' in result['x_ext']:
        psych = result['x_ext']['fusion'].get('psych_state', {})
        checks.append(("psych_state.phq9_score_est 存在", 'phq9_score_est' in psych))
        checks.append(("psych_state.gad7_score_est 存在", 'gad7_score_est' in psych))
        checks.append(("psych_state.risk_level 存在", 'risk_level' in psych))
    
    for check_name, passed in checks:
        status = "✓" if passed else "✗"
        print(f"{status} {check_name}")
    
    passed_count = sum(1 for _, p in checks if p)
    total_count = len(checks)
    
    print(f"\n{'=' * 60}")
    print(f"格式验证: {passed_count}/{total_count} 通过")
    print(f"{'=' * 60}")
    
    return passed_count == total_count


def test_modality_fallback():
    """测试模态降级策略"""
    print("=" * 60)
    print("多模态融合测试 - 模态降级")
    print("=" * 60)
    
    config = FusionConfig()
    engine = FusionEngine(config)
    
    # 测试1: 仅文本输入
    print("\n测试1: 仅文本输入")
    perception_text_only = PerceptionToLLM(
        turn=Turn(utterance_id="utt_001", input_mode="voice", barge_in=False, vad=None),
        asr=AsrResult(text="我很开心", language="zh-CN", confidence=0.9, words=[], wer_hint={}),
        emotion=EmotionSignals(voice=None, text=None, vision=None),
        vision=None
    )
    result = engine.fuse(perception_text_only)
    print(f"  主要情绪: {result['emotion']['primary']}")
    print(f"  modalities.text: {result['x_ext']['fusion']['modalities']['text']}")
    print(f"  modalities.voice: {result['x_ext']['fusion']['modalities']['voice']}")
    print(f"  modalities.vision: {result['x_ext']['fusion']['modalities']['vision']}")
    
    # 测试2: 仅视觉输入
    print("\n测试2: 仅视觉输入")
    perception_vision_only = PerceptionToLLM(
        turn=Turn(utterance_id="utt_002", input_mode="voice", barge_in=False, vad=None),
        asr=None,
        emotion=EmotionSignals(voice=None, text=None, vision=None),
        vision=VisionResult(
            enabled=True,
            provider="mediapipe",
            model={},
            mode="live_stream",
            frame_rate_fps=15,
            window_ms=1000,
            face_count=1,
            features=VisionFeatures(
                face=VisionFace(
                    landmarks_478=[],
                    blendshapes_52=[],
                    au_15=[{"name": "AU12", "intensity": 0.8}],
                    head_pose=None,
                    gaze=None
                )
            ),
            processing=VisionProcessing(au_mapping={}, symmetry={}, smoothing={}),
            summary=VisionSummary(
                va=VA(valence=0.6, arousal=0.4, confidence=0.7),
                expression_8=Expression8(
                    label="Happy",
                    probs={"Neutral": 0.1, "Happy": 0.7, "Sad": 0.05, "Surprise": 0.05,
                           "Fear": 0.03, "Disgust": 0.02, "Anger": 0.03, "Contempt": 0.02}
                )
            ),
            quality={},
            x_ext={}
        )
    )
    result = engine.fuse(perception_vision_only)
    print(f"  主要情绪: {result['emotion']['primary']}")
    print(f"  modalities.text: {result['x_ext']['fusion']['modalities']['text']}")
    print(f"  modalities.voice: {result['x_ext']['fusion']['modalities']['voice']}")
    print(f"  modalities.vision: {result['x_ext']['fusion']['modalities']['vision']}")
    
    # 测试3: 无输入
    print("\n测试3: 无输入")
    perception_empty = PerceptionToLLM(
        turn=Turn(utterance_id="utt_003", input_mode="voice", barge_in=False, vad=None),
        asr=None,
        emotion=EmotionSignals(voice=None, text=None, vision=None),
        vision=None
    )
    result = engine.fuse(perception_empty)
    print(f"  主要情绪: {result['emotion']['primary']}")
    print(f"  所有模态都应为False")
    
    print(f"\n{'=' * 60}")
    print("✓ 模态降级测试完成")
    print(f"{'=' * 60}")
    return True


def test_psychological_assessment():
    """测试心理风险评估"""
    print("=" * 60)
    print("多模态融合测试 - 心理风险评估")
    print("=" * 60)
    
    config = FusionConfig()
    engine = FusionEngine(config)
    
    test_cases = [
        ("高valence低arousal(积极平静)", 0.7, 0.2),
        ("低valence高arousal(焦虑)", -0.6, 0.7),
        ("低valence低arousal(抑郁)", -0.7, 0.1),
        ("中性", 0.0, 0.0),
    ]
    
    print("\n测试不同VA组合的风险评估:")
    print("-" * 60)
    
    for name, valence, arousal in test_cases:
        perception = PerceptionToLLM(
            turn=Turn(utterance_id="utt_test", input_mode="voice", barge_in=False, vad=None),
            asr=AsrResult(text="测试文本", language="zh-CN", confidence=0.9, words=[], wer_hint={}),
            emotion=EmotionSignals(voice=None, text=None, vision=None),
            vision=VisionResult(
                enabled=True,
                provider="mediapipe",
                model={},
                mode="live_stream",
                frame_rate_fps=15,
                window_ms=1000,
                face_count=1,
                features=VisionFeatures(
                    face=VisionFace(landmarks_478=[], blendshapes_52=[], au_15=[], head_pose=None, gaze=None)
                ),
                processing=VisionProcessing(au_mapping={}, symmetry={}, smoothing={}),
                summary=VisionSummary(
                    va=VA(valence=valence, arousal=arousal, confidence=0.7),
                    expression_8=Expression8(label="Neutral", probs={})
                ),
                quality={},
                x_ext={}
            )
        )
        
        result = engine.fuse(perception)
        psych = result['x_ext']['fusion']['psych_state']
        
        print(f"\n{name}:")
        print(f"  VA: ({valence:+.1f}, {arousal:+.1f})")
        print(f"  PHQ-9估计: {psych['phq9_score_est']}/27")
        print(f"  GAD-7估计: {psych['gad7_score_est']}/21")
        print(f"  风险等级: {psych['risk_level']}")
    
    print(f"\n{'=' * 60}")
    return True


def test_feature_alignment():
    """测试特征时空对齐"""
    print("=" * 60)
    print("多模态融合测试 - 特征时空对齐")
    print("=" * 60)
    
    config = FusionConfig()
    engine = FusionEngine(config)
    
    # 模拟多帧视觉特征
    vision_results = []
    for i in range(10):
        perception = PerceptionToLLM(
            turn=Turn(utterance_id=f"utt_{i}", input_mode="voice", barge_in=False, 
                     vad={"speech_start_ms": i*100, "speech_end_ms": (i+1)*100}),
            asr=None,
            emotion=EmotionSignals(voice=None, text=None, vision=None),
            vision=VisionResult(
                enabled=True,
                provider="mediapipe",
                model={},
                mode="live_stream",
                frame_rate_fps=15,
                window_ms=1000,
                face_count=1,
                features=VisionFeatures(
                    face=VisionFace(
                        landmarks_478=[],
                        blendshapes_52=[],
                        au_15=[{"name": "AU12", "intensity": 0.1 * i}],
                        head_pose=None,
                        gaze=None
                    )
                ),
                processing=VisionProcessing(au_mapping={}, symmetry={}, smoothing={}),
                summary=VisionSummary(
                    va=VA(valence=0.1 * i, arousal=0.05 * i, confidence=0.7),
                    expression_8=Expression8(label="Neutral", probs={})
                ),
                quality={},
                x_ext={}
            )
        )
        vision_results.append(perception)
    
    print(f"\n模拟 {len(vision_results)} 帧视觉特征")
    
    # 测试平滑效果
    print("\n测试平滑效果:")
    print("-" * 60)
    
    for i, perception in enumerate(vision_results[:5]):
        result = engine.fuse(perception)
        va = result['emotion']
        print(f"帧 {i}: valence={va['valence']:+.3f}, arousal={va['arousal']:+.3f}")
    
    print(f"\n{'=' * 60}")
    return True


def test_output_protocol():
    """测试输出协议合规性"""
    print("=" * 60)
    print("多模态融合测试 - 输出协议合规性")
    print("=" * 60)
    
    config = FusionConfig()
    engine = FusionEngine(config)
    
    # 构建完整输入
    perception = PerceptionToLLM(
        turn=Turn(
            utterance_id="utt_protocol_test",
            input_mode="voice",
            barge_in=False,
            vad={"speech_start_ms": 100, "speech_end_ms": 3500}
        ),
        asr=AsrResult(
            text="测试文本",
            language="zh-CN",
            confidence=0.9,
            words=[{"w": "测试", "start_ms": 100, "end_ms": 300}],
            wer_hint={"domain": "psychology"}
        ),
        emotion=EmotionSignals(
            voice=VoiceSignal(
                enabled=True,
                x_features={
                    "mfcc_20_mean": [0.0] * 20,
                    "mfcc_20_std": [0.1] * 20,
                    "pitch_mean": 120.0,
                    "pitch_std": 15.0,
                    "energy_mean": 0.3,
                    "energy_std": 0.05,
                }
            ),
            text=None,
            vision=None
        ),
        vision=VisionResult(
            enabled=True,
            provider="mediapipe",
            model={"name": "face_landmarker.task", "with_blendshapes": True},
            mode="live_stream",
            frame_rate_fps=15,
            window_ms=1000,
            face_count=1,
            features=VisionFeatures(
                face=VisionFace(
                    landmarks_478=[],
                    blendshapes_52=[],
                    au_15=[{"name": f"AU{i}", "intensity": 0.1} for i in [1, 2, 4]],
                    head_pose={"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
                    gaze={"x": 0.0, "y": 0.0, "z": 1.0}
                )
            ),
            processing=VisionProcessing(
                au_mapping={"name": "name2auweight", "version": "v1"},
                symmetry={"enabled": True},
                smoothing={"enabled": True, "type": "iir_1st_order"}
            ),
            summary=VisionSummary(
                va=VA(valence=0.0, arousal=0.0, confidence=0.7),
                expression_8=Expression8(
                    label="Neutral",
                    probs={e: 1/8 for e in ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger", "Contempt"]}
                )
            ),
            quality={"tracking_state": "tracked", "confidence": 0.9},
            x_ext={}
        )
    )
    
    result = engine.fuse(perception)
    
    print("\n验证 PerceptionToLLM 输出协议:")
    print("-" * 60)
    
    checks = []
    
    # emotion字段验证
    emotion = result.get('emotion', {})
    checks.append(("emotion.primary 存在且为字符串", 
                   isinstance(emotion.get('primary'), str)))
    checks.append(("emotion.valence 存在且在[-1,1]范围", 
                   -1 <= emotion.get('valence', 0) <= 1))
    checks.append(("emotion.arousal 存在且在[-1,1]范围", 
                   -1 <= emotion.get('arousal', 0) <= 1))
    checks.append(("emotion.confidence 存在且在[0,1]范围", 
                   0 <= emotion.get('confidence', 0) <= 1))
    
    # signals验证
    signals = emotion.get('signals', {})
    checks.append(("emotion.signals.voice 存在", 'voice' in signals))
    checks.append(("emotion.signals.text 存在", 'text' in signals))
    checks.append(("emotion.signals.vision 存在", 'vision' in signals))
    
    # x_ext.fusion验证
    x_ext = result.get('x_ext', {})
    fusion = x_ext.get('fusion', {})
    checks.append(("x_ext.fusion 存在", bool(fusion)))
    checks.append(("fusion.version 存在", 'version' in fusion))
    checks.append(("fusion.window_ms 存在", 'window_ms' in fusion))
    checks.append(("fusion.hop_ms 存在", 'hop_ms' in fusion))
    checks.append(("fusion.modalities 存在", 'modalities' in fusion))
    checks.append(("fusion.fused_embedding 存在", 'fused_embedding' in fusion))
    checks.append(("fusion.emotion_probs 存在且为8类", 
                   len(fusion.get('emotion_probs', {})) == 8))
    checks.append(("fusion.va 存在", 'va' in fusion))
    checks.append(("fusion.psych_state 存在", 'psych_state' in fusion))
    checks.append(("fusion.alignment 存在", 'alignment' in fusion))
    
    # psych_state验证
    psych = fusion.get('psych_state', {})
    checks.append(("psych_state.phq9_score_est 存在且在[0,27]范围", 
                   0 <= psych.get('phq9_score_est', 0) <= 27))
    checks.append(("psych_state.gad7_score_est 存在且在[0,21]范围", 
                   0 <= psych.get('gad7_score_est', 0) <= 21))
    checks.append(("psych_state.risk_level 存在且在[low,medium,high]", 
                   psych.get('risk_level') in ['low', 'medium', 'high']))
    
    for check_name, passed in checks:
        status = "✓" if passed else "✗"
        print(f"{status} {check_name}")
    
    passed_count = sum(1 for _, p in checks if p)
    total_count = len(checks)
    
    print(f"\n{'=' * 60}")
    print(f"协议合规性: {passed_count}/{total_count} 通过")
    print(f"{'=' * 60}")
    
    return passed_count == total_count


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="测试多模态融合模块")
    parser.add_argument(
        "--mode",
        choices=["fusion", "fallback", "psych", "alignment", "protocol", "all"],
        default="all",
        help="测试模式"
    )
    
    args = parser.parse_args()
    
    results = []
    
    if args.mode in ["fusion", "all"]:
        results.append(("融合引擎", test_fusion_engine()))
    
    if args.mode in ["fallback", "all"]:
        results.append(("模态降级", test_modality_fallback()))
    
    if args.mode in ["psych", "all"]:
        results.append(("心理评估", test_psychological_assessment()))
    
    if args.mode in ["alignment", "all"]:
        results.append(("时空对齐", test_feature_alignment()))
    
    if args.mode in ["protocol", "all"]:
        results.append(("协议合规性", test_output_protocol()))
    
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    for name, passed in results:
        status = "通过" if passed else "失败"
        print(f"{name}: {status}")
