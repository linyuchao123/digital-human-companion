#!/usr/bin/env python3
"""
听觉模块测试脚本
测试VAD、ASR和声学特征提取功能
"""

import sys
import wave
from pathlib import Path

import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from services.asr.audio_pipeline import AudioPipeline, AudioPipelineConfig, load_audio_file
from services.asr.vad.fsmn_vad import FsmnVad, FsmnVadConfig
from services.asr.inference.paraformer_zh import ParaformerZh, ParaformerConfig
from services.asr.inference.acoustic_features import AcousticFeaturesExtractor, AcousticFeaturesConfig


def test_vad_only():
    """单独测试VAD模块"""
    print("=" * 60)
    print("听觉模块测试 - VAD单独测试")
    print("=" * 60)
    
    # 创建测试音频（1秒静音 + 1秒正弦波 + 1秒静音）
    sample_rate = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # 生成信号：静音-声音-静音
    audio = np.zeros_like(t)
    audio[int(sample_rate*1.0):int(sample_rate*2.0)] = np.sin(2 * np.pi * 440 * t[int(sample_rate*1.0):int(sample_rate*2.0)])
    
    # 转换为int16
    pcm16 = (audio * 32767).astype(np.int16)
    
    print(f"\n测试音频: {duration}秒, {sample_rate}Hz")
    print(f"音频结构: 1秒静音 + 1秒440Hz正弦波 + 1秒静音")
    
    # 初始化VAD
    print("\n初始化VAD模型...")
    try:
        vad = FsmnVad(FsmnVadConfig())
        print("✓ VAD初始化成功")
    except Exception as e:
        print(f"✗ VAD初始化失败: {e}")
        return False
    
    # 检测语音段
    print("\n检测语音段...")
    segments = vad.detect_segments(pcm16, sample_rate)
    
    print(f"✓ 检测到 {len(segments)} 个语音段:")
    for i, seg in enumerate(segments):
        print(f"  段 {i+1}: {seg.start_ms}ms - {seg.end_ms}ms (时长: {seg.end_ms - seg.start_ms}ms)")
    
    # 验证结果
    if len(segments) > 0:
        first_seg = segments[0]
        expected_start = 1000  # 1秒
        expected_end = 2000    # 2秒
        
        print("\n验证检测结果:")
        if abs(first_seg.start_ms - expected_start) < 200:
            print(f"✓ 起始时间正确 (期望~{expected_start}ms, 实际{first_seg.start_ms}ms)")
        else:
            print(f"⚠ 起始时间偏差较大 (期望~{expected_start}ms, 实际{first_seg.start_ms}ms)")
        
        if abs(first_seg.end_ms - expected_end) < 200:
            print(f"✓ 结束时间正确 (期望~{expected_end}ms, 实际{first_seg.end_ms}ms)")
        else:
            print(f"⚠ 结束时间偏差较大 (期望~{expected_end}ms, 实际{first_seg.end_ms}ms)")
    
    print(f"\n{'=' * 60}")
    return True


def test_asr_only():
    """单独测试ASR模块"""
    print("=" * 60)
    print("听觉模块测试 - ASR单独测试")
    print("=" * 60)
    
    # 初始化ASR
    print("\n初始化ASR模型...")
    try:
        asr = ParaformerZh(ParaformerConfig())
        print("✓ ASR初始化成功")
    except Exception as e:
        print(f"✗ ASR初始化失败: {e}")
        return False
    
    # 创建测试音频（简单的语音）
    # 注意：这里使用随机噪声作为占位符，实际应该使用真实语音
    sample_rate = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # 生成模拟语音信号（多个频率的组合）
    audio = np.zeros_like(t)
    for freq in [200, 400, 600]:
        audio += 0.3 * np.sin(2 * np.pi * freq * t)
    
    # 添加包络模拟语音
    envelope = np.exp(-((t - duration/2) / (duration/4)) ** 2)
    audio = audio * envelope
    
    pcm16 = (audio * 32767).astype(np.int16)
    
    print(f"\n测试音频: {duration}秒模拟语音")
    
    # 识别
    print("\n执行语音识别...")
    result = asr.transcribe(pcm16, sample_rate)
    
    print(f"✓ 识别完成")
    print(f"  文本: '{result['text']}'")
    print(f"  置信度: {result['confidence']}")
    print(f"  词数: {len(result['words'])}")
    
    print(f"\n{'=' * 60}")
    return True


def test_acoustic_features():
    """测试声学特征提取"""
    print("=" * 60)
    print("听觉模块测试 - 声学特征提取")
    print("=" * 60)
    
    # 初始化特征提取器
    print("\n初始化特征提取器...")
    try:
        extractor = AcousticFeaturesExtractor(AcousticFeaturesConfig())
        print("✓ 特征提取器初始化成功")
    except Exception as e:
        print(f"✗ 初始化失败: {e}")
        return False
    
    # 创建测试音频
    sample_rate = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    pcm16 = (audio * 32767).astype(np.int16)
    
    print(f"\n测试音频: {duration}秒, 440Hz正弦波")
    
    # 提取特征
    print("\n提取声学特征...")
    features = extractor.extract(pcm16, sample_rate)
    
    print("✓ 特征提取完成")
    print(f"\n特征维度:")
    print(f"  MFCC: {np.array(features['mfcc']).shape}")
    print(f"  Pitch: {len(features['pitch'])}")
    print(f"  Energy: {len(features['energy'])}")
    
    # 检查聚合特征
    aggregated = features['aggregated']
    print(f"\n聚合特征:")
    print(f"  mfcc_20_mean: {len(aggregated['mfcc_20_mean'])} 维")
    print(f"  mfcc_20_std: {len(aggregated['mfcc_20_std'])} 维")
    print(f"  pitch_mean: {aggregated['pitch_mean']:.2f}")
    print(f"  pitch_std: {aggregated['pitch_std']:.2f}")
    print(f"  energy_mean: {aggregated['energy_mean']:.2f}")
    print(f"  energy_std: {aggregated['energy_std']:.2f}")
    
    # 验证输出格式
    print("\n验证输出格式...")
    required_keys = ['mfcc_20_mean', 'mfcc_20_std', 'pitch_mean', 'pitch_std', 
                     'energy_mean', 'energy_std', 'sample_rate_hz', 'feature_window_ms']
    
    for key in required_keys:
        if key in aggregated:
            print(f"✓ {key} 存在")
        else:
            print(f"✗ {key} 缺失")
    
    print(f"\n{'=' * 60}")
    return True


def test_full_pipeline():
    """测试完整管道"""
    print("=" * 60)
    print("听觉模块测试 - 完整管道")
    print("=" * 60)
    
    # 初始化管道
    print("\n初始化音频管道...")
    config = AudioPipelineConfig(
        sample_rate_hz=16000,
        enable_vad=True,
        enable_asr=True,
        enable_acoustic_features=True,
        hotwords=["失眠", "焦虑", "压力", "抑郁"]
    )
    
    try:
        pipeline = AudioPipeline(config)
        print("✓ 管道初始化成功")
    except Exception as e:
        print(f"✗ 管道初始化失败: {e}")
        return False
    
    # 创建测试音频
    sample_rate = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # 生成模拟语音（有声-静音-有声）
    audio = np.zeros_like(t)
    audio[0:int(sample_rate*1.0)] = 0.3 * np.sin(2 * np.pi * 300 * t[0:int(sample_rate*1.0)])
    audio[int(sample_rate*2.0):] = 0.3 * np.sin(2 * np.pi * 500 * t[int(sample_rate*2.0):])
    
    pcm16 = (audio * 32767).astype(np.int16)
    
    print(f"\n测试音频: {duration}秒")
    print(f"结构: 1秒300Hz + 1秒静音 + 1秒500Hz")
    
    # 处理音频
    print("\n处理音频...")
    result = pipeline.process_audio(pcm16, sample_rate)
    
    print("✓ 处理完成")
    
    # 验证输出格式
    print("\n验证输出格式 (PerceptionToLLM协议):")
    print("-" * 60)
    
    # 检查turn
    if 'turn' in result:
        print("✓ turn 存在")
        turn = result['turn']
        if 'vad' in turn:
            vad = turn['vad']
            print(f"  VAD: {vad['speech_start_ms']}ms - {vad['speech_end_ms']}ms")
    else:
        print("✗ turn 缺失")
    
    # 检查asr
    if 'asr' in result:
        print("✓ asr 存在")
        asr = result['asr']
        print(f"  文本: '{asr['text']}'")
        print(f"  语言: {asr['language']}")
        print(f"  置信度: {asr['confidence']}")
        print(f"  词数: {len(asr['words'])}")
    else:
        print("✗ asr 缺失")
    
    # 检查emotion.signals.voice
    if 'emotion' in result and 'signals' in result['emotion']:
        print("✓ emotion.signals 存在")
        voice = result['emotion']['signals'].get('voice', {})
        if voice.get('enabled'):
            print("✓ voice.enabled = True")
            x_features = voice.get('x_features', {})
            if x_features:
                print(f"  声学特征: {len(x_features)} 个字段")
        else:
            print("✗ voice.enabled = False")
    else:
        print("✗ emotion.signals 缺失")
    
    # 检查segments
    if 'segments' in result:
        print(f"✓ segments 存在 ({len(result['segments'])} 个段)")
    else:
        print("✗ segments 缺失")
    
    print(f"\n{'=' * 60}")
    return True


def test_protocol_compliance():
    """测试协议合规性"""
    print("=" * 60)
    print("听觉模块测试 - 协议合规性验证")
    print("=" * 60)
    
    config = AudioPipelineConfig()
    pipeline = AudioPipeline(config)
    
    # 创建简单测试音频
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = 0.3 * np.sin(2 * np.pi * 440 * t)
    pcm16 = (audio * 32767).astype(np.int16)
    
    result = pipeline.process_audio(pcm16, sample_rate)
    
    print("\n验证 PerceptionToLLM 协议字段:")
    print("-" * 60)
    
    checks = []
    
    # 检查turn字段
    if 'turn' in result:
        checks.append(("turn 存在", True))
        turn = result['turn']
        checks.append(("turn.utterance_id 存在", 'utterance_id' in turn))
        checks.append(("turn.input_mode = 'voice'", turn.get('input_mode') == 'voice'))
        checks.append(("turn.barge_in 存在", 'barge_in' in turn))
        checks.append(("turn.vad 存在", 'vad' in turn))
        if 'vad' in turn:
            vad = turn['vad']
            checks.append(("turn.vad.speech_start_ms 存在", 'speech_start_ms' in vad))
            checks.append(("turn.vad.speech_end_ms 存在", 'speech_end_ms' in vad))
    else:
        checks.append(("turn 结构", False))
    
    # 检查asr字段
    if 'asr' in result:
        checks.append(("asr 存在", True))
        asr = result['asr']
        checks.append(("asr.text 存在", 'text' in asr))
        checks.append(("asr.language 存在", 'language' in asr))
        checks.append(("asr.confidence 存在", 'confidence' in asr))
        checks.append(("asr.words 存在", 'words' in asr))
    else:
        checks.append(("asr 结构", False))
    
    # 检查emotion字段
    if 'emotion' in result:
        checks.append(("emotion 存在", True))
        emotion = result['emotion']
        if 'signals' in emotion:
            checks.append(("emotion.signals 存在", True))
            signals = emotion['signals']
            if 'voice' in signals:
                checks.append(("emotion.signals.voice 存在", True))
                voice = signals['voice']
                checks.append(("voice.enabled 存在", 'enabled' in voice))
                checks.append(("voice.x_features 存在", 'x_features' in voice))
            else:
                checks.append(("emotion.signals.voice 存在", False))
        else:
            checks.append(("emotion.signals 存在", False))
    else:
        checks.append(("emotion 结构", False))
    
    # 打印结果
    for check_name, passed in checks:
        status = "✓" if passed else "✗"
        print(f"{status} {check_name}")
    
    passed_count = sum(1 for _, p in checks if p)
    total_count = len(checks)
    
    print(f"\n{'=' * 60}")
    print(f"协议验证结果: {passed_count}/{total_count} 通过")
    print(f"{'=' * 60}")
    
    return passed_count == total_count


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="测试听觉模块")
    parser.add_argument(
        "--mode",
        choices=["vad", "asr", "features", "pipeline", "protocol", "all"],
        default="all",
        help="测试模式"
    )
    
    args = parser.parse_args()
    
    results = []
    
    if args.mode in ["vad", "all"]:
        results.append(("VAD", test_vad_only()))
    
    if args.mode in ["asr", "all"]:
        results.append(("ASR", test_asr_only()))
    
    if args.mode in ["features", "all"]:
        results.append(("声学特征", test_acoustic_features()))
    
    if args.mode in ["pipeline", "all"]:
        results.append(("完整管道", test_full_pipeline()))
    
    if args.mode in ["protocol", "all"]:
        results.append(("协议合规性", test_protocol_compliance()))
    
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    for name, passed in results:
        status = "通过" if passed else "失败"
        print(f"{name}: {status}")
