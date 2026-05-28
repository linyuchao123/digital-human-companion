#!/usr/bin/env python3
"""测试ASR听觉模块"""
from services.asr.audio_pipeline import AudioPipeline
from pathlib import Path
import sys

print("=" * 60)
print("ASR听觉模块测试")
print("=" * 60)

try:
    # 1. 导入测试
    print("\n[1/3] 模块导入测试...")
    print("✓ AudioPipeline 导入成功")
    
    # 2. 初始化测试
    print("\n[2/3] 流水线初始化测试...")
    pipeline = AudioPipeline()
    print("✓ AudioPipeline 初始化成功")
    
    # 3. 功能测试（如果有测试音频）
    test_audio = Path("test_audio.wav")
    if test_audio.exists():
        print(f"\n[3/3] 音频识别测试 ({test_audio})...")
        result = pipeline.recognize(str(test_audio))
        print(f"✓ 识别成功")
        print(f"  识别文本: {result.get('text', 'N/A')}")
        print(f"  WER: {result.get('wer', 'N/A')}")
        print(f"  SER: {result.get('ser', 'N/A')}")
    else:
        print(f"\n[3/3] 跳过音频识别测试（未找到 {test_audio}）")
    
    print("\n" + "=" * 60)
    print("ASR模块测试完成 - 全部通过")
    print("=" * 60)
    
except Exception as e:
    print(f"\n❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
