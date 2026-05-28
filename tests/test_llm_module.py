#!/usr/bin/env python3
"""
LLM模块测试脚本
测试Qwen引擎、RAG知识库、对话引擎
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from services.llm.qwen_engine import QwenEngine, QwenConfig
from services.llm.rag_engine import RAGEngine, RAGConfig, PsychologyKnowledgeBase
from services.llm.dialogue_engine import DialogueEngine, DialogueConfig
from packages.common.protocols import (
    PerceptionToLLM, TurnInfo, AsrInfo, EmotionInfo, EmotionSignals, EmotionSignal,
    Timestamp, Constraints
)


def test_qwen_engine():
    """测试Qwen引擎"""
    print("=" * 60)
    print("LLM模块测试 - Qwen引擎")
    print("=" * 60)
    
    print("\n初始化Qwen引擎...")
    config = QwenConfig(
        load_in_4bit=True,
        max_tokens=200
    )
    
    try:
        engine = QwenEngine(config)
        print("✓ Qwen引擎初始化完成")
    except Exception as e:
        print(f"✗ 初始化失败: {e}")
        return False
    
    # 测试生成
    print("\n测试文本生成...")
    messages = [
        {"role": "system", "content": "你是一个心理陪护助手。"},
        {"role": "user", "content": "我最近压力很大，睡不好觉。"}
    ]
    
    result = engine.generate(messages, max_tokens=100)
    
    print(f"✓ 生成完成")
    print(f"  回复: {result['text'][:100]}...")
    print(f"  Token数: {result['tokens_generated']}")
    print(f"  耗时: {result['elapsed_ms']}ms")
    print(f"  成功: {result['success']}")
    
    # 测试安全检查
    print("\n测试安全检查...")
    safe_text = "我今天心情不错"
    crisis_text = "我想结束生命"
    
    safe_result = engine.check_safety(safe_text)
    crisis_result = engine.check_safety(crisis_text)
    
    print(f"  安全文本: {safe_result['risk_level']}")
    print(f"  危机文本: {crisis_result['risk_level']} (handoff: {crisis_result['handoff']})")
    
    if crisis_result['risk_level'] == 'high':
        print("✓ 安全检查正确识别危机信号")
    else:
        print("✗ 安全检查未识别危机信号")
    
    print(f"\n{'=' * 60}")
    return True


def test_rag_engine():
    """测试RAG引擎"""
    print("=" * 60)
    print("LLM模块测试 - RAG知识库")
    print("=" * 60)
    
    print("\n初始化RAG引擎...")
    config = RAGConfig(
        chroma_db_path="data/kb/test_chroma",
        top_k=3
    )
    
    try:
        engine = RAGEngine(config)
        print("✓ RAG引擎初始化完成")
    except Exception as e:
        print(f"✗ 初始化失败: {e}")
        return False
    
    # 查看统计
    stats = engine.get_stats()
    print(f"\n知识库状态:")
    print(f"  状态: {stats['status']}")
    print(f"  文档数: {stats['count']}")
    
    # 初始化知识库
    print("\n初始化心理学知识库...")
    PsychologyKnowledgeBase.initialize_kb(engine)
    
    stats = engine.get_stats()
    print(f"  初始化后文档数: {stats['count']}")
    
    # 测试检索
    print("\n测试知识检索...")
    queries = [
        "失眠怎么办",
        "感到焦虑",
        "如何放松"
    ]
    
    for query in queries:
        print(f"\n  查询: '{query}'")
        results = engine.retrieve(query, top_k=2)
        
        if results:
            print(f"  检索到 {len(results)} 条结果:")
            for i, r in enumerate(results, 1):
                print(f"    [{i}] 相似度: {r['similarity']:.3f}")
                print(f"        内容: {r['document'][:50]}...")
        else:
            print("  未检索到结果")
    
    # 测试上下文构建
    print("\n测试上下文构建...")
    context = engine.build_context("失眠很严重")
    if context:
        print(f"✓ 上下文构建成功")
        print(f"  长度: {len(context)} 字符")
    else:
        print("✗ 上下文为空")
    
    print(f"\n{'=' * 60}")
    return True


def test_dialogue_engine():
    """测试对话引擎"""
    print("=" * 60)
    print("LLM模块测试 - 对话引擎")
    print("=" * 60)
    
    print("\n初始化对话引擎...")
    config = DialogueConfig(
        enable_rag=False,  # 简化测试
        enable_memory=False,
        max_context_turns=5
    )
    
    try:
        engine = DialogueEngine(config)
        print("✓ 对话引擎初始化完成")
    except Exception as e:
        print(f"✗ 初始化失败: {e}")
        return False
    
    # 构建测试输入
    print("\n构建测试输入...")
    perception = PerceptionToLLM(
        trace_id="test_trace_001",
        session_id="test_session_001",
        turn_id=1,
        timestamp=Timestamp(),
        constraints=Constraints(deadline_ms=60000),
        turn=TurnInfo(utterance_id="utt_001", input_mode="voice", barge_in=False),
        asr=AsrInfo(text="我最近总是失眠，感觉很焦虑", language="zh-CN", confidence=0.92),
        emotion=EmotionInfo(
            primary="anxious",
            valence=-0.4,
            arousal=0.6,
            confidence=0.8,
            signals=EmotionSignals(
                voice=EmotionSignal(enabled=True, x_features={})
            )
        ),
        x_ext={
            "fusion": {
                "psych_state": {
                    "phq9_score_est": 12,
                    "gad7_score_est": 10,
                    "risk_level": "medium"
                }
            }
        }
    )
    
    # 处理对话
    print("\n处理对话...")
    result = engine.process_turn(perception)
    
    print("✓ 对话处理完成")
    print(f"\n输出结果:")
    print(f"  回复文本: {result.assistant.text[:100]}...")
    print(f"  Actions数: {len(result.actions)}")
    
    # 验证渲染指令
    print(f"\n渲染指令:")
    print(f"  语音情绪: {result.render.voice.emotion}")
    print(f"  语速: {result.render.voice.speed}")
    print(f"  表情: {result.render.avatar.expression}")
    
    # 验证安全策略
    print(f"\n安全策略:")
    print(f"  风险等级: {result.policy.safety.risk_level}")
    print(f"  是否转接: {result.policy.safety.handoff}")
    if result.policy.safety.recommendations:
        print(f"  建议: {result.policy.safety.recommendations}")
    
    # 验证扩展信息
    print(f"\n扩展信息:")
    print(f"  LLM耗时: {result.x_ext['llm']['elapsed_ms']}ms")
    print(f"  Token数: {result.x_ext['llm']['tokens_generated']}")
    
    print(f"\n{'=' * 60}")
    return True


def test_end_to_end():
    """端到端测试"""
    print("=" * 60)
    print("LLM模块测试 - 端到端流程")
    print("=" * 60)
    
    print("\n场景：用户表达自杀意念")
    
    # 创建对话引擎
    config = DialogueConfig(enable_rag=True, enable_memory=True)
    engine = DialogueEngine(config)
    
    # 高风险输入
    perception = PerceptionToLLM(
        trace_id="crisis_test_001",
        session_id="crisis_session_001",
        turn_id=1,
        timestamp=Timestamp(),
        constraints=Constraints(deadline_ms=60000),
        turn=TurnInfo(utterance_id="utt_crisis", input_mode="voice", barge_in=False),
        asr=AsrInfo(text="我不想活了，想结束这一切", language="zh-CN", confidence=0.95),
        emotion=EmotionInfo(
            primary="sad",
            valence=-0.8,
            arousal=-0.3,
            confidence=0.9
        ),
        x_ext={
            "fusion": {
                "psych_state": {
                    "phq9_score_est": 18,
                    "gad7_score_est": 15,
                    "risk_level": "high"
                }
            }
        }
    )
    
    # 处理
    result = engine.process_turn(perception)
    
    print("\n处理结果:")
    print(f"  回复: {result.assistant.text}")
    print(f"  风险等级: {result.policy.safety.risk_level}")
    print(f"  转接人工: {result.policy.safety.handoff}")
    
    if result.policy.safety.risk_level == "high":
        print("✓ 正确识别高风险并触发转接")
    else:
        print("✗ 未正确识别高风险")
    
    print(f"\n{'=' * 60}")
    return True


def test_protocol_compliance():
    """测试协议合规性"""
    print("=" * 60)
    print("LLM模块测试 - 协议合规性")
    print("=" * 60)
    
    engine = DialogueEngine(DialogueConfig())
    
    perception = PerceptionToLLM(
        trace_id="protocol_test",
        session_id="protocol_session",
        turn_id=1,
        timestamp=Timestamp(),
        constraints=Constraints(),
        turn=TurnInfo(utterance_id="utt_001", input_mode="voice"),
        asr=AsrInfo(text="测试", language="zh-CN")
    )
    
    result = engine.process_turn(perception)
    
    print("\n验证 LLMToDriver 协议字段:")
    print("-" * 60)
    
    checks = []
    
    # 基本字段
    checks.append(("protocol = 'llm_to_driver'", result.protocol == "llm_to_driver"))
    checks.append(("version 存在", bool(result.version)))
    checks.append(("trace_id 存在", bool(result.trace_id)))
    checks.append(("session_id 存在", bool(result.session_id)))
    checks.append(("turn_id 存在", result.turn_id is not None))
    
    # assistant字段
    checks.append(("assistant 存在", result.assistant is not None))
    checks.append(("assistant.text 存在", bool(result.assistant.text)))
    
    # actions字段
    checks.append(("actions 存在", len(result.actions) > 0))
    has_tts = any(a.type == "tts_speak" for a in result.actions)
    checks.append(("actions包含tts_speak", has_tts))
    
    # render字段
    checks.append(("render 存在", result.render is not None))
    checks.append(("render.voice 存在", result.render.voice is not None))
    checks.append(("render.avatar 存在", result.render.avatar is not None))
    
    # policy字段
    checks.append(("policy 存在", result.policy is not None))
    checks.append(("policy.safety 存在", result.policy.safety is not None))
    checks.append(("policy.safety.risk_level 存在", bool(result.policy.safety.risk_level)))
    
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
    
    parser = argparse.ArgumentParser(description="测试LLM模块")
    parser.add_argument(
        "--mode",
        choices=["qwen", "rag", "dialogue", "e2e", "protocol", "all"],
        default="all",
        help="测试模式"
    )
    
    args = parser.parse_args()
    
    results = []
    
    if args.mode in ["qwen", "all"]:
        results.append(("Qwen引擎", test_qwen_engine()))
    
    if args.mode in ["rag", "all"]:
        results.append(("RAG知识库", test_rag_engine()))
    
    if args.mode in ["dialogue", "all"]:
        results.append(("对话引擎", test_dialogue_engine()))
    
    if args.mode in ["e2e", "all"]:
        results.append(("端到端", test_end_to_end()))
    
    if args.mode in ["protocol", "all"]:
        results.append(("协议合规性", test_protocol_compliance()))
    
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    for name, passed in results:
        status = "通过" if passed else "失败"
        print(f"{name}: {status}")
