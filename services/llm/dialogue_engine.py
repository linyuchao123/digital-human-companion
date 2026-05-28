#!/usr/bin/env python3
"""
情感对话引擎
整合LLM、RAG、记忆系统的完整对话流程
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from packages.common.protocols import PerceptionToLLM, LLMToDriver, AssistantInfo, Action, RenderInfo, PolicyInfo, SafetyPolicy


@dataclass(frozen=True)
class DialogueConfig:
    """对话引擎配置"""
    enable_rag: bool = True
    enable_memory: bool = True
    max_context_turns: int = 10
    response_timeout_ms: int = 55000
    safety_check: bool = True


class DialogueEngine:
    """
    情感对话引擎
    
    整合:
    - Qwen2.5 LLM
    - RAG知识检索
    - 记忆系统
    - 安全护栏
    - 渲染指令生成
    """
    
    def __init__(
        self,
        config: Optional[DialogueConfig] = None,
        llm_engine=None,
        rag_engine=None,
        memory_store=None
    ):
        self.config = config or DialogueConfig()
        self._llm = llm_engine
        self._rag = rag_engine
        self._memory = memory_store
        
        # 如果没有提供引擎，创建默认实例
        if self._llm is None:
            from services.llm.qwen_engine import QwenEngine, QwenConfig
            self._llm = QwenEngine(QwenConfig())
        
        if self._rag is None and self.config.enable_rag:
            from services.llm.rag_engine import get_rag_engine, PsychologyKnowledgeBase
            self._rag = get_rag_engine()
            # 如果知识库为空，初始化默认知识
            if self._rag.get_stats()["count"] == 0:
                PsychologyKnowledgeBase.initialize_kb(self._rag)
    
    def process_turn(self, perception: PerceptionToLLM) -> LLMToDriver:
        """
        处理一轮对话
        
        Args:
            perception: 感知输入
            
        Returns:
            LLM输出
        """
        start_time = time.time()
        
        # 1. 安全检查 - 输入
        if self.config.safety_check:
            input_safety = self._check_input_safety(perception)
            if input_safety.get("block"):
                return self._create_blocked_response(perception, input_safety)
        
        # 2. 构建对话上下文
        messages = self._build_messages(perception)
        
        # 3. 调用LLM生成回复
        llm_result = self._llm.generate(
            messages=messages,
            max_tokens=800,
            timeout_ms=self.config.response_timeout_ms
        )
        
        response_text = llm_result.get("text", "")
        
        # 4. 安全检查 - 输出
        if self.config.safety_check:
            output_safety = self._llm.check_safety(response_text)
        else:
            output_safety = {"risk_level": "low", "handoff": False}
        
        # 5. 生成记忆写入动作
        memory_actions = self._generate_memory_actions(perception, response_text)
        
        # 6. 生成渲染指令
        render = self._generate_render(perception, response_text)
        
        # 7. 生成安全策略
        policy = self._generate_policy(perception, output_safety)
        
        # 8. 构建输出
        elapsed_ms = int((time.time() - start_time) * 1000)
        
        output = LLMToDriver(
            protocol="llm_to_driver",
            version="1.0",
            trace_id=perception.trace_id,
            session_id=perception.session_id,
            turn_id=perception.turn_id,
            assistant=AssistantInfo(text=response_text),
            actions=[
                Action(type="tts_speak", params={"text": response_text})
            ] + memory_actions,
            render=render,
            policy=policy,
            x_ext={
                "llm": {
                    "model": "Qwen2.5-14B-Instruct",
                    "elapsed_ms": elapsed_ms,
                    "tokens_generated": llm_result.get("tokens_generated", 0),
                    "success": llm_result.get("success", False)
                },
                "rag": {
                    "enabled": self.config.enable_rag,
                    "context_used": self.config.enable_rag
                },
                "safety": output_safety
            }
        )
        
        return output
    
    def _build_messages(self, perception: PerceptionToLLM) -> List[Dict[str, str]]:
        """构建LLM对话消息"""
        messages = []
        
        # 系统提示词
        system_prompt = self._build_system_prompt(perception)
        messages.append({"role": "system", "content": system_prompt})
        
        # 历史对话（从记忆中获取）
        if self.config.enable_memory and perception.memory:
            history = self._get_conversation_history(perception)
            messages.extend(history)
        
        # 当前用户输入
        user_message = self._build_user_message(perception)
        messages.append({"role": "user", "content": user_message})
        
        return messages
    
    def _build_system_prompt(self, perception: PerceptionToLLM) -> str:
        """构建系统提示词"""
        base_prompt = """你是一位专业的心理陪护助手，名叫"小安"。你的职责是：

1. **共情与倾听**：理解用户的情绪，给予情感支持
2. **安全边界**：
   - 不进行医疗诊断
   - 不提供药物建议  
   - 不讨论自伤方法细节
3. **危机识别**：识别高风险信号，建议寻求专业帮助
4. **回复结构**（必须遵循）：
   - 第1句：共情确认（"我理解你..."）
   - 第2句：追问或澄清（"能告诉我..."）
   - 第3句：小步建议或总结（"你可以尝试..."）

记住：你是陪伴者，不是医生。用温暖、专业的语气对话。"""
        
        # 添加RAG上下文
        if self.config.enable_rag and self._rag and perception.asr:
            rag_context = self._rag.build_context(perception.asr.text)
            if rag_context:
                base_prompt += f"\n\n{rag_context}"
        
        return base_prompt
    
    def _build_user_message(self, perception: PerceptionToLLM) -> str:
        """构建用户消息"""
        parts = []
        
        # 用户输入
        if perception.asr and perception.asr.text:
            parts.append(f"用户说：{perception.asr.text}")
        else:
            parts.append("用户没有说话")
        
        # 情绪信息
        if perception.emotion:
            emotion_parts = []
            if perception.emotion.primary:
                emotion_parts.append(f"情绪：{perception.emotion.primary}")
            if perception.emotion.valence is not None:
                emotion_parts.append(f"愉悦度：{perception.emotion.valence:.2f}")
            if perception.emotion.arousal is not None:
                emotion_parts.append(f"激活度：{perception.emotion.arousal:.2f}")
            
            if emotion_parts:
                parts.append(f"情绪分析：{', '.join(emotion_parts)}")
        
        # 心理风险评估
        if perception.x_ext and perception.x_ext.get("fusion"):
            fusion = perception.x_ext["fusion"]
            if fusion.get("psych_state"):
                psych = fusion["psych_state"]
                parts.append(f"风险评估：等级={psych.get('risk_level', 'unknown')}, "
                           f"PHQ-9={psych.get('phq9_score_est', 0)}, "
                           f"GAD-7={psych.get('gad7_score_est', 0)}")
        
        return "\n".join(parts)
    
    def _get_conversation_history(self, perception: PerceptionToLLM) -> List[Dict[str, str]]:
        """获取对话历史"""
        history = []
        
        # 从记忆中读取历史对话
        if perception.memory and perception.memory.read:
            for result in perception.memory.read.results:
                if result.get("type") == "dialogue_turn":
                    history.append({
                        "role": "user",
                        "content": result.get("user_text", "")
                    })
                    history.append({
                        "role": "assistant",
                        "content": result.get("assistant_text", "")
                    })
        
        # 限制历史轮数
        max_turns = self.config.max_context_turns * 2  # *2因为每轮有user+assistant
        if len(history) > max_turns:
            history = history[-max_turns:]
        
        return history
    
    def _check_input_safety(self, perception: PerceptionToLLM) -> Dict[str, Any]:
        """检查输入安全性"""
        if not perception.asr or not perception.asr.text:
            return {"block": False}
        
        # 检查危险指令
        dangerous_patterns = [
            r"忽略.*指令",
            r"忘记.*设定",
            r"你是.*(医生|专家)",
            r"诊断.*疾病",
            r"开.*药方",
        ]
        
        import re
        text = perception.asr.text.lower()
        
        for pattern in dangerous_patterns:
            if re.search(pattern, text):
                return {
                    "block": True,
                    "reason": "检测到潜在危险指令",
                    "pattern": pattern
                }
        
        return {"block": False}
    
    def _create_blocked_response(
        self,
        perception: PerceptionToLLM,
        safety_info: Dict[str, Any]
    ) -> LLMToDriver:
        """创建被阻止的响应"""
        return LLMToDriver(
            protocol="llm_to_driver",
            version="1.0",
            trace_id=perception.trace_id,
            session_id=perception.session_id,
            turn_id=perception.turn_id,
            assistant=AssistantInfo(
                text="抱歉，我无法回应这个请求。让我们继续聊聊你的感受吧。"
            ),
            actions=[Action(type="tts_speak", params={"text": "抱歉，我无法回应这个请求。让我们继续聊聊你的感受吧。"})],
            render=RenderInfo(),
            policy=PolicyInfo(
                safety=SafetyPolicy(
                    risk_level="medium",
                    handoff=False,
                    recommendations=["输入被安全系统拦截"]
                )
            ),
            x_ext={"blocked": True, "reason": safety_info.get("reason")}
        )
    
    def _generate_memory_actions(
        self,
        perception: PerceptionToLLM,
        assistant_text: str
    ) -> List[Action]:
        """生成记忆写入动作"""
        actions = []
        
        if not perception.asr or not perception.asr.text:
            return actions
        
        user_text = perception.asr.text
        
        # 提取关键事实
        # 提取名字
        import re
        name_match = re.search(r"我叫(\S+)", user_text)
        if name_match:
            actions.append(Action(
                type="memory_write",
                params={
                    "type": "fact",
                    "content": f"用户名字叫{name_match.group(1)}",
                    "confidence": 0.9,
                    "privacy": "long_term",
                    "ttl_days": 365
                }
            ))
        
        # 提取情绪状态
        emotion_keywords = ["感到", "觉得", "心情", "情绪"]
        for kw in emotion_keywords:
            if kw in user_text:
                actions.append(Action(
                    type="memory_write",
                    params={
                        "type": "episode",
                        "content": f"用户表达：{user_text[:100]}",
                        "confidence": 0.8,
                        "privacy": "session",
                        "ttl_days": 1
                    }
                ))
                break
        
        # 记录当前对话轮次
        actions.append(Action(
            type="memory_write",
            params={
                "type": "dialogue_turn",
                "user_text": user_text,
                "assistant_text": assistant_text,
                "turn_id": perception.turn_id,
                "privacy": "session",
                "ttl_days": 7
            }
        ))
        
        return actions
    
    def _generate_render(
        self,
        perception: PerceptionToLLM,
        assistant_text: str
    ) -> RenderInfo:
        """生成渲染指令"""
        from packages.common.protocols import RenderVoice, RenderAvatar
        
        # 默认渲染
        voice = RenderVoice(
            speed=1.0,
            pitch=1.0,
            volume=1.0,
            emotion="neutral"
        )
        avatar = RenderAvatar(
            expression="neutral",
            gesture=None
        )
        
        # 基于情绪调整
        if perception.emotion and perception.emotion.primary:
            emotion = perception.emotion.primary.lower()
            
            if emotion in ["happy", "happy"]:
                voice.emotion = "happy"
                avatar.expression = "happy"
            elif emotion in ["sad", "sad"]:
                voice.emotion = "sad"
                avatar.expression = "sad"
                voice.speed = 0.9
                voice.pitch = 0.95
            elif emotion in ["angry", "anger"]:
                voice.emotion = "neutral"  # 保持冷静
                avatar.expression = "concerned"
                voice.speed = 0.95
            elif emotion in ["fear", "fear"]:
                voice.emotion = "gentle"
                avatar.expression = "concerned"
                voice.speed = 0.9
            elif emotion in ["surprise", "surprise"]:
                voice.emotion = "neutral"
                avatar.expression = "surprised"
        
        # 基于风险等级调整
        if perception.x_ext and perception.x_ext.get("fusion"):
            psych_state = perception.x_ext["fusion"].get("psych_state", {})
            risk_level = psych_state.get("risk_level", "low")
            
            if risk_level == "high":
                voice.emotion = "gentle"
                voice.speed = 0.85
                avatar.expression = "concerned"
            elif risk_level == "medium":
                voice.emotion = "gentle"
                voice.speed = 0.9
        
        return RenderInfo(voice=voice, avatar=avatar)
    
    def _generate_policy(
        self,
        perception: PerceptionToLLM,
        safety_check: Dict[str, Any]
    ) -> PolicyInfo:
        """生成安全策略"""
        # 基础风险等级
        risk_level = safety_check.get("risk_level", "low")
        handoff = safety_check.get("handoff", False)
        recommendations = []
        
        # 基于融合模块的风险评估
        if perception.x_ext and perception.x_ext.get("fusion"):
            psych_state = perception.x_ext["fusion"].get("psych_state", {})
            fusion_risk = psych_state.get("risk_level", "low")
            
            # 取最高风险等级
            if fusion_risk == "high" or risk_level == "high":
                risk_level = "high"
                handoff = True
                recommendations.append("建议寻求专业心理帮助")
                recommendations.append("心理危机热线：12320")
            elif fusion_risk == "medium" or risk_level == "medium":
                risk_level = "medium"
                recommendations.append("建议与亲友交流")
                recommendations.append("可以考虑寻求心理咨询")
        
        return PolicyInfo(
            safety=SafetyPolicy(
                risk_level=risk_level,
                handoff=handoff,
                recommendations=recommendations
            )
        )
