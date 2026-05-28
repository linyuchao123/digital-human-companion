from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import openai

from packages.common.protocols import PerceptionToLLM, LLMToDriver


@dataclass(frozen=True)
class LLMConfig:
    model_name: str = "gpt-3.5-turbo"
    max_tokens: int = 1000
    temperature: float = 0.7
    top_p: float = 0.9
    system_prompt: str = "你是一个专业的心理陪护助手，善于倾听和共情，能够理解用户的情绪和需求，提供支持和建议。请保持友好、专业的语气，避免使用专业术语，让用户感到被理解和支持。"


class LLMEngine:
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        # 假设已经设置了openai.api_key
        # openai.api_key = "YOUR_API_KEY"

    def generate_response(self, perception: PerceptionToLLM) -> LLMToDriver:
        # 1. 构建对话历史
        messages = self._build_messages(perception)

        # 2. 调用LLM生成响应
        response = self._call_llm(messages)

        # 3. 解析LLM响应
        assistant_text = response.choices[0].message.content.strip()

        # 4. 生成记忆写入动作
        memory_write_actions = self._generate_memory_write(perception, assistant_text)

        # 5. 生成安全策略
        safety_policy = self._generate_safety_policy(perception)

        # 6. 生成渲染指令
        render = self._generate_render(perception, assistant_text)

        # 7. 构建输出
        output = LLMToDriver(
            protocol="llm_to_driver",
            version="1.0",
            trace_id=perception.trace_id,
            session_id=perception.session_id,
            turn_id=perception.turn_id,
            assistant={"text": assistant_text},
            actions=[{"type": "tts_speak", "params": {"text": assistant_text}}] + memory_write_actions,
            render=render,
            policy={"safety": safety_policy},
            x_ext={}
        )

        return output

    def _build_messages(self, perception: PerceptionToLLM) -> List[Dict[str, str]]:
        messages = [
            {"role": "system", "content": self.config.system_prompt}
        ]

        # 添加用户的当前输入
        user_input = perception.asr.text
        if not user_input:
            user_input = "用户没有说话"

        # 添加情绪和心理状态信息
        emotion_info = ""
        if perception.emotion:
            if perception.emotion.primary:
                emotion_info += f"用户当前情绪：{perception.emotion.primary}。"
            if perception.emotion.valence is not None and perception.emotion.arousal is not None:
                emotion_info += f"情绪 valence: {perception.emotion.valence}, arousal: {perception.emotion.arousal}。"

        # 添加心理风险信息
        psych_info = ""
        if perception.x_ext and perception.x_ext.get("fusion"):
            fusion = perception.x_ext["fusion"]
            if fusion.get("psych_state"):
                psych_state = fusion["psych_state"]
                psych_info += f"心理风险等级：{psych_state.get('risk_level', 'unknown')}。"
                psych_info += f"PHQ-9估计分数：{psych_state.get('phq9_score_est', 0)}，GAD-7估计分数：{psych_state.get('gad7_score_est', 0)}。"

        # 构建用户消息
        user_message = f"用户说：{user_input}"
        if emotion_info:
            user_message += f"\n{emotion_info}"
        if psych_info:
            user_message += f"\n{psych_info}"

        messages.append({"role": "user", "content": user_message})

        # 添加历史对话（如果有）
        # 这里简化处理，实际应用中可能需要从memory中获取

        return messages

    def _call_llm(self, messages: List[Dict[str, str]]) -> openai.ChatCompletion:
        # 调用OpenAI API
        response = openai.ChatCompletion.create(
            model=self.config.model_name,
            messages=messages,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p
        )
        return response

    def _generate_memory_write(self, perception: PerceptionToLLM, assistant_text: str) -> List[Dict[str, any]]:
        # 生成记忆写入动作
        # 这里简化处理，实际应用中可能需要更复杂的逻辑
        actions = []

        # 提取用户输入中的关键信息
        user_input = perception.asr.text
        if user_input:
            # 简单的规则提取事实
            if "我叫" in user_input:
                name = user_input.split("我叫")[1].strip().split()[0]
                actions.append({
                    "type": "memory_write",
                    "params": {
                        "type": "fact",
                        "key": "name",
                        "value": name,
                        "privacy": "long_term"
                    }
                })

            # 提取情绪状态
            if "我感到" in user_input or "我觉得" in user_input:
                emotion = user_input.split("我感到")[1].strip() if "我感到" in user_input else user_input.split("我觉得")[1].strip()
                actions.append({
                    "type": "memory_write",
                    "params": {
                        "type": "episode",
                        "key": "emotion",
                        "value": emotion,
                        "privacy": "session"
                    }
                })

        return actions

    def _generate_safety_policy(self, perception: PerceptionToLLM) -> Dict[str, any]:
        # 生成安全策略
        safety = {
            "risk_level": "low",
            "handoff": False,
            "recommendations": []
        }

        # 基于心理风险评估设置安全策略
        if perception.x_ext and perception.x_ext.get("fusion"):
            fusion = perception.x_ext["fusion"]
            if fusion.get("psych_state"):
                psych_state = fusion["psych_state"]
                risk_level = psych_state.get("risk_level", "low")
                safety["risk_level"] = risk_level

                # 高风险时触发handoff
                if risk_level == "high":
                    safety["handoff"] = True
                    safety["recommendations"].append("建议寻求专业心理帮助")
                    safety["recommendations"].append("请联系心理热线：12320")
                elif risk_level == "medium":
                    safety["recommendations"].append("建议与亲友交流")
                    safety["recommendations"].append("可以考虑寻求心理咨询")

        return safety

    def _generate_render(self, perception: PerceptionToLLM, assistant_text: str) -> Dict[str, any]:
        # 生成渲染指令
        render = {
            "voice": {
                "speed": 1.0,
                "pitch": 1.0,
                "volume": 1.0,
                "emotion": "neutral"
            },
            "avatar": {
                "expression": "neutral",
                "gesture": None
            }
        }

        # 基于情绪设置语音和表情
        if perception.emotion and perception.emotion.primary:
            emotion = perception.emotion.primary.lower()
            if emotion == "happy":
                render["voice"]["emotion"] = "happy"
                render["avatar"]["expression"] = "happy"
            elif emotion == "sad":
                render["voice"]["emotion"] = "sad"
                render["avatar"]["expression"] = "sad"
                render["voice"]["speed"] = 0.9
                render["voice"]["pitch"] = 0.9
            elif emotion == "angry":
                render["voice"]["emotion"] = "angry"
                render["avatar"]["expression"] = "angry"
                render["voice"]["volume"] = 1.1
            elif emotion == "surprise":
                render["voice"]["emotion"] = "surprised"
                render["avatar"]["expression"] = "surprised"
                render["voice"]["pitch"] = 1.1

        return render
