#!/usr/bin/env python3
"""
Qwen2.5 LLM引擎
支持本地4-bit量化模型推理
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional

import torch


@dataclass(frozen=True)
class QwenConfig:
    """Qwen模型配置"""
    model_name: str = "Qwen/Qwen2.5-14B-Instruct"
    model_path: Optional[str] = None  # 本地模型路径
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_tokens: int = 1000
    temperature: float = 0.7
    top_p: float = 0.9
    load_in_4bit: bool = True  # 4-bit量化
    context_length: int = 8192


class QwenEngine:
    """
    Qwen2.5模型推理引擎
    
    支持:
    - 本地4-bit量化模型加载
    - 流式生成
    - 超时控制
    """
    
    # 系统提示词 - 心理陪护助手
    SYSTEM_PROMPT = """你是一位专业的心理陪护助手，名叫"小安"。你的职责是：

1. **共情与倾听**：理解用户的情绪，给予情感支持
2. **安全边界**：
   - 不进行医疗诊断
   - 不提供药物建议
   - 不讨论自伤方法细节
3. **危机识别**：识别高风险信号，建议寻求专业帮助
4. **回复结构**：
   - 第1句：共情确认
   - 第2句：追问或澄清
   - 第3句：小步建议或总结

记住：你是陪伴者，不是医生。用温暖、专业的语气对话。"""

    def __init__(self, config: Optional[QwenConfig] = None):
        self.config = config or QwenConfig()
        self._model = None
        self._tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """加载Qwen模型"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            model_name = self.config.model_path or self.config.model_name
            print(f"[QwenEngine] 正在加载模型: {model_name}")
            
            # 加载tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                local_files_only=True  # 离线模式
            )
            
            # 加载模型（4-bit量化）
            if self.config.load_in_4bit and self.config.device == "cuda":
                from transformers import BitsAndBytesConfig
                
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                
                self._model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True,
                    local_files_only=True
                )
            else:
                self._model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32,
                    device_map="auto" if self.config.device == "cuda" else None,
                    trust_remote_code=True,
                    local_files_only=True
                )
                if self.config.device == "cpu":
                    self._model = self._model.to(self.config.device)
            
            print(f"[QwenEngine] 模型加载成功")
            
        except Exception as e:
            print(f"[QwenEngine] 模型加载失败: {e}")
            print("[QwenEngine] 将使用模拟模式")
            self._model = None
            self._tokenizer = None
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stream: bool = False,
        timeout_ms: int = 60000
    ) -> Dict[str, Any]:
        """
        生成回复
        
        Args:
            messages: 对话历史
            max_tokens: 最大生成token数
            temperature: 温度
            stream: 是否流式输出
            timeout_ms: 超时时间
            
        Returns:
            生成结果
        """
        if self._model is None or self._tokenizer is None:
            # 模拟模式
            return self._mock_generate(messages)
        
        start_time = time.time()
        
        try:
            # 应用对话模板
            prompt = self._apply_chat_template(messages)
            
            # Tokenize
            inputs = self._tokenizer(prompt, return_tensors="pt")
            if self.config.device == "cuda":
                inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
            
            # 生成参数
            gen_kwargs = {
                "max_new_tokens": max_tokens or self.config.max_tokens,
                "temperature": temperature or self.config.temperature,
                "top_p": self.config.top_p,
                "do_sample": True,
                "pad_token_id": self._tokenizer.pad_token_id,
                "eos_token_id": self._tokenizer.eos_token_id,
            }
            
            # 生成
            with torch.no_grad():
                outputs = self._model.generate(**inputs, **gen_kwargs)
            
            # 解码
            generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            response_text = self._tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            elapsed_ms = int((time.time() - start_time) * 1000)
            
            return {
                "text": response_text.strip(),
                "tokens_generated": len(generated_tokens),
                "elapsed_ms": elapsed_ms,
                "success": True
            }
            
        except Exception as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            return {
                "text": "抱歉，我遇到了一些技术问题。让我们稍后再试。",
                "tokens_generated": 0,
                "elapsed_ms": elapsed_ms,
                "success": False,
                "error": str(e)
            }
    
    def generate_stream(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> Generator[str, None, None]:
        """
        流式生成回复
        
        Args:
            messages: 对话历史
            max_tokens: 最大生成token数
            temperature: 温度
            
        Yields:
            生成的文本片段
        """
        if self._model is None or self._tokenizer is None:
            # 模拟模式
            yield "我理解你的感受。"
            yield "能告诉我更多吗？"
            yield "我在这里陪伴你。"
            return
        
        # 流式生成实现
        try:
            prompt = self._apply_chat_template(messages)
            inputs = self._tokenizer(prompt, return_tensors="pt")
            if self.config.device == "cuda":
                inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
            
            # 使用TextIteratorStreamer实现流式输出
            from transformers import TextIteratorStreamer
            from threading import Thread
            
            streamer = TextIteratorStreamer(self._tokenizer, skip_prompt=True, skip_special_tokens=True)
            
            gen_kwargs = {
                **inputs,
                "streamer": streamer,
                "max_new_tokens": max_tokens or self.config.max_tokens,
                "temperature": temperature or self.config.temperature,
                "top_p": self.config.top_p,
                "do_sample": True,
                "pad_token_id": self._tokenizer.pad_token_id,
            }
            
            # 在后台线程生成
            thread = Thread(target=self._model.generate, kwargs=gen_kwargs)
            thread.start()
            
            # 流式输出
            for text in streamer:
                yield text
            
            thread.join()
            
        except Exception as e:
            print(f"[QwenEngine] 流式生成错误: {e}")
            yield "抱歉，我遇到了一些技术问题。"
    
    def _apply_chat_template(self, messages: List[Dict[str, str]]) -> str:
        """应用对话模板"""
        if hasattr(self._tokenizer, 'apply_chat_template') and self._tokenizer.chat_template:
            return self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            # 手动构建prompt
            prompt = ""
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    prompt += f"<|im_start|>system\n{content}<|im_end|>\n"
                elif role == "user":
                    prompt += f"<|im_start|>user\n{content}<|im_end|>\n"
                elif role == "assistant":
                    prompt += f"<|im_start|>assistant\n{content}<|im_end|>\n"
            prompt += "<|im_start|>assistant\n"
            return prompt
    
    def _mock_generate(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """模拟生成（用于测试）"""
        # 提取用户输入
        user_content = ""
        for msg in messages:
            if msg["role"] == "user":
                user_content = msg["content"]
                break
        
        # 简单的规则回复
        if "失眠" in user_content or "睡不着" in user_content:
            response = "失眠确实很难受。这种情况持续多久了？可以试试睡前放松练习。"
        elif "焦虑" in user_content or "担心" in user_content:
            response = "我能感受到你的焦虑。是什么让你感到不安？深呼吸可能会有帮助。"
        elif "难过" in user_content or "伤心" in user_content:
            response = "听到你难过，我很心疼。愿意和我聊聊发生了什么吗？"
        elif "压力" in user_content:
            response = "压力大的时候确实不容易。工作上的压力还是生活中的？"
        else:
            response = "我理解你的感受。能告诉我更多吗？我在这里倾听。"
        
        return {
            "text": response,
            "tokens_generated": len(response) // 2,
            "elapsed_ms": 100,
            "success": True,
            "mock": True
        }
    
    def check_safety(self, text: str) -> Dict[str, Any]:
        """
        安全检查
        
        检测敏感内容，返回风险等级
        """
        # 危机关键词
        crisis_keywords = [
            r"想死", r"不想活", r"自杀", r"结束生命",
            r"自残", r"伤害自己", r"没有希望",
        ]
        
        # 敏感内容
        sensitive_keywords = [
            r"杀", r"死", r"痛", r"难受", r"绝望",
        ]
        
        text_lower = text.lower()
        
        # 检查危机信号
        for pattern in crisis_keywords:
            if re.search(pattern, text_lower):
                return {
                    "risk_level": "high",
                    "handoff": True,
                    "reason": "检测到危机信号",
                    "keywords": [pattern]
                }
        
        # 检查敏感内容
        sensitive_count = 0
        for pattern in sensitive_keywords:
            if re.search(pattern, text_lower):
                sensitive_count += 1
        
        if sensitive_count >= 3:
            return {
                "risk_level": "medium",
                "handoff": False,
                "reason": "检测到较多负面内容",
                "keywords": []
            }
        
        return {
            "risk_level": "low",
            "handoff": False,
            "reason": "",
            "keywords": []
        }
