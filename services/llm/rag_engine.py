#!/usr/bin/env python3
"""
RAG知识库引擎
基于ChromaDB的心理学知识检索
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class RAGConfig:
    """RAG配置"""
    chroma_db_path: str = "data/kb/psychology_chroma"
    embedding_model: str = "BAAI/bge-large-zh"
    top_k: int = 5
    similarity_threshold: float = 0.05  # TF-IDF相似度普遍较低，使用宽松阈值
    max_context_tokens: int = 1500


class RAGEngine:
    """
    RAG检索引擎
    
    功能:
    - 文档向量化存储
    - 相似度检索
    - 上下文组装
    """
    
    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        self._client = None
        self._collection = None
        self._embedding_model = None
        self._init_chroma()
    
    def _init_chroma(self):
        """初始化ChromaDB"""
        try:
            import chromadb
            import os
            
            # 屏蔽网络请求
            os.environ['HF_HUB_OFFLINE'] = '1'
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            
            Path(self.config.chroma_db_path).parent.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self.config.chroma_db_path)
            
            # 使用轻量级本地TF-IDF Embedding（完全离线）
            from services.llm.tfidf_embedding import TFIDFEmbeddingFunction
            self._embedding_model = TFIDFEmbeddingFunction()
            
            self._collection = self._client.get_or_create_collection(
                name="psychology_kb",
                embedding_function=self._embedding_model,
                metadata={"hnsw:space": "cosine"}
            )
            
            print(f"[RAG] ChromaDB初始化成功（TF-IDF Embedding），文档数: {self._collection.count()}")
            
        except Exception as e:
            print(f"[RAG] ChromaDB初始化失败: {e}")
            self._client = None
            self._collection = None
    
    def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ):
        """
        添加文档到知识库
        
        Args:
            documents: 文档内容列表
            metadatas: 元数据列表
            ids: 文档ID列表
        """
        if self._collection is None:
            print("[RAG] 知识库未初始化，无法添加文档")
            return
        
        if ids is None:
            # 自动生成ID
            ids = [hashlib.md5(doc.encode()).hexdigest() for doc in documents]
        
        if metadatas is None:
            metadatas = [{"source": "unknown"} for _ in documents]
        
        try:
            self._collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            print(f"[RAG] 成功添加 {len(documents)} 篇文档")
        except Exception as e:
            print(f"[RAG] 添加文档失败: {e}")
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        检索相关文档
        
        Args:
            query: 查询文本
            top_k: 返回文档数
            filter_dict: 过滤条件
            
        Returns:
            检索结果列表
        """
        if self._collection is None:
            print("[RAG] 知识库未初始化，返回空结果")
            return []
        
        top_k = top_k or self.config.top_k
        
        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=top_k,
                where=filter_dict
            )
            
            # 格式化结果
            formatted_results = []
            for i in range(len(results['ids'][0])):
                distance = results['distances'][0][i]
                similarity = 1 - distance  # cosine距离转相似度
                
                if similarity >= self.config.similarity_threshold:
                    formatted_results.append({
                        "id": results['ids'][0][i],
                        "document": results['documents'][0][i],
                        "metadata": results['metadatas'][0][i],
                        "similarity": similarity
                    })
            
            return formatted_results
            
        except Exception as e:
            print(f"[RAG] 检索失败: {e}")
            return []
    
    def build_context(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> str:
        """
        构建RAG上下文
        
        Args:
            query: 查询文本
            top_k: 检索文档数
            
        Returns:
            组装后的上下文文本
        """
        results = self.retrieve(query, top_k)
        
        if not results:
            return ""
        
        context_parts = []
        context_parts.append("### 相关知识")
        
        for i, result in enumerate(results, 1):
            doc = result["document"]
            source = result["metadata"].get("source", "未知")
            context_parts.append(f"\n[{i}] 来源: {source}")
            context_parts.append(f"{doc}")
        
        return "\n".join(context_parts)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取知识库统计信息"""
        if self._collection is None:
            return {"status": "uninitialized", "count": 0}
        
        return {
            "status": "ready",
            "count": self._collection.count(),
            "db_path": self.config.chroma_db_path,
            "embedding_model": self.config.embedding_model
        }


class PsychologyKnowledgeBase:
    """
    心理学知识库 - 扩充版（约30条）
    涵盖 CBT、共情、危机干预、睡眠、焦虑、人际关系、自我接纳等领域
    """

    # ── CBT 认知行为疗法 ──────────────────────────────────────
    CBT_TECHNIQUES = [
        {
            "content": "认知重构：识别自动化负面思维，质疑其有效性，用更平衡的想法替代。步骤：1)识别负面想法 2)评估证据 3)寻找替代解释 4)验证新想法。适用于抑郁、焦虑等情绪问题。",
            "metadata": {"category": "CBT", "technique": "cognitive_restructuring", "emotion": "anxiety,depression"}
        },
        {
            "content": "行为激活：通过安排愉快活动改善情绪。抑郁时常伴随活动减少，增加活动可打破恶性循环。建议每天安排至少一项愉快活动，从小事开始，如散步、听音乐。",
            "metadata": {"category": "CBT", "technique": "behavioral_activation", "emotion": "depression"}
        },
        {
            "content": "正念呼吸：专注于当下呼吸，不评判地观察思绪。每天练习10-20分钟，可减轻焦虑，提高情绪调节能力。呼吸方法：吸气4秒，屏气4秒，呼气6秒。",
            "metadata": {"category": "mindfulness", "technique": "breathing", "emotion": "anxiety,stress"}
        },
        {
            "content": "暴露疗法：逐步面对令人恐惧的情境，从低焦虑场景开始，逐渐增加难度。每次成功应对都能降低焦虑水平，建立自信心。适用于恐惧症、社交焦虑。",
            "metadata": {"category": "CBT", "technique": "exposure_therapy", "emotion": "fear,anxiety"}
        },
        {
            "content": "思维日记：每天记录负面思维事件、引发情绪及认知扭曲类型（如灾难化、全有全无思维、读心术等），有助于识别情绪模式，逐步改变认知习惯。",
            "metadata": {"category": "CBT", "technique": "thought_diary", "emotion": "anxiety,depression"}
        },
        {
            "content": "渐进式肌肉放松：从脚部开始，依次紧绷再放松各肌肉群，持续15-20分钟。能有效减少躯体紧张感，降低生理焦虑反应，改善睡眠质量。",
            "metadata": {"category": "CBT", "technique": "progressive_relaxation", "emotion": "anxiety,stress,sleep"}
        },
    ]

    # ── 共情与情感支持 ────────────────────────────────────────
    EMPATHY_PHRASES = [
        {
            "content": "共情验证：用语言承认对方的感受是真实和合理的。如'我能感受到你现在的痛苦，这种感受完全可以理解。'避免说'你不应该这么想'或'想开点'等否定感受的话。",
            "metadata": {"category": "empathy", "type": "validation"}
        },
        {
            "content": "积极倾听技巧：保持眼神接触，用'嗯''我理解'等回应，在对话中适时复述对方的话，如'你的意思是...'。不打断、不评判，让对方感到被接纳。",
            "metadata": {"category": "empathy", "type": "active_listening"}
        },
        {
            "content": "情感命名：帮助用户识别和命名自己的情绪，如'听起来你感到很孤独和失落'。命名情绪能降低情绪强度，帮助大脑前额叶更好地调节情绪。",
            "metadata": {"category": "empathy", "type": "emotion_labeling"}
        },
        {
            "content": "支持性陪伴：有时候用户需要的不是建议，而是被陪伴的感觉。可以说'我在这里陪着你''你不是一个人'。高质量的陪伴本身就是一种治愈力量。",
            "metadata": {"category": "empathy", "type": "companionship"}
        },
    ]

    # ── 危机干预 ──────────────────────────────────────────────
    CRISIS_INTERVENTION = [
        {
            "content": "危机信号识别：当用户提到轻生、自伤、不想活时，需立即响应。中国心理危机援助热线：400-161-9995（24小时）；北京：010-82951332；上海：021-62258000。不要独自承受，请立即寻求帮助。",
            "metadata": {"category": "crisis", "type": "hotline", "priority": "high"}
        },
        {
            "content": "安全计划制定：引导用户列出在危机时刻可以采取的步骤：1)识别危机信号 2)联系信任的人 3)拨打热线 4)去安全地点。让用户感到有掌控感，减少无助感。",
            "metadata": {"category": "crisis", "type": "safety_plan", "priority": "high"}
        },
        {
            "content": "去除致死手段：危机干预的重要步骤是减少自伤工具的可及性，如药物放到他处保管、限制尖锐物品等。这能为专业帮助争取时间。",
            "metadata": {"category": "crisis", "type": "means_restriction", "priority": "high"}
        },
    ]

    # ── 睡眠问题 ──────────────────────────────────────────────
    SLEEP_HYGIENE = [
        {
            "content": "睡眠卫生建议：1)保持固定作息时间 2)睡前1小时避免电子设备 3)卧室保持凉爽(18-22°C)、安静、黑暗 4)避免咖啡因、酒精 5)如20分钟无法入睡，起床做放松活动。",
            "metadata": {"category": "sleep", "type": "hygiene"}
        },
        {
            "content": "失眠认知行为治疗(CBT-I)：限制卧床时间以增强睡眠驱动力，建立床与睡眠的联系（不在床上做其他事），睡眠限制和刺激控制是核心技术，长期效果优于安眠药。",
            "metadata": {"category": "sleep", "type": "CBT-I"}
        },
        {
            "content": "睡前放松技巧：冥想（专注呼吸10分钟）、写感恩日记（记录今天3件好事）、渐进式肌肉放松、睡前读纸质书等，都有助于平静心绪，降低皮质醇水平促进入睡。",
            "metadata": {"category": "sleep", "type": "relaxation"}
        },
    ]

    # ── 焦虑管理 ──────────────────────────────────────────────
    ANXIETY_MANAGEMENT = [
        {
            "content": "5-4-3-2-1 接地技术：焦虑发作时，注意5件看到的事、4件触摸到的东西、3种听到的声音、2种闻到的气味、1种尝到的味道。能快速将注意力拉回当下，打断焦虑循环。",
            "metadata": {"category": "anxiety", "technique": "grounding"}
        },
        {
            "content": "焦虑的本质：焦虑是大脑对感知到的威胁做出的正常反应（战或逃反应）。轻度焦虑有保护作用，但慢性焦虑会消耗心理资源。接受焦虑存在，而非抵抗，是管理焦虑的第一步。",
            "metadata": {"category": "anxiety", "technique": "psychoeducation"}
        },
        {
            "content": "担忧时间技术：把每天固定15分钟作为'担忧时间'，其他时间出现担忧思维时，告诉自己'留到担忧时间再想'。这样能减少全天被担忧占据，也不是完全压抑情绪。",
            "metadata": {"category": "anxiety", "technique": "worry_time"}
        },
    ]

    # ── 自我接纳与自尊 ────────────────────────────────────────
    SELF_ACCEPTANCE = [
        {
            "content": "自我慈悲(Self-Compassion)：当你对自己苛刻时，试着用对待好朋友的方式对待自己。包含三要素：1)善意对待自己 2)认识到不完美是人类共同经历 3)正念地感受痛苦而不被淹没。",
            "metadata": {"category": "self_acceptance", "technique": "self_compassion"}
        },
        {
            "content": "成长型思维：将失败视为学习机会，而非能力的证明。将'我失败了'改为'这次没做好，下次我可以尝试不同方法'。能力和智慧通过努力是可以发展的。",
            "metadata": {"category": "self_acceptance", "technique": "growth_mindset"}
        },
        {
            "content": "身份认同与价值观：明确自己的核心价值观（如诚实、关爱、创造力）能在困难时期提供稳定的自我认同感。当生活中某个角色失去时，核心价值观依然是你的一部分。",
            "metadata": {"category": "self_acceptance", "technique": "values_clarification"}
        },
    ]

    # ── 人际关系 ──────────────────────────────────────────────
    INTERPERSONAL = [
        {
            "content": "健康边界设定：边界是保护自己的心理和情感健康的规则。可以温和但坚定地说'我现在没有精力做这件事'或'这件事让我感到不舒服'。设定边界不是自私，是自我关爱。",
            "metadata": {"category": "interpersonal", "technique": "boundaries"}
        },
        {
            "content": "孤独感应对：孤独是一个信号，提示需要更多连接。可以主动联系一位朋友，加入兴趣社群，做志愿服务，或养宠物。即使是短暂的积极社交也能显著改善情绪。",
            "metadata": {"category": "interpersonal", "emotion": "loneliness"}
        },
    ]

    # ── 压力与情绪调节 ────────────────────────────────────────
    STRESS_EMOTION = [
        {
            "content": "情绪调节策略：1)认知重评（换角度看问题）2)表达抑制（注意不要长期压抑）3)接受情绪（不评判）4)问题解决 5)寻求社会支持。不同策略适用于不同情境，灵活使用。",
            "metadata": {"category": "stress", "technique": "emotion_regulation"}
        },
        {
            "content": "身体运动对心理健康的作用：每周150分钟中等强度有氧运动（如快走、游泳）可显著减轻抑郁和焦虑症状，效果与轻中度抗抑郁药相当，同时改善睡眠和自尊。",
            "metadata": {"category": "stress", "technique": "exercise"}
        },
        {
            "content": "社会支持网络：拥有可以倾诉的朋友或家人是心理韧性的重要保护因素。当感到困难时，不要独自承受，向信任的人寻求帮助是勇气而非软弱的表现。",
            "metadata": {"category": "stress", "technique": "social_support"}
        },
    ]

    @classmethod
    def initialize_kb(cls, rag_engine: 'RAGEngine'):
        """初始化知识库——仅在知识库为空时添加，避免重复"""
        stats = rag_engine.get_stats()
        if stats.get('count', 0) > 0:
            print(f"[PsychologyKB] 知识库已有 {stats['count']} 条记录，跳过初始化")
            return

        print("[PsychologyKB] 正在初始化心理学知识库（约30条）...")
        all_documents = []
        all_metadatas = []

        for group in [cls.CBT_TECHNIQUES, cls.EMPATHY_PHRASES, cls.CRISIS_INTERVENTION,
                      cls.SLEEP_HYGIENE, cls.ANXIETY_MANAGEMENT, cls.SELF_ACCEPTANCE,
                      cls.INTERPERSONAL, cls.STRESS_EMOTION]:
            for item in group:
                all_documents.append(item["content"])
                all_metadatas.append(item["metadata"])

        rag_engine.add_documents(all_documents, all_metadatas)
        print(f"[PsychologyKB] 知识库初始化完成，共 {len(all_documents)} 条知识")


# 全局RAG引擎实例
_rag_engine: Optional[RAGEngine] = None


def get_rag_engine() -> RAGEngine:
    """获取全局RAG引擎实例"""
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngine()
    return _rag_engine
