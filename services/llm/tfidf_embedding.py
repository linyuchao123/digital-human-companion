#!/usr/bin/env python3
"""
轻量级本地TF-IDF Embedding
完全离线，用于RAG检索

适合在无网络环境下替代BGE/BERT embedding
注意：精度低于深度学习模型，适合开发测试阶段
后期可通过下载模型后替换为BGE-large-zh
"""

from __future__ import annotations

import hashlib
import re
from typing import Dict, List, Optional

import numpy as np


def _tokenize_zh(text: str) -> List[str]:
    """
    简单中文分词
    将中文按字符分割，英文按单词分割
    """
    # 基础清洗
    text = text.strip().lower()
    
    # 尝试使用jieba分词
    try:
        import jieba
        tokens = list(jieba.cut(text))
        return [t.strip() for t in tokens if t.strip() and len(t.strip()) > 0]
    except ImportError:
        pass
    
    # 降级：按字符分割中文，按空格分割英文
    tokens = []
    i = 0
    while i < len(text):
        char = text[i]
        # 中文字符
        if '\u4e00' <= char <= '\u9fff':
            tokens.append(char)
            i += 1
        # ASCII字符（英文单词）
        elif char.isalnum():
            word = ""
            while i < len(text) and text[i].isalnum():
                word += text[i]
                i += 1
            if word:
                tokens.append(word)
        else:
            i += 1
    
    return tokens


def _make_tfidf_embedding_class():
    """
    动态创建TFIDFEmbeddingFunction类。
    若ChromaDB可用则继承其EmbeddingFunction协议，否则使用普通类。
    """
    try:
        from chromadb import EmbeddingFunction, Documents, Embeddings
        BaseClass = EmbeddingFunction
        use_chroma_base = True
    except Exception:
        BaseClass = object
        use_chroma_base = False

    class TFIDFEmbeddingFunctionImpl(BaseClass):  # type: ignore[misc]
        """
        基于TF-IDF的本地Embedding函数。
        完全离线，无需下载任何模型。
        兼容ChromaDB 1.x EmbeddingFunction协议。
        """

        def __init__(self, dim: int = 512):
            """
            Args:
                dim: 向量维度
            """
            self.dim = dim
            self._vocab: Dict[str, int] = {}  # 词汇表
            self._idf: Dict[str, float] = {}  # IDF值
            self._doc_count = 0

        # ── ChromaDB EmbeddingFunction 协议必需方法 ──

        @staticmethod
        def name() -> str:
            """返回embedding函数名称（ChromaDB协议要求）"""
            return "tfidf_local"

        @staticmethod
        def build_from_config(config: dict) -> "TFIDFEmbeddingFunctionImpl":  # type: ignore[override]
            """从配置构建实例（ChromaDB协议要求）"""
            return TFIDFEmbeddingFunctionImpl(dim=config.get("dim", 512))

        def get_config(self) -> dict:
            """获取配置（ChromaDB协议要求）"""
            return {"dim": self.dim}

        def __call__(self, input: List[str]) -> List[List[float]]:
            """
            将文本列表转换为向量列表。
            实现ChromaDB EmbeddingFunction接口。
            """
            return self.encode(input)

        def embed_query(self, input: List[str]) -> List[List[float]]:
            """查询向量化（ChromaDB 1.5.x 要求）"""
            return self.encode(input)
    
        def encode(self, texts: List[str]) -> List[List[float]]:
            """
            对文本进行编码
            
            Args:
                texts: 文本列表
                
            Returns:
                向量列表
            """
            embeddings = []
            for text in texts:
                embedding = self._encode_single(text)
                embeddings.append(embedding)
            return embeddings
        
        def _encode_single(self, text: str) -> List[float]:
            """对单个文本进行编码"""
            if not text.strip():
                return [0.0] * self.dim
            
            # 分词
            tokens = _tokenize_zh(text)
            if not tokens:
                return [0.0] * self.dim
            
            # TF计算
            tf: Dict[str, float] = {}
            for token in tokens:
                tf[token] = tf.get(token, 0.0) + 1.0
            
            # 归一化TF
            max_tf = max(tf.values())
            tf = {k: v / max_tf for k, v in tf.items()}
            
            # 构建向量
            vector = np.zeros(self.dim, dtype=np.float32)
            
            for token, tf_value in tf.items():
                # 使用哈希将词映射到向量维度
                idx = self._token_to_idx(token)
                idf = self._idf.get(token, 1.0)  # 未知词默认IDF=1
                
                # TF-IDF值
                vector[idx] += tf_value * idf
            
            # L2归一化
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
            
            return vector.tolist()
        
        def _token_to_idx(self, token: str) -> int:
            """将词映射到向量索引（哈希方法）"""
            hash1 = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim
            return hash1
        
        def update_vocab(self, texts: List[str]):
            """
            更新词汇表和IDF
            
            在添加文档时调用，动态更新IDF值
            """
            self._doc_count += len(texts)
            
            # 统计每个词出现的文档数
            doc_freq: Dict[str, int] = {}
            for text in texts:
                tokens = set(_tokenize_zh(text))
                for token in tokens:
                    doc_freq[token] = doc_freq.get(token, 0) + 1
            
            # 更新IDF
            import math
            for token, freq in doc_freq.items():
                # 光滑IDF公式
                self._idf[token] = math.log((self._doc_count + 1) / (freq + 1)) + 1.0
        
        def get_info(self) -> dict:
            """获取模型信息"""
            return {
                "type": "tfidf",
                "dim": self.dim,
                "vocab_size": len(self._vocab),
                "doc_count": self._doc_count,
                "offline": True
            }

    return TFIDFEmbeddingFunctionImpl


# 导出类（兼容旧引用 from tfidf_embedding import TFIDFEmbeddingFunction）
TFIDFEmbeddingFunction = _make_tfidf_embedding_class()
