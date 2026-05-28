#!/usr/bin/env python3
"""RAG检索诊断脚本"""
import sys
sys.path.insert(0, '.')

from services.llm.rag_engine import RAGEngine, RAGConfig, PsychologyKnowledgeBase

cfg = RAGConfig(chroma_db_path='data/kb/debug_chroma', top_k=3, similarity_threshold=0.0)
eng = RAGEngine(cfg)
PsychologyKnowledgeBase.initialize_kb(eng)

print(f"\n知识库文档数: {eng.get_stats()['count']}")

queries = ['失眠怎么办', '感到焦虑', '如何放松']
for q in queries:
    print(f"\n查询: '{q}'")
    results = eng._collection.query(query_texts=[q], n_results=3)
    for i, (doc, dist) in enumerate(zip(results['documents'][0], results['distances'][0])):
        sim = 1 - dist
        print(f"  [{i+1}] dist={dist:.4f}  sim={sim:.4f} | {doc[:50]}")
