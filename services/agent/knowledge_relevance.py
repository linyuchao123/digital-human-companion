"""Reject clear off-domain requests; uncertainty remains eligible for retrieval."""
import re
from .knowledge_intent import requests_knowledge

def clearly_unrelated(query):
    # Mixed requests about emotional impact still need psychoeducation support.
    if requests_knowledge(query) or re.search(r'焦虑|难过|悲伤|压力|情绪|心理|孤独|孤单|没人陪|睡不着|睡不好|失眠|害怕|担心|没兴趣|食欲|吃不下|不想活|自杀|伤害自己|崩溃|心跳|呼吸|迷茫|搞砸|骂自己|无法上班',query):
        return False
    return bool(re.search(
        r'天气|气温|下雨|天气预报|几点|几号|星期几|'
        r'(?:推荐|播放|找|搜).*(?:歌|音乐)|专辑|'
        r'(?:早餐|午餐|晚餐).*(?:煎|炒|煮|烤)|'
        r'(?:中午|晚上|早饭|午饭|晚饭).*(?:吃什么|吃啥)|推荐.*(?:餐厅|菜谱|食谱)|'
        r'打印机|路由器|蓝牙|wifi|安装软件|(?:电脑|手机).*(?:连接|配置|参数|内存|升级)|'
        r'股票价格|股票行情|汇率|机票|航班|酒店价格|'
        r'算一下|计算.*\d|\d+\s*[+*/=]',query,re.I))

class DomainFilteredKnowledgeRetriever:
    def __init__(self, retriever): self._retriever=retriever

    async def retrieve(self,query,top_k=3):
        if top_k<=0 or clearly_unrelated(query):
            return []
        return await self._retriever.retrieve(query,top_k)
