# 心理教育语料扩充

## 日常表达检索触发

睡不着、睡不好、不敢拒绝、丧亲、照护压力、任务过多，以及明确心理科普问题可直接进入知识检索，不额外请求模型分类。安全拦截、时间/天气/搜索及活动明确指令仍优先；“不要科普”“不需要建议”“只想倾诉”等表达跳过知识检索与分类器，走陪伴聊天。这个开关不关闭安全保护或已授权记忆检索，也不能保证模型完全不输出建议。

触发逻辑是保守的本地表达匹配，不是症状诊断或情绪分类结果。词汇覆盖仍有限；影视歌曲推荐等容易混淆场景不靠心理关键词强制检索。专项测试 `tests/test_knowledge_intent.py`。

## 回复下方参考依据

本轮知识检索结果随 `llm_reply.knowledge_sources` 发送，最多3条、每条摘录400字。主界面默认折叠显示来源与摘录，仅开放通过校验的HTTPS来源链接；文本用DOM `textContent` 渲染，不解释HTML。检索分数不是可信度百分比，因此不显示。

这些是检索参考，不是回复逐句引用或事实核验结果。语音仅朗读原回复正文，不读参考卡片。没有检索时不显示卡片；卡片目前只展示新回复，不恢复历史消息的依据。后续若实现逐句引用与持久化，需要另行验证引用对应关系和资料变更处理。

前端检查：`node tests/knowledge_references.cjs`；接口及序列化测试：`tests/test_agent_websocket_flow.py`、`tests/test_knowledge_references.py`。

保留原10条，新增20条来源可追溯的中文整理，主题包括CBT、灾难化与非黑即白思维、思维记录、担忧管理、睡眠卫生、哀伤、正念、心理韧性、人际边界、冲突沟通、照护压力、自我照顾、任务优先级、专业求助与问题解决。

来源核验日期：2026-09-13。条目是简短科普整理，不是官方中文译本、诊断量表或治疗处方，未提供用药建议。自助技巧需考虑个人情况；症状严重或安全风险时专业求助优先，不要求等待两周。部署商业服务前需重新审查原始资料许可和内容；当前不打包第三方手册全文。

新增来源：

- [NHS：认知重评](https://www.nhs.uk/every-mind-matters/mental-wellbeing-tips/self-help-cbt-techniques/reframing-unhelpful-thoughts/)
- [NHS：担忧管理](https://www.nhs.uk/every-mind-matters/mental-wellbeing-tips/self-help-cbt-techniques/tackling-your-worries/)
- [NHS：睡眠](https://www.nhs.uk/every-mind-matters/mental-wellbeing-tips/how-to-fall-asleep-faster-and-sleep-better/)
- [NHS：问题解决](https://www.nhs.uk/every-mind-matters/mental-wellbeing-tips/self-help-cbt-techniques/problem-solving/)
- [NIH：情绪健康](https://www.nih.gov/health-information/your-healthiest-self-wellness-toolkits/emotional-wellness-toolkit)
- [NIH：社会关系](https://www.nih.gov/health-information/your-healthiest-self-wellness-toolkits/social-wellness-toolkit)
- [NIMH：自我照顾](https://www.nimh.nih.gov/health/topics/caring-for-your-mental-health)
- [NIMH：何时求助](https://www.nimh.nih.gov/health/publications/my-mental-health-do-i-need-help)

测试 `tests/test_psychology_corpus.py` 检查ID、长度、来源，并覆盖8类检索的Top3命中。这是关键词提示较明确的离线冒烟测试，不代表真实用户问法的检索准确率。新增内容无需管理员令牌，随版本化JSON发布；重启服务会重建检索缓存，原管理员写入限制保持不变。
