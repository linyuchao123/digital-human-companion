# 心理教育语料扩充

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
