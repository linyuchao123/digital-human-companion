# 日常问法检索验收

评测集 `eval/rag/natural_language_cases.json` 包含12条心理教育问题和4条无关问题。期望知识ID人工指定，部分问题允许多个合理来源；负例用 `expect_no_results: true`，必须没有期望ID。

命中率与MRR仅以正例为分母；负例单独计算误命中率，任何返回的知识片段都算误命中。默认 `--max-false-positive-rate 0`，出现误命中时命令退出码为1，即使正例全命中也不通过。可查看 `failures` 中的返回ID定位问题，不应为了通过而随意放宽阈值。

离线命令：

```sh
.venv-model/bin/python scripts/evaluate_agent_rag.py --cases eval/rag/natural_language_cases.json
.venv-model/bin/python scripts/evaluate_agent_rag.py --cases eval/rag/natural_language_cases.json --embedding-model models/embedding/bge-small-zh-v1.5 --require-provider hybrid_with_fallback
```

2026-09-13基线，30条知识，Top3：

|模式|正例命中率|MRR|负例误命中率|验收|
|---|---|---|---|---|
|BM25|12/12|1.0|3/4|未通过|
|混合检索（本地BGE）|12/12|1.0|4/4|未通过|

这测的是直接调用检索器，不是主界面所有聊天都会触发知识检索；主界面仍有意图路由。混合检索可以为没有词面匹配的无关问题返回候选，不能把相对排名当可信度。下一步需扩大负例和保留独立测试集，校准相关性过滤/重排，避免用极少样本拍脑袋定阈值。

小规模人工样例只用于回归与发现问题，不是心理学效果、真实用户准确率或临床验证。本次不改线上检索阈值，不新增API调用，也不影响语音链路。
