# ASR 真实录音评测

`scripts/manual_asr_smoke.py` 不再把“模块能导入”当作准确率通过。它会逐条调用主聊天所用的 `_recognize_audio` 路径，因此遵循 `.env` 中的 `ASR_PROVIDER`、云端模型、热词及本地备用配置，并在报告中记录实际 provider 与是否发生 fallback。

1. 复制 `eval/asr/manifest.example.json` 为自己的清单。录音放在清单同目录下的 `private-recordings/`，使用真实麦克风 WAV，同一句话优化前后复用同一文件。
2. 人工逐字填写 `reference`，至少覆盖安静、噪声、短句、长句、专有词和否定表达。不要用 TTS 合成音代替真实用户录音。
3. 执行：`.venv-model/bin/python scripts/manual_asr_smoke.py eval/asr/你的清单.json --output eval/asr/reports/baseline.json`

默认按中文字符及英文单词计算微平均 CER，并按 bucket 展示 CER 与句错误率；识别调用失败按整句删除计入错误率，只记录经清洗的错误类型。CER 超过项目暂定的 10% 门槛时退出码为 2。`--no-gate` 可只采集基线。报告默认不含参考和识别原文，显式使用 `--include-text` 才会写出文本。录音和报告目录均被 Git 忽略，API Key 和厂商响应体不进入报告。

样本必须是清单目录内的非符号链接 WAV。评测会把音频发送给当前配置的云端 ASR（若启用），本地文件本身不会由脚本复制或保存；第三方服务的数据处理规则仍以厂商条款为准。没有真实标注录音时不能宣称达到 CER 10%。
