# 摄像头感知验收说明

摄像头默认关闭。主界面连接后点击“摄像头感知”，确认本机处理说明并授予摄像头权限。
需要 localhost 或 HTTPS；不支持直接双击 HTML 文件运行。预览可隐藏，关闭按钮才停止采集。
后台页面、断线、退出登录及会话切换都会停止采集，重新连接需手动开启。

## 数据与解释边界

画面和面部关键点只在浏览器及 Worker 中存在，不上传图片、不录制。
后端仅接收固定数值特征，保存于当前连接内存，五秒失效，不写数据库或日志。
简短观察描述会作为本轮模型上下文；模型回复及用户对话仍按原有历史规则保存。
表情不是心理诊断，用户描述优先。低质量、多人、无人画面不会用于情绪推断。
画面质量采用人脸大小、边界和亮度启发式检查，不等于专业检测准确率。

## 固定依赖

本地托管 @mediapipe/tasks-vision 0.10.32（Apache-2.0），包括 SIMD 与非 SIMD WASM。
来源 https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.32/ ，包元数据随资源保存。
复用 models/face_landmarker.task。运行时无 CDN 请求。
许可证 https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE 。

完整性 SHA-256：

- vision_bundle.mjs: de83c48ff329717a27aeb528d5ef5f47f077c628a5302dc483aca5b513e7464b
- vision_wasm_internal.wasm: cb3ec20026a9aecc2a81a93c25630ceb5389297ddb7a5f0bd61dd09cde606b9b
- vision_wasm_nosimd_internal.wasm: 924274fcd5ac8985f6570a8573e7971b7bd2d580ba1b8f3beb0ba8f95db6347c
- face_landmarker.task: 64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff

## 人工验收（尚未完成）

- 实际摄像头：微笑、皱眉、转头、闭眼、离开、多人与暗光。
- 对话中结合观察，但不每轮报告、不自动打断、不判断疾病或困倦。
- 浏览器网络面板确认只有 vision_control、vision_features 数值消息，无图片上传。
- 同机对比开启前后帧率和首音延迟：帧率下降目标不超过 10%，首音中位数增加目标不超过 200ms。
- 持续语音播放无明显断续，嘴型与原有动作不受摄像头驱动。

自动测试不能替代实际摄像头、模型兼容性和性能验收。
