# LiveTalking / Wav2Lip 第二数字人接入

本项目把 LiveTalking 作为独立的实时视频数字人服务接入。网页仍使用原有的对话、
Qwen3/CosyVoice 音色和会话记录；后端完成 WebRTC 协商，并把已经合成的回复音频上传给
LiveTalking 驱动口型。连接页面和“设置与工具”中的“数字人形象”可以在“小安 · Live2D”
与“播报员 · Wav2Lip”之间切换。

## 能力边界

- Wav2Lip 输出的是由真人视频素材生成的实时说话视频，不是带骨骼、网格和自由视角的
  3D 模型。它适合作为项目里的第二个写实数字人呈现方式。
- LiveTalking 开源仓库不包含商业演示站点截图中的特定人物资产。接入自定义形象时，必须
  使用自己拍摄或已经获得肖像权、著作权和部署授权的视频素材。
- 当前 Apple Silicon 开发机可以开发和显示 WebRTC 页面，但不能按官方 CUDA 路线本地运行
  实时 Wav2Lip 推理。建议在带 NVIDIA GPU 的 Linux 主机上单独部署 LiveTalking。

## GPU 服务端准备

根据 LiveTalking 官方 README，参考环境为 Ubuntu 22.04、Python 3.12、PyTorch 2.9.1 和
CUDA 12.8。Wav2Lip 256 推荐 RTX 3060 或更高规格显卡。以下命令应在 GPU 服务器上执行：

```bash
git clone https://github.com/lipku/LiveTalking.git
cd LiveTalking
# 按服务器 CUDA 版本安装匹配的 PyTorch，再安装项目依赖。
pip install -r requirements.txt
```

下载官方说明中的 `wav2lip256.pth`，放到 `models/wav2lip.pth`；将官方示例形象包解压到
`data/avatars/wav2lip256_avatar1/`。启动示例：

```bash
python app.py --transport webrtc --model wav2lip --avatar_id wav2lip256_avatar1
```

默认服务端口是 8010。跨公网部署时还需按官方说明开放 WebRTC 所需端口，并使用可信 HTTPS
反向代理、访问控制和防火墙；不要将无认证的推理接口直接暴露到互联网。

## 本项目配置

在本项目的 `.env` 中填写 LiveTalking 服务根地址，不要追加 `/offer`：

```env
LIVETALKING_BASE_URL=http://127.0.0.1:8010
LIVETALKING_AVATAR_ID=wav2lip256_avatar1
```

重启本项目后端后，访问 `/api/avatar/catalog`，`wav2lip` 条目的 `available` 应为 `true`。
随后在连接页面选择“播报员 · Wav2Lip”并点击“连接数字人”。回复仍由本项目当前选择的
Qwen3/CosyVoice 音色生成，完整音频会通过 `/humanaudio` 交给 LiveTalking 驱动。

未设置 `LIVETALKING_BASE_URL` 时，页面会保留该选项并显示“需服务”；点击连接会返回明确的
配置提示，不会影响原有 Live2D 数字人。

## 自定义真人形象

使用 LiveTalking 自带的 `/avatar.html`，上传已获授权的视频并生成 Avatar。完成后把生成的
Avatar ID 写入 `LIVETALKING_AVATAR_ID`，重启两个服务并重新连接。模型权重、Avatar 视频和
生成缓存体积较大且可能包含敏感生物特征数据，不应提交到本仓库；应放在受控的 GPU 服务端
存储中，并配置备份与访问权限。

## 验收清单

1. `/api/avatar/catalog` 返回 `wav2lip.available=true`。
2. 选择 Wav2Lip 后可以建立 WebRTC 连接并看到待机视频。
3. 三个 Qwen3 音色都能驱动口型，切换音色后声音和口型同步。
4. 点击停止朗读或断开连接后，远端说话立即中止。
5. LiveTalking 不可用时显示连接错误，Live2D 模式仍能正常使用。

参考：

- <https://github.com/lipku/livetalking>
- <https://github.com/lipku/livetalking/blob/main/docs/api.md>
