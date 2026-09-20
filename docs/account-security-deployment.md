# 账号、安全与部署验收

## 本阶段

- 前端渲染只消费自己 `/ws/main` 的驱动帧，不再消费跨会话 `/ws/drive` 广播。旧接口保留心跳兼容，但不推送其他会话参数。
- 新注册密码 8～64 位、UTF-8 不超过 bcrypt 的 72 字节限制。旧短密码账号仍可以登录并自行修改密码。bcrypt 缺失时拒绝注册，绝不回退明文保存。
- `/api/profile` GET/PATCH 仅操作当前账号。姓名、生日、头像选填；用户名修改需要当前密码。生日验证范围，头像限制 JPEG/PNG 和大小，前端裁切压缩为 192×192。
- `/api/profile/password` 验证当前密码，修改后撤销全部令牌；WebSocket 后续输入也检查令牌是否有效。密码永不回传或展示。
- SMTP 可用时支持邮箱验证码注册。正式公开部署会强制完成邮箱验证；验证码仅保存 HMAC 摘要，10 分钟失效、最多尝试 5 次，并限制单 IP 与单邮箱发送频率。开发环境在 SMTP 未配置时仍保留用户名快速注册。
- 公开部署开启 `PUBLIC_DEPLOYMENT=true`：TTS、Agent HTTP、视频上传和 WebSocket 的模型输入必须登录。开发默认 false，保留游客体验；仍有请求预算。
- HTTP 默认最大请求体 256 KiB，视频 20 MiB（含 multipart 开销），超限 413；不支持的视频后缀拒绝。视频任务公开部署按账号隔离。
- 单进程内存滑动窗口限流：公开认证 20 次/分钟/IP；开发认证 120 次；计费接口合计 30 次/分钟/IP/账号；WS 音频/文字各 60 次，视频帧 1800 次。超限返回错误；该方案不替代 Redis 共享限流或反向代理连接限制。
- WS 输入限制 3 MiB；单会话后台任务数限制为 4，断开时取消。浏览器 WS 来源必须与 `ALLOWED_ORIGINS` 匹配。

## Docker

统一使用 `pyproject.toml` 安装，默认包含 cloud/ml/speech/rag/vision，完整镜像依赖较大。使用非 root 用户、只读根文件系统、临时目录、持久数据卷，不自动公开评测服务。8800 仅绑定宿主机回环地址，需 HTTPS 反向代理对外提供服务。

在项目根目录执行（先安装并启动 Docker）：

```bash
docker compose --env-file .env -f infra/docker/docker-compose.yml config --quiet
docker compose --env-file .env -f infra/docker/docker-compose.yml build
docker compose --env-file .env -f infra/docker/docker-compose.yml up -d
curl --fail http://127.0.0.1:8800/api/health/ready
```

仅使用 `config --quiet`，避免展开配置时在终端泄露 API Key。`.env` 的 `ALLOWED_ORIGINS` 必须匹配实际访问来源，例如本机 8800 或正式 HTTPS 域名。Docker 强制公开模式；游客不能使用云模型服务。

`app-data` 命名卷持久化数据库及知识库。该卷首次创建为空，不会自动导入开发机 `data/users.db`；现有本地数据不被删除。迁移需在停止写入后备份、导入并检查文件所有权 UID 10001。不要使用 `down -v`，该命令会删除数据卷。模型权重/语义模型由只读挂载提供，ASR 和模型下载缓存独立可写卷。

就绪检查仅检查密码依赖和数据库，不代表云端模型可调用或所有模型已加载。上线还必须验收域名 HTTPS、备份恢复、多人并发、模型权重挂载、云模型调用及成本限额。

## 验收状态与限制

本机没有 Docker，镜像构建、依赖解析、容器权限、健康检查及 Linux 模型运行尚未实测，不能认定完成 Docker 部署验收。已实现的是配置整改和静态检查。

仍需后续加固：验证码/找回密码、可信反向代理 IP 配置、共享限流、多副本状态存储、账号数据彻底删除与缓存过期清理。现有令牌由 JS 可读 cookie 存储（SameSite/HTTPS Secure 已补），后续应迁移 HttpOnly 会话 cookie；目前接口凭证通过请求头和 WS init 传递，不能把这一版本视为完整生产安全审计。
