# =============================================
# 数智心伴 - AI数字人情感陪护系统
# =============================================
# 项目名称：数智心伴 —— 多模态大模型驱动的AI数字人情感陪护系统
#
# 【快速启动】
#   方式一（推荐）：docker-compose up -d
#   方式二：
#     docker build -t shuzhixinban:latest .
#     docker run -d -p 8800:8800 \
#       -e DASHSCOPE_API_KEY=<your_key> \
#       -v $(pwd)/data/kb:/app/data/kb \
#       -v $(pwd)/digital_human_engine/checkpoints_v2:/app/digital_human_engine/checkpoints_v2:ro \
#       -v $(pwd)/models:/app/models:ro \
#       --name shuzhixinban shuzhixinban:latest
#
# 【服务端口】
#   8800 - 主服务（Web界面 + WebSocket对话 + 数字人驱动）
#   8801 - ASR语音识别评测服务
#
# 【访问地址】
#   主界面：  http://localhost:8800/
#   数字人：  http://localhost:8800/ （打开后点击"新开连接"）
#   评测页：  http://localhost:8801/eval
# =============================================

FROM python:3.10-slim

LABEL maintainer="数智心伴团队"
LABEL description="多模态大模型驱动的AI数字人情感陪护系统"
LABEL version="1.0.0"

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ \
    PIP_TRUSTED_HOST=mirrors.aliyun.com \
    # FunASR / ModelScope 模型缓存目录
    MODELSCOPE_CACHE=/app/.cache/modelscope \
    # 默认关闭 GPU（纯 CPU 推理）
    CUDA_VISIBLE_DEVICES=""

# 安装系统依赖（MediaPipe、OpenCV、ffmpeg 需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 先复制依赖文件，利用 Docker 层缓存加速后续构建
COPY requirements.txt .

# 安装 Python 依赖
# torch CPU 版本（减小镜像体积；如需 GPU 请改为对应 cuda 版本）
RUN pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 \
    --index-url https://download.pytorch.org/whl/cpu
RUN pip install -r requirements.txt
RUN pip install python-multipart

# 复制项目源码（不含大文件，由 .dockerignore 控制）
COPY . .

# 创建必要的数据目录
RUN mkdir -p data/kb data/processed data/raw data/eval .cache/modelscope

# 暴露服务端口
EXPOSE 8800
EXPOSE 8801

# 健康检查（等待60s让FunASR/MediaPipe模型完成首次加载）
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=5 \
    CMD curl -f http://localhost:8800/api/status || exit 1

# 默认启动主服务（同时在后台拉起评测服务）
CMD ["sh", "-c", \
    "python -m uvicorn apps.api.eval_server:app --host 0.0.0.0 --port 8801 --workers 1 & \
     python -m uvicorn apps.api.integrated_server:app --host 0.0.0.0 --port 8800 --workers 1"]
