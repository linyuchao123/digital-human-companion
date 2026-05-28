import zipfile
from pathlib import Path

print("=" * 60)
print("交付物检查报告")
print("=" * 60)

# 检查Docker包
print("\n[1] 数智心伴_Docker镜像部署包.zip")
print("-" * 60)
docker_zip = Path("数智心伴_Docker镜像部署包.zip")
if docker_zip.exists():
    with zipfile.ZipFile(docker_zip, 'r') as zf:
        files = zf.namelist()
        print(f"文件数量: {len(files)}")
        print(f"文件大小: {docker_zip.stat().st_size / 1024 / 1024:.2f} MB")
        
        # 检查关键文件
        checks = {
            "docker-compose.yml": any("docker-compose" in f for f in files),
            "Dockerfile": any("Dockerfile" in f for f in files),
            "requirements.txt": any("requirements.txt" in f for f in files),
            "README.md": any("README" in f for f in files),
            "start.bat": any("start.bat" in f for f in files),
            "start.sh": any("start.sh" in f for f in files),
            "integrated_server.py": any("integrated_server.py" in f for f in files),
            "eval_server.py": any("eval_server.py" in f for f in files),
        }
        
        print("\n关键文件检查:")
        for name, exists in checks.items():
            status = "存在" if exists else "缺失"
            icon = "+" if exists else "x"
            print(f"  [{icon}] {name}: {status}")
        
        # 检查docker-compose.yml内容
        if checks["docker-compose.yml"]:
            compose_file = [f for f in files if "docker-compose" in f][0]
            content = zf.read(compose_file).decode('utf-8')
            if "8800" in content and "8766" in content:
                print("\n  [OK] docker-compose.yml 端口配置正确 (8800, 8766)")
            else:
                print("\n  [WARN] docker-compose.yml 端口配置可能有问题")
else:
    print("文件不存在!")

# 检查ASR包
print("\n[2] 数智心伴_语音识别模型工程文件.zip")
print("-" * 60)
asr_zip = Path("数智心伴_语音识别模型工程文件.zip")
if asr_zip.exists():
    with zipfile.ZipFile(asr_zip, 'r') as zf:
        files = zf.namelist()
        print(f"文件数量: {len(files)}")
        print(f"文件大小: {asr_zip.stat().st_size / 1024 / 1024:.2f} MB")
        
        checks = {
            "eval_server.py": any("eval_server.py" in f for f in files),
            "eval.html": any("eval.html" in f for f in files),
            "README.md": any("README" in f for f in files),
            "requirements.txt": any("requirements.txt" in f for f in files),
            "paraformer_zh.py": any("paraformer_zh.py" in f for f in files),
            "fsmn_vad.py": any("fsmn_vad.py" in f for f in files),
            "audio_pipeline.py": any("audio_pipeline.py" in f for f in files),
        }
        
        print("\n关键文件检查:")
        for name, exists in checks.items():
            status = "存在" if exists else "缺失"
            icon = "+" if exists else "x"
            print(f"  [{icon}] {name}: {status}")
else:
    print("文件不存在!")

print("\n" + "=" * 60)
print("检查完成")
print("=" * 60)
