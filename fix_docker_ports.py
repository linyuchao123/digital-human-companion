#!/usr/bin/env python3
"""
修复Docker部署包端口配置
将8801改为8766以匹配实际eval_server.py端口
"""
import zipfile
import os
from pathlib import Path

WORK_DIR = Path(r"d:\AI数字人情感陪护项目\交付物_v2")
DOCKER_ZIP = WORK_DIR / "数智心伴_Docker镜像部署包.zip"
OUTPUT_ZIP = WORK_DIR / "数智心伴_Docker镜像部署包_修复版.zip"

print("=" * 60)
print("Docker部署包端口修复")
print("=" * 60)

if not DOCKER_ZIP.exists():
    print("源文件不存在!")
    exit(1)

# 读取原zip
with zipfile.ZipFile(DOCKER_ZIP, 'r') as src_zf:
    files = src_zf.namelist()
    print(f"\n原文件数: {len(files)}")
    
    # 创建新zip
    with zipfile.ZipFile(OUTPUT_ZIP, 'w', zipfile.ZIP_DEFLATED) as dst_zf:
        for file_info in src_zf.infolist():
            filename = file_info.filename
            content = src_zf.read(filename)
            
            # 修复docker-compose.yml端口
            if 'docker-compose.yml' in filename:
                print(f"\n修复: {filename}")
                text = content.decode('utf-8')
                # 替换端口
                text = text.replace('8801:8801', '8766:8766')
                text = text.replace('localhost:8801', 'localhost:8766')
                print("  8801 -> 8766")
                dst_zf.writestr(file_info, text.encode('utf-8'))
            else:
                dst_zf.writestr(file_info, content)

new_size = OUTPUT_ZIP.stat().st_size / 1024 / 1024
print(f"\n生成文件: {OUTPUT_ZIP.name}")
print(f"文件大小: {new_size:.2f} MB")
print("\n修复完成!")
