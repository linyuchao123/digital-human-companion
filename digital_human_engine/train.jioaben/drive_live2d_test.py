#!/usr/bin/env python3
"""
Live2D 驱动测试脚本

使用训练好的情感反应预测模型(v2)生成25维情感预测，
通过 emotion_to_live2d.py 映射为 24个雫人物 Live2D 真实参数，
并生成可视化图表 + 启动 WebSocket 实时驱动。

用法:
  # 1. 仅生成可视化图
  python drive_live2d_test.py --mode plot

  # 2. 启动 WebSocket 服务 (配合 live2d_viewer.html)
  python drive_live2d_test.py --mode websocket --port 8765

  # 3. 导出参数 JSON 文件 (可用于 Live2D Viewer 回放)
  python drive_live2d_test.py --mode export
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

# 添加路径
sys.path.insert(0, str(Path(__file__).parent))

from emotion_to_live2d import (
    LIVE2D_PARAMS,
    PARAM_NAMES,
    NUM_LIVE2D_PARAMS,
    emotion25_to_live2d,
    emotion25_sequence_to_live2d,
    live2d_to_dict,
    live2d_sequence_to_dicts,
)
from model import EmotionReactionTransformer


def load_model(checkpoint_path: str, device: torch.device):
    """加载训练好的模型"""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    stats = ckpt["stats"]

    model = EmotionReactionTransformer(
        audio_dim=768,
        emotion_dim=25,
        output_dim=25,
        hidden_dim=config["hidden_dim"],
        latent_dim=config["latent_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        dropout=0.0,  # 推理关闭dropout
        use_audio=config.get("use_audio", True),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"模型加载成功 (epoch {ckpt['epoch']}, val_loss {ckpt['val_loss']:.4f})")
    return model, stats, config


def load_emotion_csv(path: str, max_len: int = 750) -> np.ndarray:
    """加载情感CSV，处理空文件"""
    try:
        data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32)
        if data.ndim < 2 or data.shape[0] == 0:
            return np.zeros((max_len, 25), dtype=np.float32)
        return data[:max_len]
    except Exception:
        return np.zeros((max_len, 25), dtype=np.float32)


def pad_or_trim(arr: np.ndarray, target_len: int) -> np.ndarray:
    if arr.shape[0] >= target_len:
        return arr[:target_len]
    pad = np.zeros((target_len - arr.shape[0], arr.shape[1]), dtype=arr.dtype)
    return np.concatenate([arr, pad], axis=0)


def run_inference_single(
    model, speaker_emotion: np.ndarray, stats: dict, device: torch.device, K: int = 1
) -> np.ndarray:
    """
    对单个样本运行推理
    
    Args:
        speaker_emotion: [T, 25] 说话者情感
        stats: 归一化统计量
        K: 生成候选数
    
    Returns:
        [K, T, 25] 反归一化后的预测结果
    """
    T = speaker_emotion.shape[0]

    # 归一化
    sp_mean = stats["speaker_emotion"]["mean"]
    sp_std = stats["speaker_emotion"]["std"]
    sp_norm = ((speaker_emotion - sp_mean) / sp_std).astype(np.float32)

    # 构建输入 (无音频)
    audio = np.zeros((T, 768), dtype=np.float32)

    sp_t = torch.from_numpy(sp_norm).unsqueeze(0).to(device)
    au_t = torch.from_numpy(audio).unsqueeze(0).to(device)
    mask = torch.ones(1, T, dtype=torch.bool, device=device)

    with torch.no_grad():
        preds = model.generate(au_t, sp_t, mask=mask, num_candidates=K)  # [1, K, T, 25]

    preds_np = preds.cpu().numpy()[0]  # [K, T, 25]

    # 反归一化
    li_mean = stats["listener_emotion"]["mean"]
    li_std = stats["listener_emotion"]["std"]
    preds_np = preds_np * li_std + li_mean

    return preds_np


def find_sample_emotion(val_root: str) -> str | None:
    """找一个有效的验证集情感CSV"""
    val_path = Path(val_root)
    emotion_dir = val_path / "Emotion" / "NoXI"
    if not emotion_dir.exists():
        return None

    for session_dir in sorted(emotion_dir.iterdir()):
        if not session_dir.is_dir():
            continue
        for role_dir in sorted(session_dir.iterdir()):
            if not role_dir.is_dir():
                continue
            for csv_file in sorted(role_dir.glob("*.csv")):
                # 检查是否非空
                data = load_emotion_csv(str(csv_file))
                if data.shape[0] > 100:  # 至少100帧
                    return str(csv_file)
    return None


def plot_live2d_params(
    live2d_seq: np.ndarray,
    output_path: str = "live2d_params_plot.png",
    fps: int = 25,
    title: str = "Live2D 参数时序可视化",
):
    """
    生成 Live2D 参数时序可视化图
    
    Args:
        live2d_seq: [T, 24] Live2D参数序列
        output_path: 输出图片路径
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = live2d_seq.shape[0]
    time_axis = np.arange(T) / fps  # 秒

    # 按类别分组绘制
    groups = {
        "头部姿态": [0, 1, 2],        # ANGLE_X/Y/Z
        "眼睛": [3, 4, 7],             # EYE_L/R_OPEN, EYE_BALL_FORM
        "眉毛": [8, 9, 12, 13],        # BROW_L/R_Y, BROW_L/R_ANGLE
        "嘴巴": [16, 17, 18],          # MOUTH_OPEN_Y, FORM, SIZE
        "身体+特效": [19, 20, 21, 23], # TERE, BODY_X/Y, BREATH
    }

    fig, axes = plt.subplots(len(groups), 1, figsize=(16, 3 * len(groups)), sharex=True)
    fig.suptitle(title, fontsize=16, fontweight="bold")

    for ax, (group_name, indices) in zip(axes, groups.items()):
        for idx in indices:
            label = PARAM_NAMES[idx].replace("PARAM_", "")
            ax.plot(time_axis, live2d_seq[:, idx], label=label, linewidth=1.2, alpha=0.85)
        ax.set_ylabel(group_name, fontsize=11)
        ax.legend(loc="upper right", fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, time_axis[-1])

    axes[-1].set_xlabel("时间 (秒)", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"参数可视化图已保存: {output_path}")


def export_params_json(
    live2d_dicts: list,
    output_path: str = "live2d_params.json",
    fps: int = 25,
):
    """导出为JSON格式，可用于Live2D Viewer回放"""
    export_data = {
        "fps": fps,
        "total_frames": len(live2d_dicts),
        "duration_seconds": len(live2d_dicts) / fps,
        "param_names": PARAM_NAMES,
        "frames": live2d_dicts,
    }
    with open(output_path, "w") as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    print(f"参数JSON已导出: {output_path} ({len(live2d_dicts)} 帧)")


async def websocket_server(live2d_dicts: list, host: str, port: int, fps: int = 25):
    """WebSocket服务器，循环播放Live2D参数 + 支持动作触发"""
    import websockets

    frame_idx = 0
    total_frames = len(live2d_dicts)
    
    # 动作触发状态
    motion_queue = []
    last_motion_time = 0

    async def handler(websocket):
        nonlocal frame_idx, motion_queue, last_motion_time
        print(f"客户端已连接: {websocket.remote_address}")
        
        # 发送欢迎消息，告知可用动作
        await websocket.send(json.dumps({
            "type": "info",
            "motions": ["FlickUp", "Tap", "Flick3"],
            "motion_desc": {
                "FlickUp": "哭泣/感动 - 检测到悲伤话题",
                "Tap": "惊讶/害羞 - 意外或甜蜜的话",
                "Flick3": "否认/不认同 - 用户自我否定时摇头鼓励"
            }
        }))
        
        try:
            while True:
                # 检查是否有动作需要触发 (每5秒随机触发一个动作用于演示)
                current_time = time.time()
                if current_time - last_motion_time > 8:  # 每8秒触发一次演示
                    import random
                    motion_name = random.choice(["FlickUp", "Tap", "Flick3"])
                    motion_queue.append({
                        "name": motion_name,
                        "priority": 1,
                        "timestamp": current_time
                    })
                    last_motion_time = current_time
                    print(f"[动作触发] {motion_name}")
                
                # 构建消息
                params = live2d_dicts[frame_idx % total_frames]
                # 转换参数名格式 (PARAM_ANGLE_X -> ParamAngleX)
                ws_params = {}
                for k, v in params.items():
                    parts = k.replace("PARAM_", "").split("_")
                    camel = "Param" + "".join(p.capitalize() for p in parts)
                    ws_params[camel] = v
                
                message = {
                    "type": "params",
                    "data": ws_params,
                    "frame": frame_idx,
                }
                
                # 如果有待触发的动作，添加到消息
                if motion_queue:
                    motion = motion_queue.pop(0)
                    message["motion"] = motion["name"]
                    message["motion_priority"] = motion["priority"]
                
                await websocket.send(json.dumps(message))
                frame_idx += 1
                await asyncio.sleep(1.0 / fps)
        except Exception as e:
            print(f"客户端断开: {e}")

    print(f"\nWebSocket服务器启动: ws://{host}:{port}")
    print(f"共 {total_frames} 帧 ({total_frames / fps:.1f}秒), 循环播放")
    print("请在浏览器中打开 live2d_viewer.html 并点击'连接 WebSocket'")
    print("按 Ctrl+C 停止\n")

    async with websockets.serve(handler, host, port):
        await asyncio.Future()


def print_param_summary(live2d_seq: np.ndarray):
    """打印参数统计摘要"""
    print("\n=== Live2D 参数统计摘要 ===")
    print(f"{'参数名':<25} {'最小值':>8} {'最大值':>8} {'均值':>8} {'标准差':>8}")
    print("-" * 65)
    for i, name in enumerate(PARAM_NAMES):
        col = live2d_seq[:, i]
        if np.abs(col).max() > 0.001:  # 只显示有变化的参数
            print(
                f"{name:<25} {col.min():>8.3f} {col.max():>8.3f} "
                f"{col.mean():>8.3f} {col.std():>8.3f}"
            )


def main():
    parser = argparse.ArgumentParser(description="Live2D 驱动测试")
    parser.add_argument(
        "--checkpoint", type=str,
        default="/root/autodl-tmp/checkpoints_v2/best_model.pt",
        help="模型检查点路径",
    )
    parser.add_argument(
        "--val_root", type=str,
        default="/root/autodl-tmp/val",
        help="验证数据根目录",
    )
    parser.add_argument(
        "--emotion_csv", type=str, default=None,
        help="指定speaker emotion CSV文件 (不指定则自动找一个)",
    )
    parser.add_argument(
        "--mode", type=str, default="all",
        choices=["plot", "export", "websocket", "all"],
        help="运行模式",
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument(
        "--output_dir", type=str, default="/root/autodl-tmp/predict",
        help="输出目录",
    )
    parser.add_argument("--K", type=int, default=1, help="生成候选数")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载模型
    print("=" * 60)
    print("Live2D 驱动测试")
    print("=" * 60)
    model, stats, config = load_model(args.checkpoint, device)

    # 2. 找到一个speaker emotion样本
    if args.emotion_csv:
        emotion_path = args.emotion_csv
    else:
        print("\n自动搜索有效的验证集样本...")
        emotion_path = find_sample_emotion(args.val_root)
        if emotion_path is None:
            print("错误: 找不到有效的情感CSV文件")
            return

    print(f"\n使用样本: {emotion_path}")
    speaker_emotion = load_emotion_csv(emotion_path)
    speaker_emotion = pad_or_trim(speaker_emotion, 750)
    print(f"  Speaker emotion shape: {speaker_emotion.shape}")
    print(f"  数值范围: [{speaker_emotion.min():.3f}, {speaker_emotion.max():.3f}]")

    # 3. 模型推理 → 25维预测
    print("\n运行模型推理...")
    preds_25 = run_inference_single(model, speaker_emotion, stats, device, K=args.K)
    print(f"  预测结果 shape: {preds_25.shape}")  # [K, T, 25]
    pred_best = preds_25[0]  # 取第一个候选 [T, 25]
    print(f"  25维预测范围: [{pred_best.min():.3f}, {pred_best.max():.3f}]")

    # 4. 25维 → 24维 Live2D 参数 (情感陪护模式: 低强度+自然生理行为)
    print("\n映射为 Live2D 参数 (情感陪护模式)...")
    live2d_seq = emotion25_sequence_to_live2d(
        pred_best,
        smooth_window=15,     # 大窗口平滑, 消除抖动
        fps=args.fps,
        intensity=0.3,        # 低强度: 安静/倾听状态
        add_idle_behaviors=True,  # 自然眨眼/呼吸/眼球微动/头部微动
    )
    print(f"  Live2D参数 shape: {live2d_seq.shape}")  # [T, 24]

    # 打印参数摘要
    print_param_summary(live2d_seq)

    # 转为字典列表
    live2d_dicts = live2d_sequence_to_dicts(live2d_seq)

    # 5. 根据模式执行
    if args.mode in ("plot", "all"):
        plot_path = str(output_dir / "live2d_params_plot.png")
        plot_live2d_params(
            live2d_seq, output_path=plot_path, fps=args.fps,
            title=f"Live2D 参数时序 (模型v2, 样本: {Path(emotion_path).parent.parent.name})"
        )

    if args.mode in ("export", "all"):
        json_path = str(output_dir / "live2d_params.json")
        export_params_json(live2d_dicts, output_path=json_path, fps=args.fps)

    if args.mode in ("websocket", "all"):
        try:
            asyncio.run(websocket_server(live2d_dicts, "0.0.0.0", args.port, args.fps))
        except KeyboardInterrupt:
            print("\n服务已停止")

    if args.mode not in ("websocket",):
        print("\n驱动测试完成!")


if __name__ == "__main__":
    main()
