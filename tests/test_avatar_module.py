#!/usr/bin/env python3
"""
数字人面部行为驱动模块 - 完整测试套件
测试项目:
  1. 数据集扫描 (scan_dataset)
  2. 模型前向推理形状验证
  3. 多任务损失计算
  4. VRM/Live2D映射器
  5. 评测预测文件生成 [N, K=10, T=750, 25]
  6. DriveEngine端到端流程
  7. 验证集CSV结构验证
"""
from __future__ import annotations

import os
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

# 路径设置
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ─────────────────────────────────────────────────
# 辅助：测试报告
# ─────────────────────────────────────────────────
_results: list[tuple[str, bool, str]] = []

def _ok(name: str, msg: str = ""):
    _results.append((name, True, msg))
    print(f"  [PASS] {name}" + (f" — {msg}" if msg else ""))

def _fail(name: str, msg: str = ""):
    _results.append((name, False, msg))
    print(f"  [FAIL] {name} — {msg}")

def _section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ─────────────────────────────────────────────────
# 训练集/验证集路径
# ─────────────────────────────────────────────────
TRAIN_ROOT = r"D:\服创比赛\数字人面部行为驱动模型训练集"
VAL_ROOT   = r"D:\AI数字人情感陪护项目\数字人面部行为驱动模型验证脚本"
VAL_CSV    = os.path.join(VAL_ROOT, "person_specific_val.csv")
VAL_NBR    = os.path.join(VAL_ROOT, "person_specific_masked_neighbour_emotion_val.npy")
EVAL_SCRIPT = os.path.join(VAL_ROOT, "eval_emotion_metrics.py")


# ═══════════════════════════════════════════════
# 测试1：数据集扫描
# ═══════════════════════════════════════════════
def test_scan_dataset():
    _section("TEST 1: scan_dataset — 训练集三元组扫描")
    from services.avatar.face_drive.training.dataset import scan_dataset

    train_exists = Path(TRAIN_ROOT).exists()
    if not train_exists:
        _fail("训练集目录存在", f"路径不存在: {TRAIN_ROOT}")
        return

    _ok("训练集目录存在", TRAIN_ROOT)

    samples = scan_dataset(TRAIN_ROOT)
    if len(samples) == 0:
        _fail("三元组数量>0", "未找到任何 (audio, face, emotion) 三元组")
    else:
        _ok("三元组数量>0", f"共 {len(samples)} 个有效三元组")

    if samples:
        s = samples[0]
        required_keys = {"audio", "face", "emotion", "session", "role", "idx", "dataset"}
        missing = required_keys - set(s.keys())
        if missing:
            _fail("三元组字段完整", f"缺少字段: {missing}")
        else:
            _ok("三元组字段完整", f"示例: {s['dataset']}/{s['session']}/{s['role']}/{s['idx']}")

        # 验证文件实际存在
        for key in ("audio", "face", "emotion"):
            if Path(s[key]).exists():
                _ok(f"样本文件存在({key})", s[key][-60:])
            else:
                _fail(f"样本文件存在({key})", s[key])


# ═══════════════════════════════════════════════
# 测试2：音频/面部/情绪数据加载
# ═══════════════════════════════════════════════
def test_data_loading():
    _section("TEST 2: 数据加载函数验证")
    from services.avatar.face_drive.training.dataset import (
        load_audio_mel, load_face_params, load_emotion_csv,
        align_sequences, FEATURE_ORDER
    )

    # FEATURE_ORDER 完整性
    if len(FEATURE_ORDER) == 25:
        _ok("FEATURE_ORDER 25维", str(FEATURE_ORDER[:5]) + "...")
    else:
        _fail("FEATURE_ORDER 25维", f"实际维度: {len(FEATURE_ORDER)}")

    # 用随机数据测试（不需要真实文件）
    with tempfile.TemporaryDirectory() as tmpdir:
        # 生成假wav（静音）
        wav_path = os.path.join(tmpdir, "test.wav")
        try:
            import wave, struct
            with wave.open(wav_path, 'w') as f:
                f.setnchannels(1); f.setsampwidth(2); f.setframerate(16000)
                f.writeframes(struct.pack('<' + '0h' * 16000))  # 1秒静音
            mel = load_audio_mel(wav_path)
            if mel.shape[1] == 80:
                _ok("load_audio_mel 输出形状", f"{mel.shape}")
            else:
                _fail("load_audio_mel 输出形状", f"期望 (T, 80), 实际 {mel.shape}")
        except Exception as e:
            _fail("load_audio_mel", str(e))

        # 生成假 face npy
        face_path = os.path.join(tmpdir, "test.npy")
        fake_face = np.random.rand(100, 58).astype(np.float32)
        np.save(face_path, fake_face)
        try:
            face = load_face_params(face_path)
            if face.shape == (100, 58):
                _ok("load_face_params 输出形状", f"{face.shape}")
            else:
                _fail("load_face_params 输出形状", f"期望 (100, 58), 实际 {face.shape}")
        except Exception as e:
            _fail("load_face_params", str(e))

        # 生成假情绪csv
        emo_path = os.path.join(tmpdir, "test.csv")
        header = ",".join(FEATURE_ORDER)
        rows = [header]
        for _ in range(100):
            vals = [str(np.random.rand()) for _ in FEATURE_ORDER]
            rows.append(",".join(vals))
        with open(emo_path, "w") as f:
            f.write("\n".join(rows))

        try:
            emo = load_emotion_csv(emo_path)
            if emo.shape == (100, 25):
                _ok("load_emotion_csv 输出形状", f"{emo.shape}")
            else:
                _fail("load_emotion_csv 输出形状", f"期望 (100, 25), 实际 {emo.shape}")
        except Exception as e:
            _fail("load_emotion_csv", str(e))

        # 对齐测试
        try:
            mel_a = np.random.rand(120, 80).astype(np.float32)
            face_a = np.random.rand(110, 58).astype(np.float32)
            emo_a = np.random.rand(130, 25).astype(np.float32)
            m, f, e = align_sequences(mel_a, face_a, emo_a)
            if m.shape[0] == f.shape[0] == e.shape[0]:
                _ok("align_sequences 对齐", f"T={m.shape[0]}, mel={m.shape}, face={f.shape}, emo={e.shape}")
            else:
                _fail("align_sequences 对齐", f"长度不一致: {m.shape[0]}, {f.shape[0]}, {e.shape[0]}")
        except Exception as e:
            _fail("align_sequences", str(e))


# ═══════════════════════════════════════════════
# 测试3：模型前向推理
# ═══════════════════════════════════════════════
def test_model_forward():
    _section("TEST 3: FaceReactionModel 前向推理")
    from services.avatar.face_drive.inference.model import FaceReactionModel, ModelConfig

    cfg = ModelConfig()
    model = FaceReactionModel(cfg)
    _ok("模型实例化成功")

    B, T = 2, 75
    mel_np = np.random.rand(B, T, 80).astype(np.float32)
    emotion_cond_np = np.random.rand(B, 10).astype(np.float32)

    try:
        import torch
        mel_t = torch.from_numpy(mel_np)
        emo_t = torch.from_numpy(emotion_cond_np)
        outputs = model(mel_t, emo_t)

        checks = [
            ("face_58", (B, T, 58)),
            ("emotion_25", (B, T, 25)),
            ("au_15", (B, T, 15)),
            ("va_2", (B, T, 2)),
            ("exp_8", (B, T, 8)),
        ]

        for key, expected_shape in checks:
            if key not in outputs:
                _fail(f"输出包含{key}", "键缺失")
            else:
                arr = outputs[key]
                actual = tuple(arr.shape) if hasattr(arr, 'shape') else None
                if actual == expected_shape:
                    _ok(f"输出{key}形状", f"{actual}")
                else:
                    _fail(f"输出{key}形状", f"期望{expected_shape}, 实际{actual}")

    except ImportError:
        # PyTorch不可用，用numpy降级测试
        outputs = model(mel_np, emotion_cond_np)
        if "face_58" in outputs:
            _ok("模型前向推理(降级)", "numpy模式")
        else:
            _fail("模型前向推理(降级)", "输出缺少face_58")
    except Exception as e:
        _fail("模型前向推理", traceback.format_exc()[-300:])

    # 不带情绪条件
    try:
        import torch
        mel_t = torch.from_numpy(mel_np)
        outputs2 = model(mel_t, None)
        if "face_58" in outputs2:
            _ok("无情绪条件推理", "正常")
        else:
            _fail("无情绪条件推理", "输出缺少face_58")
    except ImportError:
        _ok("无情绪条件推理", "跳过（PyTorch不可用）")
    except Exception as e:
        _fail("无情绪条件推理", str(e))

    # 参数量
    n_params = model.count_parameters()
    _ok("模型参数量", f"{n_params:,} 参数")


# ═══════════════════════════════════════════════
# 测试4：多任务损失
# ═══════════════════════════════════════════════
def test_multitask_loss():
    _section("TEST 4: MultiTaskLoss 多任务损失")
    try:
        from services.avatar.face_drive.training.train import MultiTaskLoss
        from services.avatar.face_drive.inference.model import FaceReactionModel, ModelConfig

        B, T = 2, 75
        criterion = MultiTaskLoss()
        model = FaceReactionModel(ModelConfig())

        face_gt = np.random.rand(B, T, 58).astype(np.float32)
        emo_gt  = np.random.rand(B, T, 25).astype(np.float32)
        # EXP归一化
        emo_gt[:, :, 17:] = np.abs(emo_gt[:, :, 17:])
        emo_gt[:, :, 17:] /= emo_gt[:, :, 17:].sum(axis=-1, keepdims=True)
        # AU clip到[0,1]
        emo_gt[:, :, :15] = np.clip(emo_gt[:, :, :15], 0.0, 1.0)
        # VA clip到[-1,1]
        emo_gt[:, :, 15:17] = np.clip(emo_gt[:, :, 15:17], -1.0, 1.0)

        try:
            import torch
            mel_t = torch.from_numpy(np.random.rand(B, T, 80).astype(np.float32))
            emo_cond_t = torch.from_numpy(np.random.rand(B, 10).astype(np.float32))
            face_gt_t = torch.from_numpy(face_gt)
            emo_gt_t  = torch.from_numpy(emo_gt)
            outputs = model(mel_t, emo_cond_t)
            losses = criterion(outputs, face_gt_t, emo_gt_t)
        except ImportError:
            mel_np = np.random.rand(B, T, 80).astype(np.float32)
            outputs = model(mel_np, None)
            losses = criterion(outputs, face_gt, emo_gt)

        total = losses.get("total", None)
        if total is not None:
            loss_val = float(total) if not hasattr(total, 'item') else total.item()
            if loss_val > 0:
                _ok("多任务损失计算", f"L_total={loss_val:.4f}")
            else:
                _fail("多任务损失计算", f"损失不应为0或负: {loss_val}")

            for key in ("face", "smooth", "au", "va", "expr"):
                if key in losses:
                    v = losses[key]
                    _ok(f"  子损失L_{key}", f"{float(v) if not hasattr(v,'item') else v.item():.4f}")
        else:
            _fail("多任务损失计算", "losses字典中无'total'键")

    except Exception as e:
        _fail("MultiTaskLoss", traceback.format_exc()[-400:])


# ═══════════════════════════════════════════════
# 测试5：VRM/Live2D 映射器
# ═══════════════════════════════════════════════
def test_avatar_mappers():
    _section("TEST 5: VRM/Live2D 映射器")
    try:
        from services.avatar.face_drive.mapping.avatar_mapper import VRMMapper, Live2DMapper

        vrm = VRMMapper()
        live2d = Live2DMapper()
        _ok("VRMMapper/Live2DMapper 实例化")

        face_params = np.random.rand(58).astype(np.float32)
        emotion_25  = np.random.rand(25).astype(np.float32)
        # EXP归一化
        emotion_25[17:] = np.abs(emotion_25[17:])
        emotion_25[17:] /= emotion_25[17:].sum()

        # VRM映射
        vrm_out = vrm.map(face_params, emotion_25, emotion_intent="Happy", emotion_intensity=0.8)
        if isinstance(vrm_out, dict) and len(vrm_out) > 0:
            _ok("VRMMapper.map() 输出", f"{len(vrm_out)} 个参数: {list(vrm_out.keys())[:4]}...")
        else:
            _fail("VRMMapper.map() 输出", f"期望非空dict, 实际: {vrm_out}")

        # 数值范围验证（HeadPitch/HeadYaw/HeadRoll可以为负，只检查blendshape）
        bs_params = {k: v for k, v in vrm_out.items() if not k.startswith("Head")}
        out_of_range = [(k, v) for k, v in bs_params.items() if not (-0.01 <= v <= 1.01)]
        if out_of_range:
            _fail("VRM Blendshape参数在[0,1]", f"越界: {out_of_range[:3]}")
        else:
            _ok("VRM Blendshape参数在[0,1]")

        # Live2D映射
        live2d_out = live2d.map(face_params, emotion_25, emotion_intent="Sad", emotion_intensity=0.6)
        if isinstance(live2d_out, dict) and len(live2d_out) > 0:
            _ok("Live2DMapper.map() 输出", f"{len(live2d_out)} 个参数: {list(live2d_out.keys())[:4]}...")
        else:
            _fail("Live2DMapper.map() 输出", f"期望非空dict, 实际: {live2d_out}")

        # 关键参数存在
        for param in ("ParamEyeLOpen", "ParamMouthOpenY", "ParamMouthForm"):
            if param in live2d_out:
                _ok(f"  Live2D参数存在({param})", f"{live2d_out[param]:.3f}")
            else:
                _fail(f"  Live2D参数存在({param})", "缺失")

    except Exception as e:
        _fail("映射器测试", traceback.format_exc()[-400:])


# ═══════════════════════════════════════════════
# 测试6：评测预测文件生成
# ═══════════════════════════════════════════════
def test_generate_prediction():
    _section("TEST 6: 评测预测文件生成 [N, K=10, T=750, 25]")

    val_csv_exists = Path(VAL_CSV).exists()
    if not val_csv_exists:
        _fail("验证集CSV存在", f"未找到: {VAL_CSV}")
        return
    _ok("验证集CSV存在", VAL_CSV)

    # 验证neighbor matrix
    if Path(VAL_NBR).exists():
        nbr = np.load(VAL_NBR)
        _ok("邻居矩阵存在", f"形状: {nbr.shape}, dtype: {nbr.dtype}")
    else:
        _fail("邻居矩阵存在", f"未找到: {VAL_NBR}")

    # 读取CSV行数
    import csv
    with open(VAL_CSV, "r", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    N_raw = len(rows) - 1  # 减去header
    N_expanded = N_raw * 2  # 双向展开
    _ok("CSV行数读取", f"原始 {N_raw} 对 → 展开 N={N_expanded}")

    # 生成预测（使用随机模型，快速测试）
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "prediction_emotion.npy")

        try:
            from services.avatar.face_drive.inference.generate_prediction import (
                load_val_index, generate_diverse_candidates, _normalize_emotion
            )
            from services.avatar.face_drive.inference.model import FaceReactionModel, ModelConfig

            speaker_order, listener_order = load_val_index(VAL_CSV)
            N = len(speaker_order)
            K, T, D = 10, 750, 25

            _ok("load_val_index", f"N={N} (展开后)")

            # 使用随机模型生成少量样本验证形状
            model = FaceReactionModel(ModelConfig())
            N_TEST = min(5, N)
            pred_test = np.zeros((N_TEST, K, T, D), dtype=np.float32)

            for i in range(N_TEST):
                mel = np.random.rand(T, 80).astype(np.float32)
                base = np.random.rand(T, 25).astype(np.float32)
                base = _normalize_emotion(base)
                pred_test[i] = generate_diverse_candidates(base, k=K)

            np.save(output_path, pred_test)
            _ok("generate_diverse_candidates", f"形状: {pred_test.shape}")

            # 验证形状
            loaded = np.load(output_path)
            if loaded.shape == (N_TEST, K, T, D):
                _ok("prediction_emotion.npy 形状验证", f"{loaded.shape} ✓")
            else:
                _fail("prediction_emotion.npy 形状验证", f"期望({N_TEST},{K},{T},{D}), 实际{loaded.shape}")

            # 数值范围
            au_range  = (loaded[:, :, :, :15].min(), loaded[:, :, :, :15].max())
            va_range  = (loaded[:, :, :, 15:17].min(), loaded[:, :, :, 15:17].max())
            exp_sum   = loaded[:, :, :, 17:].sum(axis=-1)

            if au_range[0] >= -0.01 and au_range[1] <= 1.01:
                _ok("AU维度值域[0,1]", f"min={au_range[0]:.3f}, max={au_range[1]:.3f}")
            else:
                _fail("AU维度值域[0,1]", f"越界: {au_range}")

            if va_range[0] >= -1.01 and va_range[1] <= 1.01:
                _ok("VA维度值域[-1,1]", f"min={va_range[0]:.3f}, max={va_range[1]:.3f}")
            else:
                _fail("VA维度值域[-1,1]", f"越界: {va_range}")

            if np.allclose(exp_sum, 1.0, atol=0.05):
                _ok("EXP维度Softmax归一化(和≈1)", f"mean_sum={exp_sum.mean():.4f}")
            else:
                _fail("EXP维度Softmax归一化", f"和不为1: {exp_sum.mean():.4f}")

            # 多样性验证 (K条候选不完全相同)
            for i in range(N_TEST):
                diffs = [np.abs(loaded[i, k] - loaded[i, 0]).mean() for k in range(1, K)]
                avg_diff = np.mean(diffs)
                if avg_diff > 1e-6:
                    _ok(f"  样本{i} K条候选多样性", f"avg_diff={avg_diff:.4f}")
                    break
            else:
                _fail("K条候选多样性", "所有K条候选完全相同")

        except Exception as e:
            _fail("评测预测生成", traceback.format_exc()[-500:])


# ═══════════════════════════════════════════════
# 测试7：DriveEngine 端到端
# ═══════════════════════════════════════════════
def test_drive_engine():
    _section("TEST 7: DriveEngine 端到端驱动")
    try:
        from services.avatar.drive_engine import DriveEngine

        engine = DriveEngine()
        _ok("DriveEngine 实例化")

        # 获取映射器中注册的数字人 ID
        avatar_ids = list(engine._mappers.keys())
        if len(avatar_ids) >= 2:
            _ok("数字人数量 ≥ 2", str(avatar_ids))
        else:
            _fail("数字人数量 ≥ 2", f"只有 {len(avatar_ids)} 个: {avatar_ids}")

        # 构造 LLMToDriver 协议对象
        try:
            from packages.common.protocols import LLMToDriver, AssistantInfo, RenderInfo

            llm_out = LLMToDriver(
                trace_id="trace_test_001",
                session_id="test_session",
                turn_id=1,
                assistant=AssistantInfo(text="我今天感觉很开心！"),
            )

            result = engine.drive(llm_out)
            if isinstance(result, dict) and "frames" in result:
                frames = result["frames"]
                _ok("drive() 返回帧序列", f"生成 {len(frames)} 帧")
                if "avatar_outputs" in result:
                    _ok("drive() 包含 avatar_outputs",
                        f"数字人: {list(result['avatar_outputs'].keys())}")
                if frames:
                    frame0 = frames[0]
                    if "face_params_58" in frame0 and "emotion_25" in frame0:
                        _ok("帧结构验证", "包含face_params_58/emotion_25")
                    else:
                        _fail("帧结构验证", f"缺少字段: {set(frame0.keys())}")
            else:
                _fail("drive() 返回帧序列", f"返回不包含frames: {type(result)}")

        except Exception as e:
            _fail("drive()", traceback.format_exc()[-300:])

        # Idle动画
        try:
            idle_result = engine.drive_idle(duration_s=0.5)
            if isinstance(idle_result, dict) and "frames" in idle_result:
                idle_frames = idle_result["frames"]
                _ok("drive_idle() Idle动画", f"生成 {len(idle_frames)} 帧")
            elif isinstance(idle_result, list) and len(idle_result) > 0:
                _ok("drive_idle() Idle动画", f"生成 {len(idle_result)} 帧")
            else:
                _fail("drive_idle() Idle动画", f"返回类型: {type(idle_result)}")
        except Exception as e:
            _fail("drive_idle()", str(e)[:200])

    except Exception as e:
        _fail("DriveEngine 端到端", traceback.format_exc()[-500:])


# ═══════════════════════════════════════════════
# 测试8：验证集结构校验
# ═══════════════════════════════════════════════
def test_val_set_structure():
    _section("TEST 8: 验证集文件结构")

    # person_specific_val.csv
    if Path(VAL_CSV).exists():
        import csv
        with open(VAL_CSV, "r", encoding="utf-8-sig") as f:
            rows = list(csv.reader(f))
        header = rows[0] if rows else []
        n_rows = len(rows) - 1

        _ok("person_specific_val.csv 存在", f"{n_rows} 行数据")
        if len(header) >= 3:
            _ok("CSV列结构", f"header={header}")
        else:
            _fail("CSV列结构", f"需要≥3列, 实际: {header}")

        # 检查路径格式
        sample_row = rows[1] if len(rows) > 1 else []
        if len(sample_row) >= 3:
            spk = sample_row[1].strip()
            lst = sample_row[2].strip()
            if "/" in spk and "/" in lst:
                _ok("路径格式", f"spk={spk}, lst={lst}")
            else:
                _fail("路径格式", f"期望 dataset/session/role/idx 格式")
    else:
        _fail("person_specific_val.csv 存在", VAL_CSV)

    # 邻居矩阵
    if Path(VAL_NBR).exists():
        nbr = np.load(VAL_NBR)
        _ok("邻居矩阵 .npy 存在", f"形状={nbr.shape}")

        # 与CSV行数对应
        import csv
        with open(VAL_CSV, "r", encoding="utf-8-sig") as f:
            n_rows = len(list(csv.reader(f))) - 1
        N_expanded = n_rows * 2
        if nbr.shape[0] == N_expanded:
            _ok("邻居矩阵行数=N_expanded", f"{nbr.shape[0]} == {N_expanded}")
        else:
            _fail("邻居矩阵行数=N_expanded",
                  f"矩阵行数{nbr.shape[0]} ≠ CSV展开数{N_expanded}")
    else:
        _fail("邻居矩阵 .npy 存在", VAL_NBR)

    # eval脚本
    if Path(EVAL_SCRIPT).exists():
        _ok("eval_emotion_metrics.py 存在")
    else:
        _fail("eval_emotion_metrics.py 存在", EVAL_SCRIPT)


# ─────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────
def main():
    print("\n" + "█" * 60)
    print("  数字人面部行为驱动模块 完整测试")
    print("█" * 60)

    test_scan_dataset()
    test_data_loading()
    test_model_forward()
    test_multitask_loss()
    test_avatar_mappers()
    test_generate_prediction()
    test_drive_engine()
    test_val_set_structure()

    # 汇总
    print("\n" + "="*60)
    print("  测试汇总")
    print("="*60)
    passed = sum(1 for _, ok, _ in _results if ok)
    failed = sum(1 for _, ok, _ in _results if not ok)
    total  = len(_results)

    print(f"  通过: {passed}/{total}   失败: {failed}/{total}")
    if failed:
        print("\n  失败项目:")
        for name, ok, msg in _results:
            if not ok:
                print(f"    ✗ {name}: {msg[:120]}")

    print("\n" + "─"*60)
    if failed == 0:
        print("  ✅ 所有测试通过！数字人面部行为驱动模块就绪。")
    else:
        print(f"  ⚠️  {failed} 项测试失败（通常是数据集路径或依赖库未安装）。")
    print("─"*60 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
