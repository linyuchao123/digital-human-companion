#!/usr/bin/env python3
"""
视觉模块测试脚本
测试MediaPipe面部特征提取功能
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from services.vision.inference.mediapipe_face import VisionExtractor, VisionConfig


def test_with_webcam():
    """使用摄像头测试视觉模块"""
    print("=" * 60)
    print("视觉模块测试 - 摄像头模式")
    print("=" * 60)
    
    # 配置
    model_path = "d:/face_landmarker.task"
    config = VisionConfig(
        model_path=model_path,
        frame_rate_fps=15,
        window_ms=1000,
        face_count=1,
        au_mapping_path=str(project_root / "services" / "vision" / "config" / "au_mapping_v1.json").replace("\\", "/"),
        smoothing_alpha=0.7,
        enable_landmarks=False,
        enable_gaze=True,
    )
    
    print(f"模型路径: {config.model_path}")
    print(f"帧率: {config.frame_rate_fps} fps")
    print(f"AU映射: {config.au_mapping_path}")
    
    # 初始化提取器
    print("\n正在初始化MediaPipe FaceLandmarker...")
    try:
        extractor = VisionExtractor(config)
        print("✓ 初始化成功")
    except Exception as e:
        print(f"✗ 初始化失败: {e}")
        return False
    
    # 打开摄像头
    print("\n正在打开摄像头...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("✗ 无法打开摄像头")
        return False
    print("✓ 摄像头已打开")
    
    print("\n按 'q' 键退出测试")
    print("-" * 60)
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("✗ 无法读取帧")
                break
            
            timestamp_ms = int(time.time() * 1000)
            result = extractor.process_frame(frame, timestamp_ms)
            
            if result:
                frame_count += 1
                
                # 显示基本信息
                if frame_count % 30 == 0:  # 每30帧打印一次
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed
                    
                    print(f"\n帧 #{frame_count} | 实际FPS: {fps:.1f}")
                    print(f"  跟踪状态: {result['quality']['tracking_state']}")
                    print(f"  置信度: {result['quality']['confidence']:.2f}")
                    
                    if result['features']['face']['au_15']:
                        au_summary = []
                        for au in result['features']['face']['au_15'][:5]:  # 只显示前5个
                            if au['intensity'] > 0.1:
                                au_summary.append(f"{au['name']}:{au['intensity']:.2f}")
                        if au_summary:
                            print(f"  AU(>0.1): {', '.join(au_summary)}")
                    
                    if result['summary']['expression_8']:
                        expr = result['summary']['expression_8']
                        print(f"  表情: {expr['label']} (置信度: {max(expr['probs'].values()):.2f})")
                    
                    if result['summary']['va']:
                        va = result['summary']['va']
                        print(f"  VA: valence={va['valence']:.2f}, arousal={va['arousal']:.2f}")
                    
                    if result['features']['face']['head_pose']:
                        hp = result['features']['face']['head_pose']
                        print(f"  头部姿态: pitch={hp['pitch']:.1f}, yaw={hp['yaw']:.1f}, roll={hp['roll']:.1f}")
            
            # 显示视频
            cv2.imshow('Vision Module Test', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n用户退出")
                break
                
    except KeyboardInterrupt:
        print("\n测试被中断")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        extractor.close()
        
        elapsed = time.time() - start_time
        print(f"\n{'=' * 60}")
        print(f"测试完成")
        print(f"总帧数: {frame_count}")
        print(f"运行时间: {elapsed:.1f}秒")
        print(f"平均FPS: {frame_count/elapsed:.1f}")
        print(f"{'=' * 60}")
    
    return True


def test_with_image():
    """使用静态图片测试视觉模块"""
    print("=" * 60)
    print("视觉模块测试 - 静态图片模式")
    print("=" * 60)
    
    # 创建一个测试图片（模拟人脸）
    print("\n创建测试图片...")
    image = np.ones((480, 640, 3), dtype=np.uint8) * 200  # 灰色背景
    # 画一个简单的"脸"
    cv2.circle(image, (320, 240), 100, (180, 150, 120), -1)  # 脸
    cv2.circle(image, (280, 210), 15, (50, 50, 50), -1)  # 左眼
    cv2.circle(image, (360, 210), 15, (50, 50, 50), -1)  # 右眼
    cv2.ellipse(image, (320, 280), (40, 20), 0, 0, 180, (100, 50, 50), 3)  # 嘴
    
    # 配置
    model_path = "d:/face_landmarker.task"  # MediaPipe 不支持含中文的路径
    config = VisionConfig(
        model_path=model_path,
        frame_rate_fps=15,
        au_mapping_path=str(project_root / "services" / "vision" / "config" / "au_mapping_v1.json").replace("\\", "/"),
    )
    
    print(f"模型路径: {config.model_path}")
    
    # 初始化提取器
    print("\n正在初始化...")
    try:
        extractor = VisionExtractor(config)
        print("✓ 初始化成功")
    except Exception as e:
        print(f"✗ 初始化失败: {e}")
        return False
    
    # 处理图片
    print("\n处理测试图片...")
    timestamp_ms = int(time.time() * 1000)
    result = extractor.process_frame(image, timestamp_ms)
    
    if result:
        print("\n✓ 成功获取结果")
        print(f"\n输出JSON结构:")
        print(f"  enabled: {result['enabled']}")
        print(f"  provider: {result['provider']}")
        print(f"  frame_rate_fps: {result['frame_rate_fps']}")
        print(f"  face_count: {result['face_count']}")
        print(f"  quality.tracking_state: {result['quality']['tracking_state']}")
        
        if result['features']['face']['au_15']:
            print(f"\n  AU特征 (15维):")
            for au in result['features']['face']['au_15']:
                print(f"    {au['name']}: {au['intensity']:.3f}")
        
        if result['summary']['va']:
            va = result['summary']['va']
            print(f"\n  VA估计:")
            print(f"    valence: {va['valence']:.3f}")
            print(f"    arousal: {va['arousal']:.3f}")
        
        if result['summary']['expression_8']:
            expr = result['summary']['expression_8']
            print(f"\n  表情分类:")
            print(f"    主要表情: {expr['label']}")
            print(f"    概率分布:")
            for name, prob in sorted(expr['probs'].items(), key=lambda x: -x[1])[:3]:
                print(f"      {name}: {prob:.3f}")
        
        # 验证输出格式
        print("\n" + "-" * 60)
        print("验证输出格式...")
        
        required_keys = ['enabled', 'provider', 'model', 'mode', 'frame_rate_fps', 
                        'window_ms', 'face_count', 'features', 'processing', 
                        'summary', 'quality']
        
        missing_keys = [k for k in required_keys if k not in result]
        if missing_keys:
            print(f"✗ 缺少必要字段: {missing_keys}")
        else:
            print("✓ 所有必要字段都存在")
        
        # 验证features结构
        face_features = result['features']['face']
        feature_keys = ['blendshapes_52', 'au_15', 'head_pose', 'gaze']
        for key in feature_keys:
            if key in face_features:
                print(f"✓ features.face.{key} 存在")
            else:
                print(f"✗ features.face.{key} 缺失")
    else:
        print("✗ 未获取到结果")
    
    extractor.close()
    print(f"\n{'=' * 60}")
    return True


def test_output_protocol():
    """测试输出协议是否符合规范"""
    print("=" * 60)
    print("视觉模块测试 - 协议格式验证")
    print("=" * 60)
    
    model_path = "d:/face_landmarker.task"  # MediaPipe 不支持含中文的路径，使用纯英文路径
    config = VisionConfig(
        model_path=model_path,
        au_mapping_path=str(project_root / "services" / "vision" / "config" / "au_mapping_v1.json").replace("\\", "/"),
    )
    extractor = VisionExtractor(config)
    
    # 创建测试图片
    image = np.ones((480, 640, 3), dtype=np.uint8) * 200
    cv2.circle(image, (320, 240), 100, (180, 150, 120), -1)
    cv2.circle(image, (280, 210), 15, (50, 50, 50), -1)
    cv2.circle(image, (360, 210), 15, (50, 50, 50), -1)
    
    result = extractor.process_frame(image, int(time.time() * 1000))
    extractor.close()
    
    if not result:
        print("✗ 无法获取结果")
        return False
    
    print("\n验证 PerceptionToLLM.vision 协议格式:")
    print("-" * 60)
    
    checks = []
    
    # 检查基本字段
    checks.append(("enabled 为布尔值", isinstance(result.get('enabled'), bool)))
    checks.append(("provider 为 'mediapipe'", result.get('provider') == 'mediapipe'))
    checks.append(("mode 为 'live_stream'", result.get('mode') == 'live_stream'))
    checks.append(("frame_rate_fps 为整数", isinstance(result.get('frame_rate_fps'), int)))
    checks.append(("window_ms 为整数", isinstance(result.get('window_ms'), int)))
    
    # 检查features
    face = result.get('features', {}).get('face', {})
    checks.append(("features.face 存在", bool(face)))
    checks.append(("blendshapes_52 为列表", isinstance(face.get('blendshapes_52'), list)))
    checks.append(("au_15 为列表", isinstance(face.get('au_15'), list)))
    
    # 检查au_15格式
    au_list = face.get('au_15', [])
    if au_list:
        au_item = au_list[0]
        checks.append(("au_15 项有 name 字段", 'name' in au_item))
        checks.append(("au_15 项有 intensity 字段", 'intensity' in au_item))
    
    # 检查processing
    proc = result.get('processing', {})
    checks.append(("processing.au_mapping 存在", 'au_mapping' in proc))
    checks.append(("processing.smoothing 存在", 'smoothing' in proc))
    
    # 检查summary
    summary = result.get('summary', {})
    checks.append(("summary.va 存在", 'va' in summary))
    checks.append(("summary.expression_8 存在", 'expression_8' in summary))
    
    # 检查quality
    quality = result.get('quality', {})
    checks.append(("quality.tracking_state 存在", 'tracking_state' in quality))
    checks.append(("quality.confidence 存在", 'confidence' in quality))
    
    # 打印结果
    for check_name, passed in checks:
        status = "✓" if passed else "✗"
        print(f"{status} {check_name}")
    
    passed_count = sum(1 for _, p in checks if p)
    total_count = len(checks)
    
    print(f"\n{'=' * 60}")
    print(f"协议验证结果: {passed_count}/{total_count} 通过")
    print(f"{'=' * 60}")
    
    return passed_count == total_count


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="测试视觉模块")
    parser.add_argument(
        "--mode",
        choices=["webcam", "image", "protocol", "all"],
        default="all",
        help="测试模式: webcam=摄像头, image=静态图片, protocol=协议验证, all=全部"
    )
    
    args = parser.parse_args()
    
    results = []
    
    if args.mode in ["image", "all"]:
        results.append(("静态图片", test_with_image()))
    
    if args.mode in ["protocol", "all"]:
        results.append(("协议验证", test_output_protocol()))
    
    if args.mode in ["webcam", "all"]:
        results.append(("摄像头", test_with_webcam()))
    
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    for name, passed in results:
        status = "通过" if passed else "失败"
        print(f"{name}: {status}")
