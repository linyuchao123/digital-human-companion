import os, zipfile, traceback
from pathlib import Path

ROOT = Path(r"d:\AI数字人情感陪护项目")
asr_root = ROOT / "services" / "asr"

outpath = ROOT / "交付物_v2" / "test_asr2.zip"
try:
    with zipfile.ZipFile(outpath, 'w', zipfile.ZIP_DEFLATED) as zf:
        _ASR_FILES = [
            ("inference/paraformer_zh.py", "inference/paraformer_zh.py"),
            ("inference/acoustic_features.py", "inference/acoustic_features.py"),
            ("inference/__init__.py", "inference/__init__.py"),
            ("vad/fsmn_vad.py", "vad/fsmn_vad.py"),
            ("vad/__init__.py", "vad/__init__.py"),
            ("audio_pipeline.py", "audio_pipeline.py"),
            ("__init__.py", "__init__.py"),
        ]
        for rel_src, rel_dst in _ASR_FILES:
            fp = asr_root / rel_src
            arcname = f"ASR_Package/{rel_dst}"   # 用英文前缀
            if fp.exists():
                try:
                    zf.write(fp, arcname)
                    print(f"  OK: {arcname}")
                except Exception as e:
                    print(f"  FAIL: {arcname} -> {e}")
            else:
                print(f"  MISSING: {fp}")
    print(f"\nResult: {outpath.stat().st_size} bytes")
except Exception as e:
    traceback.print_exc()
