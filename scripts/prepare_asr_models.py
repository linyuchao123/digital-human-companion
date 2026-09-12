"""显式准备 FunASR 模型，可选用本地 WAV 做识别验收。"""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, help="可选：识别已有本地 WAV")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env", override=False)
    os.environ.setdefault("MODELSCOPE_CACHE", str(root / "models" / "asr"))
    if args.audio and not args.audio.is_file():
        parser.error("本地音频文件不存在")
    from funasr import AutoModel

    model = AutoModel(
        model=os.getenv("ASR_MODEL", "paraformer-zh"),
        vad_model="fsmn-vad",
        punc_model="ct-punc",
        device=os.getenv("ASR_DEVICE", "cpu"),
        disable_update=True,
    )
    print("FunASR / VAD / 标点模型准备完成")
    if args.audio:
        result = model.generate(input=str(args.audio), batch_size_s=30)
        print("识别结果：", "".join(item.get("text", "") for item in result))


if __name__ == "__main__":
    main()
