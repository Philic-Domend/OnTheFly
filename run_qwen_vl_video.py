import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from decord import VideoReader, cpu
import numpy as np


# =========================
# 1. 配置
# =========================

MODEL_PATH = "models/Qwen3-VL-8B-Instruct"
VIDEO_PATH = "/data/huangdi/huangruixiang/SAFE/openpi/rollouts1/rollouts/pi0-libero_10/env_records/task0--ep0--succ1.mp4"

NUM_FRAMES = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# 2. 读取视频并抽帧
# =========================

def load_video_frames(video_path, num_frames=8):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)

    indices = np.linspace(0, total_frames - 1, num_frames).astype(int)

    print(f"indices: {indices}")

    frames = vr.get_batch(indices).asnumpy()  # (T, H, W, C)

    return frames


# =========================
# 3. 主逻辑
# =========================

def main():
    print("Loading video...")
    frames = load_video_frames(VIDEO_PATH, NUM_FRAMES)

    print("Loading model and processor...")
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True
    )

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    # =========================
    # 4. 构造多模态 messages（关键）
    # =========================

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video"},
                {
                    "type": "text",
                    "text": (
                        "任务指令：描述机械臂执行任务的异常情况\n"
                        "你的角色：专业的任务描述与异常鉴定人员，能够为每一帧任务执行时的图像给出精准的描述。\n"
                        "你的任务：对8帧的视频进行观看与分析，对每一帧的视频图像都给出描述，并判断这一帧是否存在任务异常。。\n"
                        "请严格遵循以下步骤与格式要求执行：\n"
                        "1. 观看与分析视频：仔细观看机械臂执行任务的视频，理解该任务。\n2. 为视频中的每一帧图像，都给出精准的描述，描述需要包括下列几点：当前图像中机械臂执行任务的情况、是否存在卡住的情况、机械臂是否存在异常(包括抖动、抓空、卡住等等)、是否抓错了物体。\n3. 对每一帧图像，需要结合相邻的前后图像来进行分析，最终得到精确的描述。\n"
                        "输出格式与语言要求：请以Excel表格形式输出表格必须包含以下两列，且列标题与顺序必须为：\n"
                        "第几帧 描述\n"
                        "请严格按照以上规则，为提供的视频进行分析，并输出8行2列的Excel表格。"
                    )
                }
            ]
        }
    ]

    # 自动生成 <image> / <video> 占位 token
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # =========================
    # 5. Processor 打包输入
    # =========================

    inputs = processor(
        text=[text],
        videos=[frames],
        return_tensors="pt"
    ).to(model.device)

    # =========================
    # 6. 推理
    # =========================

    print("Generating response...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=4096
        )

    # =========================
    # 7. 解码输出
    # =========================

    result = processor.batch_decode(
        outputs,
        skip_special_tokens=True
    )[0]

    print("\n===== Model Output =====")
    print(result)


# =========================
# 8. 入口
# =========================

if __name__ == "__main__":
    main()
