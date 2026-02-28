import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor


# =========================
# 1. 配置（完全保留）
# =========================

MODEL_PATH = "models/Qwen3-VL-8B-Instruct"
VIDEO_PATH = "/data/huangdi/huangruixiang/SAFE/openpi/rollouts1/rollouts/pi0-libero_10/env_records/task0--ep3--succ1.mp4"
VIDEO_PATH1 = "/data/huangdi/huangruixiang/SAFE/openpi/rollouts1/rollouts/pi0-libero_10/env_records/task2--ep0--succ1.mp4"

NUM_FRAMES = 8
NUM_SEGMENTS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROMPT = (
    "你的任务：对这几个视频进行观看与分析，对两个视频分别给出描述。\n"
)

# =========================
# 2. 主逻辑
# =========================

def main():
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

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": VIDEO_PATH,
                },
                {
                    "type": "video",
                    "video": VIDEO_PATH1,
                },
                {
                    "type": "text",
                    "text": PROMPT,
                }
            ]
        }
    ]

    print("Processing video input...")
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    print("Generating response...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=4096
        )

    result = processor.batch_decode(
        outputs,
        skip_special_tokens=True
    )[0]

    print(f"\n===== Video Analysis Output =====")
    print(result)


# =========================
# 3. 入口（不动）
# =========================

if __name__ == "__main__":
    main()