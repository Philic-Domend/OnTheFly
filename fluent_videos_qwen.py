import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import os
import glob
import subprocess


MODEL_PATH = "models/Qwen3-VL-8B-Instruct"
VIDEO_PATH = "/data/huangdi/huangruixiang/SAFE/openpi/rollouts1/rollouts/pi0-libero_10/env_records/task2--ep4--succ0.mp4"
VIDEO_PATH1 = "/data/huangdi/huangruixiang/SAFE/openpi/rollouts1/rollouts/pi0-libero_10/env_records/task2--ep0--succ1.mp4"

OUTPUT_DIR = "./task2--ep4/"
SEGMENT_LENGTH = 3  # 秒

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def split_video_ffmpeg(video_path, segment_length, output_dir):
    
    os.makedirs(output_dir, exist_ok=True)
    for f in glob.glob(os.path.join(output_dir, "segment_*.mp4")):
        os.remove(f)
    output_pattern = os.path.join(output_dir, "segment_%03d.mp4")

    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-force_key_frames", f"expr:gte(t,n_forced*{segment_length})",
        "-map", "0",
        "-f", "segment",
        "-segment_time", str(segment_length),
        "-reset_timestamps", "1",
        output_pattern,
    ]

    print(f"[INFO] Splitting video every {segment_length}s ...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("[ERROR] FFmpeg failed")
        print(result.stderr)
        return []

    segments = sorted(glob.glob(os.path.join(output_dir, "segment_*.mp4")))
    print(f"[INFO] Generated {len(segments)} segments")
    return segments


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

    segment_paths = split_video_ffmpeg(
        VIDEO_PATH, SEGMENT_LENGTH, OUTPUT_DIR
    )
    if len(segment_paths) == 0:
        raise RuntimeError("No video segments generated")
    
    for i, path in enumerate(segment_paths):

        print(f"{i}th: This is segment {i}")

        video_content = {
            "type": "video",
            "video": path,
            "id": f"segment_{i}"
        }

        messages = [
            {
                "role": "user",
                "content": [video_content] + [
                    {
                        "type": "text",
                        "text": (
                            f"这个视频是机械臂执行任务的第{i}个片段，不是完整任务进程，只是一个部分。\n"
                            f"视频中任务要求是：打开炉灶旋钮，抓取桌面上的摩卡壶放在炉灶上。\n"
                            f"你现在的任务：\n"
                            f"(1) 描述这个视频中机械臂的执行情况。\n"
                            f"(2) 对这个视频进行分类，给出分析，最终输出一个数字(1到4)。"
                            f"数字1表示：这个视频非常正常，机械臂在正常、持续性的执行任务；"
                            f"数字2表示：这个视频中，夹爪弄倒了物体，或者桌面上翻倒的物体；"
                            f"数字3表示：这个视频中，机械臂夹爪没有继续操作，而是保持 抖动/静止/待命/不知所措 等状态；"
                            f"数字4表示：这个视频中，机械臂操作有语义错误，也就是没有按照指令要求执行，比如：抓取的物体不属于任务要求的物体等。"
                            f"你输出的最后一句话必须是'最终分类结果是i'，其中i表示1-4其中的某个数字。"
                        )
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
                max_new_tokens=2048
            )

        result = processor.batch_decode(
            outputs,
            skip_special_tokens=True
        )[0]

        print(f"\n===== Video Analysis Output =====")
        print(result)


if __name__ == "__main__":
    main()