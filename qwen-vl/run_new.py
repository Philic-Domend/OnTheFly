import os
import glob
import subprocess
import torch

from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info

# =========================================================
# 1. 配置
# =========================================================

MODEL_PATH = "models/Qwen3-VL-8B-Instruct"

VIDEO_PATH = "/data/huangdi/huangruixiang/SAFE/openpi/rollouts1/rollouts/pi0-libero_10/env_records/task0--ep37--succ0.mp4"
OUTPUT_DIR = "./task0--ep37/"
SEGMENT_LENGTH = 3  # 秒

PROMPT = (
    "你的任务：对若干个连续的视频进行观看与分析，对视频给出描述，并判断是否存在任务异常。\n"
    "输入的视频按照顺序可以连接成一个完整的视频。\n"
    # "视频中的任务名称：put both the alphabet soup and the tomato sauce in the basket。但是这个视频不一定正确执行了该任务。\n"
    "请你针对我提供的任务名称，分析整个视频，并对视频进行描述。\n"
    "如果视频存在异常，那么请你分别给出任务正常执行的时间段和异常对应的时间段。"
)

# =========================================================
# 2. 视频分割（严格时间、强制关键帧）
# =========================================================

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

# =========================================================
# 3. 主流程
# =========================================================

def main():
    print("[INFO] Loading model & processor ...")

    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True
    )

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    # -----------------------------------------------------
    # 分割视频
    # -----------------------------------------------------

    segment_paths = split_video_ffmpeg(
        VIDEO_PATH, SEGMENT_LENGTH, OUTPUT_DIR
    )

    if len(segment_paths) == 0:
        raise RuntimeError("No video segments generated")

    # -----------------------------------------------------
    # 构造 messages（多视频）
    # -----------------------------------------------------

    video_contents = []
    for i, path in enumerate(segment_paths):
        video_contents.append({
            "type": "video",
            "video": path,
            "id": f"segment_{i}"
        })

    messages = [
        {
            "role": "user",
            "content": video_contents + [
                {
                    "type": "text",
                    "text": (
                        f"{PROMPT}\n\n"
                        f"注意：这是原视频被分成的 {len(segment_paths)} 个片段，"
                        f"每个片段 {SEGMENT_LENGTH} 秒，请对每个片段都进行分析。"
                    )
                }
            ]
        }
    ]

    # -----------------------------------------------------
    # 正确的多模态处理流程（核心）
    # -----------------------------------------------------

    # ① 只生成文本 prompt（不 tokenize）
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # ② 让 qwen-vl-utils 处理视频
    images, videos, video_kwargs = process_vision_info(
        messages,
        image_patch_size=16,              # Qwen3-VL 必须 16
        return_video_kwargs=True,
        return_video_metadata=True
    )

    # 拆 video + metadata
    if videos is not None:
        videos, video_metadatas = zip(*videos)
        videos = list(videos)
        video_metadatas = list(video_metadatas)
    else:
        video_metadatas = None

    # ③ 最终 processor 调用
    inputs = processor(
        text=text,
        images=images,
        videos=videos,
        video_metadata=video_metadatas,
        do_resize=False,                 # 已由 qwen-vl-utils 处理
        return_tensors="pt",
        **video_kwargs
    ).to(model.device)

    # -----------------------------------------------------
    # 推理
    # -----------------------------------------------------

    print("[INFO] Generating response ...")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=4096
        )

    output_text = processor.batch_decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    print("\n================== MODEL OUTPUT ==================\n")
    print(output_text)

# =========================================================

if __name__ == "__main__":
    main()
