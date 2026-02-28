import os
import re
import glob
import subprocess
import pickle
from typing import List, Dict

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor


# =========================
# 基本配置
# =========================

MODEL_PATH = "models/Qwen3-VL-8B-Instruct"
VIDEOS_PATH = "/data/huangdi/huangruixiang/SAFE/openpi/rollouts/pi0-libero_10/env_records"

SEGMENT_LENGTHS = [2, 2.5, 3, 3.5, 4]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TASK_VIDEO_PATTERN = re.compile(r"(task\d+--ep\d+--succ[01])\.mp4$")

LABEL_SAVE_PATH = os.path.join(VIDEOS_PATH, "segment_labels.pkl")
LABEL_SAVE_TXT_PATH = os.path.join(VIDEOS_PATH, "segment_labels.txt")


# =========================
# task requirement 列表
# =========================

TASK_REQUIREMENTS = {
    0: "put both the alphabet soup and the tomato sauce in the basket",
    1: "put both the cream cheese box and the butter in the basket",
    2: "turn on the stove and put the moka pot on it",
    3: "put the black bowl in the bottom drawer of the cabinet and close it",
    4: "put the white mug on the left plate and put the yellow and white mug on the right plate",
    5: "pick up the book and place it in the back compartment of the caddy",
    6: "put the white mug on the plate and put the chocolate pudding to the right of the plate",
    7: "put both the alphabet soup and the cream cheese box in the basket",
    8: "put both moka pots on the stove",
    9: "put the yellow and white mug in the microwave and close it",
}


# =========================
# 实时保存工具函数（新增）
# =========================

def save_labels_realtime(results: dict, save_path: str):
    """
    实时保存当前 results 到磁盘（pkl）
    """
    with open(save_path, "wb") as f:
        pickle.dump(results, f)
    print(f"[SAVE] labels saved to {save_path}")


def save_labels_realtime_txt(results: dict, save_path: str):
    """
    实时保存当前 results 到 TXT（人类可读）
    """
    with open(save_path, "w") as f:
        for task_name, seg_dict in results.items():
            f.write(f"TASK: {task_name}\n")
            for seg_len, labels in sorted(seg_dict.items()):
                f.write(f"  SEGMENT_LENGTH = {seg_len}\n")
                f.write(f"    labels = {labels}\n")
            f.write("\n")
    print(f"[SAVE] labels saved to {save_path}")


# =========================
# FFmpeg 拆分（单视频）
# =========================

def split_single_video_ffmpeg(
    video_path: str,
    segment_length: float,
    output_dir: str,
):
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

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr)

    return sorted(glob.glob(os.path.join(output_dir, "segment_*.mp4")))


# =========================
# 批量拆分 videos_path
# =========================

def batch_split_videos(
    videos_path: str,
    segment_lengths: List[float],
):
    segments_root = os.path.join(videos_path, "segments")
    os.makedirs(segments_root, exist_ok=True)

    mp4_files = glob.glob(os.path.join(videos_path, "*.mp4"))

    for video_path in mp4_files:
        fname = os.path.basename(video_path)
        match = TASK_VIDEO_PATTERN.match(fname)
        if not match:
            continue

        task_name = match.group(1)
        print(f"[SPLIT] {task_name}")

        for seg_len in segment_lengths:
            out_dir = os.path.join(
                segments_root,
                task_name,
                str(seg_len)
            )
            print(f"  └─ SEGMENT_LENGTH={seg_len}")
            split_single_video_ffmpeg(
                video_path,
                seg_len,
                out_dir
            )


# =========================
# VLM 标签提取
# =========================

def extract_class_label(text: str, default: int = 3) -> int:
    """
    从 VLM 输出中提取分类标签。
    如果无法匹配，返回默认值（默认 3），并给出警告。
    """

    # 最严格匹配（原来的规则）
    match = re.search(r"Final classification result is\s*([1-5])", text)
    if match:
        return int(match.group(1))

    # 兜底匹配：Number X
    match = re.search(r"Number\s*([1-5])", text)
    if match:
        print("[WARN] Fallback match using 'Number X'")
        return int(match.group(1))

    # 兜底匹配：最后一行 / 结尾数字
    match = re.search(r"([1-5])\s*$", text.strip())
    if match:
        print("[WARN] Fallback match using trailing number")
        return int(match.group(1))

    # 最终兜底：默认值
    print(
        "[WARN] Cannot extract class label, using default = "
        f"{default}\n----- RAW OUTPUT -----\n{text}\n----------------------"
    )
    return default


# =========================
# VLM 推理：一个 SEGMENT_LENGTH
# =========================

def classify_segments_with_vlm(
    segment_dir: str,
    task_requirement: str,
    processor,
    model,
) -> List[int]:

    segment_paths = sorted(glob.glob(os.path.join(segment_dir, "segment_*.mp4")))
    labels = []

    for i, path in enumerate(segment_paths):
        print(f"    [VLM] {os.path.basename(path)}")

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
                            f"This video is the {i}th segment of a robotic arm performing a task, not the complete task process, just a part.\n"
                            f"The task requirement in the video is: {task_requirement}\n"
                            f"Your current task:\n"
                            f"(1) Describe the execution of the robotic arm in this video.\n"
                            f"(2) Classify this video, provide analysis, and finally output a number (1 to 5)."
                            f"Number 1 indicates: This video is very normal and correct, the robotic arm is performing the task normally and continuously;"
                            f"Number 2 indicates: Overturned objects on the table;"
                            f"Number 3 indicates: The robotic arm gripper does not continue to operate, but remains in a state of shaking/stationary/standby/confused, etc.;"
                            f"Number 4 indicates: The robotic arm operation has semantic errors, not executing according to instruction requirements, for example: the grabbed object does not belong to the task requirements, etc."
                            f"Number 5 indicates: The robotic arm gripper attempted to grab but missed the object, starting to move the gripper without actually grasping anything."
                            f"Be attention: the last sentence of your output must be 'Final classification result is i', where i represents a certain number from 1-5."
                        )
                    }
                ]
            }
        ]

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024
            )

        reply = processor.batch_decode(
            outputs,
            skip_special_tokens=True
        )[0]

        cls = extract_class_label(reply)
        labels.append(1 if cls == 1 else 0)

    return labels


# =========================
# 构建最终大数组（实时保存）
# =========================

def build_labels_from_segments(
    segments_root: str,
    processor,
    model,
) -> Dict[str, Dict[float, List[int]]]:

    results = {}

    task_dirs = sorted(glob.glob(os.path.join(segments_root, "task*--ep*--succ*")))

    for task_dir in task_dirs:
        task_name = os.path.basename(task_dir)
        succ_flag = int(task_name.split("succ")[-1])

        task_id = int(re.search(r"task(\d+)", task_name).group(1))
        task_requirement = TASK_REQUIREMENTS[task_id]

        print(f"[TASK] {task_name}")
        results[task_name] = {}

        seg_len_dirs = sorted(glob.glob(os.path.join(task_dir, "*")))

        for seg_len_dir in seg_len_dirs:
            seg_len = float(os.path.basename(seg_len_dir))
            segments = glob.glob(os.path.join(seg_len_dir, "segment_*.mp4"))

            print(f"  [LEN={seg_len}] segments={len(segments)}")

            if succ_flag == 1:
                labels = [1] * len(segments)
            else:
                labels = classify_segments_with_vlm(
                    seg_len_dir,
                    task_requirement,
                    processor,
                    model
                )

            results[task_name][seg_len] = labels

            # ⭐ 实时保存（pkl + txt）
            save_labels_realtime(results, LABEL_SAVE_PATH)
            save_labels_realtime_txt(results, LABEL_SAVE_TXT_PATH)

    return results


# =========================
# 主入口
# =========================

def main():
    '''
    batch_split_videos(
        VIDEOS_PATH,
        SEGMENT_LENGTHS
    )
    '''

    print("[MODEL] Loading VLM...")
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

    segments_root = os.path.join(VIDEOS_PATH, "segments")
    labels = build_labels_from_segments(
        segments_root,
        processor,
        model
    )

    print("\n===== FINAL LABEL STRUCTURE =====")
    for task, v in labels.items():
        print(task)
        for k, arr in v.items():
            print(f"  SEGMENT_LENGTH={k}: {arr}")


if __name__ == "__main__":
    main()
