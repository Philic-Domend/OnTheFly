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

VIDEOS_PATHS = [
    # "/data/huangdi/huangruixiang/SAFE/openpi/rollouts/pi0-libero_goal/env_records",
    "/data/huangdi/huangruixiang/SAFE/openpi/rollouts/pi0-libero_spatial/env_records",
    # "/data/huangdi/huangruixiang/SAFE/openpi/rollouts/pi0-libero_90/env_videos",
    # "/data/huangdi/huangruixiang/SAFE/openpi/rollouts/pi0-libero_10/env_records",
]

SEGMENT_LENGTHS = [2, 2.5, 3, 3.5, 4]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TASK_VIDEO_PATTERN = re.compile(r"(task\d+--ep\d+--succ[01])\.mp4$")

# ⭐ task id 白名单
TASK_ID_FILTER = [9]   # 只需要改这里


# =========================
# TASK REQUIREMENTS
# =========================

TASK_REQUIREMENTS_10 = {
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

TASK_REQUIREMENTS_GOAL = {
    0: "open the middle drawer of the cabinet",
    1: "put the bowl on the stove",
    2: "put the wine bottle on top of the cabinet",
    3: "open the top drawer and put the bowl inside",
    4: "put the bowl on top of the cabinet",
    5: "push the plate to the front of the stove",
    6: "put the cream cheese in the bowl",
    7: "turn on the stove",
    8: "put the bowl on the plate",
    9: "put the wine bottle on the rack",
}

TASK_REQUIREMENTS_SPATIAL = {
    0: "pick up the black bowl between the plate and the ramekin and place it on the plate",
    1: "pick up the black bowl next to the ramekin and place it on the plate",
    2: "pick up the black bowl from table center and place it on the plate",
    3: "pick up the black bowl on the cookie box and place it on the plate",
    4: "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
    5: "pick up the black bowl on the ramekin and place it on the plate",
    6: "pick up the black bowl next to the cookie box and place it on the plate",
    7: "pick up the black bowl on the stove and place it on the plate",
    8: "pick up the black bowl next to the plate and place it on the plate",
    9: "pick up the black bowl on the wooden cabinet and place it on the plate",
}

TASK_REQUIREMENTS_90 = {
    0: "close the top drawer of the cabinet",
    1: "close the top drawer of the cabinet and put the black bowl on top of it",
    2: "put the black bowl in the top drawer of the cabinet",
    3: "put the butter at the back in the top drawer of the cabinet and close it",
    4: "put the butter at the front in the top drawer of the cabinet and close it",
    5: "put the chocolate pudding in the top drawer of the cabinet and close it",
    6: "open the bottom drawer of the cabinet",
    7: "open the top drawer of the cabinet",
    8: "open the top drawer of the cabinet and put the bowl in it",
    9: "put the black bowl on the plate",
    10: "put the black bowl on top of the cabinet",
    11: "open the top drawer of the cabinet",
    12: "put the black bowl at the back on the plate",
    13: "put the black bowl at the front on the plate",
    14: "put the middle black bowl on the plate",
    15: "put the middle black bowl on top of the cabinet",
    16: "stack the black bowl at the front on the black bowl in the middle",
    17: "stack the middle black bowl on the back black bowl",
    18: "put the frying pan on the stove",
    19: "put the moka pot on the stove",
    20: "turn on the stove",
    21: "turn on the stove and put the frying pan on it",
    22: "close the bottom drawer of the cabinet",
    23: "close the bottom drawer of the cabinet and open the top drawer",
    24: "put the black bowl in the bottom drawer of the cabinet",
    25: "put the black bowl on top of the cabinet",
    26: "put the wine bottle in the bottom drawer of the cabinet",
    27: "put the wine bottle on the wine rack",
    28: "close the top drawer of the cabinet",
    29: "put the black bowl in the top drawer of the cabinet",
    30: "put the black bowl on the plate",
    31: "put the black bowl on top of the cabinet",
    32: "put the ketchup in the top drawer of the cabinet",
    33: "close the microwave",
    34: "put the yellow and white mug to the front of the white mug",
    35: "open the microwave",
    36: "put the white bowl on the plate",
    37: "put the white bowl to the right of the plate",
    38: "put the right moka pot on the stove",
    39: "turn off the stove",
    40: "put the frying pan on the cabinet shelf",
    41: "put the frying pan on top of the cabinet",
    42: "put the frying pan under the cabinet shelf",
    43: "put the white bowl on top of the cabinet",
    44: "turn on the stove",
    45: "turn on the stove and put the frying pan on it",
    46: "pick up the alphabet soup and put it in the basket",
    47: "pick up the cream cheese box and put it in the basket",
    48: "pick up the ketchup and put it in the basket",
    49: "pick up the tomato sauce and put it in the basket",
    50: "pick up the alphabet soup and put it in the basket",
    51: "pick up the butter and put it in the basket",
    52: "pick up the milk and put it in the basket",
    53: "pick up the orange juice and put it in the basket",
    54: "pick up the tomato sauce and put it in the basket",
    55: "pick up the alphabet soup and put it in the tray",
    56: "pick up the butter and put it in the tray",
    57: "pick up the cream cheese and put it in the tray",
    58: "pick up the ketchup and put it in the tray",
    59: "pick up the tomato sauce and put it in the tray",
    60: "pick up the black bowl on the left and put it in the tray",
    61: "pick up the chocolate pudding and put it in the tray",
    62: "pick up the salad dressing and put it in the tray",
    63: "stack the left bowl on the right bowl and place them in the tray",
    64: "stack the right bowl on the left bowl and place them in the tray",
    65: "put the red mug on the left plate",
    66: "put the red mug on the right plate",
    67: "put the white mug on the left plate",
    68: "put the yellow and white mug on the right plate",
    69: "put the chocolate pudding to the left of the plate",
    70: "put the chocolate pudding to the right of the plate",
    71: "put the red mug on the plate",
    72: "put the white mug on the plate",
    73: "pick up the book and place it in the front compartment of the caddy",
    74: "pick up the book and place it in the left compartment of the caddy",
    75: "pick up the book and place it in the right compartment of the caddy",
    76: "pick up the yellow and white mug and place it to the right of the caddy",
    77: "pick up the book and place it in the back compartment of the caddy",
    78: "pick up the book and place it in the front compartment of the caddy",
    79: "pick up the book and place it in the left compartment of the caddy",
    80: "pick up the book and place it in the right compartment of the caddy",
    81: "pick up the book and place it in the front compartment of the caddy",
    82: "pick up the book and place it in the left compartment of the caddy",
    83: "pick up the book and place it in the right compartment of the caddy",
    84: "pick up the red mug and place it to the right of the caddy",
    85: "pick up the white mug and place it to the right of the caddy",
    86: "pick up the book in the middle and place it on the cabinet shelf",
    87: "pick up the book on the left and place it on top of the shelf",
    88: "pick up the book on the right and place it on the cabinet shelf",
    89: "pick up the book on the right and place it under the cabinet shelf",
}


# =========================
# SUITE → REQUIREMENTS 映射
# =========================

def get_task_requirements(videos_path: str):
    if "libero_10" in videos_path:
        return TASK_REQUIREMENTS_10
    if "libero_goal" in videos_path:
        return TASK_REQUIREMENTS_GOAL
    if "libero_spatial" in videos_path:
        return TASK_REQUIREMENTS_SPATIAL
    if "libero_90" in videos_path:
        return TASK_REQUIREMENTS_90
    raise ValueError(f"Unknown suite for path: {videos_path}")


# =========================
# 实时保存工具函数
# =========================

def save_labels_realtime(results: dict, save_path: str):
    with open(save_path, "wb") as f:
        pickle.dump(results, f)
    print(f"[SAVE] labels saved to {save_path}")


def save_labels_realtime_txt(results: dict, save_path: str):
    with open(save_path, "w") as f:
        for task_name, seg_dict in results.items():
            f.write(f"TASK: {task_name}\n")
            for seg_len, labels in sorted(seg_dict.items()):
                f.write(f"  SEGMENT_LENGTH = {seg_len}\n")
                f.write(f"    labels = {labels}\n")
            f.write("\n")
    print(f"[SAVE] labels saved to {save_path}")


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
    match = re.search(r"Final classification result is\s*([1-5])", text)
    if match:
        return int(match.group(1))

    match = re.search(r"Number\s*([1-5])", text)
    if match:
        print("[WARN] Fallback match using 'Number X'")
        return int(match.group(1))

    match = re.search(r"([1-5])\s*$", text.strip())
    if match:
        print("[WARN] Fallback match using trailing number")
        return int(match.group(1))

    print(
        "[WARN] Cannot extract class label, using default = "
        f"{default}\n----- RAW OUTPUT -----\n{text}\n----------------------"
    )
    return default


# =========================
# VLM 推理
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

        '''
        for item in messages[0]["content"]:
            if item.get("type") == "text":
                print("\n========== VLM INPUT TEXT ==========")
                print(item["text"])
                print("====================================\n")
        '''

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=1024)

        reply = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        cls = extract_class_label(reply)

        labels.append(1 if cls == 1 else 0)

    return labels


# =========================
# 构建最终大数组（断点续写）
# =========================

def build_labels_from_segments(
    segments_root: str,
    processor,
    model,
) -> Dict[str, Dict[float, List[int]]]:

    # ⭐ 如果已有结果，先 load
    if os.path.exists(LABEL_SAVE_PATH):
        with open(LABEL_SAVE_PATH, "rb") as f:
            results = pickle.load(f)
        print(f"[LOAD] Existing labels loaded from {LABEL_SAVE_PATH}")
    else:
        results = {}

    task_dirs = sorted(glob.glob(os.path.join(segments_root, "task*--ep*--succ*")))

    for task_dir in task_dirs:
        task_name = os.path.basename(task_dir)

        task_id = int(re.search(r"task(\d+)", task_name).group(1))
        if task_id not in TASK_ID_FILTER:
            print(f"[SKIP] {task_name} (task_id not in filter)")
            continue

        if task_name in results:
            print(f"[SKIP] {task_name} (already processed)")
            continue

        succ_flag = int(task_name.split("succ")[-1])
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

        # ⭐ 每个 task 结束后立刻保存（不断点）
        save_labels_realtime(results, LABEL_SAVE_PATH)
        save_labels_realtime_txt(results, LABEL_SAVE_TXT_PATH)

    return results


# =========================
# 主入口
# =========================

def main():
    global LABEL_SAVE_PATH, LABEL_SAVE_TXT_PATH, TASK_REQUIREMENTS

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

    for videos_path in VIDEOS_PATHS:
        print(f"\n==============================")
        print(f"[PROCESSING SUITE]")
        print(videos_path)
        print(f"==============================")

        LABEL_SAVE_PATH = os.path.join(videos_path, "segment_labels.pkl")
        LABEL_SAVE_TXT_PATH = os.path.join(videos_path, "segment_labels.txt")

        TASK_REQUIREMENTS = get_task_requirements(videos_path)

        # batch_split_videos(videos_path, SEGMENT_LENGTHS)

        segments_root = os.path.join(videos_path, "segments")

        build_labels_from_segments(
            segments_root,
            processor,
            model
        )


if __name__ == "__main__":
    main()