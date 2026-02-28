import os
import pickle
import numpy as np
import math
import time
import cv2


# =========================
# 路径配置
# =========================

ROOT_DIR = "/data/huangdi/huangruixiang/SAFE/openpi/rollouts/pi0-libero_10/env_records"

SEGMENT_LABELS_PATH = os.path.join(ROOT_DIR, "segment_labels.pkl")

OUTPUT_PKL_PATH = os.path.join(ROOT_DIR, "final_scores.pkl")
OUTPUT_TXT_PATH = os.path.join(ROOT_DIR, "final_scores.txt")


# =========================
# 你给定的融合方法（不改一行）
# =========================

def fuse_multiscale_segment_predictions(
    video_duration: float,
    segment_lengths: list,
    segment_level_preds: dict,
    time_step: float,
    alpha: float = 10.0
):
    num_steps = int(math.ceil(video_duration / time_step))
    times = np.array([(i + 0.5) * time_step for i in range(num_steps)])

    fused_scores = []

    for t in times:
        anomaly_scores = []

        for seg_len in segment_lengths:
            preds = segment_level_preds[seg_len]
            seg_idx = int(t // seg_len)

            if seg_idx >= len(preds):
                seg_idx = len(preds) - 1

            anomaly_score = 1.0 - preds[seg_idx]
            anomaly_scores.append(anomaly_score)

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        anomaly_scores = np.array(anomaly_scores)
        mean_score = anomaly_scores.mean()
        fused = sigmoid(alpha * (mean_score - 0.5))
        fused_scores.append(fused)

    return times, np.array(fused_scores)


def gaussian_kernel(size, sigma):
    x = np.linspace(-size // 2, size // 2, size)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()


def gaussian_smooth_1d(data, sigma, pad_mode="edge"):
    size = int(6 * sigma) + 1
    kernel = gaussian_kernel(size, sigma)
    pad = size // 2
    padded_data = np.pad(data, pad_width=pad, mode=pad_mode)
    smoothed = np.convolve(padded_data, kernel, mode="valid")
    return smoothed


def temporal_scores_to_frame_scores(
    temporal_scores: np.ndarray,
    video_duration: float,
    num_frames: int,
    fps: int,
    smooth_sigma_ratio: float = 0.1
):
    frames_per_step = int(num_frames / len(temporal_scores))

    frame_scores = []
    for score in temporal_scores:
        frame_scores.extend([score] * frames_per_step)

    if len(frame_scores) < num_frames:
        frame_scores.extend([temporal_scores[-1]] * (num_frames - len(frame_scores)))

    frame_scores = np.array(frame_scores[:num_frames])

    sigma = max(1, int(smooth_sigma_ratio * num_frames))
    frame_scores = gaussian_smooth_1d(frame_scores, sigma)

    return frame_scores


# =========================
# 读取真实视频信息（关键）
# =========================

def load_video_metadata(video_path: str):
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if fps <= 0 or num_frames <= 0:
        raise RuntimeError(f"Invalid video metadata: {video_path}")

    video_duration = num_frames / fps
    return video_duration, num_frames, fps


# =========================
# 实时保存
# =========================

def save_realtime(results: dict):
    with open(OUTPUT_PKL_PATH, "wb") as f:
        pickle.dump(results, f)

    with open(OUTPUT_TXT_PATH, "w") as f:
        for task, v in results.items():
            f.write(f"{task}\n")
            f.write(f"  times: {v['times']}\n")
            f.write(f"  temporal_scores: {v['temporal_scores']}\n")
            f.write(f"  frame_scores(len={len(v['frame_scores'])})\n")
            f.write(f"  frame_scores: {[f'{score:.2f}' for score in v['frame_scores']]})\n\n")

    print(f"[SAVE] realtime saved")


# =========================
# 主流程
# =========================

def main():
    print("[LOAD] segment_labels.pkl")
    with open(SEGMENT_LABELS_PATH, "rb") as f:
        segment_labels = pickle.load(f)

    final_results = {}
    TIME_STEP = 0.5

    for task_name, seg_dict in segment_labels.items():
        print(f"[PROCESS] {task_name}")

        video_path = os.path.join(ROOT_DIR, f"{task_name}.mp4")
        video_duration, num_frames, fps = load_video_metadata(video_path)

        segment_lengths = sorted(seg_dict.keys())
        segment_level_preds = seg_dict

        times, temporal_scores = fuse_multiscale_segment_predictions(
            video_duration=video_duration,
            segment_lengths=segment_lengths,
            segment_level_preds=segment_level_preds,
            time_step=TIME_STEP
        )

        frame_scores = temporal_scores_to_frame_scores(
            temporal_scores=temporal_scores,
            video_duration=video_duration,
            num_frames=num_frames,
            fps=fps
        )

        final_results[task_name] = {
            "video_path": video_path,
            "video_duration": video_duration,
            "fps": fps,
            "num_frames": num_frames,
            "times": times.tolist(),
            "temporal_scores": temporal_scores.tolist(),
            "frame_scores": frame_scores.tolist(),
        }

        save_realtime(final_results)
        time.sleep(0.05)

    print("\n[DONE] All tasks processed with REAL video metadata.")


if __name__ == "__main__":
    main()
