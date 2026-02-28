import torch
import cv2
import numpy as np
from PIL import Image
from diffsynth.utils.data import save_video
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig


def load_first_frame(video_path):
    """从视频中读取第一帧图片"""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"无法读取视频: {video_path}")
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


def sample_frames_evenly(video_path, num_frames=4):
    """从视频中均等采样 num_frames 帧，返回 (PIL Image 列表, 对应帧索引列表)"""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if total < num_frames:
        raise ValueError(f"视频总帧数 {total} 小于采样数 {num_frames}")
    # 均等采样的帧索引（0-based）
    indices = [int((i + 0.5) * total / num_frames) for i in range(num_frames)]
    frames = []
    cap = cv2.VideoCapture(video_path)
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            raise ValueError(f"无法读取视频第 {idx} 帧: {video_path}")
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))
    cap.release()
    return frames, indices


def load_video_clip(video_path, start_frame_idx, num_frames, target_size=None):
    """从视频中读取从 start_frame_idx 开始的连续 num_frames 帧，返回 PIL Image 列表"""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if start_frame_idx + num_frames > total:
        cap.release()
        raise ValueError(f"视频从第 {start_frame_idx} 帧起不足 {num_frames} 帧 (总帧数 {total})")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)
    frames = []
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        if target_size is not None:
            img = img.resize(target_size, Image.Resampling.LANCZOS)
        frames.append(img)
    cap.release()
    return frames


def make_left_right_frames(gen_frames, real_frames, gap=4):
    """将生成帧与真实帧左右拼接，中间留 gap 像素白线。两者需同尺寸。返回可写入视频的帧列表（numpy RGB）。"""
    out_frames = []
    for g, r in zip(gen_frames, real_frames):
        ga = np.array(g) if isinstance(g, Image.Image) else g
        ra = np.array(r) if isinstance(r, Image.Image) else r
        if ga.shape[:2] != ra.shape[:2]:
            r_pil = Image.fromarray(ra)
            r_pil = r_pil.resize((ga.shape[1], ga.shape[0]), Image.Resampling.LANCZOS)
            ra = np.array(r_pil)
        h, w = ga.shape[0], ga.shape[1]
        white = np.ones((h, gap, 3), dtype=ga.dtype) * 255
        combined = np.concatenate([ga, white, ra], axis=1)
        out_frames.append(combined)
    return out_frames


# 本地模型路径配置
MODEL_BASE_PATH = "/mnt/shared-storage-user/mahaoxiang/code/DiffSynth-Studio/pretrained_model/Wan2.2-TI2V-5B"
TOKENIZER_PATH = "/mnt/shared-storage-user/mahaoxiang/code/DiffSynth-Studio/pretrained_model/google/umt5-xxl"

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        # 使用 path 参数直接指定本地路径，避免下载
        ModelConfig(path=f"{MODEL_BASE_PATH}/models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(path=f"/mnt/shared-storage-user/internvla2/mahaoxiang/models/train/Wan2.2-TI2V-5B_full/epoch-1.safetensors"),
        # ModelConfig(path=[
        #     f"{MODEL_BASE_PATH}/diffusion_pytorch_model-00001-of-00003.safetensors",
        #     f"{MODEL_BASE_PATH}/diffusion_pytorch_model-00002-of-00003.safetensors",
        #     f"{MODEL_BASE_PATH}/diffusion_pytorch_model-00003-of-00003.safetensors",
        # ]),
        ModelConfig(path=f"{MODEL_BASE_PATH}/Wan2.2_VAE.pth"),
    ],
    # tokenizer 也使用本地路径
    tokenizer_config=ModelConfig(path=TOKENIZER_PATH),
)

# Image-to-video：从视频中均等采样 4 帧，分别作为初始帧生成，保存为生成 vs 真实左右对比
video_path = "/mnt/shared-storage-user/internvla/Users/mahaoxiang/LIBERO_collected_dataset/videos/libero_goal/task0--ep0--succ1.mp4"
num_sample_frames = 4
num_gen_frames = 41
input_frames, frame_indices = sample_frames_evenly(video_path, num_frames=num_sample_frames)

prompt = "open the middle drawer of the cabinet"
height, width = 224, 224
target_size = (width, height)

for i, input_image in enumerate(input_frames):
    input_image = input_image.resize(target_size)
    # 生成 121 帧
    gen_frames = pipe(
        prompt=prompt,
        negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        seed=0, tiled=True,
        height=height, width=width,
        input_image=input_image,
        num_frames=num_gen_frames,
    )
    # 从原视频读取从同一初始帧开始的 121 帧（真实）
    start_idx = frame_indices[i]
    real_frames = load_video_clip(video_path, start_idx, num_gen_frames, target_size=target_size)
    # 左右拼接：左=生成，右=真实
    comparison_frames = make_left_right_frames(gen_frames, real_frames, gap=4)
    out_path = f"video_2_Wan2.2-TI2V-5B_frame{i}_compare.mp4"
    save_video(comparison_frames, out_path, fps=15, quality=5)
    print(f"已保存 (左生成/右真实): {out_path}")
