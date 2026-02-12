import os
import random
import hydra
import imageio
from omegaconf import OmegaConf
from tqdm import tqdm, trange

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

import wandb

from failure_prob.data import load_rollouts, split_rollouts, load_rollouts_plus, split_rollouts_plus
from failure_prob.data.utils import Rollout, RolloutDataset, normalize_rollouts_hidden_states, TimestepRolloutDataset, PlusRolloutDataset
from failure_prob.model import get_model
from failure_prob.model.base import BaseModel
from failure_prob.utils.constants import MANUAL_METRICS, EVAL_TIME_QUANTILES
from failure_prob.utils.timer import Timer
from failure_prob.utils.routines import (
    eval_model_and_log,
    eval_metrics_and_log,
    eval_save_timing_plots,
    eval_model_and_log_timestep,
    eval_model_and_log_plus,
)
from failure_prob.utils.video import eval_save_videos, eval_save_videos_functional_cp
from failure_prob.utils.random import seed_everything

from failure_prob.conf import Config, process_cfg

import json
import re
from pathlib import Path
import pickle
from typing import List, Dict, Any
from collections import defaultdict

def load_step_labels(success_labels_paths: List[Path], task_names: List[str]) -> List[Dict[str, Any]]:
    """加载步级标签"""
    labels = []
    
    for path in success_labels_paths:
        task_name = next((n for n in task_names if n in str(path)), None)
        if not task_name:
            continue
            
        scores_file = path / "final_scores.txt"
        if not scores_file.exists():
            continue
            
        with open(scores_file, 'r') as f:
            content = f.read()
        
        # 提取success_id (succ后面的数字)
        for match in re.finditer(r'task(\d+)--ep(\d+)--succ(\d+).*?frame_scores:\s*\[([^\]]+)\]', content, re.DOTALL):
            task_id, ep_id, succ_id, scores_str = match.groups()
            scores = [float(s.strip().strip("'\"")) for s in scores_str.split(',') if s.strip()]
            
            labels.append({
                'task_name': task_name,
                'task_id': int(task_id),
                'episode_id': int(ep_id),
                'success_id': int(succ_id),
                'frame_scores': scores,
                'is_success': int(succ_id) > 0  # 通常succ0表示失败，succ1表示成功
            })
    
    return labels


def save_rollouts(rollouts, filename="rollouts_saved.pkl"):
    """保存rollouts到文件"""
    with open(filename, 'wb') as f:
        pickle.dump(rollouts, f)
    print(f"已保存 {len(rollouts)} 条rollouts到 {filename}")

def load_saved_rollouts(filename="rollouts_saved.pkl"):
    """从文件加载rollouts"""
    with open(filename, 'rb') as f:
        rollouts = pickle.load(f)
    print(f"从 {filename} 加载了 {len(rollouts)} 条rollouts")
    return rollouts

def print_rollouts_details(rollouts, save_to_file=False):
    """打印rollouts详情 可选保存到文件"""
    
    output_lines = []
    
    for i, r in enumerate(rollouts):
        output_lines.append(f"\n[{i}] ==========")
        for attr in sorted(dir(r)):
            if not attr.startswith('_'):
                val = getattr(r, attr)
                if not callable(val):
                    if attr == 'mp4_path':
                        line = f"  {attr}: {val}"
                    elif attr == 'step_labels' and val is not None:
                        line = f"  {attr}: shape{list(val.shape)} values{val.tolist()}"
                    elif hasattr(val, 'shape'):
                        line = f"  {attr}: shape{list(val.shape)}"
                    else:
                        line = f"  {attr}: {val}"
                    
                    output_lines.append(line)
                    print(line)  # 同时打印到控制台
    
    # 保存到文件
    if save_to_file:
        with open('./my_rollouts.txt', 'w') as f:
            f.write('\n'.join(output_lines))
        print(f"\n已保存到 ./my_rollouts.txt")
    
    return output_lines


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: Config) -> None:
    cfg = process_cfg(cfg)
    print(OmegaConf.to_yaml(cfg))
    
    if (
        cfg.train.eval_save_logs 
        or cfg.train.eval_save_video 
        or cfg.train.eval_save_ckpt 
        or cfg.train.eval_save_video_functional
        or cfg.train.eval_save_timing_plots
    ):
        os.makedirs(cfg.train.logs_save_path, exist_ok=True)
        # Save the config
        with open(os.path.join(cfg.train.logs_save_path, "config.yaml"), "w") as f:
            f.write(OmegaConf.to_yaml(cfg))

    cfg.model.cumsum = False
            
    # Set seed for randomness in loading rollouts
    seed_everything(0)

    success_labels_root = [Path(path) for path in cfg.dataset.success_labels_paths]
    task_names = cfg.dataset.task_names

    # print(success_labels_root)
    # print(task_names)
            
    step_labels = load_step_labels(success_labels_root, task_names)

    # Load and preprocess the rollout data
    with Timer("Loading rollouts"):
        # 修改点：从.pkl那边加载rollouts的方法写在pizero.py里面
        
        # all_rollouts = load_rollouts_plus(cfg.dataset.data_paths, cfg, step_labels, cfg.dataset.task_names)
    
        # pkl_path = "./all_rollouts.pkl"
        pkl_path = "./all_rollouts_updated_step_labels_1.pkl"
        print(f"从 {pkl_path} 加载rollouts...")
        
        with open(pkl_path, 'rb') as f:
            all_rollouts = pickle.load(f)

        print(f"Loaded {len(all_rollouts)} rollouts")
        if cfg.dataset.load_to_cuda:
            all_rollouts = [r.to("cuda") for r in all_rollouts]
    
    if len(all_rollouts) == 0:
        raise ValueError(f"No rollouts loaded from {cfg.dataset.data_path}")
    
    if cfg.dataset.normalize_hidden_states:
        # Normalize the hidden states to zero mean and unit variance
        all_rollouts = normalize_rollouts_hidden_states(all_rollouts)
    
    # Seed again for splitting rollouts
    if isinstance(cfg.train.seed, int):
        seeds = [cfg.train.seed]
    elif cfg.train.seed.isnumeric() and "-" not in cfg.train.seed:
        seeds = [int(cfg.train.seed)]
    else:
        # assert there are only integer seeds separated by "-" in cfg.train.seed
        assert all(s.isdigit() for s in cfg.train.seed.split("-")), "All seeds must be integers separated by '_'"

        # Run different seeds in the same call, to speed up 
        seeds = [int(s) for s in cfg.train.seed.split("-")]
        
    for seed in seeds:
        print(f"Running seed {seed}")
        cfg.train.seed = seed
        seed_everything(0)

        np.random.seed(cfg.train.seed)
        torch.manual_seed(cfg.train.seed)
        
        wandb.init(
            project = cfg.train.wandb_project, 
            dir = cfg.train.wandb_dir,
            name = cfg.train.exp_name,
            group = cfg.train.wandb_group_name,
            config = OmegaConf.to_container(cfg, resolve=True),
            reinit=True,
            # mode="offline",
        )
        
        rollouts_by_split_name = split_rollouts_plus(cfg, all_rollouts)

        '''
        rollouts_by_split_name = {
            "train": [Rollout, Rollout, ...],
            "val_seen": [Rollout, Rollout, ...],
            "val_unseen": [Rollout, Rollout, ...],
        }

        [Example rollout attributes]:
        hidden_states        → Tensor(shape=(T, 4096), dtype=torch.float32)
        task_suite_name      → str: libero_10
        task_id              → int: 0
        task_description     → str(ex.): put both the alphabet soup and the tomato sauce in the basket
        episode_idx          → int: 0
        episode_success      → int: 0
        mp4_path             → str: /data/huangdi/huangruixiang/SAFE/openvla/rollouts/single-foward/libero_10/task0--ep0--succ0.mp4
        logs                 → NoneType: None
        task_min_step        → int(ex.): 244
        exec_horizon         → NoneType: None
        action_vectors       → Tensor(shape=(T, 7), dtype=torch.float32)

        ==================================================
        [Rollout Loading Info] (Total Loaded: 1)
        - action_vectors       : Shape [56, 350]          | Dtype: torch.float32
        - episode_idx          : Value 0                  | Type: int
        - episode_success      : Value 1                  | Type: int
        - exec_horizon         : Value 5                  | Type: int
        - hidden_states        : Shape [56, 1024]         | Dtype: torch.float32
        - logs                 : Value None               | Type: NoneType
        - mp4_path             : Value /data/huangdi/huangruixiang/SAFE/openpi/rollouts/pi0-libero_10/env_records/task0--ep0--succ1.mp4 | Type: str
        - task_description     : Value put both the alphabet soup and the tomato sauce... | Type: str
        - task_id              : Value 0                  | Type: int
        - task_min_step        : Value None               | Type: NoneType
        - task_suite_name      : Value libero_10          | Type: str
        ==================================================
        '''
        
        train_rollouts = rollouts_by_split_name["train"]
        val_seen_rollouts = rollouts_by_split_name["val_seen"]

        input_dim = train_rollouts[0].hidden_states.shape[-1]
        print(f"hidden_feature: {train_rollouts[0].hidden_states.shape} {train_rollouts[0].hidden_states.dtype}")

        # Construct datasets and dataloaders from the rollouts
        # 修改！使用新定义的 Dataset 类
        dataset_by_split_name = {

            # 修改点：在utils.py里面的TimestepRolloutDataset类中新增vl_states属性，注意pad也要处理。
            # TimestepRolloutDataset类中的属性形状都是(B, ...)。

            k: PlusRolloutDataset(cfg, v)
            for k, v in rollouts_by_split_name.items()
        }
        dataloader_by_split_name = {
            k: DataLoader(
                v, 
                batch_size=cfg.model.batch_size, 
                shuffle="train" in k, 
                num_workers=0)
            for k, v in dataset_by_split_name.items()
        }

        # Plot and log the precomputed metrics
        if cfg.train.log_precomputed or cfg.train.log_precomputed_only:
            to_be_logged = eval_metrics_and_log(
                cfg,
                rollouts_by_split_name, 
                MANUAL_METRICS[cfg.dataset.name],
                EVAL_TIME_QUANTILES[cfg.dataset.name],
            )
            to_be_logged['epoch'] = 0
            wandb.log(to_be_logged)
        
        if cfg.train.log_precomputed_only:
            wandb.finish(quiet=True)
            continue
        
        model: BaseModel = get_model(cfg, input_dim)
        print(model)
        model.to("cuda")
        
        optimizer, lr_scheduler = model.get_optimizer()
        
        n_epochs = cfg.model.n_epochs
        epoch_losses = []
        pbar = trange(n_epochs)
        for epoch in pbar:
            to_be_logged = {"epoch": epoch + 1}
            
            # Training
            model.train()
            avg_loss = model.train_epoch_step(optimizer, dataloader_by_split_name["train"])
            epoch_losses.append(avg_loss)
            pbar.set_description(f"Avg Loss: {avg_loss:.4f}")
            to_be_logged["train_loss"] = avg_loss

            if lr_scheduler is not None:
                lr_scheduler.step()
                to_be_logged["learning_rate"] = optimizer.param_groups[0]['lr']
            
            # Evaluation
            model.eval()
            if epoch % 10 == 0 or epoch == n_epochs - 1:
                logs_timestep = eval_model_and_log_plus(
                    cfg=cfg,
                    model=model,
                    rollouts_by_split_name=rollouts_by_split_name,
                    dataloader_by_split_name=dataloader_by_split_name
                )
                wandb.log(logs_timestep)
                
            wandb.log(to_be_logged)
        
        # Final evaluation
        if cfg.train.eval_save_video:
            for split, dataloader in dataloader_by_split_name.items():
                if split == "train":
                    continue
                video_save_folder = os.path.join(cfg.train.logs_save_path, f"videos_{split}")
                print("Saving videos to", os.path.abspath(video_save_folder))
                os.makedirs(video_save_folder, exist_ok=True)
                eval_save_videos(dataloader, model, cfg, video_save_folder)
                
        if cfg.train.eval_save_video_functional:
            video_save_folder = os.path.join(cfg.train.logs_save_path, f"videos_functional")
            os.makedirs(video_save_folder, exist_ok=True)
            eval_save_videos_functional_cp(
                cfg, model, 
                rollouts_by_split_name,
                dataloader_by_split_name,
                video_save_folder,
                alpha = cfg.train.eval_cp_alpha,
            )
            
        if cfg.train.eval_save_timing_plots:
            plot_save_folder = os.path.join(cfg.train.logs_save_path, f"timing_plots")
            os.makedirs(plot_save_folder, exist_ok=True)
            eval_save_timing_plots(
                cfg, model, 
                rollouts_by_split_name,
                dataloader_by_split_name,
                plot_save_folder,
                alpha= cfg.train.eval_cp_alpha,
            )
            
                
        if cfg.train.eval_save_ckpt:
            os.makedirs(cfg.train.logs_save_path, exist_ok=True)
            ckpt_save_path = os.path.join(cfg.train.logs_save_path, "model_final.ckpt")
            print("Saving model checkpoint to", os.path.abspath(ckpt_save_path))
            torch.save(model.state_dict(), ckpt_save_path)
        
        wandb.finish(quiet=True)
        

if __name__ == "__main__":
    main()
