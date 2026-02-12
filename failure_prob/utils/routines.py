import json
import os
from typing import Optional
import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader
import wandb

from failure_prob.conf import Config
from failure_prob.model.base import BaseModel

# Use a non-interactive backend for multiprocessing (if needed)
import matplotlib

from failure_prob.utils.torch import move_to_device
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from failure_prob.data.utils import Rollout
import cv2
from .metrics import (
    eval_fixed_threshold,
    eval_scores_roc_prc, 
    get_metrics_curve, 
    eval_split_conformal,
    eval_functional_conformal,
    eval_det_time_vs_classification,
    eval_functional_conformal_timestep,
)

'''
def model_forward_dataloader(
    model: BaseModel,
    loader: DataLoader,
):
    device = model.get_device()
    
    scores = []
    valid_masks = []
    labels = []
    
    for batch in loader:
        batch = move_to_device(batch, device)
        scores.append(model(batch))
        valid_masks.append(batch["valid_masks"])
        labels.append(batch["success_labels"])
        
    scores = torch.cat(scores, dim=0).squeeze(-1)
    valid_masks = torch.cat(valid_masks, dim=0)
    labels = torch.cat(labels, dim=0)
    
    return scores, valid_masks, labels
'''
def model_forward_dataloader(
    model: BaseModel,
    loader: DataLoader,
):
    device = model.get_device()
    
    scores = []
    valid_masks = []
    rollout_labels = []
    step_labels = []
    mp4_paths = []
    has_step_labels = False
    
    for batch_idx, batch in enumerate(loader):
        batch = move_to_device(batch, device)
        
        # ======= 打印 batch 信息 =======
        print(f"\n===== Batch {batch_idx} =====")
        if isinstance(batch, dict):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    print(f"batch[{k}]: type={type(v)}, shape={v.shape}, dtype={v.dtype}")
                else:
                    print(f"batch[{k}]: type={type(v)}, value={v}")
        else:
            print(f"batch: type={type(batch)}, value={batch}")
        
        # ======= forward =======
        batch_scores = model(batch)
        scores.append(batch_scores)
        
        # ======= 打印 scores 信息 =======
        if isinstance(batch_scores, torch.Tensor):
            print(f"scores: type={type(batch_scores)}, shape={batch_scores.shape}, dtype={batch_scores.dtype}")
        else:
            print(f"scores: type={type(batch_scores)}, value={batch_scores}")
        
        # 获取掩码
        if "masks" in batch:
            valid_masks.append(batch["masks"])  # TimestepRolloutDataset 用 "masks"
        else:
            valid_masks.append(batch["valid_masks"])  # RolloutDataset 用 "valid_masks"
        
        # 获取标签
        if "step_labels" in batch:
            step_labels.append(batch["step_labels"])  # 时间步级标签 [B, T]
            has_step_labels = True
            rollout_labels.append(batch["rollout_labels"])
            
        else:
            # 回退到 rollout 级标签
            step_labels.append(None)
            rollout_labels.append(batch["success_labels"])
        
        mp4_paths.append(batch["mp4_path"])
        
    scores = torch.cat(scores, dim=0).squeeze(-1)  # [total_batch, max_seq_len]
    valid_masks = torch.cat(valid_masks, dim=0)    # [total_batch, max_seq_len]
    rollout_labels = torch.cat(rollout_labels, dim=0)  # [total_batch]
    
    # 处理 step_labels
    if has_step_labels:
        # 过滤掉 None
        valid_step_labels = [s for s in step_labels if s is not None]
        if valid_step_labels:
            step_labels = torch.cat(valid_step_labels, dim=0)  # [total_batch, max_seq_len]
        else:
            step_labels = None
    else:
        step_labels = None

    print(f"scores.shape(): {scores.shape}")
    print(f"valid_masks.shape(): {valid_masks.shape}")
    print(f"rollout_labels.shape(): {rollout_labels.shape}")
    # print(f"step_labels.shape(): {step_labels.shape}")

    print("\n=== Rollout Details ===")
    for i in range(len(rollout_labels)):
        # print(f"\n[{i}] mp4_path: {loader.dataset.mp4_paths[i]}")
        
        # 获取有效长度
        seq_len = int(valid_masks[i].sum().item())
        
        # 打印有效部分的scores（保留1位小数）
        valid_scores = scores[i, :seq_len].cpu().numpy()
        scores_str = [f"{x:.1f}" for x in valid_scores]
        print(f"  scores ({seq_len} steps): [{', '.join(scores_str)}]")
        
        # 打印step_labels
        if step_labels is not None:
            valid_step_labels = step_labels[i, :seq_len].cpu().numpy()
            labels_str = [f"{x:.1f}" for x in valid_step_labels]
            print(f"  step_labels: [{', '.join(labels_str)}]")
        else:
            print(f"  step_labels: None")

        print()
    
    return scores, valid_masks, rollout_labels, step_labels, mp4_paths


def eval_metrics_and_log(
    cfg: Config,
    rollouts_by_split_name: dict[str, list[Rollout]],
    metric_keys: Optional[list[str]] = None, 
    time_quantiles: list[str] = [1.0]
):
    to_be_logged = {}
    classification_logs = []
    
    if metric_keys is None:
        metric_keys = rollouts_by_split_name['train'][0].logs.columns
        
    for metric_key in metric_keys:
        if metric_key not in rollouts_by_split_name['train'][0].logs.columns:
            print(f"Skipping {metric_key}")
            continue
        
        metric_name = metric_key.split("/")[-1]
        # scores_by_split_name will be a dict: split name -> list of np arrays
        scores_by_split_name = {
            k: get_metrics_curve(v, metric_key) 
            for k, v in rollouts_by_split_name.items()
        }
        
        #### Evaluate ROC and PRC metrics at certain timesteps ####
        metrics_logs = eval_scores_roc_prc(
            rollouts_by_split_name, 
            scores_by_split_name, 
            metric_name, 
            time_quantiles,
            plot_auc_curves=True,
            plot_score_curves=True,
        )
        to_be_logged.update(metrics_logs)
        
        #### Evaluate the classification performance using different thresholding methods ####
        # Split Conformal Prediction: val_seen for calibration, val_unseen for testing
        # Here we only use val_seen for calibration, to make it comparable to learned methods
        split_cp_logs = eval_split_conformal(
            rollouts_by_split_name, scores_by_split_name, metric_name,
            calib_split_names=["val_seen"], test_split_names=["val_unseen"]
        )
        split_cp_logs = pd.DataFrame(split_cp_logs)
        to_be_logged[f"classify_cp_maxsofar/{metric_name}"] = wandb.Table(dataframe=split_cp_logs)
        
        # Functional Conformal Prediction
        df, cp_bands_by_alpha = eval_functional_conformal(
            rollouts_by_split_name, scores_by_split_name, metric_name,
            calib_split_names=["val_seen"], test_split_names=["val_unseen"]
        )
        to_be_logged[f"classify_cp_functional/{metric_name}"] = wandb.Table(dataframe=df)
        
        # Compute the classification metrics vs. detection time
        df, logs = eval_perf_det_time_curves(
            rollouts_by_split_name, scores_by_split_name, metric_name
        )
        to_be_logged.update(logs)
        
        if cfg.train.eval_save_logs:
            os.makedirs(cfg.train.logs_save_path, exist_ok=True)
            df.to_csv(f"{cfg.train.logs_save_path}/{metric_name}_perf_vs_det.csv", index=False)
        
    # Convert the classification logs to a wandb table
    classification_logs = pd.DataFrame(classification_logs)
    to_be_logged["classify/metrics"] = wandb.Table(dataframe=classification_logs)
    
    return to_be_logged


def eval_model_and_log(
    cfg: Config,
    model: BaseModel, 
    rollouts_by_split_name: dict[str, list[Rollout]],
    dataloader_by_split_name: dict[str, DataLoader],
    eval_time_quantiles: list[float], 
    plot_auc_curves: bool = True,
    plot_score_curves: bool = True,
    log_classification_metrics: bool = True,
):
    to_be_logged = {}
    method_name = "model"
    
    #### Forward the model and compute the scores ####
    scores_by_split_name = {}
    for split, dataloader in dataloader_by_split_name.items():
        # Re-create a dataset to disable shuffling
        dataloader = DataLoader(dataloader.dataset, batch_size=cfg.model.batch_size, shuffle=False, num_workers=0)
        with torch.no_grad():
            scores, valid_masks, _, _, _ = model_forward_dataloader(model, dataloader)
        scores = scores.detach().cpu().numpy()
        seq_lengths = valid_masks.sum(dim=-1).cpu().numpy() # (B,)
        scores_by_split_name[split] = [scores[i, :int(seq_lengths[i])] for i in range(len(seq_lengths))]

    if log_classification_metrics == True:
        ### !
        # 打印详细结构
        print("\n=== scores_by_split_name 结构分析 ===")
        print(f"包含的split数量: {len(scores_by_split_name)}")
        print(f"split名称: {list(scores_by_split_name.keys())}")

        for split_name, score_list in scores_by_split_name.items():
            print(f"\n{split_name}:")
            print(f"  样本数量: {len(score_list)}")
            
            # 打印前3个样本的详细信息
            for i in range(min(3, len(score_list))):
                sample_scores = score_list[i]
                print(f"  样本{i}: shape={sample_scores.shape}, 长度={len(sample_scores)}")
                print(f"    前5个分数: {sample_scores[:5]}")
            
            # 统计信息
            if len(score_list) > 0:
                lengths = [len(seq) for seq in score_list]
                print(f"  序列长度统计: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.1f}")

        # 打印 rollouts_by_split_name 的 mp4_path 和 hidden_states
        print("\n" + "="*50)
        print("=== rollouts_by_split_name - mp4_path 和 hidden_states ===")
        print("="*50)

        for split_name, rollout_list in rollouts_by_split_name.items():
            print(f"\n{split_name}:")
            print(f"  rollout数量: {len(rollout_list)}")
            
            for i in range(len(rollout_list)):
                rollout = rollout_list[i]
                print(f"  rollout{i}:")
                
                # 打印 mp4_path
                if hasattr(rollout, 'mp4_path'):
                    print(f"    mp4_path: {rollout.mp4_path}")
                elif isinstance(rollout, dict) and 'mp4_path' in rollout:
                    print(f"    mp4_path: {rollout['mp4_path']}")
                
                # 打印 hidden_states 形状
                if hasattr(rollout, 'hidden_states'):
                    hidden_states = rollout.hidden_states
                    if hasattr(hidden_states, 'shape'):
                        print(f"    hidden_states: shape={hidden_states.shape}")
                    else:
                        print(f"    hidden_states: 类型={type(hidden_states)}")
                elif isinstance(rollout, dict) and 'hidden_states' in rollout:
                    hidden_states = rollout['hidden_states']
                    if hasattr(hidden_states, 'shape'):
                        print(f"    hidden_states: shape={hidden_states.shape}")
                    else:
                        print(f"    hidden_states: 类型={type(hidden_states)}")
                else:
                    print(f"    hidden_states: 未找到")
                
                print()  # 空行分隔
        ### !


    #### Evaluate ROC and PRC metrics at certain timesteps ####
    roc_rpc_logs = eval_scores_roc_prc(
        rollouts_by_split_name, 
        scores_by_split_name, 
        method_name,
        eval_time_quantiles,
        plot_auc_curves, 
        plot_score_curves
    )
    to_be_logged.update(roc_rpc_logs)
    
    #### Evaluate the classification performance using different thresholding methods ####
    if log_classification_metrics:
        # Compute the metrics at fixed thresholds
        df = eval_fixed_threshold(
            rollouts_by_split_name, scores_by_split_name, method_name,
        )
        to_be_logged[f"classify_fixed_thresh/{method_name}"] = wandb.Table(dataframe=df)

        # Split Conformal Prediction: val_seen for calibration, val_unseen for testing
        split_cp_logs = eval_split_conformal(
            rollouts_by_split_name, scores_by_split_name, method_name,
            calib_split_names=["val_seen"], test_split_names=["val_unseen"]
        )
        split_cp_logs = pd.DataFrame(split_cp_logs)
        to_be_logged[f"classify_cp_maxsofar/{method_name}"] = wandb.Table(dataframe=split_cp_logs)
        
        # Functional Conformal Prediction
        df, cp_bands_by_alpha = eval_functional_conformal(
            rollouts_by_split_name, scores_by_split_name, method_name,
            calib_split_names=["val_seen"], test_split_names=["val_unseen"]
        )
        to_be_logged[f"classify_cp_functional/{method_name}"] = wandb.Table(dataframe=df)
        
        # Compute the classification metrics vs. detection time, by varying thresholds
        df, logs = eval_perf_det_time_curves(
            rollouts_by_split_name, scores_by_split_name, method_name
        )
        to_be_logged.update(logs)

        if cfg.train.eval_save_logs:
            os.makedirs(cfg.train.logs_save_path, exist_ok=True)
            df.to_csv(f"{cfg.train.logs_save_path}/{method_name}_perf_vs_det.csv", index=False)
    
    return to_be_logged


def eval_model_and_log_timestep(
    cfg: Config,
    model: BaseModel, 
    rollouts_by_split_name: dict[str, list[Rollout]],
    dataloader_by_split_name: dict[str, DataLoader],
    log_classification_metrics: bool = True,
):
    to_be_logged = {}
    method_name = "model"
    
    #### Forward the model and compute the scores ####
    scores_by_split_name = {}
    step_scores_by_split_name = {}  # 新增：存储时间步级数据
    step_labels_by_split_name = {}  # 新增：存储时间步级标签

    for split, dataloader in dataloader_by_split_name.items():
        # Re-create a dataset to disable shuffling
        dataloader = DataLoader(dataloader.dataset, batch_size=cfg.model.batch_size, shuffle=False, num_workers=0)

        with torch.no_grad():
            # 修改：接收 step_labels
            scores, valid_masks, rollout_labels, step_labels, mp4_paths = model_forward_dataloader(model, dataloader)
        
        print("111111111")
        print(len(mp4_paths))

        if split == "val_seen":
            # 确保数据在CPU上且是numpy数组
            scores_np = scores.cpu().numpy() if torch.is_tensor(scores) else scores
            step_labels_np = step_labels.cpu().numpy() if torch.is_tensor(step_labels) else step_labels
            
            # 评估异常检测
            print(f"\n正在评估 {split} 数据集...")
            stats, error_samples = evaluate_anomaly_detection(
                scores=scores_np,
                step_labels=step_labels_np,
                mp4_paths=mp4_paths,
                threshold=0.7,
                consecutive_count=3
            )
            
            # 打印结果
            print_evaluation_results(stats, error_samples)
            
            # 可以保存结果到wandb
            to_be_logged.update({
                f"{split}_anomaly_acc": stats["acc"],
                f"{split}_anomaly_tpr": stats["tpr"],
                f"{split}_anomaly_tnr": stats["tnr"],
                f"{split}_anomaly_fpr": stats["fpr"],
            })

        if split == "val_unseen":
            # 确保数据在CPU上且是numpy数组
            scores_np = scores.cpu().numpy() if torch.is_tensor(scores) else scores
            step_labels_np = step_labels.cpu().numpy() if torch.is_tensor(step_labels) else step_labels
            
            # 评估异常检测
            print(f"\n正在评估 {split} 数据集...")
            stats, error_samples = evaluate_anomaly_detection(
                scores=scores_np,
                step_labels=step_labels_np,
                mp4_paths=mp4_paths,
                threshold=0.55,
                consecutive_count=3
            )
            
            # 打印结果
            print_evaluation_results(stats, error_samples)
            
            # 可以保存结果到wandb
            to_be_logged.update({
                f"{split}_anomaly_acc": stats["acc"],
                f"{split}_anomaly_tpr": stats["tpr"],
                f"{split}_anomaly_tnr": stats["tnr"],
                f"{split}_anomaly_fpr": stats["fpr"],
            })

    return to_be_logged


def evaluate_anomaly_detection(scores, step_labels, mp4_paths, threshold=0.5, consecutive_count=2):
    B = scores.shape[0]
    mp4_paths = mp4_paths[0] if len(mp4_paths) == 1 and isinstance(mp4_paths[0], list) else mp4_paths
    
    detect_times = np.full(B, -1.0)
    fail_times = np.full(B, -1.0)
    detect_errors = np.full(B, -1.0)  # 新增：存储每个样本的检测误差
    true_labels = []
    pred_labels = []
    error_samples = {"fp": [], "fn": []}
    
    for i in range(B):
        seq_scores = scores[i]
        seq_labels = step_labels[i]

        valid_length = len([x for x in seq_labels if x != -1])
        
        # 真实标签和失败时间
        true_label = 1 if np.any(seq_labels == 0) else 0
        true_labels.append(true_label)
        
        if true_label == 1:
            fail_indices = np.where(seq_labels == 0)[0]
            first_fail_time = fail_indices[0] / valid_length
            fail_times[i] = first_fail_time
        
        # 预测标签和检测时间
        pred_label = 0
        relative_time = -1.0
        for t in range(valid_length - consecutive_count + 1):
            if all(seq_scores[t + k] > threshold for k in range(consecutive_count)):
                pred_label = 1
                relative_time = t / valid_length
                detect_times[i] = relative_time
                break
        pred_labels.append(pred_label)
        
        # 计算并存储检测误差
        if true_label == 1 and pred_label == 1:
            detect_errors[i] = relative_time - fail_times[i]  # 正数表示检测晚了
        
        # 记录错误样本
        if pred_label == 1 and true_label == 0:
            error_samples["fp"].append({
                "index": i,
                "mp4_path": mp4_paths[i],
                "scores": seq_scores.tolist(),
                "detect_time": relative_time
            })
        elif pred_label == 0 and true_label == 1:
            error_samples["fn"].append({
                "index": i,
                "mp4_path": mp4_paths[i],
                "scores": seq_scores.tolist(),
                "labels": seq_labels.tolist(),
                "fail_time": fail_times[i]
            })
    
    # 计算指标
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)
    
    tp = np.sum((pred_labels == 1) & (true_labels == 1))
    tn = np.sum((pred_labels == 0) & (true_labels == 0))
    fp = np.sum((pred_labels == 1) & (true_labels == 0))
    fn = np.sum((pred_labels == 0) & (true_labels == 1))
    
    with np.errstate(divide='ignore', invalid='ignore'):
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        acc = (tp + tn) / B if B > 0 else 0.0
    
    # 检测时间误差统计
    valid_errors = detect_errors[detect_errors != -1]
    detect_time_stats = {}
    if len(valid_errors) > 0:
        detect_time_stats = {
            "n_valid_detections": len(valid_errors),
            "mean_abs_detect_error": float(np.mean(np.abs(valid_errors))),  # 绝对值平均
            "mean_raw_detect_error": float(np.mean(valid_errors)),          # 原始平均
            "std_detect_error": float(np.std(valid_errors)),
            "min_detect_error": float(np.min(valid_errors)),
            "max_detect_error": float(np.max(valid_errors)),
            "median_detect_error": float(np.median(valid_errors)),
        }
    else:
        detect_time_stats = {
            "n_valid_detections": 0,
            "mean_abs_detect_error": 0.0,
            "mean_raw_detect_error": 0.0,
            "std_detect_error": 0.0,
            "min_detect_error": 0.0,
            "max_detect_error": 0.0,
            "median_detect_error": 0.0,
        }
    
    stats = {
        "total_samples": B,
        "true_anomalies": np.sum(true_labels == 1),
        "true_normal": np.sum(true_labels == 0),
        "detected_anomalies": np.sum(pred_labels == 1),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "tpr": float(tpr), "tnr": float(tnr), "fpr": float(fpr), "fnr": float(fnr), "acc": float(acc),
        "detect_times": detect_times.tolist(),
        "fail_times": fail_times.tolist(),
        "detect_errors": detect_errors.tolist(),  # 新增：返回每个样本的检测误差
        **detect_time_stats
    }
    
    return stats, error_samples


def print_evaluation_results(stats, error_samples):
    """打印评估结果"""
    print("\n" + "="*60)
    print("异常检测评估结果")
    print("="*60)
    
    print(f"\n数据集统计:")
    print(f"  总样本数: {stats['total_samples']}")
    print(f"  真实异常: {stats['true_anomalies']}")
    print(f"  真实正常: {stats['true_normal']}")
    print(f"  检测为异常: {stats['detected_anomalies']}")
    
    print(f"\n混淆矩阵:")
    print(f"         预测正常  预测异常")
    print(f"实际正常  {stats['tn']:>6}   {stats['fp']:>6}")
    print(f"实际异常  {stats['fn']:>6}   {stats['tp']:>6}")
    
    print(f"\n评估指标:")
    print(f"  准确率 (ACC): {stats['acc']:.4f}")
    print(f"  召回率 (TPR): {stats['tpr']:.4f}")
    print(f"  特异度 (TNR): {stats['tnr']:.4f}")
    print(f"  误报率 (FPR): {stats['fpr']:.4f}")
    print(f"  漏报率 (FNR): {stats['fnr']:.4f}")
    
    # 新增：检测时间误差统计
    if stats['n_valid_detections'] > 0:
        print(f"\n检测时间误差统计 ({stats['n_valid_detections']}个正确检测):")
        print(f"  平均绝对值误差: {stats['mean_abs_detect_error']:.4f} "
              f"(正值表示检测晚了，负值表示检测早了)")
        print(f"  最小误差: {stats['min_detect_error']:.4f}")
        print(f"  最大误差: {stats['max_detect_error']:.4f}")
        print(f"  中位数误差: {stats['median_detect_error']:.4f}")
    else:
        print(f"\n检测时间误差: 无正确检测的异常样本")

    print(f"\ndetect_errors: {stats['detect_errors']}")
    print(f"\nfail_times(real): {stats['fail_times']}")
    
    # 打印错误样本
    if error_samples["fp"]:
        print(f"\n误报样本 (FP, {len(error_samples['fp'])}个):")
        for i, sample in enumerate(error_samples["fp"][:5]):
            print(f"  [{i}] {sample['mp4_path']}")
            print(f"      检测时间: {sample['detect_time']:.3f}")
            scores_str = ' '.join([f"{s:.2f}" for s in sample['scores'][:20]])
            if len(sample['scores']) > 20:
                scores_str += " ..."
            print(f"      分数: {scores_str}")
    
    if error_samples["fn"]:
        print(f"\n漏报样本 (FN, {len(error_samples['fn'])}个):")
        for i, sample in enumerate(error_samples["fn"][:5]):
            print(f"  [{i}] {sample['mp4_path']}")
            if 'fail_time' in sample:
                print(f"      真实失败时间: {sample['fail_time']:.3f}")
            labels_str = ' '.join([f"{int(l)}" for l in sample['labels'][:20]])
            if len(sample['labels']) > 20:
                labels_str += " ..."
            print(f"      标签: {labels_str}")
            scores_str = ' '.join([f"{s:.2f}" for s in sample['scores'][:20]])
            if len(sample['scores']) > 20:
                scores_str += " ..."
            print(f"      分数: {scores_str}")
    
    print("="*60)


def eval_perf_det_time_curves(
    rollouts_by_split_name: dict[str, list[Rollout]],
    scores_by_split_name: dict[str, list[np.ndarray]],
    method_name: str,
):
    # Compute the classification metrics vs. detection time
    dfs = []
    logs = {}
    fig, ax = plt.subplots()
    y_key = "bal_acc"
    
    for split_name in rollouts_by_split_name:
        rollouts = rollouts_by_split_name[split_name]
        scores_all = scores_by_split_name[split_name]
        labels = np.asarray([1-r.episode_success for r in rollouts])
        results = eval_det_time_vs_classification(
            rollouts, scores_all, labels
        )
        df = pd.DataFrame(results)
        df['method_name'] = method_name
        df['split_name'] = split_name
        dfs.append(df)
        
        df.plot(x="avg_det_time", y=y_key, ax=ax, label=f"{split_name}")

    ax.set_xlabel("Mean detection time of GT failure")
    ax.set_ylabel(y_key)
    fig.tight_layout()
    logs[f"perf_vs_det/{method_name}_{y_key}_vs_Tdet"] = fig
    plt.close(fig)

    dfs = pd.concat(dfs)
    return dfs, logs


def eval_save_timing_plots(
    cfg: Config, 
    model: BaseModel, 
    rollouts_by_split_name: dict[str, list[Rollout]],
    dataloader_by_split_name: dict[str, DataLoader],
    save_folder: str,
    alpha: float = 0.2,
    calib_split_names=["val_seen"], 
    test_split_names=["val_unseen"],
):
    # Load the failure timestep annotations
    label_path = str(cfg.dataset.failure_time_label_path)
    assert os.path.exists(label_path), f"Label file not found: {label_path}"
    labels = json.load(open(label_path, "r"))
    
    # Convert the labels to a dict, indexed by task id, episode_id
    labels_dict = {
        (r['task_id'], r['episode_id']): r for r in labels
    }
    
    #### Forward the model and compute the scores ####
    scores_by_split_name = {}
    for split, dataloader in dataloader_by_split_name.items():
        # Re-create a dataset to disable shuffling.
        dataloader = DataLoader(
            dataloader.dataset, 
            batch_size=cfg.model.batch_size, 
            shuffle=False, 
            num_workers=0
        )
        with torch.no_grad():
            scores, valid_masks, _ = model_forward_dataloader(model, dataloader)
        scores = scores.detach().cpu().numpy()
        seq_lengths = valid_masks.sum(dim=-1).cpu().numpy()  # (B,)
        scores_by_split_name[split] = [
            scores[i, :int(seq_lengths[i])] for i in range(len(seq_lengths))
        ]
        
    df, cp_bands_by_alpha = eval_functional_conformal(
        rollouts_by_split_name, scores_by_split_name, "model",
        calib_split_names=calib_split_names, test_split_names=test_split_names
    )
    
    # Retrieve the CP band for the given alpha.
    cp_band = cp_bands_by_alpha[alpha][0]  # Shape: (T,)
    
    # Gather test rollouts and their corresponding scores.
    test_rollouts = sum([rollouts_by_split_name[k] for k in test_split_names], [])
    test_scores = sum([scores_by_split_name[k] for k in test_split_names], [])
    
    # Determine detection times for each rollout
    records = []
    for i, r in enumerate(test_rollouts):
        score = test_scores[i]
        task_id, episode_idx = r.task_id, r.episode_idx

        # Whether the rollouts is a failure or not
        gt_fail_flag = not bool(r.episode_success)

        detection_mask = score > cp_band[:len(score)]
        # detection_mask = score[:r.task_min_step] >= cp_band[:r.task_min_step]
        
        pred_fail_flag = detection_mask.any() if len(detection_mask) > 0 else False
        
        # predicted failure time
        pred_fail_time = np.argmax(detection_mask) if pred_fail_flag else 2 * len(score)
        pred_fail_time_rel = pred_fail_time / len(score)

        if (task_id, episode_idx) not in labels_dict:
            if gt_fail_flag:
                print(f"Label not found for task_id {task_id}, episode_idx {episode_idx}")
            gt_fail_frame = 0
        else:
            label = labels_dict[(task_id, episode_idx)]
            gt_fail_frame = label["frame"]
        
        # Get the total number of frames from mp4_path
        cap = cv2.VideoCapture(r.mp4_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        gt_fail_time_rel = gt_fail_frame / total_frames
        
        records.append({
            "task_id": task_id,
            "episode_idx": episode_idx,
            "gt_fail_flag": gt_fail_flag,
            "pred_fail_flag": pred_fail_flag,
            "gt_fail_time_rel": gt_fail_time_rel,
            "pred_fail_time_rel": pred_fail_time_rel,
        })
        
    df = pd.DataFrame(records)
    
    # Save the dataframe to a CSV file
    save_path = os.path.join(save_folder, f"timing_data_alpha_{alpha}.csv")
    os.makedirs(save_folder, exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Saved timing data to {save_path}")
    
    # Plot the scatter of predicted vs GT failure time
    fig, ax = plt.subplots(figsize=(3,3), dpi=300)
    df_gt_fail = df[df["gt_fail_flag"]]
    df_tp = df_gt_fail[df_gt_fail["pred_fail_flag"]]
    df_fn = df_gt_fail[~df_gt_fail["pred_fail_flag"]]
    ax.scatter(
        df_tp["gt_fail_time_rel"], 
        df_tp["pred_fail_time_rel"], 
        s=9,
        alpha=0.8, 
        label="TP",
        color="red",
    )
    ax.scatter(
        df_fn["gt_fail_time_rel"], 
        [1.0] * len(df_fn), 
        s=9,
        alpha=0.8, 
        marker="x",
        label="FN",
    )
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
    
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("GT failure time (relative)")
    ax.set_ylabel("Detected failure time\n(relative)")
    ax.set_aspect('equal', adjustable='box')
    ax.legend()
    plt.tight_layout()
    
    save_path = os.path.join(save_folder, f"timing_plot_alpha_{alpha}.pdf")
    os.makedirs(save_folder, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved timing plot to {save_path}")
    
    # Plot the cumulative GT and detected failure over time
    gt_fail_time_rel = df_gt_fail["gt_fail_time_rel"].values
    pred_fail_time_rel = df_gt_fail["pred_fail_time_rel"].values
    fig, ax = plt.subplots(figsize=(4.5,3), dpi=300)
    
    x = np.linspace(0, 1, 100)
    y_gt = np.array([np.mean(gt_fail_time_rel <= t) for t in x])
    y_pred = np.array([np.mean(pred_fail_time_rel <= t) for t in x])
    ax.plot(x, y_pred, label="TP", color="red")
    ax.plot(x, y_gt, label="GT Positives", color="blue")
    ax.set_xlabel("Time (relative)")
    ax.set_ylabel("Cumulative failures\n(proportion)")
    ax.legend()
    plt.tight_layout()
    
    save_path = os.path.join(save_folder, f"cumulative_failures_alpha_{alpha}.pdf")
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved cumulative failure plot to {save_path}")


def compute_conformal_bands(
    scores: torch.Tensor,        # (B, T)
    valid_masks: torch.Tensor,   # (B, T), bool
    rollout_labels: torch.Tensor,# (B,)
    a_list=None,
):
    valid_masks = valid_masks.bool()

    if a_list is None:
        a_list = [1, 0.95, 0.9, 0.8, 0.75, 0.7]

    device = scores.device
    B, T = scores.shape
    A = len(a_list)

    pos_idx = rollout_labels == 1
    scores_pos = scores[pos_idx]      # (B_pos, T)
    masks_pos  = valid_masks[pos_idx] # (B_pos, T)

    bands = torch.empty((A, T), device=device)

    qs = torch.tensor(a_list, device=device)

    for t in range(T):
        valid_t = masks_pos[:, t]
        values = scores_pos[valid_t, t]   # (N_t,)

        # 新增：显式丢弃 NaN
        values = values[~torch.isnan(values)]

        print(values)

        if values.numel() == 0:
            bands[:, t] = torch.nan
            continue

        bands[:, t] = torch.quantile(values, qs)

    # max_true = valid_masks.sum(dim=1).max().item()
    # print(f"max_true: {max_true}")

    bands[torch.isnan(bands)] = 0.5

    return bands, a_list


def eval_model_and_log_plus(
    cfg: Config,
    model: BaseModel, 
    rollouts_by_split_name: dict[str, list[Rollout]],
    dataloader_by_split_name: dict[str, DataLoader],
    log_classification_metrics: bool = True,
):
    to_be_logged = {}
    method_name = "model"
    
    #### Forward the model and compute the scores ####
    scores_by_split_name = {}
    step_scores_by_split_name = {}  # 新增：存储时间步级数据
    step_labels_by_split_name = {}  # 新增：存储时间步级标签

    for split, dataloader in dataloader_by_split_name.items():
        # Re-create a dataset to disable shuffling
        dataloader = DataLoader(dataloader.dataset, batch_size=cfg.model.batch_size, shuffle=False, num_workers=0)

        with torch.no_grad():
            # 修改：接收 step_labels
            scores, valid_masks, rollout_labels, step_labels, mp4_paths = model_forward_dataloader(model, dataloader)
        
        print("111111111")
        print(len(mp4_paths))

        if split == "val_seen":
            # 确保数据在CPU上且是numpy数组
            scores_np = scores.cpu().numpy() if torch.is_tensor(scores) else scores
            step_labels_np = step_labels.cpu().numpy() if torch.is_tensor(step_labels) else step_labels

            print(f"scores shape: {scores.shape}")
            print(f"scores_np shape: {scores_np.shape}")
            print(f"valid_masks shape: {valid_masks.shape}")
            print(f"rollout_labels shape: {rollout_labels.shape}")
            print(f"step_labels shape: {step_labels.shape}")
            print(f"step_labels_np shape: {step_labels_np.shape}")
            
            # 评估异常检测
            print(f"\n正在评估 {split} 数据集...")
            stats, error_samples = evaluate_anomaly_detection_plus(
                scores=scores_np,
                step_labels=step_labels_np,
                mp4_paths=mp4_paths,
                threshold=0.5,
                consecutive_count=3
            )
            
            # 打印结果
            print_evaluation_results(stats, error_samples)
            
            # 可以保存结果到wandb
            to_be_logged.update({
                f"{split}_anomaly_acc": stats["acc"],
                f"{split}_anomaly_tpr": stats["tpr"],
                f"{split}_anomaly_tnr": stats["tnr"],
                f"{split}_anomaly_fpr": stats["fpr"],
            })

            bands, a_list = compute_conformal_bands(scores, valid_masks, rollout_labels)
            print(f"bands: {bands}")
            print(f"bands shape: {bands.shape}")

        if split == "val_unseen":
            # 确保数据在CPU上且是numpy数组
            scores_np = scores.cpu().numpy() if torch.is_tensor(scores) else scores
            step_labels_np = step_labels.cpu().numpy() if torch.is_tensor(step_labels) else step_labels
            
            print(f"scores shape: {scores.shape}")
            print(f"scores_np shape: {scores_np.shape}")
            print(f"valid_masks shape: {valid_masks.shape}")
            print(f"rollout_labels shape: {rollout_labels.shape}")
            print(f"step_labels shape: {step_labels.shape}")
            print(f"step_labels_np shape: {step_labels_np.shape}")

            # 评估异常检测
            print(f"\n正在评估 {split} 数据集...")
            stats, error_samples = evaluate_anomaly_detection_plus(
                scores=scores_np,
                step_labels=step_labels_np,
                mp4_paths=mp4_paths,
                threshold=0.5,
                consecutive_count=3
            )

            summary = evaluate_anomaly_detection_plus_cp(
                scores=scores_np,
                step_labels=step_labels_np,
                mp4_paths=mp4_paths,
                bands=bands,
                a_list=a_list,
                consecutive_count=1
            )
            
            # 打印结果
            print_evaluation_results(stats, error_samples)
            print_evaluation_results_cp(summary)
            
            # 可以保存结果到wandb
            to_be_logged.update({
                f"{split}_anomaly_acc": stats["acc"],
                f"{split}_anomaly_tpr": stats["tpr"],
                f"{split}_anomaly_tnr": stats["tnr"],
                f"{split}_anomaly_fpr": stats["fpr"],
            })

    return to_be_logged


def evaluate_anomaly_detection_plus(scores, step_labels, mp4_paths, threshold=0.5, consecutive_count=2):
    B = scores.shape[0]
    mp4_paths = mp4_paths[0] if len(mp4_paths) == 1 and isinstance(mp4_paths[0], list) else mp4_paths
    
    detect_times = np.full(B, -1.0)
    fail_times = np.full(B, -1.0)
    detect_errors = np.full(B, -1.0)  # 新增：存储每个样本的检测误差
    true_labels = []
    pred_labels = []
    error_samples = {"fp": [], "fn": []}
    
    for i in range(B):
        seq_scores = scores[i]
        seq_labels = step_labels[i]

        valid_length = len([x for x in seq_labels if x != -1])
        
        # 真实标签和失败时间
        true_label = 1 if np.any(seq_labels > 0.01) else 0
        true_labels.append(true_label)
        
        if true_label == 1:
            fail_indices = np.where(seq_labels > 0.45)[0]
            if len(fail_indices) == 0:
                fail_indices = np.where(seq_labels > np.mean(seq_labels))[0] # 重点关注！！！
            first_fail_time = fail_indices[0] / valid_length
            fail_times[i] = first_fail_time
        
        # 预测标签和检测时间
        pred_label = 0
        relative_time = -1.0
        for t in range(valid_length - consecutive_count + 1):
            if all(seq_scores[t + k] > threshold for k in range(consecutive_count)):
                pred_label = 1
                relative_time = t / valid_length
                detect_times[i] = relative_time
                break
        pred_labels.append(pred_label)
        
        # 计算并存储检测误差
        if true_label == 1 and pred_label == 1:
            detect_errors[i] = relative_time - fail_times[i]  # 正数表示检测晚了
        
        # 记录错误样本
        if pred_label == 1 and true_label == 0:
            error_samples["fp"].append({
                "index": i,
                "mp4_path": mp4_paths[i],
                "scores": seq_scores.tolist(),
                "detect_time": relative_time
            })
        elif pred_label == 0 and true_label == 1:
            error_samples["fn"].append({
                "index": i,
                "mp4_path": mp4_paths[i],
                "scores": seq_scores.tolist(),
                "labels": seq_labels.tolist(),
                "fail_time": fail_times[i]
            })
    
    # 计算指标
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)
    
    tp = np.sum((pred_labels == 1) & (true_labels == 1))
    tn = np.sum((pred_labels == 0) & (true_labels == 0))
    fp = np.sum((pred_labels == 1) & (true_labels == 0))
    fn = np.sum((pred_labels == 0) & (true_labels == 1))
    
    with np.errstate(divide='ignore', invalid='ignore'):
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        acc = (tp + tn) / B if B > 0 else 0.0
    
    # 检测时间误差统计
    valid_errors = detect_errors[detect_errors != -1]
    detect_time_stats = {}
    if len(valid_errors) > 0:
        detect_time_stats = {
            "n_valid_detections": len(valid_errors),
            "mean_abs_detect_error": float(np.mean(np.abs(valid_errors))),  # 绝对值平均
            "mean_raw_detect_error": float(np.mean(valid_errors)),          # 原始平均
            "std_detect_error": float(np.std(valid_errors)),
            "min_detect_error": float(np.min(valid_errors)),
            "max_detect_error": float(np.max(valid_errors)),
            "median_detect_error": float(np.median(valid_errors)),
        }
    else:
        detect_time_stats = {
            "n_valid_detections": 0,
            "mean_abs_detect_error": 0.0,
            "mean_raw_detect_error": 0.0,
            "std_detect_error": 0.0,
            "min_detect_error": 0.0,
            "max_detect_error": 0.0,
            "median_detect_error": 0.0,
        }
    
    stats = {
        "total_samples": B,
        "true_anomalies": np.sum(true_labels == 1),
        "true_normal": np.sum(true_labels == 0),
        "detected_anomalies": np.sum(pred_labels == 1),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "tpr": float(tpr), "tnr": float(tnr), "fpr": float(fpr), "fnr": float(fnr), "acc": float(acc),
        "detect_times": detect_times.tolist(),
        "fail_times": fail_times.tolist(),
        "detect_errors": detect_errors.tolist(),  # 新增：返回每个样本的检测误差
        **detect_time_stats
    }
    
    return stats, error_samples


def evaluate_anomaly_detection_plus_cp(scores, step_labels, mp4_paths, bands, a_list, consecutive_count=2):
    """
    使用多条评判条带（bands）评估异常检测性能
    
    参数:
        scores: 形状为[B, T]的异常得分序列
        step_labels: 形状为[B, T]的真实标签序列
        mp4_paths: 样本的视频文件路径
        bands: 形状为[N, T]的评判条带
        a_list: 形状为[N]的条带标识列表
        consecutive_count: 连续帧数要求
    """
    B = scores.shape[0]
    N = bands.shape[0]  # 条带数量
    
    mp4_paths = mp4_paths[0] if len(mp4_paths) == 1 and isinstance(mp4_paths[0], list) else mp4_paths
    
    # 存储所有条带的评估结果
    all_stats = []
    all_error_samples = []
    
    for n in range(N):
        band = bands[n]
        a_value = a_list[n]
        
        detect_times = np.full(B, -1.0)
        fail_times = np.full(B, -1.0)
        detect_errors = np.full(B, -1.0)
        true_labels = []
        pred_labels = []
        error_samples = {"fp": [], "fn": []}
        
        for i in range(B):
            seq_scores = scores[i]
            seq_labels = step_labels[i]

            valid_length = len([x for x in seq_labels if x != -1])
            
            # 真实标签和失败时间（与原始代码相同）
            true_label = 1 if np.any(seq_labels > 0.01) else 0
            true_labels.append(true_label)
            
            if true_label == 1:
                fail_indices = np.where(seq_labels > 0.45)[0]
                if len(fail_indices) == 0:
                    fail_indices = np.where(seq_labels > np.mean(seq_labels))[0]
                first_fail_time = fail_indices[0] / valid_length
                fail_times[i] = first_fail_time
            
            # 预测标签和检测时间（修改部分：使用band而不是固定threshold）
            pred_label = 0
            relative_time = -1.0
            
            # 使用对应的band值作为动态阈值
            for t in range(valid_length - consecutive_count + 1):
                # 检查连续consecutive_count个点是否都超过对应的band值
                all_above_band = True
                for k in range(consecutive_count):
                    if seq_scores[t + k] <= band[t + k]:
                        all_above_band = False
                        break
                
                if all_above_band:
                    pred_label = 1
                    relative_time = t / valid_length
                    detect_times[i] = relative_time
                    break
            
            pred_labels.append(pred_label)
            
            # 计算并存储检测误差
            if true_label == 1 and pred_label == 1:
                detect_errors[i] = relative_time - fail_times[i]
            
            # 记录错误样本
            if pred_label == 1 and true_label == 0:
                error_samples["fp"].append({
                    "index": i,
                    "mp4_path": mp4_paths[i],
                    "scores": seq_scores.tolist(),
                    "band_values": band.tolist(),  # 新增：记录使用的band值
                    "detect_time": relative_time,
                    "a_value": a_value
                })
            elif pred_label == 0 and true_label == 1:
                error_samples["fn"].append({
                    "index": i,
                    "mp4_path": mp4_paths[i],
                    "scores": seq_scores.tolist(),
                    "band_values": band.tolist(),  # 新增：记录使用的band值
                    "labels": seq_labels.tolist(),
                    "fail_time": fail_times[i],
                    "a_value": a_value
                })
        
        # 计算指标（与原始代码相同）
        true_labels = np.array(true_labels)
        pred_labels = np.array(pred_labels)
        
        tp = np.sum((pred_labels == 1) & (true_labels == 1))
        tn = np.sum((pred_labels == 0) & (true_labels == 0))
        fp = np.sum((pred_labels == 1) & (true_labels == 0))
        fn = np.sum((pred_labels == 0) & (true_labels == 1))
        
        with np.errstate(divide='ignore', invalid='ignore'):
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
            acc = (tp + tn) / B if B > 0 else 0.0
        
        # 检测时间误差统计
        valid_errors = detect_errors[detect_errors != -1]
        detect_time_stats = {}
        if len(valid_errors) > 0:
            detect_time_stats = {
                "n_valid_detections": len(valid_errors),
                "mean_abs_detect_error": float(np.mean(np.abs(valid_errors))),
                "mean_raw_detect_error": float(np.mean(valid_errors)),
                "std_detect_error": float(np.std(valid_errors)),
                "min_detect_error": float(np.min(valid_errors)),
                "max_detect_error": float(np.max(valid_errors)),
                "median_detect_error": float(np.median(valid_errors)),
            }
        else:
            detect_time_stats = {
                "n_valid_detections": 0,
                "mean_abs_detect_error": 0.0,
                "mean_raw_detect_error": 0.0,
                "std_detect_error": 0.0,
                "min_detect_error": 0.0,
                "max_detect_error": 0.0,
                "median_detect_error": 0.0,
            }
        
        stats = {
            "a_value": a_value,
            "total_samples": B,
            "true_anomalies": np.sum(true_labels == 1),
            "true_normal": np.sum(true_labels == 0),
            "detected_anomalies": np.sum(pred_labels == 1),
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
            "tpr": float(tpr), "tnr": float(tnr), "fpr": float(fpr), "fnr": float(fnr), "acc": float(acc),
            "detect_times": detect_times.tolist(),
            "fail_times": fail_times.tolist(),
            "detect_errors": detect_errors.tolist(),
            **detect_time_stats
        }
        
        all_stats.append(stats)
        all_error_samples.append({
            "a_value": a_value,
            "fp_samples": error_samples["fp"],
            "fn_samples": error_samples["fn"]
        })
    
    # 汇总结果
    summary = {
        "all_stats": all_stats,  # 包含N个条带的统计结果
        "all_error_samples": all_error_samples,  # 包含N个条带的错误样本
        "n_bands": N,
        "a_list": a_list.tolist() if hasattr(a_list, 'tolist') else list(a_list)
    }
    
    return summary


def print_evaluation_results_cp(summary, max_error_samples=3):
    """
    打印多条带评估结果
    
    参数:
        summary: evaluate_anomaly_detection_plus_cp 返回的汇总结果
        max_error_samples: 每个条带最多显示的误报/漏报样本数
    """
    all_stats = summary["all_stats"]
    all_error_samples = summary["all_error_samples"]
    a_list = summary["a_list"]
    n_bands = summary["n_bands"]
    
    print("\n" + "="*80)
    print("异常检测评估结果 - 多条带分析")
    print(f"条带数量: {n_bands}")
    print("="*80)
    
    # 汇总表格
    print("\n" + "-"*80)
    print(f"{'条带ID':<8} {'a值':<10} {'ACC':<8} {'TPR':<8} {'TNR':<8} {'FPR':<8} {'检测数':<8} {'平均误差':<10}")
    print("-"*80)
    
    for i, stats in enumerate(all_stats):
        a_value = stats["a_value"]
        n_detected = stats["n_valid_detections"]
        mean_error = stats["mean_abs_detect_error"] if n_detected > 0 else 0.0
        
        print(f"{i:<8} {a_value:<10.4f} {stats['acc']:<8.4f} {stats['tpr']:<8.4f} "
              f"{stats['tnr']:<8.4f} {stats['fpr']:<8.4f} {n_detected:<8} {mean_error:<10.4f}")
    
    print("-"*80)
    
    # 详细打印每个条带的结果
    for i, (stats, error_info) in enumerate(zip(all_stats, all_error_samples)):
        a_value = stats["a_value"]
        error_samples = {"fp": error_info["fp_samples"], "fn": error_info["fn_samples"]}
        
        print(f"\n\n{'+'*60}")
        print(f"条带 {i} (a={a_value:.4f}) 详细结果")
        print(f"{'+'*60}")
        
        print(f"\n数据集统计:")
        print(f"  总样本数: {stats['total_samples']}")
        print(f"  真实异常: {stats['true_anomalies']}")
        print(f"  真实正常: {stats['true_normal']}")
        print(f"  检测为异常: {stats['detected_anomalies']}")
        
        print(f"\n混淆矩阵:")
        print(f"         预测正常  预测异常")
        print(f"实际正常  {stats['tn']:>6}   {stats['fp']:>6}")
        print(f"实际异常  {stats['fn']:>6}   {stats['tp']:>6}")
        
        print(f"\n评估指标:")
        print(f"  准确率 (ACC): {stats['acc']:.4f}")
        print(f"  召回率 (TPR): {stats['tpr']:.4f}")
        print(f"  特异度 (TNR): {stats['tnr']:.4f}")
        print(f"  误报率 (FPR): {stats['fpr']:.4f}")
        print(f"  漏报率 (FNR): {stats['fnr']:.4f}")
        
        # 检测时间误差统计
        if stats['n_valid_detections'] > 0:
            print(f"\n检测时间误差统计 ({stats['n_valid_detections']}个正确检测):")
            print(f"  平均绝对值误差: {stats['mean_abs_detect_error']:.4f}")
            print(f"  原始平均误差: {stats['mean_raw_detect_error']:.4f} "
                  f"(正=检测晚，负=检测早)")
            print(f"  标准差: {stats['std_detect_error']:.4f}")
            print(f"  最小误差: {stats['min_detect_error']:.4f}")
            print(f"  最大误差: {stats['max_detect_error']:.4f}")
            print(f"  中位数误差: {stats['median_detect_error']:.4f}")
        else:
            print(f"\n检测时间误差: 无正确检测的异常样本")
        
        # 显示部分检测误差和时间信息（可选）
        if stats['n_valid_detections'] > 0:
            valid_errors = [e for e in stats['detect_errors'] if e != -1]
            if len(valid_errors) <= 10:
                print(f"\n检测误差列表: {[f'{e:.3f}' for e in valid_errors]}")
        
        # 打印错误样本
        if error_samples["fp"]:
            print(f"\n误报样本 (FP, {len(error_samples['fp'])}个):")
            for j, sample in enumerate(error_samples["fp"][:max_error_samples]):
                print(f"  [{j}] {sample['mp4_path']}")
                print(f"      检测时间: {sample['detect_time']:.3f}")
                print(f"      a值: {sample['a_value']:.4f}")
                # 显示前20个得分和对应的band值
                scores_preview = sample['scores'][:20]
                band_preview = sample['band_values'][:20]
                score_str = ' '.join([f"{s:.2f}" for s in scores_preview])
                band_str = ' '.join([f"{b:.2f}" for b in band_preview])
                if len(sample['scores']) > 20:
                    score_str += " ..."
                    band_str += " ..."
                print(f"      分数: {score_str}")
                print(f"      Band: {band_str}")
        
        if error_samples["fn"]:
            print(f"\n漏报样本 (FN, {len(error_samples['fn'])}个):")
            for j, sample in enumerate(error_samples["fn"][:max_error_samples]):
                print(f"  [{j}] {sample['mp4_path']}")
                if 'fail_time' in sample:
                    print(f"      真实失败时间: {sample['fail_time']:.3f}")
                print(f"      a值: {sample['a_value']:.4f}")
                # 显示前20个标签、得分和band值
                labels_preview = sample['labels'][:20]
                scores_preview = sample['scores'][:20]
                band_preview = sample['band_values'][:20]
                
                labels_str = ' '.join([f"{int(l)}" for l in labels_preview])
                scores_str = ' '.join([f"{s:.2f}" for s in scores_preview])
                band_str = ' '.join([f"{b:.2f}" for b in band_preview])
                
                if len(sample['labels']) > 20:
                    labels_str += " ..."
                    scores_str += " ..."
                    band_str += " ..."
                    
                print(f"      标签: {labels_str}")
                print(f"      分数: {scores_str}")
                print(f"      Band: {band_str}")
    
    print("\n" + "="*80)
    print("评估完成")
    print("="*80)


def eval_model_and_log_plus_mlp(
    cfg: Config,
    model: BaseModel, 
    rollouts_by_split_name: dict[str, list[Rollout]],
    dataloader_by_split_name: dict[str, DataLoader],
    log_classification_metrics: bool = True,
):
    to_be_logged = {}
    method_name = "model"
    
    #### Forward the model and compute the scores ####
    scores_by_split_name = {}
    step_scores_by_split_name = {}  # 新增：存储时间步级数据
    step_labels_by_split_name = {}  # 新增：存储时间步级标签

    for split, dataloader in dataloader_by_split_name.items():
        # Re-create a dataset to disable shuffling
        dataloader = DataLoader(dataloader.dataset, batch_size=cfg.model.batch_size, shuffle=False, num_workers=0)

        with torch.no_grad():
            # 修改：接收 step_labels
            scores, valid_masks, rollout_labels, step_labels, mp4_paths = model_forward_dataloader(model, dataloader)
        
        print("111111111")
        print(len(mp4_paths))

        if split == "val_seen":
            # 确保数据在CPU上且是numpy数组
            scores_np = scores.cpu().numpy() if torch.is_tensor(scores) else scores
            step_labels_np = step_labels.cpu().numpy() if torch.is_tensor(step_labels) else step_labels

            print(f"scores shape: {scores.shape}")
            print(f"scores_np shape: {scores_np.shape}")
            print(f"valid_masks shape: {valid_masks.shape}")
            print(f"rollout_labels shape: {rollout_labels.shape}")
            print(f"step_labels shape: {step_labels.shape}")
            print(f"step_labels_np shape: {step_labels_np.shape}")
            
            # 生成条带
            print(f"\n正在生成 {split} 条带...")

            bands, a_list = compute_conformal_bands(scores, valid_masks, rollout_labels)
            print(f"bands: {bands}")
            print(f"bands shape: {bands.shape}")

        if split == "val_unseen":
            # 确保数据在CPU上且是numpy数组
            scores_np = scores.cpu().numpy() if torch.is_tensor(scores) else scores
            step_labels_np = step_labels.cpu().numpy() if torch.is_tensor(step_labels) else step_labels
            
            print(f"scores shape: {scores.shape}")
            print(f"scores_np shape: {scores_np.shape}")
            print(f"valid_masks shape: {valid_masks.shape}")
            print(f"rollout_labels shape: {rollout_labels.shape}")
            print(f"step_labels shape: {step_labels.shape}")
            print(f"step_labels_np shape: {step_labels_np.shape}")

            # 评估异常检测
            print(f"\n正在评估 {split} 数据集...")

            summary = evaluate_anomaly_detection_plus_cp(
                scores=scores_np,
                step_labels=step_labels_np,
                mp4_paths=mp4_paths,
                bands=bands,
                a_list=a_list,
                consecutive_count=1
            )
            
            # 打印结果
            print_evaluation_results_cp(summary)

    return to_be_logged