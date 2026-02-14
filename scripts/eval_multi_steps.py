import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Optional

# OpenPI imports
from openpi.training import config as _config
from openpi.policies import policy_config

# ==========================================
# 配置区域
# ==========================================

# 模型配置
CONFIG_KEY = "pi05_lora_tactile"
CHECKPOINT_DIR = "/home/ps/VTLA/VTLA/checkpoints/pi05_lora_tactile/ACTION_EXPERT_INPUT/19999"

# 数据集配置
DATASET_ROOT = "/home/ps/dataset/VTLA/Pick_and_Place_two_Tennis_Balls"
TEST_EPISODE_PATH = "/home/ps/dataset/VTLA/Pick_and_Place_two_Tennis_Balls/data/chunk-000/episode_000162.parquet"
OUTPUT_DIR = "/home/ps/VTLA/eval_results"

CAMERA_MAPPING = {
    "camera_top": "camera_top", 
}

# 执行步长: 一次推理后，连续执行多少步
EXECUTION_HORIZON = 30 

# ==========================================
# 1. 绘图工具函数 (统一颜色版)
# ==========================================

def plot_chunk_execution(
    gt_array: np.ndarray,      # shape: [T_total, Dim]
    pred_chunks: List[Dict],   # List of {t: start_time, data: [Horizon, Dim]}
    out_path: str,
    title: str = None,
):
    """
    绘制分段执行的对比图。
    GT 是黑色实线。
    预测是橙色实线。
    """
    dims = gt_array.shape[1]
    
    if dims == 8:
        dim_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'joint_7', 'gripper']
    else:
        dim_names = [f"Dim {i}" for i in range(dims)]

    fig, axes = plt.subplots(dims, 1, sharex=True, figsize=(16, 4 * dims))
    if dims == 1: axes = [axes]
    
    total_steps = len(gt_array)
    x_gt = range(total_steps)

    # --- 颜色设置 ---
    GT_COLOR = 'black'
    PRED_COLOR = 'tab:orange'
    # ----------------

    for d, ax in enumerate(axes):
        # 1. 画 Ground Truth
        ax.plot(x_gt, gt_array[:, d], label="Ground Truth", color=GT_COLOR, linewidth=2.5, alpha=0.5)
        
        # 2. 画分段预测
        for idx, chunk in enumerate(pred_chunks):
            t_start = chunk['t']
            chunk_data = chunk['data'] # shape [30, Dim]
            
            # 计算时间轴
            t_end = min(t_start + len(chunk_data), total_steps)
            valid_len = t_end - t_start
            
            x_pred = range(t_start, t_end)
            y_pred = chunk_data[:valid_len, d]
            
            # 统一使用橙色 (PRED_COLOR)
            ax.plot(x_pred, y_pred, linestyle="-", linewidth=2.0, alpha=0.9, color=PRED_COLOR)
            
            # 在推理起点画个小点，标记接缝处
            ax.scatter(t_start, y_pred[0], s=20, color=PRED_COLOR, marker='o', zorder=5)
            
            # 画竖虚线标记推理时刻 (仅在第一个图画，且稍微淡一点)
            if d == 0:
                ax.axvline(x=t_start, color='gray', linestyle=':', alpha=0.2)

        ax.set_ylabel(dim_names[d], fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        if d == 0:
            from matplotlib.lines import Line2D
            custom_lines = [
                Line2D([0], [0], color=GT_COLOR, lw=2.5, alpha=0.5),
                Line2D([0], [0], color=PRED_COLOR, lw=2.0),
                Line2D([0], [0], marker='o', color=PRED_COLOR, linestyle='None')
            ]
            ax.legend(custom_lines, ['Ground Truth', 'Executed Prediction', 'Inference Point'], loc="upper right")

    axes[-1].set_xlabel("Timestep")
    if title:
        fig.suptitle(title, y=0.995, fontsize=16)

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"已保存分段预测图: {out_path}")

# ==========================================
# 2. 视频读取工具
# ==========================================
def get_frames_from_video(video_path: str, timestamps: List[float], tolerance_s=0.1):
    if not os.path.exists(video_path):
        print(f"警告: 视频文件不存在 {video_path}")
        return [np.zeros((224, 224, 3), dtype=np.uint8) for _ in timestamps]

    try:
        torchvision.set_video_backend("pyav")
    except:
        pass
        
    reader = torchvision.io.VideoReader(str(video_path), "video")
    frames = []
    current_idx = 0
    target_ts = timestamps[current_idx]
    
    for frame in reader:
        pts = frame['pts']
        if abs(pts - target_ts) < tolerance_s:
            img_tensor = frame['data'] 
            img_np = img_tensor.permute(1, 2, 0).numpy()
            frames.append(img_np)
            current_idx += 1
            if current_idx >= len(timestamps):
                break
            target_ts = timestamps[current_idx]
            
    while len(frames) < len(timestamps):
        frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))
        
    return frames

# ==========================================
# 3. 评估主逻辑 (分段执行逻辑)
# ==========================================

def evaluate_episode(policy, parquet_path, dataset_root, output_dir):
    path_obj = Path(parquet_path)
    chunk_name = path_obj.parent.name
    episode_name = path_obj.name
    
    print(f"正在处理: {episode_name}")
    
    # 1. 读取 Parquet
    try:
        df = pd.read_parquet(parquet_path, engine="pyarrow")
        if df.isna().any().any():
            df = df.dropna()
        if 'timestamp' in df.columns:
            df['timestamp'] = df['timestamp'].astype(float)
        timestamps = df['timestamp'].tolist()
    except Exception as e:
        print(f"读取失败: {e}")
        return

    # 2. 读取视频
    video_frames_map = {}
    video_root_base = os.path.join(dataset_root, "videos", chunk_name)
    video_filename = episode_name.replace(".parquet", ".mp4")

    print("加载视频帧...")
    for model_key, dataset_suffix in CAMERA_MAPPING.items():
        video_dir_name = f"observation.images.{dataset_suffix}"
        video_path = os.path.join(video_root_base, video_dir_name, video_filename)
        if not os.path.exists(video_path):
             video_path = os.path.join(dataset_root, "videos", video_dir_name, video_filename)
        frames = get_frames_from_video(video_path, timestamps)
        video_frames_map[model_key] = frames

    # 3. 推理循环
    gt_actions_list = []
    pred_chunks_to_plot = []
    action_buffer = [] 
    
    print(f"开始推理 (策略: 每 {EXECUTION_HORIZON} 步推理一次)...")
    
    records = df.to_dict("records")
    for i, row in tqdm(enumerate(records), total=len(records)):
        
        gt_action = row['action'] if 'action' in row else row.get('actions')
        if gt_action is None: continue
        gt_actions_list.append(gt_action)
        
        if len(action_buffer) == 0:
            current_images = {}
            for key, frames in video_frames_map.items():
                current_images[key] = frames[i]
                
            state = np.array(row['observation.state'], dtype=np.float32)
            
            tactile = None
            if 'observation.tactile' in row and row['observation.tactile'] is not None:
                 raw_tactile = row['observation.tactile']
                 if not (isinstance(raw_tactile, float) and np.isnan(raw_tactile)):
                     tactile = np.array(raw_tactile, dtype=np.float32)
                     if tactile.size == 65: tactile = tactile.reshape(5, 13)

            example = {
                "images": current_images,
                "state": state,
                "tactile": tactile,
                "prompt": "Pick and place two green tennis balls into the bowl",
            }
            
            with torch.no_grad():
                result = policy.infer(example)
            
            action_chunk = result["actions"]
            if isinstance(action_chunk, torch.Tensor):
                action_chunk = action_chunk.cpu().numpy()
            else:
                action_chunk = np.array(action_chunk)
            
            valid_chunk = action_chunk[:EXECUTION_HORIZON]
            action_buffer = [valid_chunk[k] for k in range(len(valid_chunk))]
            
            pred_chunks_to_plot.append({
                't': i,
                'data': valid_chunk
            })

        if len(action_buffer) > 0:
            _ = action_buffer.pop(0) 

    # 4. 绘图
    gt_array = np.array(gt_actions_list)
    out_plot_path = os.path.join(output_dir, f"eval_exec{EXECUTION_HORIZON}_{chunk_name}_{episode_name.replace('.parquet', '.png')}")

    plot_chunk_execution(
        gt_array,
        pred_chunks_to_plot,
        out_path=out_plot_path,
        title=f"Pi0 Chunk Execution (Horizon={EXECUTION_HORIZON})"
    )

# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"正在加载配置: {CONFIG_KEY}")
    config = _config.get_config(CONFIG_KEY)
    
    print(f"正在加载 Checkpoint: {CHECKPOINT_DIR}")
    policy = policy_config.create_trained_policy(config, CHECKPOINT_DIR)
    
    evaluate_episode(policy, TEST_EPISODE_PATH, DATASET_ROOT, OUTPUT_DIR)