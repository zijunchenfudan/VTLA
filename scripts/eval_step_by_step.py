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
TEST_EPISODE_PATH = "/home/ps/dataset/VTLA/Pick_and_Place_two_Tennis_Balls/data/chunk-000/episode_000001.parquet"
OUTPUT_DIR = "/home/ps/VTLA/eval_results"

CAMERA_MAPPING = {
    "camera_top": "camera_top", 
}

# ==========================================
# 1. 绘图工具函数
# ==========================================

def plot_dims_compare(
    gt_list: List[List[float]],
    pred_list: List[List[float]],
    out_path: str,
    title: str = None,
):
    """绘制 Ground Truth vs Prediction 对比图"""
    gt_array = np.array(gt_list)
    pred_array = np.array(pred_list)
    
    dims = gt_array.shape[1]
    
    # 定义维度名称 (假设前3是位置, 后3是旋转, 最后是夹爪, 如果不是则用 Dim X)
    if dims == 8:
        dim_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'joint_7', 'gripper']
    else:
        dim_names = [f"Dim {i}" for i in range(dims)]

    fig, axes = plt.subplots(dims, 1, sharex=True, figsize=(12, 3 * dims))
    if dims == 1: axes = [axes] # Handle single dimension case
    
    x_values = range(len(gt_list))

    for d, ax in enumerate(axes):
        y_gt = gt_array[:, d]
        y_pr = pred_array[:, d]

        ax.plot(x_values, y_gt, label="Ground Truth", color='tab:blue', linewidth=2)
        ax.plot(x_values, y_pr, linestyle="--", label="Prediction", color='tab:orange', linewidth=2)
        ax.fill_between(x_values, y_gt, y_pr, alpha=0.1, color='red', label="Error")

        ax.set_ylabel(dim_names[d], fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 计算 MAE
        mae = np.mean(np.abs(y_gt - y_pr))
        ax.text(0.98, 0.92, f"MAE: {mae:.4f}", transform=ax.transAxes, ha="right", 
                bbox=dict(boxstyle="round", fc="white", alpha=0.8))

        if d == 0:
            ax.legend(loc="upper right")

    axes[-1].set_xlabel("Timestep")
    if title:
        fig.suptitle(title, y=0.995, fontsize=14)

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"已保存对比图: {out_path}")

# ==========================================
# 2. 视频读取工具 (保持简单有效)
# ==========================================

def get_frames_from_video(video_path: str, timestamps: List[float], tolerance_s=0.1):
    """读取视频并提取对应时间戳的帧"""
    if not os.path.exists(video_path):
        print(f"警告: 视频文件不存在 {video_path}")
        # 返回全黑图像作为 fallback
        return [np.zeros((224, 224, 3), dtype=np.uint8) for _ in timestamps]

    # 尝试使用 PyAV 后端
    try:
        torchvision.set_video_backend("pyav")
    except:
        pass
        
    reader = torchvision.io.VideoReader(str(video_path), "video")
    frames = []
    
    # 简化读取逻辑：遍历寻找
    # 注意：对于长视频，这种方式可能较慢，生产环境建议优化 Seek 逻辑
    current_idx = 0
    target_ts = timestamps[current_idx]
    
    for frame in reader:
        pts = frame['pts']
        if abs(pts - target_ts) < tolerance_s:
            # 转换为 HWC numpy array, uint8 [0, 255]
            img_tensor = frame['data'] # C, H, W
            img_np = img_tensor.permute(1, 2, 0).numpy()
            frames.append(img_np)
            
            current_idx += 1
            if current_idx >= len(timestamps):
                break
            target_ts = timestamps[current_idx]
            
    # 补齐丢失的帧
    while len(frames) < len(timestamps):
        frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))
        
    return frames

# ==========================================
# 3. 评估主逻辑
# ==========================================

def evaluate_episode(policy, parquet_path, dataset_root, output_dir):
    path_obj = Path(parquet_path)
    chunk_name = path_obj.parent.name
    episode_name = path_obj.name
    
    print(f"正在处理: {episode_name}")
    
    # 1. 读取 Parquet 数据
    df = pd.read_parquet(parquet_path, engine="pyarrow")
    timestamps = df['timestamp'].tolist()

    # 2. 读取所有摄像头的视频帧
    video_frames_map = {}
    video_root_base = os.path.join(dataset_root, "videos", chunk_name)
    video_filename = episode_name.replace(".parquet", ".mp4")

    print("正在加载视频帧...")
    for model_key, dataset_suffix in CAMERA_MAPPING.items():
        # 构造类似 .../videos/chunk-000/observation.images.image_0/episode_xxxx.mp4 的路径
        video_dir_name = f"observation.images.{dataset_suffix}"
        video_path = os.path.join(video_root_base, video_dir_name, video_filename)
        
        frames = get_frames_from_video(video_path, timestamps)
        video_frames_map[model_key] = frames

    # 3. 逐帧推理
    gt_actions = []
    pred_actions = []
    
    print("开始推理...")
    records = df.to_dict("records")

    for i, row in tqdm(enumerate(records), total=len(records)):
        # --- 构造 Ground Truth ---
        gt_action = row['action']
        
        # --- 构造 OpenPI 输入 Example ---
        current_images = {}
        for key, frames in video_frames_map.items():
            current_images[key] = frames[i]
            
        # 构造状态
        state = np.array(row['observation.state'], dtype=np.float32)
        
        # 构造触觉 (如果存在)
        tactile = None
        if 'observation.tactile' in row and row['observation.tactile'] is not None:
            # 假设 Parquet 里存的是扁平数组或者 list，转为 numpy
            tactile = np.array(row['observation.tactile'], dtype=np.float32)
        
        example = {
            "images": current_images,
            "state": state,
            "tactile": tactile,
            "prompt": "Pick and place two green tennis balls into the bowl",
        }
        
        # --- 模型推理 ---
        result = policy.infer(example)
        action_chunk = result["actions"]
        
        # action_chunk 可能是 numpy array 或 tensor
        if isinstance(action_chunk, torch.Tensor):
            action_chunk = action_chunk.cpu().numpy()
        else:
            action_chunk = np.array(action_chunk)
            
        # 取当前步的预测动作
        current_pred_action = action_chunk[0]
        
        gt_actions.append(gt_action)
        pred_actions.append(current_pred_action)

    # 4. 绘图保存
    out_plot_path = os.path.join(output_dir, f"eval_{chunk_name}_{episode_name.replace('.parquet', '.png')}")
    plot_dims_compare(
        gt_actions,
        pred_actions,
        out_plot_path,
        title=f"Pi0 Prediction vs GT"
    )

# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 加载配置
    print(f"正在加载配置: {CONFIG_KEY}")
    config = _config.get_config(CONFIG_KEY)

    # 2. 创建并加载策略
    print(f"正在加载 Checkpoint: {CHECKPOINT_DIR}")
    # 注意：policy_config.create_trained_policy 内部通常会自动处理设备 (GPU)
    policy = policy_config.create_trained_policy(config, CHECKPOINT_DIR)
    
    # 3. 执行评估
    evaluate_episode(policy, TEST_EPISODE_PATH, DATASET_ROOT, OUTPUT_DIR)