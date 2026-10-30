import os
import logging
import pandas as pd
import torch

def create_exp_dir(base_output_dir: str, exp_name: str, seed: int):
    """
    创建本次实验的唯一输出目录
    返回实验根目录路径
    """
    exp_dir = os.path.join(base_output_dir, f"{exp_name}_seed{seed}")
    sub_dirs = ["logs", "checkpoints", "predictions", "metrics"]
    for sub in sub_dirs:
        os.makedirs(os.path.join(exp_dir, sub), exist_ok=True)
    return exp_dir

def setup_logger(log_file: str):
    """同时输出到控制台和log文件"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # 清除旧handler防止重复打印
    logger.handlers.clear()

    fh = logging.FileHandler(log_file, encoding="utf-8")
    ch = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def save_metrics(metrics: dict, csv_path: str):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    pd.DataFrame([metrics]).to_csv(csv_path, index=False)

def save_predictions(pred_df, csv_path: str):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    pred_df.to_csv(csv_path, index=False)

def save_checkpoint(model, ckpt_path: str):
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)
