import yaml
import os

def load_config(config_path: str):
    """加载yaml配置文件为python字典"""
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg

def save_config(config: dict, save_path: str):
    """把本次实验配置保存到输出目录，用于回溯"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, sort_keys=False, allow_unicode=True)
