import argparse
import torch
from torch.utils.data import DataLoader
from eduworld.utils import (
    load_config, save_config,
    create_exp_dir, setup_logger
)
from eduworld.data import BaseKTDataset
from eduworld.models.kt import DummyKTModel
from eduworld.training import BaseTrainer


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to yaml config")
    args = parser.parse_args()

    # 1.加载配置
    cfg = load_config(args.config)
    seed = cfg["reproducibility"]["seed"]
    set_seed(seed)

    # 2.创建实验目录
    exp_dir = create_exp_dir(
        base_output_dir=cfg["output"]["base_output_dir"],
        exp_name=cfg["output"]["exp_name"],
        seed=seed
    )
    # 保存本次运行的配置
    save_config(cfg, f"{exp_dir}/config.yaml")

    # 3.设置日志
    log_path = f"{exp_dir}/logs/train.log"
    setup_logger(log_path)

    # 4.加载伪造数据集与dataloader
    train_dataset = BaseKTDataset(cfg)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True
    )

    # 5.实例化dummy模型
    model = DummyKTModel(cfg)

    # 6.初始化训练器并执行
    trainer = BaseTrainer(model, train_loader, cfg, exp_dir)
    trainer.train()


if __name__ == "__main__":
    main()
