"""
dataset.py
Week2：对接苏浩天 EduWorldBench_v1
支持两种模式：
1. use_dummy=True：生成伪造假数据，冒烟测试
2. use_dummy=False：调用 load_eduworldbench.load()读取ednet/junyi真实数据集
"""
import sys
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# 项目根目录
BASE_DIR = Path(__file__).parent.resolve()
BENCH_DIR = BASE_DIR / "EduWorldBench_v1"
sys.path.insert(0, str(BASE_DIR))

# 先覆盖苏浩天原来写死的ROOT
import EduWorldBench_v1.load_eduworldbench as bench_mod
bench_mod.ROOT = BENCH_DIR
bench_mod.PATHS = {
    "ednet": bench_mod.ROOT / "ednet" / "interactions_with_splits.parquet",
    "junyi": bench_mod.ROOT / "junyi" / "interactions_with_splits.parquet",
}


class SimpleDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def get_dataloader(data_path: str, batch_size: int, use_dummy: bool = False):
    if use_dummy:
        print("===== 【测试模式】使用 dummy 假数据 =====")
        dummy_samples = [
            {"action_type": "Problem"},
            {"action_type": "Hint"},
            {"action_type": "Review"},
            {"action_type": "Explanation"},
            {"action_type": "Unknown"},
            {"action_type": "BadType"}
        ] * 10
        dataset = SimpleDataset(dummy_samples)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return loader

    else:
        print("===== 加载 EduWorldBench_v1 真实数据集 =====")
        # 示例：加载 junyi，seen_student 的 train 划分，你后面可以改成ednet
        rel = bench_mod.load(dataset="junyi", split_type="seen_student", split="train")
        df = rel.df()
        # 转成字典列表，对齐我们的接口
        samples = df.to_dict("records")
        dataset = SimpleDataset(samples)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return loader


if __name__ == "__main__":
    print("---自测 dummy dataloader---")
    dl = get_dataloader("", batch_size=2, use_dummy=True)
    for batch in dl:
        print(batch)
        break
