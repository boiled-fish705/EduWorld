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

# 覆盖苏浩天原来写死的ROOT
import EduWorldBench_v1.load_eduworldbench as bench_mod
bench_mod.ROOT = BENCH_DIR
bench_mod.PATHS = {
    "ednet": bench_mod.ROOT / "ednet" / "interactions_with_splits.parquet",
    "junyi": bench_mod.ROOT / "junyi" / "interactions_with_splits.parquet",
}

# 真实action -> 任务规定5大类映射
ACTION_MAP = {
    "answer": "Problem",
    "hint": "Hint",
    "review": "Review",
    "explanation": "Explanation",
}

class SimpleDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        raw_act = item["action"]
        # 映射，不在字典里就归为Unknown
        mapped_act = ACTION_MAP.get(raw_act, "Unknown")
        return {"action_type": mapped_act}


def custom_collate(batch):
    """自定义collate，只返回action_type列表，规避Timestamp等异常类型"""
    action_list = [x["action_type"] for x in batch]
    return {"action_type": action_list}


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
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
        return loader

    else:
        print("===== 加载 EduWorldBench_v1 真实数据集 =====")
        rel = bench_mod.load(dataset="junyi", split_type="seen_student", split="train")
        df = rel.df()
        # 只保留我们需要的action列，减少内存占用
        df = df[["action"]].copy()
        samples = df.to_dict("records")
        dataset = SimpleDataset(samples)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
        return loader


if __name__ == "__main__":
    print("---自测 dummy dataloader---")
    dl = get_dataloader("", batch_size=2, use_dummy=True)
    for batch in dl:
        print(batch)
        break
