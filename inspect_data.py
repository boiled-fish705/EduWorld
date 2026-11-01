# -*- coding: utf-8 -*-
"""诊断脚本：查看 EduWorldBench_v1 真实数据集的列名和样例"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()
BENCH_DIR = BASE_DIR / "EduWorldBench_v1"
sys.path.insert(0, str(BASE_DIR))

import EduWorldBench_v1.load_eduworldbench as bench_mod

# 1) 覆盖写死的 E 盘路径，ROOT 和 PATHS 一起改
bench_mod.ROOT = BENCH_DIR
bench_mod.PATHS = {
    "ednet": bench_mod.ROOT / "ednet" / "interactions_with_splits.parquet",
    "junyi": bench_mod.ROOT / "junyi" / "interactions_with_splits.parquet",
}

print("ROOT =", bench_mod.ROOT)
print("PATHS =", bench_mod.PATHS)

# 2) 先检查文件是否真实存在
for name, p in bench_mod.PATHS.items():
    print(name, "文件存在？", p.exists(), "->", p)

# 3) 加载 junyi train 子集，打印列名和前3行
rel = bench_mod.load(dataset="junyi", split_type="seen_student", split="train")
df = rel.df()

print("\n========== 所有列名 ==========")
print(df.columns.tolist())

print("\n========== 数据形状 ==========")
print(df.shape)

print("\n========== 前3行样例 ==========")
print(df.head(3).to_string())
