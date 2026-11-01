```
# W02_D_encoder_action_v0.1

## 1. 模块简介
Action Encoder 是 EduWorld 底座的公共交互动作编码组件，负责把学生交互动作类型（action）映射为固定维度的 embedding 向量，供上层知识追踪（KT）模型使用。

- 模块文件：`action_encoder.py`
- 类名：`ActionEncoder`
- 定位：独立公共组件，不耦合任何具体模型；DKT / AKT / GRU / Transformer 等模型直接 `import` 复用，不复制编码器代码。

## 2. Action 类别定义（共 5 类）

| 标准类别 | 含义 | 数据集原始取值映射 |
|---|---|---|
| Problem | 作答习题 | `answer` |
| Hint | 查看提示 | `hint` |
| Review | 复习回顾 | `review` |
| Explanation | 查看解析 | `explanation` |
| Unknown | 未知 / 无法识别 | 以上之外的所有取值 |

不在映射表内的原始 action 一律兜底归为 `Unknown`，保证程序不崩溃。

## 3. 对外接口

- 初始化：`ActionEncoder(embed_dim=16)`
- 输入：`action_text_list: List[str]`，一批动作字符串，取值为上述 5 类之一
- 输出：`torch.Tensor`，shape 为 `[batch_size, embed_dim]`

### 调用示例

```python
from action_encoder import ActionEncoder

encoder = ActionEncoder(embed_dim=16)
emb = encoder(["Problem", "Hint", "Unknown"])
# emb.shape == torch.Size([3, 16])
```

## 4. 与其他模块的关系

- `dataset.py`：读取 EduWorldBench_v1 中 `action` 字段，按下表映射为标准类别后交给编码器。
- `model_factory.py`：实例化模型时可注入同一个编码器实例。
- `run.py`：全局只初始化一次编码器，各模型复用。

### 原始字段映射表

```
answer      -> Problem
hint        -> Hint
review      -> Review
explanation -> Explanation
其他        -> Unknown
```

## 5. 依赖

- `torch`：embedding 层与张量运算
- `pandas`、`duckdb`：数据集读取（由 dataset 层使用）

## 6. 测试方式

```
# 1) 编码器单独自测
python action_encoder.py

# 2) 冒烟测试（dummy 假数据，仅验证链路，不作为正式实验结果）
python run.py --use_dummy

# 3) 真实数据测试（需本地放入 EduWorldBench_v1 数据集）
python run.py
```